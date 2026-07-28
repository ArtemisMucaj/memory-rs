//! DuckDB-backed [`MemoryRepository`](crate::application::MemoryRepository).
//!
//! Implemented **on [`DuckdbStore`]** rather than as a second store,
//! so memories, nodes, sessions and namespaces all share one connection to one
//! `memory.duckdb`. A separate handle would mean two writers on one file, and
//! DuckDB allows a single writer — the lock conflict that produces is a startup
//! failure the desktop app has to explain to a user.
//!
//! Storage conventions are inherited from the memory adapter rather than
//! reinvented: vectors are `FLOAT[dimensions]` literals scanned with
//! `array_cosine_distance`, read back through `to_json(...)::VARCHAR` (duckdb-rs
//! cannot fetch fixed-width arrays natively), and global scope is the **empty
//! string**, never `NULL` — SQL treats `NULL`s as distinct, which is precisely
//! how the item store once ended up holding the same memory twice.

use async_trait::async_trait;
use duckdb::{params, Row};

use crate::application::MemoryRepository;
use crate::domain::{
    DomainError, EdgeOrigin, EdgeType, Entity, EntityRef, Memory, MemoryEdge, MemoryKind,
    MemoryStatus, MemoryStoreStats, SourceKind,
};

use super::duckdb_store::{
    project_from_column, project_scope_clause, project_to_column, sql_quote, DuckdbStore,
};

/// Memory columns in DDL order. [`memory_from_row`] reads by position, so the two
/// must stay in lockstep — the indices there are annotated with these names.
const MEMORY_COLUMNS: &str = "id, kind, subject_entity_id, subject_literal, predicate, \
     object_entity_id, object_literal, statement, project, recorded_at, valid_from, valid_to, \
     source_session_id, source_message_index, source_kind, confidence, status, derived, \
     derived_from";

/// Number of columns in [`MEMORY_COLUMNS`], and therefore the position of the
/// `score` column the two search queries append after them.
///
/// Kept as a named constant rather than a literal because adding a column
/// shifts every index in [`memory_from_row`] *and* the score index in both
/// search legs — and a stale score index reads a memory field as a float instead
/// of failing, which is a silently wrong ranking rather than an error. The
/// count is checked against the string itself in the tests below.
const MEMORY_COLUMN_COUNT: usize = 19;

const ENTITY_COLUMNS: &str = "id, entity_type, canonical_name, created_at, updated_at";

/// Number of columns in [`ENTITY_COLUMNS`]; the score index for
/// `search_entities_semantic`, for the same reason as [`MEMORY_COLUMN_COUNT`].
const ENTITY_COLUMN_COUNT: usize = 5;

const EDGE_COLUMNS: &str = "from_memory, to_memory, edge_type, created_at, created_by, confidence";

/// Most query terms honoured by keyword search (mirrors the item store).
const MAX_KEYWORD_TERMS: usize = 16;

fn memory_from_row(row: &Row<'_>) -> Result<Memory, duckdb::Error> {
    // An unparseable enum falls back rather than failing the read: a row written
    // by a newer build must not make the whole store unreadable by an older one.
    let kind: String = row.get(1)?;
    let source_kind: String = row.get(14)?;
    let status: String = row.get(16)?;
    let derived_from: String = row.get(18)?;
    Ok(Memory {
        id: row.get(0)?,                                            // 0  id
        kind: MemoryKind::parse(&kind).unwrap_or(MemoryKind::Fact), // 1  kind
        subject: EntityRef::from_columns(row.get(2)?, row.get(3)?), // 2,3 subject
        predicate: row.get(4)?,                                     // 4  predicate
        object: EntityRef::from_columns(row.get(5)?, row.get(6)?),  // 5,6 object
        statement: row.get(7)?,                                     // 7  statement
        project: project_from_column(row.get::<_, String>(8)?),     // 8  project
        recorded_at: row.get(9)?,                                   // 9  recorded_at
        valid_from: row.get(10)?,                                   // 10 valid_from
        valid_to: row.get(11)?,                                     // 11 valid_to
        source_session_id: row.get(12)?,                            // 12 source_session_id
        source_message_index: row.get(13)?,                         // 13 source_message_index
        source_kind: SourceKind::parse(&source_kind).unwrap_or(SourceKind::AssistantInferred), // 14
        confidence: row.get::<_, f64>(15)? as f32,                  // 15 confidence
        status: MemoryStatus::parse(&status).unwrap_or(MemoryStatus::Active), // 16 status
        derived: row.get(17)?,                                      // 17 derived
        derived_from: serde_json::from_str(&derived_from).unwrap_or_default(), // 18 derived_from
    })
}

fn entity_from_row(row: &Row<'_>) -> Result<Entity, duckdb::Error> {
    Ok(Entity {
        id: row.get(0)?,
        entity_type: row.get(1)?,
        canonical_name: row.get(2)?,
        aliases: Vec::new(), // filled by a second query; see `load_aliases`
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
    })
}

fn edge_from_row(row: &Row<'_>) -> Result<MemoryEdge, duckdb::Error> {
    let edge_type: String = row.get(2)?;
    let created_by: String = row.get(4)?;
    Ok(MemoryEdge {
        from_memory: row.get(0)?,
        to_memory: row.get(1)?,
        edge_type: EdgeType::parse(&edge_type).unwrap_or(EdgeType::RelatesTo),
        created_at: row.get(3)?,
        created_by: EdgeOrigin::parse(&created_by).unwrap_or(EdgeOrigin::Ingestion),
        confidence: row.get::<_, f64>(5)? as f32,
    })
}

/// `IN ('a', 'b')` list from ids, or `None` when there is nothing to match.
fn id_in_list(ids: &[String]) -> Option<String> {
    if ids.is_empty() {
        return None;
    }
    Some(
        ids.iter()
            .map(|id| format!("'{}'", sql_quote(id)))
            .collect::<Vec<_>>()
            .join(", "),
    )
}

/// `AND <column> = '…'`, or empty when unfiltered. The column is passed in
/// because the semantic search aliases the table, and rewriting a finished
/// clause by string replacement would also hit any project *value* containing
/// the column's name.
fn kind_clause(column: &str, kind: Option<MemoryKind>) -> String {
    match kind {
        Some(k) => format!(" AND {column} = '{}'", k.as_str()),
        None => String::new(),
    }
}

/// `AND (<column> = '' OR <column> IN (…))`, or empty when unscoped.
fn scope_clause(column: &str, projects: Option<&[String]>) -> String {
    match project_scope_clause(column, projects) {
        Some(clause) => format!(" AND {clause}"),
        None => String::new(),
    }
}

impl DuckdbStore {
    /// Aliases for one entity, ordered for stable output.
    fn load_aliases(conn: &duckdb::Connection, id: &str) -> Result<Vec<String>, DomainError> {
        let mut stmt = conn
            .prepare("SELECT alias FROM entity_aliases WHERE entity_id = ?1 ORDER BY alias")
            .map_err(|e| DomainError::storage(format!("Failed to prepare alias query: {e}")))?;
        let rows = stmt
            .query_map(params![id], |row| row.get::<_, String>(0))
            .map_err(|e| DomainError::storage(format!("Failed to query aliases: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read aliases: {e}")))
    }
}

#[async_trait]
impl MemoryRepository for DuckdbStore {
    // ── Memories ───────────────────────────────────────────────────────────

    async fn append_memory(
        &self,
        memory: &Memory,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError> {
        let literal = vector.map(|v| self.vector_literal(v)).transpose()?;
        let derived_from = serde_json::to_string(&memory.derived_from)
            .map_err(|e| DomainError::storage(format!("Failed to encode derived_from: {e}")))?;
        let conn = self.conn.lock().await;

        // Re-appending the same id replaces the row; the ingestion path always
        // mints a fresh id, so this only fires on a deliberate rewrite.
        conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![memory.id],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear memory vector: {e}")))?;
        conn.execute("DELETE FROM memories WHERE id = ?1", params![memory.id])
            .map_err(|e| DomainError::storage(format!("Failed to clear memory: {e}")))?;

        conn.execute(
            &format!(
                "INSERT INTO memories ({MEMORY_COLUMNS}) VALUES \
                 (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)"
            ),
            params![
                memory.id,
                memory.kind.as_str(),
                memory.subject.entity_id(),
                memory.subject.literal(),
                memory.predicate,
                memory.object.entity_id(),
                memory.object.literal(),
                memory.statement,
                project_to_column(memory.project.as_deref()),
                memory.recorded_at,
                memory.valid_from,
                memory.valid_to,
                memory.source_session_id,
                memory.source_message_index,
                memory.source_kind.as_str(),
                memory.confidence as f64,
                memory.status.as_str(),
                memory.derived,
                derived_from,
            ],
        )
        .map_err(|e| DomainError::storage(format!("Failed to insert memory: {e}")))?;

        if let Some(literal) = literal {
            conn.execute(
                &format!(
                    "INSERT INTO memory_embeddings (memory_id, vector) VALUES (?1, {literal})"
                ),
                params![memory.id],
            )
            .map_err(|e| DomainError::storage(format!("Failed to insert memory vector: {e}")))?;
        }
        Ok(())
    }

    async fn find_memory(&self, id: &str) -> Result<Option<Memory>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {MEMORY_COLUMNS} FROM memories WHERE id = ?1"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare memory query: {e}")))?;
        let mut rows = stmt
            .query_map(params![id], memory_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to query memory: {e}")))?;
        match rows.next() {
            Some(row) => Ok(Some(row.map_err(|e| {
                DomainError::storage(format!("Failed to read memory: {e}"))
            })?)),
            None => Ok(None),
        }
    }

    async fn find_memories(&self, ids: &[String]) -> Result<Vec<Memory>, DomainError> {
        let Some(in_list) = id_in_list(ids) else {
            return Ok(Vec::new());
        };
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {MEMORY_COLUMNS} FROM memories WHERE id IN ({in_list})"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare memory batch: {e}")))?;
        let rows = stmt
            .query_map([], memory_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to query memory batch: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read memory batch: {e}")))
    }

    async fn list_memories(
        &self,
        kind: Option<MemoryKind>,
        status: Option<MemoryStatus>,
        projects: Option<&[String]>,
    ) -> Result<Vec<Memory>, DomainError> {
        let status_clause = match status {
            Some(s) => format!(" AND status = '{}'", s.as_str()),
            None => String::new(),
        };
        let sql = format!(
            "SELECT {MEMORY_COLUMNS} FROM memories WHERE 1 = 1{}{}{} ORDER BY recorded_at DESC",
            kind_clause("kind", kind),
            status_clause,
            scope_clause("project", projects),
        );
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| DomainError::storage(format!("Failed to prepare memory list: {e}")))?;
        let rows = stmt
            .query_map([], memory_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to list memories: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read memories: {e}")))
    }

    async fn set_memory_status(
        &self,
        id: &str,
        status: MemoryStatus,
        valid_to: Option<i64>,
    ) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        // `valid_to = None` leaves the stored value alone (see the port docs):
        // retracting a memory must not rewrite when it stopped being true.
        let updated = match valid_to {
            Some(ts) => conn.execute(
                "UPDATE memories SET status = ?1, valid_to = ?2 WHERE id = ?3",
                params![status.as_str(), ts, id],
            ),
            None => conn.execute(
                "UPDATE memories SET status = ?1 WHERE id = ?2",
                params![status.as_str(), id],
            ),
        }
        .map_err(|e| DomainError::storage(format!("Failed to set memory status: {e}")))?;
        Ok(updated > 0)
    }

    async fn set_memory_confidence(&self, id: &str, confidence: f32) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let updated = conn
            .execute(
                "UPDATE memories SET confidence = ?1 WHERE id = ?2",
                params![confidence as f64, id],
            )
            .map_err(|e| DomainError::storage(format!("Failed to set memory confidence: {e}")))?;
        Ok(updated > 0)
    }

    async fn reopen_memory(&self, id: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let updated = conn
            .execute(
                "UPDATE memories SET status = ?1, valid_to = NULL WHERE id = ?2",
                params![MemoryStatus::Active.as_str(), id],
            )
            .map_err(|e| DomainError::storage(format!("Failed to reopen memory: {e}")))?;
        Ok(updated > 0)
    }

    async fn delete_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError> {
        let conn = self.conn.lock().await;
        // Edges and vectors first, so nothing is left pointing at a memory that
        // is about to disappear.
        conn.execute(
            "DELETE FROM memory_edges WHERE from_memory IN \
             (SELECT id FROM memories WHERE source_session_id = ?1) \
                OR to_memory IN (SELECT id FROM memories WHERE source_session_id = ?1)",
            params![session_id],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear session edges: {e}")))?;
        conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id IN \
             (SELECT id FROM memories WHERE source_session_id = ?1)",
            params![session_id],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear session vectors: {e}")))?;
        let deleted = conn
            .execute(
                "DELETE FROM memories WHERE source_session_id = ?1",
                params![session_id],
            )
            .map_err(|e| DomainError::storage(format!("Failed to delete session memories: {e}")))?;
        Ok(deleted)
    }

    async fn count_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError> {
        let conn = self.conn.lock().await;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE source_session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|e| DomainError::storage(format!("Failed to count session memories: {e}")))?;
        Ok(count as usize)
    }

    // ── Edges ────────────────────────────────────────────────────────────

    async fn add_edge(&self, edge: &MemoryEdge) -> Result<(), DomainError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM memory_edges WHERE from_memory = ?1 AND to_memory = ?2 AND edge_type = ?3",
            params![edge.from_memory, edge.to_memory, edge.edge_type.as_str()],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear edge: {e}")))?;
        conn.execute(
            &format!("INSERT INTO memory_edges ({EDGE_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5, ?6)"),
            params![
                edge.from_memory,
                edge.to_memory,
                edge.edge_type.as_str(),
                edge.created_at,
                edge.created_by.as_str(),
                edge.confidence as f64,
            ],
        )
        .map_err(|e| DomainError::storage(format!("Failed to insert edge: {e}")))?;
        Ok(())
    }

    async fn edges_from(&self, memory_id: &str) -> Result<Vec<MemoryEdge>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {EDGE_COLUMNS} FROM memory_edges WHERE from_memory = ?1 ORDER BY created_at"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare edge query: {e}")))?;
        let rows = stmt
            .query_map(params![memory_id], edge_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to query edges: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read edges: {e}")))
    }

    async fn edges_to(&self, memory_id: &str) -> Result<Vec<MemoryEdge>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {EDGE_COLUMNS} FROM memory_edges WHERE to_memory = ?1 ORDER BY created_at"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare edge query: {e}")))?;
        let rows = stmt
            .query_map(params![memory_id], edge_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to query edges: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read edges: {e}")))
    }

    async fn edges_for(&self, ids: &[String]) -> Result<Vec<MemoryEdge>, DomainError> {
        let Some(in_list) = id_in_list(ids) else {
            return Ok(Vec::new());
        };
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {EDGE_COLUMNS} FROM memory_edges \
                 WHERE from_memory IN ({in_list}) OR to_memory IN ({in_list}) \
                 ORDER BY created_at"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare edge batch: {e}")))?;
        let rows = stmt
            .query_map([], edge_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to query edge batch: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read edge batch: {e}")))
    }

    async fn list_edges(
        &self,
        edge_type: Option<EdgeType>,
    ) -> Result<Vec<MemoryEdge>, DomainError> {
        let filter = match edge_type {
            Some(t) => format!(" WHERE edge_type = '{}'", t.as_str()),
            None => String::new(),
        };
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {EDGE_COLUMNS} FROM memory_edges{filter} ORDER BY created_at"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare edge list: {e}")))?;
        let rows = stmt
            .query_map([], edge_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to list edges: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read edges: {e}")))
    }

    // ── Entities ─────────────────────────────────────────────────────────

    async fn upsert_entity(
        &self,
        entity: &Entity,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError> {
        let literal = vector.map(|v| self.vector_literal(v)).transpose()?;
        let conn = self.conn.lock().await;

        conn.execute(
            "DELETE FROM entity_vectors WHERE entity_id = ?1",
            params![entity.id],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear entity vector: {e}")))?;
        conn.execute(
            "DELETE FROM entity_aliases WHERE entity_id = ?1",
            params![entity.id],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear entity aliases: {e}")))?;
        conn.execute("DELETE FROM entities WHERE id = ?1", params![entity.id])
            .map_err(|e| DomainError::storage(format!("Failed to clear entity: {e}")))?;

        conn.execute(
            &format!("INSERT INTO entities ({ENTITY_COLUMNS}) VALUES (?1, ?2, ?3, ?4, ?5)"),
            params![
                entity.id,
                entity.entity_type,
                entity.canonical_name,
                entity.created_at,
                entity.updated_at,
            ],
        )
        .map_err(|e| DomainError::storage(format!("Failed to insert entity: {e}")))?;

        // The canonical name resolves like any other alias, and duplicates are
        // folded away by the primary key rather than rejected.
        let mut seen = std::collections::HashSet::new();
        for alias in std::iter::once(&entity.canonical_name).chain(entity.aliases.iter()) {
            let alias = alias.trim();
            if alias.is_empty() || !seen.insert(alias.to_lowercase()) {
                continue;
            }
            conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?1, ?2)",
                params![alias, entity.id],
            )
            .map_err(|e| DomainError::storage(format!("Failed to insert entity alias: {e}")))?;
        }

        if let Some(literal) = literal {
            conn.execute(
                &format!("INSERT INTO entity_vectors (entity_id, vector) VALUES (?1, {literal})"),
                params![entity.id],
            )
            .map_err(|e| DomainError::storage(format!("Failed to insert entity vector: {e}")))?;
        }
        Ok(())
    }

    async fn find_entity(&self, id: &str) -> Result<Option<Entity>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {ENTITY_COLUMNS} FROM entities WHERE id = ?1"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare entity query: {e}")))?;
        let mut rows = stmt
            .query_map(params![id], entity_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to query entity: {e}")))?;
        let Some(row) = rows.next() else {
            return Ok(None);
        };
        let mut entity =
            row.map_err(|e| DomainError::storage(format!("Failed to read entity: {e}")))?;
        drop(rows);
        drop(stmt);
        entity.aliases = Self::load_aliases(&conn, id)?;
        Ok(Some(entity))
    }

    async fn find_entity_by_alias(&self, alias: &str) -> Result<Option<Entity>, DomainError> {
        let conn = self.conn.lock().await;
        let id: Option<String> = {
            let mut stmt = conn
                .prepare(
                    "SELECT entity_id FROM entity_aliases WHERE lower(alias) = lower(?1) LIMIT 1",
                )
                .map_err(|e| {
                    DomainError::storage(format!("Failed to prepare alias lookup: {e}"))
                })?;
            let mut rows = stmt
                .query_map(params![alias], |row| row.get::<_, String>(0))
                .map_err(|e| DomainError::storage(format!("Failed to query alias: {e}")))?;
            match rows.next() {
                Some(row) => Some(
                    row.map_err(|e| DomainError::storage(format!("Failed to read alias: {e}")))?,
                ),
                None => None,
            }
        };
        drop(conn);
        match id {
            Some(id) => self.find_entity(&id).await,
            None => Ok(None),
        }
    }

    async fn list_entities(&self) -> Result<Vec<Entity>, DomainError> {
        let conn = self.conn.lock().await;
        let mut entities: Vec<Entity> = {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT {ENTITY_COLUMNS} FROM entities ORDER BY created_at DESC"
                ))
                .map_err(|e| DomainError::storage(format!("Failed to prepare entity list: {e}")))?;
            let rows = stmt
                .query_map([], entity_from_row)
                .map_err(|e| DomainError::storage(format!("Failed to list entities: {e}")))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| DomainError::storage(format!("Failed to read entities: {e}")))?
        };
        for entity in &mut entities {
            entity.aliases = Self::load_aliases(&conn, &entity.id)?;
        }
        Ok(entities)
    }

    async fn search_entities_semantic(
        &self,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<(Entity, f32)>, DomainError> {
        let literal = self.vector_literal(vector)?;
        let sql = format!(
            "SELECT {ENTITY_COLUMNS}, 1.0 - array_cosine_distance(v.vector, {literal}) AS score \
             FROM entities e JOIN entity_vectors v ON v.entity_id = e.id \
             ORDER BY score DESC LIMIT {limit}"
        );
        // The scan is O(rows); it must not run on a Tokio worker thread.
        let scored: Vec<(Entity, f32)> = self
            .query_blocking(move |conn| {
                let mut stmt = conn.prepare(&sql).map_err(|e| {
                    DomainError::storage(format!("Failed to prepare entity search: {e}"))
                })?;
                let rows = stmt
                    .query_map([], |row| {
                        Ok((
                            entity_from_row(row)?,
                            row.get::<_, f64>(ENTITY_COLUMN_COUNT)? as f32,
                        ))
                    })
                    .map_err(|e| DomainError::storage(format!("Failed to search entities: {e}")))?;
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|e| DomainError::storage(format!("Failed to read entities: {e}")))
            })
            .await?;

        let conn = self.conn.lock().await;
        let mut out = Vec::with_capacity(scored.len());
        for (mut entity, score) in scored {
            entity.aliases = Self::load_aliases(&conn, &entity.id)?;
            out.push((entity, score));
        }
        Ok(out)
    }

    async fn repoint_entity(&self, from: &str, to: &str) -> Result<usize, DomainError> {
        let conn = self.conn.lock().await;
        // Both ref columns: an entity can be the subject of one memory and the
        // object of another, and a merge that moved only one would silently
        // strand half the graph.
        let subjects = conn
            .execute(
                "UPDATE memories SET subject_entity_id = ?1 WHERE subject_entity_id = ?2",
                params![to, from],
            )
            .map_err(|e| DomainError::storage(format!("Failed to repoint subjects: {e}")))?;
        let objects = conn
            .execute(
                "UPDATE memories SET object_entity_id = ?1 WHERE object_entity_id = ?2",
                params![to, from],
            )
            .map_err(|e| DomainError::storage(format!("Failed to repoint objects: {e}")))?;
        Ok(subjects + objects)
    }

    async fn delete_entity(&self, id: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM entity_vectors WHERE entity_id = ?1",
            params![id],
        )
        .map_err(|e| DomainError::storage(format!("Failed to delete entity vector: {e}")))?;
        conn.execute(
            "DELETE FROM entity_aliases WHERE entity_id = ?1",
            params![id],
        )
        .map_err(|e| DomainError::storage(format!("Failed to delete entity aliases: {e}")))?;
        let deleted = conn
            .execute("DELETE FROM entities WHERE id = ?1", params![id])
            .map_err(|e| DomainError::storage(format!("Failed to delete entity: {e}")))?;
        Ok(deleted > 0)
    }

    // ── Retrieval ────────────────────────────────────────────────────────

    async fn search_memories_semantic(
        &self,
        vector: &[f32],
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError> {
        let literal = self.vector_literal(vector)?;
        // Only `active` memories surface. Superseded, retracted and — critically —
        // Retired memories are excluded here rather than downstream, so
        // no caller can forget to filter an unsettled conflict out of recall.
        let sql = format!(
            "SELECT {}, 1.0 - array_cosine_distance(v.vector, {literal}) AS score \
             FROM memories c JOIN memory_embeddings v ON v.memory_id = c.id \
             WHERE c.status = 'active'{}{} \
             ORDER BY score DESC LIMIT {limit}",
            MEMORY_COLUMNS
                .split(", ")
                .map(|c| format!("c.{c}"))
                .collect::<Vec<_>>()
                .join(", "),
            kind_clause("c.kind", kind),
            scope_clause("c.project", projects),
        );
        self.query_blocking(move |conn| {
            let mut stmt = conn.prepare(&sql).map_err(|e| {
                DomainError::storage(format!("Failed to prepare memory search: {e}"))
            })?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        memory_from_row(row)?,
                        row.get::<_, f64>(MEMORY_COLUMN_COUNT)? as f32,
                    ))
                })
                .map_err(|e| DomainError::storage(format!("Failed to search memories: {e}")))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| DomainError::storage(format!("Failed to read memories: {e}")))
        })
        .await
    }

    async fn search_memories_keyword(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .take(MAX_KEYWORD_TERMS)
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        // Score = fraction of query terms found in the statement.
        let escape = |t: &str| {
            t.replace('\\', "\\\\")
                .replace('\'', "''")
                .replace('%', "\\%")
                .replace('_', "\\_")
        };
        let match_cases: Vec<String> = terms
            .iter()
            .map(|t| {
                let e = escape(t);
                format!("(CASE WHEN lower(statement) LIKE '%{e}%' ESCAPE '\\' THEN 1 ELSE 0 END)")
            })
            .collect();
        let score_expr = format!("({}) / {}.0", match_cases.join(" + "), terms.len());
        let sql = format!(
            "SELECT {MEMORY_COLUMNS}, {score_expr} AS score FROM memories \
             WHERE status = 'active' AND {score_expr} > 0{}{} \
             ORDER BY score DESC, recorded_at DESC LIMIT {limit}",
            kind_clause("kind", kind),
            scope_clause("project", projects),
        );
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| DomainError::storage(format!("Failed to prepare keyword search: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    memory_from_row(row)?,
                    row.get::<_, f64>(MEMORY_COLUMN_COUNT)? as f32,
                ))
            })
            .map_err(|e| DomainError::storage(format!("Failed to keyword-search memories: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read memories: {e}")))
    }

    async fn list_memory_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, DomainError> {
        // duckdb-rs cannot fetch a `FLOAT[n]` column natively, so the array is
        // rendered to JSON in SQL and parsed back here.
        self.query_blocking(|conn| {
            let mut stmt = conn
                .prepare("SELECT memory_id, to_json(vector)::VARCHAR FROM memory_embeddings")
                .map_err(|e| DomainError::storage(format!("Failed to prepare vector list: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| DomainError::storage(format!("Failed to list memory vectors: {e}")))?;
            let mut out = Vec::new();
            for row in rows {
                let (id, json) =
                    row.map_err(|e| DomainError::storage(format!("Failed to read vector: {e}")))?;
                let vector: Vec<f32> = serde_json::from_str(&json)
                    .map_err(|e| DomainError::storage(format!("Failed to parse vector: {e}")))?;
                out.push((id, vector));
            }
            Ok(out)
        })
        .await
    }

    async fn memory_stats(&self) -> Result<MemoryStoreStats, DomainError> {
        let conn = self.conn.lock().await;
        let total_memories: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|e| DomainError::storage(format!("Failed to count memories: {e}")))?;
        let total_entities: i64 = conn
            .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))
            .map_err(|e| DomainError::storage(format!("Failed to count entities: {e}")))?;
        let total_edges: i64 = conn
            .query_row("SELECT COUNT(*) FROM memory_edges", [], |row| row.get(0))
            .map_err(|e| DomainError::storage(format!("Failed to count edges: {e}")))?;

        let by = |column: &str| -> Result<Vec<(String, u64)>, DomainError> {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT {column}, COUNT(*) FROM memories GROUP BY {column} ORDER BY {column}"
                ))
                .map_err(|e| DomainError::storage(format!("Failed to prepare grouping: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
                })
                .map_err(|e| DomainError::storage(format!("Failed to group memories: {e}")))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| DomainError::storage(format!("Failed to read grouping: {e}")))
        };
        let memories_by_status = by("status")?;
        let memories_by_kind = by("kind")?;

        Ok(MemoryStoreStats {
            total_memories: total_memories as u64,
            memories_by_status,
            memories_by_kind,
            total_entities: total_entities as u64,
            total_edges: total_edges as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The column-list constants are interpolated into SQL and then read back
    /// **by position**, so they encode an invariant no type check can see: the
    /// Nth name in the string must be the Nth `row.get` in the mapper. Adding
    /// `kind` to the memory list already shifted seventeen indices once. These
    /// assertions turn the next such shift into a failing unit test rather than
    /// a mis-mapped field.
    #[test]
    fn column_counts_match_their_column_lists() {
        assert_eq!(
            MEMORY_COLUMNS.split(", ").count(),
            MEMORY_COLUMN_COUNT,
            "MEMORY_COLUMNS changed without updating MEMORY_COLUMN_COUNT — the \
             score index in both search legs is now wrong",
        );
        assert_eq!(
            ENTITY_COLUMNS.split(", ").count(),
            ENTITY_COLUMN_COUNT,
            "ENTITY_COLUMNS changed without updating ENTITY_COLUMN_COUNT",
        );
    }

    /// `MEMORY_COLUMNS` is written as a multi-line literal joined by `\`
    /// line-continuations. Rust strips the newline *and* the following
    /// indentation, but if that ever stopped holding, `split(", ")` would yield
    /// names with leading spaces — and `search_memories_semantic`, which builds
    /// its select list by prefixing each with `c.`, would emit `c. object_id`.
    /// That is a SQL error rather than a wrong answer, but it would only show
    /// up at runtime on the semantic path.
    #[test]
    fn column_lists_have_no_stray_whitespace() {
        for (label, columns) in [
            ("MEMORY_COLUMNS", MEMORY_COLUMNS),
            ("ENTITY_COLUMNS", ENTITY_COLUMNS),
            ("EDGE_COLUMNS", EDGE_COLUMNS),
        ] {
            for column in columns.split(", ") {
                assert_eq!(
                    column,
                    column.trim(),
                    "{label} column {column:?} carries whitespace",
                );
                assert!(
                    !column.is_empty(),
                    "{label} has an empty column — a doubled separator?",
                );
            }
        }
    }

    /// The memory mapper's field order is load-bearing documentation; if the
    /// list and the DDL ever disagree on *which* columns exist, every
    /// `SELECT {MEMORY_COLUMNS}` fails at once. Pin the names so a rename has to
    /// be deliberate.
    #[test]
    fn memory_columns_are_the_expected_names_in_order() {
        assert_eq!(
            MEMORY_COLUMNS.split(", ").collect::<Vec<_>>(),
            [
                "id",
                "kind",
                "subject_entity_id",
                "subject_literal",
                "predicate",
                "object_entity_id",
                "object_literal",
                "statement",
                "project",
                "recorded_at",
                "valid_from",
                "valid_to",
                "source_session_id",
                "source_message_index",
                "source_kind",
                "confidence",
                "status",
                "derived",
                "derived_from",
            ],
        );
    }
}
