//! DuckDB-backed [`MemoryRepository`](crate::application::MemoryRepository).
//!
//! Implemented on [`DuckdbStore`] rather than as a second store, so memories,
//! sessions and namespaces all share one connection to one `memory.duckdb`.
//!
//! Storage conventions: vectors are `FLOAT[dimensions]` literals scanned with
//! `array_cosine_distance`, and global scope is the **empty string**, never
//! `NULL` — SQL treats `NULL`s as distinct, which is precisely how the
//! store would otherwise end up holding the same global memory twice.
//!
//! The memory↔entity link is a join table (`memory_entities`), not a column
//! pair: a memory mentions zero or more entities, and an entity is mentioned
//! by zero or more memories.

use async_trait::async_trait;
use duckdb::{params, Connection, Row};

use crate::application::MemoryRepository;
use crate::domain::{
    entity_name_key, DomainError, Entity, ImportedSession, Memory, MemoryKind, MemoryResource,
    SessionStatus, SourceKind,
};

use super::duckdb_store::{
    project_from_column, project_scope_clause, project_to_column, DuckdbStore,
};

/// Memory columns in DDL order. [`memory_from_row`] reads by position, so
/// the two must stay in lockstep.
const MEMORY_COLUMNS: &str = "id, kind, statement, project, recorded_at, \
     source_session_id, source_message_index, source_kind, confidence";

/// Number of columns in [`MEMORY_COLUMNS`], and therefore the position of the
/// `score` column the two search queries append after them.
const MEMORY_COLUMN_COUNT: usize = 9;

const ENTITY_COLUMNS: &str = "id, entity_type, canonical_name, created_at, updated_at";

const RESOURCE_COLUMNS: &str = "uri, source, name, abstract, overview, content, created_at";
const RESOURCE_COLUMN_COUNT: usize = 7;

const SESSION_COLUMNS: &str =
    "id, source, imported_at, message_count, items_written, status, last_error, project";

/// Most query terms honoured by keyword search.
const MAX_KEYWORD_TERMS: usize = 16;

fn memory_from_row(row: &Row<'_>) -> Result<Memory, duckdb::Error> {
    // An unparseable enum falls back rather than failing the read: a row
    // written by a newer build must not make the whole store unreadable by
    // an older one.
    let kind: String = row.get(1)?;
    let source_kind: String = row.get(7)?;
    Ok(Memory {
        id: row.get(0)?,                                                               // 0 id
        kind: MemoryKind::parse(&kind).unwrap_or(MemoryKind::Fact),                    // 1 kind
        statement: row.get(2)?,                                 // 2 statement
        entity_ids: Vec::new(), // filled by a second query; see `load_entity_ids`
        project: project_from_column(row.get::<_, String>(3)?), // 3 project
        recorded_at: row.get(4)?, // 4 recorded_at
        source_session_id: row.get(5)?, // 5 source_session_id
        source_message_index: row.get(6)?, // 6 source_message_index
        source_kind: SourceKind::parse(&source_kind).unwrap_or(SourceKind::Extracted), // 7
        confidence: row.get::<_, f64>(8)? as f32, // 8 confidence
    })
}

fn entity_from_row(row: &Row<'_>) -> Result<Entity, duckdb::Error> {
    Ok(Entity {
        id: row.get(0)?,
        entity_type: row.get(1)?,
        canonical_name: row.get(2)?,
        names: Vec::new(), // filled by a second query; see `load_names`
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
    })
}

fn resource_from_row(row: &Row<'_>) -> Result<MemoryResource, duckdb::Error> {
    Ok(MemoryResource {
        uri: row.get(0)?,
        source: row.get(1)?,
        name: row.get(2)?,
        abstract_: row.get(3)?,
        overview: row.get(4)?,
        content: row.get(5)?,
        created_at: row.get(6)?,
    })
}

fn session_from_row(row: &Row<'_>) -> Result<ImportedSession, duckdb::Error> {
    let status: String = row.get(5)?;
    Ok(ImportedSession {
        id: row.get(0)?,
        source: row.get(1)?,
        imported_at: row.get(2)?,
        message_count: row.get::<_, i64>(3)? as usize,
        items_written: row.get::<_, i64>(4)? as usize,
        status: SessionStatus::parse(&status),
        last_error: row.get(6)?,
        project: row.get(7)?,
    })
}

#[async_trait]
impl MemoryRepository for DuckdbStore {
    // ── Memories ─────────────────────────────────────────────────────────

    async fn append_memory(
        &self,
        memory: &Memory,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError> {
        // Validate before opening the transaction: a wrong-width vector must
        // fail the call without having touched the store.
        if let Some(vector) = vector {
            if vector.len() != self.dimensions {
                return Err(DomainError::invalid_input(format!(
                    "vector width {} does not match pinned dimensions {}",
                    vector.len(),
                    self.dimensions
                )));
            }
        }
        let mut conn = self.conn.lock().await;
        // Atomic: a crash between the row write and the link / embedding
        // writes must not leave a memory without its entities. DuckDB's
        // `Transaction` rolls back on drop if not committed.
        let tx = conn
            .transaction()
            .map_err(|e| DomainError::storage(format!("failed to begin transaction: {e}")))?;

        tx.execute(
            &format!(
                "INSERT OR REPLACE INTO memories ({MEMORY_COLUMNS}) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            params![
                memory.id,
                memory.kind.as_str(),
                memory.statement,
                project_to_column(memory.project.as_deref()),
                memory.recorded_at,
                memory.source_session_id,
                memory.source_message_index,
                memory.source_kind.as_str(),
                memory.confidence as f64,
            ],
        )
        .map_err(|e| DomainError::storage(format!("failed to append memory: {e}")))?;

        // Replace the entity links wholesale.
        tx.execute(
            "DELETE FROM memory_entities WHERE memory_id = ?",
            params![memory.id],
        )
        .map_err(|e| DomainError::storage(format!("failed to clear memory entities: {e}")))?;
        for entity_id in &memory.entity_ids {
            tx.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                params![memory.id, entity_id],
            )
            .map_err(|e| DomainError::storage(format!("failed to link memory to entity: {e}")))?;
        }

        tx.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?",
            params![memory.id],
        )
        .map_err(|e| DomainError::storage(format!("failed to clear memory embedding: {e}")))?;
        if let Some(vector) = vector {
            let literal = vector_literal(vector);
            tx.execute(
                &format!("INSERT INTO memory_embeddings (memory_id, vector) VALUES (?, {literal})"),
                params![memory.id],
            )
            .map_err(|e| DomainError::storage(format!("failed to write memory embedding: {e}")))?;
        }

        tx.commit()
            .map_err(|e| DomainError::storage(format!("failed to commit memory write: {e}")))?;
        Ok(())
    }

    async fn find_memory(&self, id: &str) -> Result<Option<Memory>, DomainError> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {MEMORY_COLUMNS} FROM memories WHERE id = ?");
        let out = query_collect(
            &conn,
            &sql,
            &[&id as &dyn duckdb::ToSql],
            memory_from_row,
            "find memory",
        )?;
        let Some(memory) = out.into_iter().next() else {
            return Ok(None);
        };
        let mut by_memory = load_entity_ids_for(&conn, std::slice::from_ref(&memory.id))?;
        let entity_ids = by_memory.remove(&memory.id).unwrap_or_default();
        Ok(Some(Memory {
            entity_ids,
            ..memory
        }))
    }

    async fn find_memories(&self, ids: &[String]) -> Result<Vec<Memory>, DomainError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {MEMORY_COLUMNS} FROM memories WHERE id IN ({placeholders})");
        let params: Vec<&dyn duckdb::ToSql> = ids.iter().map(|s| s as &dyn duckdb::ToSql).collect();
        let out = query_collect(&conn, &sql, &params, memory_from_row, "find memories")?;
        attach_entity_ids(&conn, out)
    }

    async fn list_memories(&self, projects: Option<&[String]>) -> Result<Vec<Memory>, DomainError> {
        let mut scope_params: Vec<String> = Vec::new();
        let scope = project_scope_clause(projects, "project", &mut scope_params);
        let sql = format!(
            "SELECT {MEMORY_COLUMNS} FROM memories WHERE TRUE{scope} ORDER BY recorded_at DESC"
        );
        let conn = self.conn.lock().await;
        let params_ref: Vec<&dyn duckdb::ToSql> = scope_params
            .iter()
            .map(|s| s as &dyn duckdb::ToSql)
            .collect();
        let out = query_collect(&conn, &sql, &params_ref, memory_from_row, "list memories")?;
        attach_entity_ids(&conn, out)
    }

    async fn delete_memory(&self, id: &str) -> Result<bool, DomainError> {
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| DomainError::storage(format!("failed to begin transaction: {e}")))?;
        tx.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?",
            params![id],
        )
        .map_err(|e| DomainError::storage(format!("failed to delete memory embedding: {e}")))?;
        tx.execute(
            "DELETE FROM memory_entities WHERE memory_id = ?",
            params![id],
        )
        .map_err(|e| DomainError::storage(format!("failed to delete memory links: {e}")))?;
        let n = tx
            .execute("DELETE FROM memories WHERE id = ?", params![id])
            .map_err(|e| DomainError::storage(format!("failed to delete memory: {e}")))?;
        tx.commit()
            .map_err(|e| DomainError::storage(format!("failed to commit memory delete: {e}")))?;
        Ok(n > 0)
    }

    async fn delete_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError> {
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| DomainError::storage(format!("failed to begin transaction: {e}")))?;
        tx.execute(
            "DELETE FROM memory_embeddings WHERE memory_id IN \
             (SELECT id FROM memories WHERE source_session_id = ?)",
            params![session_id],
        )
        .map_err(|e| DomainError::storage(format!("failed to delete session embeddings: {e}")))?;
        tx.execute(
            "DELETE FROM memory_entities WHERE memory_id IN \
             (SELECT id FROM memories WHERE source_session_id = ?)",
            params![session_id],
        )
        .map_err(|e| DomainError::storage(format!("failed to delete session links: {e}")))?;
        let n = tx
            .execute(
                "DELETE FROM memories WHERE source_session_id = ?",
                params![session_id],
            )
            .map_err(|e| DomainError::storage(format!("failed to delete session memories: {e}")))?;
        tx.commit()
            .map_err(|e| DomainError::storage(format!("failed to commit session delete: {e}")))?;
        Ok(n)
    }

    async fn count_memories_for_session(&self, session_id: &str) -> Result<usize, DomainError> {
        let conn = self.conn.lock().await;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE source_session_id = ?",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|e| DomainError::storage(format!("failed to count session memories: {e}")))?;
        Ok(count as usize)
    }

    // ── Entities ─────────────────────────────────────────────────────────

    async fn upsert_entity(&self, entity: &Entity) -> Result<(), DomainError> {
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| DomainError::storage(format!("failed to begin transaction: {e}")))?;
        tx.execute(
            "INSERT OR REPLACE INTO entities (id, entity_type, canonical_name, created_at, updated_at) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                entity.id,
                entity.entity_type,
                entity.canonical_name,
                entity.created_at,
                entity.updated_at,
            ],
        )
        .map_err(|e| DomainError::storage(format!("failed to upsert entity: {e}")))?;

        // Replace the name rows wholesale: canonical + every alias.
        tx.execute(
            "DELETE FROM entity_names WHERE entity_id = ?",
            params![entity.id],
        )
        .map_err(|e| DomainError::storage(format!("failed to clear entity names: {e}")))?;
        let mut names = vec![entity.canonical_name.clone()];
        for alias in &entity.names {
            if !names.iter().any(|n| n.eq_ignore_ascii_case(alias)) {
                names.push(alias.clone());
            }
        }
        for name in names {
            tx.execute(
                "INSERT OR REPLACE INTO entity_names (name, name_key, entity_id) \
                 VALUES (?, ?, ?)",
                params![name, entity_name_key(&name), entity.id],
            )
            .map_err(|e| DomainError::storage(format!("failed to write entity name: {e}")))?;
        }
        tx.commit()
            .map_err(|e| DomainError::storage(format!("failed to commit entity write: {e}")))?;
        Ok(())
    }

    async fn find_entity(&self, id: &str) -> Result<Option<Entity>, DomainError> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {ENTITY_COLUMNS} FROM entities WHERE id = ?");
        let out = query_collect(
            &conn,
            &sql,
            &[&id as &dyn duckdb::ToSql],
            entity_from_row,
            "find entity",
        )?;
        let Some(entity) = out.into_iter().next() else {
            return Ok(None);
        };
        let names = load_names(&conn, &entity.id)?;
        Ok(Some(Entity { names, ..entity }))
    }

    async fn find_entities(&self, ids: &[String]) -> Result<Vec<Entity>, DomainError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat_n("?", ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {ENTITY_COLUMNS} FROM entities WHERE id IN ({placeholders})");
        let params_ref: Vec<&dyn duckdb::ToSql> =
            ids.iter().map(|s| s as &dyn duckdb::ToSql).collect();
        let out = query_collect(&conn, &sql, &params_ref, entity_from_row, "find entities")?;
        attach_names(&conn, out)
    }

    async fn find_entities_by_name(&self, name: &str) -> Result<Vec<Entity>, DomainError> {
        let key = entity_name_key(name);
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT DISTINCT {ENTITY_COLUMNS} FROM entities e \
             JOIN entity_names n ON n.entity_id = e.id \
             WHERE n.name_key = ?"
        );
        let out = query_collect(
            &conn,
            &sql,
            &[&key as &dyn duckdb::ToSql],
            entity_from_row,
            "find entities by name",
        )?;
        attach_names(&conn, out)
    }

    async fn list_entities(&self) -> Result<Vec<Entity>, DomainError> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {ENTITY_COLUMNS} FROM entities ORDER BY updated_at DESC");
        let out = query_collect(&conn, &sql, &[], entity_from_row, "list entities")?;
        attach_names(&conn, out)
    }

    async fn memories_for_entity(&self, entity_id: &str) -> Result<Vec<Memory>, DomainError> {
        let conn = self.conn.lock().await;
        let sql = format!(
            "SELECT {MEMORY_COLUMNS} FROM memories m \
             JOIN memory_entities me ON me.memory_id = m.id \
             WHERE me.entity_id = ? \
             ORDER BY m.recorded_at DESC"
        );
        let out = query_collect(
            &conn,
            &sql,
            &[&entity_id as &dyn duckdb::ToSql],
            memory_from_row,
            "list entity memories",
        )?;
        attach_entity_ids(&conn, out)
    }

    // ── Retrieval ────────────────────────────────────────────────────────

    async fn search_memories_semantic(
        &self,
        vector: &[f32],
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError> {
        if vector.len() != self.dimensions {
            return Err(DomainError::invalid_input(format!(
                "query vector width {} does not match pinned dimensions {}",
                vector.len(),
                self.dimensions
            )));
        }
        let literal = vector_literal(vector);
        let mut scope_params: Vec<String> = Vec::new();
        let scope = project_scope_clause(projects, "project", &mut scope_params);
        let sql = format!(
            "SELECT {MEMORY_COLUMNS}, \
                    1.0 - array_cosine_distance(e.vector, {literal}::FLOAT[{d}]) AS score \
             FROM memories m JOIN memory_embeddings e ON e.memory_id = m.id \
             WHERE TRUE{scope} \
             ORDER BY score DESC LIMIT ?",
            d = self.dimensions,
        );
        // Bind order: scope projects, then limit. The limit binds as a real
        // integer, not text: DuckDB coerces a bound string to the column type,
        // and callers pass `usize::MAX` to mean "no limit", which overflows
        // INT64 as the string "18446744073709551615".
        let conn = self.conn.lock().await;
        let mut params_ref: Vec<&dyn duckdb::ToSql> = scope_params
            .iter()
            .map(|s| s as &dyn duckdb::ToSql)
            .collect();
        let limit = sql_limit(limit);
        params_ref.push(&limit);
        let out = query_collect(
            &conn,
            &sql,
            &params_ref,
            |row| {
                let memory = memory_from_row(row)?;
                let score: f64 = row.get(MEMORY_COLUMN_COUNT)?;
                Ok((memory, score as f32))
            },
            "search memories",
        )?;
        let ids: Vec<String> = out.iter().map(|(m, _)| m.id.clone()).collect();
        let mut by_memory = load_entity_ids_for(&conn, &ids)?;
        Ok(out
            .into_iter()
            .map(|(m, score)| {
                let entity_ids = by_memory.remove(&m.id).unwrap_or_default();
                (Memory { entity_ids, ..m }, score)
            })
            .collect())
    }

    async fn search_memories_keyword(
        &self,
        query: &str,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(Memory, f32)>, DomainError> {
        // Split into lowercase terms; each term must appear in the statement
        // for the row to score at all.
        let terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .take(MAX_KEYWORD_TERMS)
            .map(|t| t.to_lowercase())
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut scope_params: Vec<String> = Vec::new();
        let mut term_clauses = String::new();
        for _ in &terms {
            term_clauses.push_str(" AND statement ILIKE '%' || ? || '%'");
        }
        let scope = project_scope_clause(projects, "project", &mut scope_params);
        // Bind order follows the SQL's textual order: terms, then scope
        // projects, then limit (bound as an integer — see `search_by_vector`).
        let mut ordered_params: Vec<String> = terms.clone();
        ordered_params.extend(scope_params);

        let sql = format!(
            "SELECT {MEMORY_COLUMNS}, 1.0 AS score \
             FROM memories \
             WHERE TRUE{term_clauses}{scope} \
             ORDER BY recorded_at DESC LIMIT ?"
        );
        let conn = self.conn.lock().await;
        let mut params_ref: Vec<&dyn duckdb::ToSql> = ordered_params
            .iter()
            .map(|s| s as &dyn duckdb::ToSql)
            .collect();
        let limit = sql_limit(limit);
        params_ref.push(&limit);
        let out = query_collect(
            &conn,
            &sql,
            &params_ref,
            |row| {
                let memory = memory_from_row(row)?;
                let score: f64 = row.get(MEMORY_COLUMN_COUNT)?;
                Ok((memory, score as f32))
            },
            "keyword-search memories",
        )?;
        let ids: Vec<String> = out.iter().map(|(m, _)| m.id.clone()).collect();
        let mut by_memory = load_entity_ids_for(&conn, &ids)?;
        Ok(out
            .into_iter()
            .map(|(m, score)| {
                let entity_ids = by_memory.remove(&m.id).unwrap_or_default();
                (Memory { entity_ids, ..m }, score)
            })
            .collect())
    }

    async fn list_memories_by_recency(
        &self,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<Memory>, DomainError> {
        let mut scope_params: Vec<String> = Vec::new();
        let scope = project_scope_clause(projects, "project", &mut scope_params);
        let sql = format!(
            "SELECT {MEMORY_COLUMNS} FROM memories WHERE TRUE{scope} \
             ORDER BY recorded_at DESC LIMIT ?"
        );
        // Bind order: scope projects, then limit (bound as an integer — see
        // `search_by_vector`).
        let conn = self.conn.lock().await;
        let mut params_ref: Vec<&dyn duckdb::ToSql> = scope_params
            .iter()
            .map(|s| s as &dyn duckdb::ToSql)
            .collect();
        let limit = sql_limit(limit);
        params_ref.push(&limit);
        let out = query_collect(
            &conn,
            &sql,
            &params_ref,
            memory_from_row,
            "list memories by recency",
        )?;
        attach_entity_ids(&conn, out)
    }

    // ── Resources ────────────────────────────────────────────────────────

    async fn upsert_resource(
        &self,
        resource: &MemoryResource,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError> {
        // Validate before opening the transaction.
        if let Some(vector) = vector {
            if vector.len() != self.dimensions {
                return Err(DomainError::invalid_input(format!(
                    "vector width {} does not match pinned dimensions {}",
                    vector.len(),
                    self.dimensions
                )));
            }
        }
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| DomainError::storage(format!("failed to begin transaction: {e}")))?;
        tx.execute(
            &format!(
                "INSERT OR REPLACE INTO memory_resources ({RESOURCE_COLUMNS}) \
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            params![
                resource.uri,
                resource.source,
                resource.name,
                resource.abstract_,
                resource.overview,
                resource.content,
                resource.created_at,
            ],
        )
        .map_err(|e| DomainError::storage(format!("failed to upsert resource: {e}")))?;

        tx.execute(
            "DELETE FROM memory_resource_embeddings WHERE uri = ?",
            params![resource.uri],
        )
        .map_err(|e| DomainError::storage(format!("failed to clear resource embedding: {e}")))?;
        if let Some(vector) = vector {
            let literal = vector_literal(vector);
            tx.execute(
                &format!(
                    "INSERT INTO memory_resource_embeddings (uri, vector) VALUES (?, {literal})"
                ),
                params![resource.uri],
            )
            .map_err(|e| {
                DomainError::storage(format!("failed to write resource embedding: {e}"))
            })?;
        }
        tx.commit()
            .map_err(|e| DomainError::storage(format!("failed to commit resource write: {e}")))?;
        Ok(())
    }

    async fn find_resource(&self, uri: &str) -> Result<Option<MemoryResource>, DomainError> {
        let conn = self.conn.lock().await;
        let sql = format!("SELECT {RESOURCE_COLUMNS} FROM memory_resources WHERE uri = ?");
        let out = query_collect(
            &conn,
            &sql,
            &[&uri as &dyn duckdb::ToSql],
            resource_from_row,
            "find resource",
        )?;
        Ok(out.into_iter().next())
    }

    async fn list_resources(&self) -> Result<Vec<MemoryResource>, DomainError> {
        let conn = self.conn.lock().await;
        let sql =
            format!("SELECT {RESOURCE_COLUMNS} FROM memory_resources ORDER BY created_at DESC");
        query_collect(&conn, &sql, &[], resource_from_row, "list resources")
    }

    async fn search_resources_semantic(
        &self,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryResource, f32)>, DomainError> {
        if vector.len() != self.dimensions {
            return Err(DomainError::invalid_input(format!(
                "query vector width {} does not match pinned dimensions {}",
                vector.len(),
                self.dimensions
            )));
        }
        let literal = vector_literal(vector);
        let sql = format!(
            "SELECT {RESOURCE_COLUMNS}, \
                    1.0 - array_cosine_distance(e.vector, {literal}::FLOAT[{d}]) AS score \
             FROM memory_resources r JOIN memory_resource_embeddings e ON e.uri = r.uri \
             ORDER BY score DESC LIMIT ?",
            d = self.dimensions,
        );
        let conn = self.conn.lock().await;
        query_collect(
            &conn,
            &sql,
            &[&(limit as i64) as &dyn duckdb::ToSql],
            |row| {
                let resource = resource_from_row(row)?;
                let score: f64 = row.get(RESOURCE_COLUMN_COUNT)?;
                Ok((resource, score as f32))
            },
            "search resources",
        )
    }

    // ── Namespaces ───────────────────────────────────────────────────────

    async fn create_namespace(&self, name: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        // A namespace with no projects is still recorded — by inserting a
        // marker row whose `project` is the sentinel `""`.
        let n = conn
            .execute(
                "INSERT OR IGNORE INTO memory_namespaces (namespace, project, created_at) \
                 VALUES (?, '', ?)",
                params![name, now],
            )
            .map_err(|e| DomainError::storage(format!("failed to create namespace: {e}")))?;
        Ok(n > 0)
    }

    async fn delete_namespace(&self, name: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let n = conn
            .execute(
                "DELETE FROM memory_namespaces WHERE namespace = ?",
                params![name],
            )
            .map_err(|e| DomainError::storage(format!("failed to delete namespace: {e}")))?;
        Ok(n > 0)
    }

    async fn list_namespaces(&self) -> Result<Vec<(String, u64)>, DomainError> {
        let conn = self.conn.lock().await;
        query_collect(
            &conn,
            "SELECT namespace, COUNT(*) FILTER (project <> '') AS count \
             FROM memory_namespaces GROUP BY namespace ORDER BY namespace",
            &[],
            |row| {
                let name: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                Ok((name, count as u64))
            },
            "list namespaces",
        )
    }

    async fn namespace_projects(&self, name: &str) -> Result<Vec<String>, DomainError> {
        let conn = self.conn.lock().await;
        query_collect(
            &conn,
            "SELECT project FROM memory_namespaces \
             WHERE namespace = ? AND project <> '' ORDER BY project",
            &[&name as &dyn duckdb::ToSql],
            |row| row.get::<_, String>(0),
            "list namespace projects",
        )
    }

    async fn namespace_created_at(&self, name: &str) -> Result<Option<i64>, DomainError> {
        let conn = self.conn.lock().await;
        let value: Option<i64> = conn
            .query_row(
                "SELECT MIN(created_at) FROM memory_namespaces WHERE namespace = ?",
                params![name],
                |row| row.get(0),
            )
            .map_err(|e| DomainError::storage(format!("failed to read namespace cutoff: {e}")))?;
        Ok(value)
    }

    async fn assign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        // Inherit the namespace's existing cutoff so re-adding a project does
        // not silently widen the auto-import window. Fresh namespaces use now.
        // Inline the cutoff read rather than calling `namespace_created_at`:
        // that method takes the same lock, and we already hold it.
        let cutoff: Option<i64> = conn
            .query_row(
                "SELECT MIN(created_at) FROM memory_namespaces WHERE namespace = ?",
                params![namespace],
                |row| row.get(0),
            )
            .map_err(|e| DomainError::storage(format!("failed to read namespace cutoff: {e}")))?;
        let cutoff = cutoff.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });
        let n = conn
            .execute(
                "INSERT OR IGNORE INTO memory_namespaces (namespace, project, created_at) \
                 VALUES (?, ?, ?)",
                params![namespace, project, cutoff],
            )
            .map_err(|e| DomainError::storage(format!("failed to assign project: {e}")))?;
        Ok(n > 0)
    }

    async fn unassign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let n = conn
            .execute(
                "DELETE FROM memory_namespaces WHERE namespace = ? AND project = ?",
                params![namespace, project],
            )
            .map_err(|e| DomainError::storage(format!("failed to unassign project: {e}")))?;
        Ok(n > 0)
    }

    // ── Sessions ─────────────────────────────────────────────────────────

    async fn record_session(&self, session: &ImportedSession) -> Result<(), DomainError> {
        let conn = self.conn.lock().await;
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO memory_sessions ({SESSION_COLUMNS}) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            params![
                session.id,
                session.source,
                session.imported_at,
                session.message_count as i64,
                session.items_written as i64,
                session.status.as_str(),
                session.last_error,
                session.project,
            ],
        )
        .map_err(|e| DomainError::storage(format!("failed to record session: {e}")))?;
        Ok(())
    }

    async fn find_session(
        &self,
        source: &str,
        id: &str,
    ) -> Result<Option<ImportedSession>, DomainError> {
        let conn = self.conn.lock().await;
        let sql =
            format!("SELECT {SESSION_COLUMNS} FROM memory_sessions WHERE source = ? AND id = ?");
        let out = query_collect(
            &conn,
            &sql,
            &[&source as &dyn duckdb::ToSql, &id as &dyn duckdb::ToSql],
            session_from_row,
            "find session",
        )?;
        Ok(out.into_iter().next())
    }

    async fn list_sessions(
        &self,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<ImportedSession>, DomainError> {
        list_sessions_inner(self, None, projects, limit).await
    }

    async fn list_sessions_by_status(
        &self,
        status: SessionStatus,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<ImportedSession>, DomainError> {
        list_sessions_inner(self, Some(status), projects, limit).await
    }
}

/// Shared body of `list_sessions` / `list_sessions_by_status` — free
/// function rather than an inherent method, so the trait impl stays a list
/// of port methods.
async fn list_sessions_inner(
    store: &DuckdbStore,
    status: Option<SessionStatus>,
    projects: Option<&[String]>,
    limit: usize,
) -> Result<Vec<ImportedSession>, DomainError> {
    let mut params: Vec<String> = Vec::new();
    // Sessions store project as a real NULL/Optional column rather than
    // the `''` convention the memories table uses.
    let scope = match projects {
        None => String::new(),
        Some([]) => " AND project IS NULL".to_string(),
        Some(ps) => {
            let placeholders = std::iter::repeat_n("?", ps.len())
                .collect::<Vec<_>>()
                .join(", ");
            for p in ps {
                params.push(p.clone());
            }
            format!(" AND project IN ({placeholders})")
        }
    };
    let status_clause = match status {
        Some(_) => " AND status = ?",
        None => "",
    };
    let sql = format!(
        "SELECT {SESSION_COLUMNS} FROM memory_sessions \
             WHERE TRUE{status_clause}{scope} \
             ORDER BY imported_at DESC LIMIT ?"
    );
    // Bind order: status (if any), scope projects, limit. The limit binds as a
    // real integer, not text: DuckDB coerces a bound string to the column type,
    // and callers pass `usize::MAX` to mean "no limit", which overflows INT64
    // as the string "18446744073709551615" and failed the whole query.
    let mut ordered: Vec<String> = Vec::new();
    if let Some(s) = status {
        ordered.push(s.as_str().to_string());
    }
    ordered.extend(params);

    let conn = store.conn.lock().await;
    let mut params_ref: Vec<&dyn duckdb::ToSql> =
        ordered.iter().map(|s| s as &dyn duckdb::ToSql).collect();
    let limit = sql_limit(limit);
    params_ref.push(&limit);
    query_collect(&conn, &sql, &params_ref, session_from_row, "list sessions")
}

fn load_names(conn: &Connection, entity_id: &str) -> Result<Vec<String>, DomainError> {
    let mut by_entity = load_names_for(conn, std::slice::from_ref(&entity_id.to_string()))?;
    Ok(by_entity.remove(entity_id).unwrap_or_default())
}

/// Load every alias for a batch of entities in one query, grouped by entity
/// id. The per-row `load_names` was the N+1 half of every list/find method
/// on this adapter.
fn load_names_for(
    conn: &Connection,
    entity_ids: &[String],
) -> Result<std::collections::HashMap<String, Vec<String>>, DomainError> {
    use std::collections::HashMap;
    let mut by_entity: HashMap<String, Vec<String>> = HashMap::new();
    if entity_ids.is_empty() {
        return Ok(by_entity);
    }
    let placeholders = std::iter::repeat_n("?", entity_ids.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT entity_id, name FROM entity_names \
         WHERE entity_id IN ({placeholders}) ORDER BY entity_id, name"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| DomainError::storage(format!("failed to load entity names: {e}")))?;
    let params_ref: Vec<&dyn duckdb::ToSql> =
        entity_ids.iter().map(|s| s as &dyn duckdb::ToSql).collect();
    let mut rows = stmt
        .query(&params_ref[..])
        .map_err(|e| DomainError::storage(format!("failed to load entity names: {e}")))?;
    while let Some(row) = rows
        .next()
        .map_err(|e| DomainError::storage(format!("failed to load entity names: {e}")))?
    {
        let entity_id: String = row
            .get(0)
            .map_err(|e| DomainError::storage(format!("failed to decode entity id: {e}")))?;
        let name: String = row
            .get(1)
            .map_err(|e| DomainError::storage(format!("failed to decode name: {e}")))?;
        by_entity.entry(entity_id).or_default().push(name);
    }
    Ok(by_entity)
}

/// Attach `names` to a list of entities, in place. One query for the whole
/// batch.
fn attach_names(conn: &Connection, entities: Vec<Entity>) -> Result<Vec<Entity>, DomainError> {
    let ids: Vec<String> = entities.iter().map(|e| e.id.clone()).collect();
    let mut by_entity = load_names_for(conn, &ids)?;
    Ok(entities
        .into_iter()
        .map(|e| {
            let names = by_entity.remove(&e.id).unwrap_or_default();
            Entity { names, ..e }
        })
        .collect())
}

/// Load `entity_ids` for many memories in one query, grouped by memory id.
///
/// Callers handing out `Vec<Memory>` were resolving the link one row at a
/// time — an N+1 against a connection every other table shares. One
/// `WHERE memory_id IN (…)` and a hash-map group replaces that.
fn load_entity_ids_for(
    conn: &Connection,
    memory_ids: &[String],
) -> Result<std::collections::HashMap<String, Vec<String>>, DomainError> {
    use std::collections::HashMap;
    let mut by_memory: HashMap<String, Vec<String>> = HashMap::new();
    if memory_ids.is_empty() {
        return Ok(by_memory);
    }
    let placeholders = std::iter::repeat_n("?", memory_ids.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT memory_id, entity_id FROM memory_entities \
         WHERE memory_id IN ({placeholders}) ORDER BY memory_id, entity_id"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| DomainError::storage(format!("failed to load memory entities: {e}")))?;
    let params_ref: Vec<&dyn duckdb::ToSql> =
        memory_ids.iter().map(|s| s as &dyn duckdb::ToSql).collect();
    let mut rows = stmt
        .query(&params_ref[..])
        .map_err(|e| DomainError::storage(format!("failed to load memory entities: {e}")))?;
    while let Some(row) = rows
        .next()
        .map_err(|e| DomainError::storage(format!("failed to load memory entities: {e}")))?
    {
        let memory_id: String = row
            .get(0)
            .map_err(|e| DomainError::storage(format!("failed to decode memory id: {e}")))?;
        let entity_id: String = row
            .get(1)
            .map_err(|e| DomainError::storage(format!("failed to decode entity id: {e}")))?;
        by_memory.entry(memory_id).or_default().push(entity_id);
    }
    Ok(by_memory)
}

/// Attach entity ids to a list of memories, in place. One query for the
/// whole batch.
fn attach_entity_ids(conn: &Connection, memories: Vec<Memory>) -> Result<Vec<Memory>, DomainError> {
    let ids: Vec<String> = memories.iter().map(|m| m.id.clone()).collect();
    let mut by_memory = load_entity_ids_for(conn, &ids)?;
    Ok(memories
        .into_iter()
        .map(|m| {
            let entity_ids = by_memory.remove(&m.id).unwrap_or_default();
            Memory { entity_ids, ..m }
        })
        .collect())
}

/// Render a `&[f32]` as a DuckDB array literal `[x, y, z]`. Used because
/// duckdb-rs cannot bind fixed-width arrays natively.
fn vector_literal(vector: &[f32]) -> String {
    let mut s = String::from("[");
    for (i, v) in vector.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{v}"));
    }
    s.push(']');
    s
}

/// Prepare `sql`, bind `params`, and collect every row through `decode`.
///
/// The largest `LIMIT` DuckDB's binder accepts (2^62). Anything above this is
/// rejected with "Max value … for LIMIT/OFFSET is 4611686018427387904".
const MAX_SQL_LIMIT: i64 = 4_611_686_018_427_387_904;

/// A `LIMIT` value DuckDB will accept.
///
/// Callers spell "no limit" as `usize::MAX`, which survives none of the obvious
/// spellings: bound as text it overflows INT64 during coercion, cast to `i64`
/// it wraps to -1 (a negative LIMIT is rejected), and even `i64::MAX` is past
/// the binder's own ceiling. Clamping keeps the sentinel meaning "everything"
/// while staying inside the range DuckDB will take — 2^62 rows is unreachable
/// for any real store.
fn sql_limit(limit: usize) -> i64 {
    i64::try_from(limit)
        .unwrap_or(MAX_SQL_LIMIT)
        .min(MAX_SQL_LIMIT)
}

/// Every read method on this adapter used to spell the same six lines of
/// prepare/query/loop inline; the shape was identical, only the SQL, the
/// binds, and the row decoder varied. One helper keeps the variations
/// visible instead of repeating the ritual.
fn query_collect<T>(
    conn: &Connection,
    sql: &str,
    params: &[&dyn duckdb::ToSql],
    decode: impl Fn(&Row<'_>) -> Result<T, duckdb::Error>,
    op: &'static str,
) -> Result<Vec<T>, DomainError> {
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| DomainError::storage(format!("failed to {op}: {e}")))?;
    let mut rows = stmt
        .query(params)
        .map_err(|e| DomainError::storage(format!("failed to {op}: {e}")))?;
    let mut out = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|e| DomainError::storage(format!("failed to {op}: {e}")))?
    {
        out.push(
            decode(row)
                .map_err(|e| DomainError::storage(format!("failed to decode {op} row: {e}")))?,
        );
    }
    Ok(out)
}
