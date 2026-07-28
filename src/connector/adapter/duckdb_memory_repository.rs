//! DuckDB-backed [`MemoryRepository`](crate::application::MemoryRepository).
//!
//! Memory lives in its own database file (`memory.duckdb`), so the store can be
//! inspected, backed up, or wiped independently. Vectors are stored as
//! fixed-width `FLOAT[dimensions]` columns and recalled with DuckDB's native
//! `array_cosine_distance`. `dimensions` is pinned on first open — a later open
//! at a different width is rejected, since vectors would be incomparable. The
//! embedding *model* is also recorded on first open, but an existing store keeps
//! its original: it is authoritative for retrieval, so queries are embedded with
//! it (see [`MemoryRepository::embedding_model`]) rather than being rejected.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use duckdb::{params, Connection, Row};
use tokio::sync::Mutex;
use tracing::debug;

use crate::application::{MemoryRepository, MemoryStats};
use crate::domain::{
    DomainError, DreamRun, ImportedSession, MemoryItem, MemoryKind, MemoryNode, NodeKind,
    SessionStatus,
};

/// File name of the memory database inside the data directory.
pub const MEMORY_DB_FILE: &str = "memory.duckdb";

pub struct DuckdbMemoryRepository {
    conn: Arc<Mutex<Connection>>,
    dimensions: usize,
    /// The embedding model that wrote the stored vectors — the value seeded on
    /// first open (an existing store keeps its original, ignoring the argument).
    /// Authoritative for retrieval; exposed via [`Self::stored_embedding_model`].
    stored_embedding_model: String,
}

impl DuckdbMemoryRepository {
    /// Open (or create) the memory database at `db_path`.
    ///
    /// `dimensions` and `embedding_model` describe the embedding setup. Both are
    /// persisted on first open. A later open at a different `dimensions` is
    /// rejected (incomparable vector widths); a different `embedding_model` is
    /// *not* — the stored model wins and is returned by
    /// [`Self::stored_embedding_model`], since it must be used to embed queries.
    pub fn new(
        db_path: &Path,
        dimensions: usize,
        embedding_model: &str,
    ) -> Result<Self, DomainError> {
        let conn = Connection::open(db_path)
            .map_err(|e| DomainError::storage(format!("Failed to open memory database: {e}")))?;
        Self::initialize(conn, dimensions, embedding_model)
    }

    /// In-memory database for tests.
    pub fn in_memory(dimensions: usize, embedding_model: &str) -> Result<Self, DomainError> {
        let conn = Connection::open_in_memory().map_err(|e| {
            DomainError::storage(format!("Failed to open in-memory memory database: {e}"))
        })?;
        Self::initialize(conn, dimensions, embedding_model)
    }

    fn initialize(
        conn: Connection,
        dimensions: usize,
        embedding_model: &str,
    ) -> Result<Self, DomainError> {
        if dimensions == 0 {
            return Err(DomainError::invalid_input(
                "embedding dimensions must be greater than 0",
            ));
        }
        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                source_session_id TEXT,
                project TEXT NOT NULL DEFAULT '',
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL,
                update_count BIGINT NOT NULL DEFAULT 0,
                UNIQUE (kind, name, project)
            );
            CREATE TABLE IF NOT EXISTS memory_vectors (
                item_id TEXT PRIMARY KEY,
                vector FLOAT[{dimensions}] NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                imported_at BIGINT NOT NULL,
                message_count BIGINT NOT NULL,
                items_written BIGINT NOT NULL,
                status TEXT NOT NULL DEFAULT 'imported',
                last_error TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_nodes (
                uri TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                parent_uri TEXT,
                label TEXT,
                abstract TEXT NOT NULL,
                overview TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_node_vectors (
                node_uri TEXT PRIMARY KEY,
                vector FLOAT[{dimensions}] NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_dream_runs (
                id TEXT PRIMARY KEY,
                started_at BIGINT NOT NULL,
                finished_at BIGINT NOT NULL,
                sessions_imported BIGINT NOT NULL,
                clusters_found BIGINT NOT NULL,
                operations_applied BIGINT NOT NULL,
                operations_skipped BIGINT NOT NULL,
                status TEXT NOT NULL DEFAULT 'completed'
            );
            CREATE TABLE IF NOT EXISTS memory_namespaces (
                namespace TEXT NOT NULL,
                project TEXT NOT NULL,
                UNIQUE (namespace, project)
            );
            "#
        ))
        .map_err(|e| DomainError::storage(format!("Failed to initialize memory schema: {e}")))?;

        // Migrate databases created before `memory_nodes.label` existed. DuckDB
        // has no `ADD COLUMN IF NOT EXISTS`, so add it and swallow the
        // already-exists error (idempotent across restarts).
        if let Err(e) = conn.execute_batch("ALTER TABLE memory_nodes ADD COLUMN label TEXT;") {
            let msg = e.to_string().to_lowercase();
            if !msg.contains("already exists") && !msg.contains("duplicate") {
                return Err(DomainError::storage(format!(
                    "Failed to add memory_nodes.label column: {e}"
                )));
            }
        }

        // Older databases predate the harvest failure marker. Same idempotent
        // add-and-swallow pattern as `memory_nodes.label` — but DuckDB rejects
        // `ADD COLUMN` carrying a constraint ("Adding columns with constraints
        // not yet supported"), so the column is added bare and backfilled.
        // Every pre-existing row is by definition a successful import.
        for column in ["status", "last_error"] {
            if let Err(e) = conn.execute_batch(&format!(
                "ALTER TABLE memory_sessions ADD COLUMN {column} TEXT;"
            )) {
                let msg = e.to_string().to_lowercase();
                if !msg.contains("already exists") && !msg.contains("duplicate") {
                    return Err(DomainError::storage(format!(
                        "Failed to add memory_sessions column '{column}': {e}"
                    )));
                }
            }
        }
        conn.execute_batch("UPDATE memory_sessions SET status = 'imported' WHERE status IS NULL;")
            .map_err(|e| {
                DomainError::storage(format!("Failed to backfill memory_sessions.status: {e}"))
            })?;

        Self::migrate_item_identity(&conn)?;

        // `dimensions` is a hard pin: vectors of different widths cannot be
        // compared, so a mismatch is a genuine incompatibility and is rejected.
        Self::check_meta(&conn, "dimensions", &dimensions.to_string())?;
        // The embedding *model* is seeded on a fresh store but NOT rejected on
        // reopen: the stored model is authoritative for retrieval (queries must
        // be embedded with whatever wrote the vectors). Callers read it back via
        // [`MemoryRepository::embedding_model`] and embed queries with it.
        Self::seed_meta(&conn, "embedding_model", embedding_model)?;
        // Read back the effective model: for a fresh store this is the argument
        // just seeded; for an existing store it is the original, which wins.
        let stored_embedding_model = Self::read_meta(&conn, "embedding_model")?
            .unwrap_or_else(|| embedding_model.to_string());

        debug!(
            "memory database schema initialized ({dimensions} dims, model '{stored_embedding_model}')"
        );
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            dimensions,
            stored_embedding_model,
        })
    }

    /// The embedding model that wrote the stored vectors, read at open time.
    /// Synchronous companion to [`MemoryRepository::embedding_model`], so the
    /// container can build the retrieval embedder without an async hop.
    pub fn stored_embedding_model(&self) -> &str {
        &self.stored_embedding_model
    }

    /// Widen item identity from `(kind, name)` to `(kind, name, project)` on
    /// databases written by an earlier version.
    ///
    /// Under the old key a memory extracted in project B silently overwrote a
    /// same-named memory from project A *and* relabelled it with B's project,
    /// so the store ended up asserting A's content about B. Widening the key
    /// keeps them apart. DuckDB cannot drop a constraint in place, so the table
    /// is rebuilt; the old key is strictly narrower than the new one, so every
    /// existing row already satisfies it and nothing is dropped. Rows already
    /// merged by the old behaviour cannot be un-merged — their pre-collision
    /// content is gone — so they carry over as-is.
    ///
    /// `project` also changes from nullable to `NOT NULL DEFAULT ''` (empty
    /// means global), because SQL treats NULLs as distinct in a UNIQUE
    /// constraint — with NULLs, two global items of the same name would both be
    /// allowed and the constraint would not bind where it matters most.
    ///
    /// Note that this key permits one more thing than the model allows: a name
    /// held *both* globally and by a project. That combination is meaningless
    /// (recall for that project returns both rows, with nothing to choose
    /// between them) but it cannot be excluded by a `UNIQUE`, so it is enforced
    /// on the write path instead — see `resolve_scope` in `memory_support`.
    fn migrate_item_identity(conn: &Connection) -> Result<(), DomainError> {
        // `project` is nullable only on the pre-migration schema, so it is a
        // reliable, cheap marker that avoids rebuilding on every open.
        let nullable: Option<String> = conn
            .query_row(
                "SELECT is_nullable FROM information_schema.columns \
                 WHERE table_name = 'memory_items' AND column_name = 'project'",
                [],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                duckdb::Error::QueryReturnedNoRows => Ok(None),
                other => Err(DomainError::storage(format!(
                    "Failed to inspect memory_items.project: {other}"
                ))),
            })?;
        if !matches!(nullable.as_deref(), Some("YES")) {
            return Ok(());
        }

        conn.execute_batch(
            "BEGIN TRANSACTION;
             CREATE TABLE memory_items_migrated (
                 id TEXT PRIMARY KEY,
                 kind TEXT NOT NULL,
                 name TEXT NOT NULL,
                 content TEXT NOT NULL,
                 source_session_id TEXT,
                 project TEXT NOT NULL DEFAULT '',
                 created_at BIGINT NOT NULL,
                 updated_at BIGINT NOT NULL,
                 update_count BIGINT NOT NULL DEFAULT 0,
                 UNIQUE (kind, name, project)
             );
             INSERT INTO memory_items_migrated
                 SELECT id, kind, name, content, source_session_id,
                        COALESCE(project, ''), created_at, updated_at, update_count
                 FROM memory_items;
             DROP TABLE memory_items;
             ALTER TABLE memory_items_migrated RENAME TO memory_items;
             COMMIT;",
        )
        .map_err(|e| {
            DomainError::storage(format!("Failed to migrate memory item identity: {e}"))
        })?;

        debug!("migrated memory_items identity to (kind, name, project)");
        Ok(())
    }

    /// Persist a meta value on first open; reject a mismatch on later opens.
    fn check_meta(conn: &Connection, key: &str, expected: &str) -> Result<(), DomainError> {
        let stored: Option<String> = conn
            .query_row(
                "SELECT value FROM memory_meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                duckdb::Error::QueryReturnedNoRows => Ok(None),
                other => Err(DomainError::storage(format!(
                    "Failed to read memory meta '{key}': {other}"
                ))),
            })?;
        match stored {
            Some(value) if value == expected => Ok(()),
            Some(value) => Err(DomainError::invalid_input(format!(
                "memory database was created with {key}='{value}' but the current configuration \
                 uses '{expected}'; use the original embedding setup or delete the memory \
                 database to start over"
            ))),
            None => {
                conn.execute(
                    "INSERT INTO memory_meta (key, value) VALUES (?1, ?2)",
                    params![key, expected],
                )
                .map_err(|e| {
                    DomainError::storage(format!("Failed to write memory meta '{key}': {e}"))
                })?;
                Ok(())
            }
        }
    }

    /// Read a `memory_meta` value, or `None` when the key is absent.
    fn read_meta(conn: &Connection, key: &str) -> Result<Option<String>, DomainError> {
        conn.query_row(
            "SELECT value FROM memory_meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )
        .map(Some)
        .or_else(|e| match e {
            duckdb::Error::QueryReturnedNoRows => Ok(None),
            other => Err(DomainError::storage(format!(
                "Failed to read memory meta '{key}': {other}"
            ))),
        })
    }

    /// Persist `value` under `key` only if the key is absent — used to record
    /// the embedding model on a fresh store while leaving an existing store's
    /// recorded value untouched (the stored value stays authoritative).
    fn seed_meta(conn: &Connection, key: &str, value: &str) -> Result<(), DomainError> {
        if Self::read_meta(conn, key)?.is_none() {
            conn.execute(
                "INSERT INTO memory_meta (key, value) VALUES (?1, ?2)",
                params![key, value],
            )
            .map_err(|e| {
                DomainError::storage(format!("Failed to write memory meta '{key}': {e}"))
            })?;
        }
        Ok(())
    }

    /// Render a vector as a DuckDB `[..]::FLOAT[n]` literal (FLOAT arrays
    /// cannot be bound as parameters).
    fn vector_literal(&self, vector: &[f32]) -> Result<String, DomainError> {
        if vector.len() != self.dimensions {
            return Err(DomainError::invalid_input(format!(
                "vector has {} dimensions, memory database expects {}",
                vector.len(),
                self.dimensions
            )));
        }
        let mut s = String::with_capacity(vector.len() * 8);
        s.push('[');
        for (i, v) in vector.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{v}"));
        }
        s.push(']');
        s.push_str(&format!("::FLOAT[{}]", self.dimensions));
        Ok(s)
    }

    /// Run a blocking DuckDB query off the async runtime. DuckDB calls are
    /// synchronous I/O, so they must not execute on a Tokio worker thread; this
    /// clones the connection handle, runs `f` under the blocking lock inside
    /// `spawn_blocking`, and propagates a join failure as a storage error.
    async fn query_blocking<T, F>(&self, f: F) -> Result<T, DomainError>
    where
        F: FnOnce(&Connection) -> Result<T, DomainError> + Send + 'static,
        T: Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || f(&conn.blocking_lock()))
            .await
            .map_err(|e| DomainError::storage(format!("Blocking task panicked: {e}")))?
    }

    fn item_from_row(row: &Row<'_>) -> Result<MemoryItem, duckdb::Error> {
        let kind_str: String = row.get(1)?;
        let kind = MemoryKind::parse(&kind_str).unwrap_or(MemoryKind::Fact);
        Ok(MemoryItem::new(
            row.get(0)?,
            kind,
            row.get(2)?,
            row.get(3)?,
            row.get::<_, Option<String>>(4)?,
            project_from_column(row.get::<_, String>(5)?),
            row.get(6)?,
            row.get(7)?,
            row.get::<_, i64>(8)? as u32,
        ))
    }

    fn node_from_row(row: &Row<'_>) -> Result<MemoryNode, duckdb::Error> {
        let kind_str: String = row.get(1)?;
        let kind = NodeKind::parse(&kind_str).unwrap_or(NodeKind::Resource);
        let label: Option<String> = row.get::<_, Option<String>>(3)?;
        let mut node = MemoryNode::new(
            row.get(0)?,
            kind,
            row.get::<_, Option<String>>(2)?,
            row.get(4)?, // abstract
            row.get(5)?, // overview
            row.get(6)?, // content
            row.get(7)?, // created_at
            row.get(8)?, // updated_at
        );
        if let Some(label) = label.filter(|l| !l.is_empty()) {
            node = node.with_label(label);
        }
        Ok(node)
    }
}

const ITEM_COLUMNS: &str =
    "id, kind, name, content, source_session_id, project, created_at, updated_at, update_count";

/// `memory_items.project` is `NOT NULL` with `''` standing for "global", so the
/// `(kind, name, project)` unique key also binds global items (SQL treats NULLs
/// as distinct). The domain models "no project" as `None`; these two functions
/// are the only places that translation happens.
fn project_from_column(stored: String) -> Option<String> {
    (!stored.is_empty()).then_some(stored)
}

fn project_to_column(project: Option<&str>) -> &str {
    project.unwrap_or("")
}

const NODE_COLUMNS: &str =
    "uri, kind, parent_uri, label, abstract, overview, content, created_at, updated_at";

#[async_trait]
impl MemoryRepository for DuckdbMemoryRepository {
    async fn upsert_item(
        &self,
        item: &MemoryItem,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError> {
        let vector_literal = vector.map(|v| self.vector_literal(v)).transpose()?;
        let conn = self.conn.lock().await;

        // Replace any previous item with the same identity (by id or by the
        // (kind, name, project) key) so both unique constraints stay
        // conflict-free. Matching on project too is what keeps a same-named
        // memory from another project from being clobbered here.
        let project = project_to_column(item.project());
        conn.execute(
            "DELETE FROM memory_vectors WHERE item_id IN \
             (SELECT id FROM memory_items \
              WHERE id = ?1 OR (kind = ?2 AND name = ?3 AND project = ?4))",
            params![item.id(), item.kind().as_str(), item.name(), project],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear memory vector: {e}")))?;
        conn.execute(
            "DELETE FROM memory_items \
             WHERE id = ?1 OR (kind = ?2 AND name = ?3 AND project = ?4)",
            params![item.id(), item.kind().as_str(), item.name(), project],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear memory item: {e}")))?;

        conn.execute(
            &format!(
                "INSERT INTO memory_items ({ITEM_COLUMNS}) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
            ),
            params![
                item.id(),
                item.kind().as_str(),
                item.name(),
                item.content(),
                item.source_session_id(),
                project,
                item.created_at(),
                item.updated_at(),
                item.update_count() as i64,
            ],
        )
        .map_err(|e| DomainError::storage(format!("Failed to insert memory item: {e}")))?;

        if let Some(literal) = vector_literal {
            conn.execute(
                &format!("INSERT INTO memory_vectors (item_id, vector) VALUES (?1, {literal})"),
                params![item.id()],
            )
            .map_err(|e| DomainError::storage(format!("Failed to insert memory vector: {e}")))?;
        }
        Ok(())
    }

    async fn find_item(
        &self,
        kind: MemoryKind,
        name: &str,
        project: Option<&str>,
    ) -> Result<Option<MemoryItem>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {ITEM_COLUMNS} FROM memory_items \
                 WHERE kind = ?1 AND name = ?2 AND project = ?3"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare find_item: {e}")))?;
        match stmt.query_row(
            params![kind.as_str(), name, project_to_column(project)],
            Self::item_from_row,
        ) {
            Ok(item) => Ok(Some(item)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DomainError::storage(format!(
                "Failed to query memory item: {e}"
            ))),
        }
    }

    async fn find_items_named(
        &self,
        kind: MemoryKind,
        name: &str,
    ) -> Result<Vec<MemoryItem>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {ITEM_COLUMNS} FROM memory_items \
                 WHERE kind = ?1 AND name = ?2 ORDER BY updated_at DESC"
            ))
            .map_err(|e| {
                DomainError::storage(format!("Failed to prepare find_items_named: {e}"))
            })?;
        let rows = stmt
            .query_map(params![kind.as_str(), name], Self::item_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to query memory items: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read memory item row: {e}")))
    }

    async fn find_item_by_id(&self, id: &str) -> Result<Option<MemoryItem>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {ITEM_COLUMNS} FROM memory_items WHERE id = ?1"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare find_item_by_id: {e}")))?;
        match stmt.query_row(params![id], Self::item_from_row) {
            Ok(item) => Ok(Some(item)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DomainError::storage(format!(
                "Failed to query memory item by id: {e}"
            ))),
        }
    }

    async fn delete_item(
        &self,
        kind: MemoryKind,
        name: &str,
        project: Option<&str>,
    ) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let project = project_to_column(project);
        conn.execute(
            "DELETE FROM memory_vectors WHERE item_id IN \
             (SELECT id FROM memory_items \
              WHERE kind = ?1 AND name = ?2 AND project = ?3)",
            params![kind.as_str(), name, project],
        )
        .map_err(|e| DomainError::storage(format!("Failed to delete memory vector: {e}")))?;
        let deleted = conn
            .execute(
                "DELETE FROM memory_items WHERE kind = ?1 AND name = ?2 AND project = ?3",
                params![kind.as_str(), name, project],
            )
            .map_err(|e| DomainError::storage(format!("Failed to delete memory item: {e}")))?;
        Ok(deleted > 0)
    }

    async fn delete_item_by_id(&self, id: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM memory_vectors WHERE item_id = ?1", params![id])
            .map_err(|e| DomainError::storage(format!("Failed to delete memory vector: {e}")))?;
        let deleted = conn
            .execute("DELETE FROM memory_items WHERE id = ?1", params![id])
            .map_err(|e| DomainError::storage(format!("Failed to delete memory item: {e}")))?;
        Ok(deleted > 0)
    }

    async fn list_items(&self, kind: Option<MemoryKind>) -> Result<Vec<MemoryItem>, DomainError> {
        let conn = self.conn.lock().await;
        let (sql, kind_param) = match kind {
            Some(k) => (
                format!(
                    "SELECT {ITEM_COLUMNS} FROM memory_items WHERE kind = ?1 \
                     ORDER BY updated_at DESC, name"
                ),
                Some(k.as_str().to_string()),
            ),
            None => (
                format!("SELECT {ITEM_COLUMNS} FROM memory_items ORDER BY updated_at DESC, name"),
                None,
            ),
        };
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| DomainError::storage(format!("Failed to prepare list_items: {e}")))?;
        let rows = match kind_param {
            Some(k) => stmt.query_map(params![k], Self::item_from_row),
            None => stmt.query_map([], Self::item_from_row),
        }
        .map_err(|e| DomainError::storage(format!("Failed to list memory items: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read memory item row: {e}")))
    }

    async fn search_semantic(
        &self,
        vector: &[f32],
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(MemoryItem, f32)>, DomainError> {
        let literal = self.vector_literal(vector)?;
        let mut conditions: Vec<String> = Vec::new();
        if let Some(k) = kind {
            conditions.push(format!("i.kind = '{}'", k.as_str()));
        }
        if let Some(clause) = project_scope_clause("i.project", projects) {
            conditions.push(clause);
        }
        let kind_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };
        let sql = format!(
            "SELECT {cols}, 1.0 - array_cosine_distance(v.vector, {literal}) AS score \
             FROM memory_items i \
             JOIN memory_vectors v ON v.item_id = i.id \
             {kind_clause} \
             ORDER BY score DESC \
             LIMIT {limit}",
            cols = ITEM_COLUMNS
                .split(", ")
                .map(|c| format!("i.{c}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| DomainError::storage(format!("Failed to prepare semantic search: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                let item = Self::item_from_row(row)?;
                // Score is the column appended after ITEM_COLUMNS' 9 fields.
                let score: f32 = row.get(ITEM_COLUMNS.split(", ").count())?;
                Ok((item, score))
            })
            .map_err(|e| DomainError::storage(format!("Semantic memory search failed: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read semantic search row: {e}")))
    }

    async fn search_keyword(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<(MemoryItem, f32)>, DomainError> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .take(16)
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Score = fraction of query terms found in name or content.
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
                format!(
                    "(CASE WHEN lower(name) LIKE '%{e}%' ESCAPE '\\' \
                       OR lower(content) LIKE '%{e}%' ESCAPE '\\' THEN 1 ELSE 0 END)"
                )
            })
            .collect();
        let score_expr = format!("({}) / {}.0", match_cases.join(" + "), terms.len());
        let mut kind_clause = match kind {
            Some(k) => format!("AND kind = '{}'", k.as_str()),
            None => String::new(),
        };
        if let Some(clause) = project_scope_clause("project", projects) {
            kind_clause.push_str(&format!(" AND {clause}"));
        }
        let sql = format!(
            "SELECT {ITEM_COLUMNS}, {score_expr} AS score \
             FROM memory_items \
             WHERE {score_expr} > 0 {kind_clause} \
             ORDER BY score DESC, updated_at DESC \
             LIMIT {limit}"
        );
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| DomainError::storage(format!("Failed to prepare keyword search: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                let item = Self::item_from_row(row)?;
                // Score is the column appended after ITEM_COLUMNS' 9 fields.
                let score: f64 = row.get(ITEM_COLUMNS.split(", ").count())?;
                Ok((item, score as f32))
            })
            .map_err(|e| DomainError::storage(format!("Keyword memory search failed: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read keyword search row: {e}")))
    }

    async fn list_item_vectors(&self) -> Result<Vec<(String, Vec<f32>)>, DomainError> {
        // The full vector table is scanned and every row JSON-decoded here, so
        // this must not run on a Tokio worker thread.
        self.query_blocking(|conn| {
            // FLOAT[n] values cannot be fetched as a native Rust type through
            // duckdb-rs, so round-trip them through JSON text.
            let mut stmt = conn
                .prepare("SELECT item_id, to_json(vector)::VARCHAR FROM memory_vectors")
                .map_err(|e| {
                    DomainError::storage(format!("Failed to prepare list_item_vectors: {e}"))
                })?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| DomainError::storage(format!("Failed to list item vectors: {e}")))?;
            let mut vectors = Vec::new();
            for row in rows {
                let (item_id, json) = row
                    .map_err(|e| DomainError::storage(format!("Failed to read vector row: {e}")))?;
                let vector: Vec<f32> = serde_json::from_str(&json).map_err(|e| {
                    DomainError::storage(format!("Failed to parse vector for '{item_id}': {e}"))
                })?;
                vectors.push((item_id, vector));
            }
            Ok(vectors)
        })
        .await
    }

    async fn find_item_vector(&self, id: &str) -> Result<Option<Vec<f32>>, DomainError> {
        let id = id.to_string();
        self.query_blocking(move |conn| {
            let mut stmt = conn
                .prepare("SELECT to_json(vector)::VARCHAR FROM memory_vectors WHERE item_id = ?1")
                .map_err(|e| {
                    DomainError::storage(format!("Failed to prepare find_item_vector: {e}"))
                })?;
            match stmt.query_row(params![id], |row| row.get::<_, String>(0)) {
                Ok(json) => {
                    let vector: Vec<f32> = serde_json::from_str(&json).map_err(|e| {
                        DomainError::storage(format!("Failed to parse vector for '{id}': {e}"))
                    })?;
                    Ok(Some(vector))
                }
                Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(DomainError::storage(format!(
                    "Failed to query item vector: {e}"
                ))),
            }
        })
        .await
    }

    async fn record_session(&self, session: &ImportedSession) -> Result<(), DomainError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO memory_sessions \
                 (id, source, imported_at, message_count, items_written, status, last_error) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
             ON CONFLICT (id) DO UPDATE SET \
                 source = excluded.source, \
                 imported_at = excluded.imported_at, \
                 message_count = excluded.message_count, \
                 items_written = excluded.items_written, \
                 status = excluded.status, \
                 last_error = excluded.last_error",
            params![
                session.id,
                session.source,
                session.imported_at,
                session.message_count as i64,
                session.items_written as i64,
                session.status.as_str(),
                session.last_error,
            ],
        )
        .map_err(|e| DomainError::storage(format!("Failed to record session: {e}")))?;
        Ok(())
    }

    async fn find_session(&self, id: &str) -> Result<Option<ImportedSession>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                "SELECT id, source, imported_at, message_count, items_written, status, last_error \
                 FROM memory_sessions WHERE id = ?1",
            )
            .map_err(|e| DomainError::storage(format!("Failed to prepare find_session: {e}")))?;
        match stmt.query_row(params![id], session_from_row) {
            Ok(session) => Ok(Some(session)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DomainError::storage(format!(
                "Failed to query session: {e}"
            ))),
        }
    }

    async fn list_sessions(&self) -> Result<Vec<ImportedSession>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                "SELECT id, source, imported_at, message_count, items_written, status, last_error \
                 FROM memory_sessions ORDER BY imported_at DESC",
            )
            .map_err(|e| DomainError::storage(format!("Failed to prepare list_sessions: {e}")))?;
        let rows = stmt
            .query_map([], session_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to list sessions: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read session row: {e}")))
    }

    async fn upsert_node(
        &self,
        node: &MemoryNode,
        vector: Option<&[f32]>,
    ) -> Result<(), DomainError> {
        let vector_literal = vector.map(|v| self.vector_literal(v)).transpose()?;
        let conn = self.conn.lock().await;

        // Replace any previous node with the same URI so both tables stay
        // conflict-free (URI is the primary key on each).
        conn.execute(
            "DELETE FROM memory_node_vectors WHERE node_uri = ?1",
            params![node.uri()],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear node vector: {e}")))?;
        conn.execute(
            "DELETE FROM memory_nodes WHERE uri = ?1",
            params![node.uri()],
        )
        .map_err(|e| DomainError::storage(format!("Failed to clear node: {e}")))?;

        conn.execute(
            &format!(
                "INSERT INTO memory_nodes ({NODE_COLUMNS}) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
            ),
            params![
                node.uri(),
                node.kind().as_str(),
                node.parent_uri(),
                node.label(),
                node.abstract_(),
                node.overview(),
                node.content(),
                node.created_at(),
                node.updated_at(),
            ],
        )
        .map_err(|e| DomainError::storage(format!("Failed to insert node: {e}")))?;

        if let Some(literal) = vector_literal {
            conn.execute(
                &format!(
                    "INSERT INTO memory_node_vectors (node_uri, vector) VALUES (?1, {literal})"
                ),
                params![node.uri()],
            )
            .map_err(|e| DomainError::storage(format!("Failed to insert node vector: {e}")))?;
        }
        Ok(())
    }

    async fn find_node(&self, uri: &str) -> Result<Option<MemoryNode>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {NODE_COLUMNS} FROM memory_nodes WHERE uri = ?1"
            ))
            .map_err(|e| DomainError::storage(format!("Failed to prepare find_node: {e}")))?;
        match stmt.query_row(params![uri], Self::node_from_row) {
            Ok(node) => Ok(Some(node)),
            Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DomainError::storage(format!("Failed to query node: {e}"))),
        }
    }

    async fn delete_node(&self, uri: &str) -> Result<bool, DomainError> {
        let conn = self.conn.clone();
        let uri = uri.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();
            conn.execute(
                "DELETE FROM memory_node_vectors WHERE node_uri = ?1",
                params![&uri],
            )
            .map_err(|e| DomainError::storage(format!("Failed to delete node vector: {e}")))?;
            let deleted = conn
                .execute("DELETE FROM memory_nodes WHERE uri = ?1", params![&uri])
                .map_err(|e| DomainError::storage(format!("Failed to delete node: {e}")))?;
            Ok(deleted > 0)
        })
        .await
        .map_err(|e| DomainError::storage(format!("Blocking task panicked: {e}")))?
    }

    async fn list_child_nodes(&self, parent_uri: &str) -> Result<Vec<MemoryNode>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT {NODE_COLUMNS} FROM memory_nodes WHERE parent_uri = ?1 \
                 ORDER BY updated_at DESC, uri"
            ))
            .map_err(|e| {
                DomainError::storage(format!("Failed to prepare list_child_nodes: {e}"))
            })?;
        let rows = stmt
            .query_map(params![parent_uri], Self::node_from_row)
            .map_err(|e| DomainError::storage(format!("Failed to list child nodes: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read node row: {e}")))
    }

    async fn list_nodes(&self, kind: Option<NodeKind>) -> Result<Vec<MemoryNode>, DomainError> {
        let conn = self.conn.lock().await;
        let (sql, kind_param) = match kind {
            Some(k) => (
                format!(
                    "SELECT {NODE_COLUMNS} FROM memory_nodes WHERE kind = ?1 \
                     ORDER BY updated_at DESC, uri"
                ),
                Some(k.as_str().to_string()),
            ),
            None => (
                format!("SELECT {NODE_COLUMNS} FROM memory_nodes ORDER BY updated_at DESC, uri"),
                None,
            ),
        };
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| DomainError::storage(format!("Failed to prepare list_nodes: {e}")))?;
        let rows = match kind_param {
            Some(k) => stmt.query_map(params![k], Self::node_from_row),
            None => stmt.query_map([], Self::node_from_row),
        }
        .map_err(|e| DomainError::storage(format!("Failed to list nodes: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read node row: {e}")))
    }

    async fn search_nodes_semantic(
        &self,
        vector: &[f32],
        kind: Option<NodeKind>,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f32)>, DomainError> {
        let literal = self.vector_literal(vector)?;
        let kind_clause = match kind {
            Some(k) => format!("WHERE n.kind = '{}'", k.as_str()),
            None => String::new(),
        };
        let sql = format!(
            "SELECT {cols}, 1.0 - array_cosine_distance(v.vector, {literal}) AS score \
             FROM memory_nodes n \
             JOIN memory_node_vectors v ON v.node_uri = n.uri \
             {kind_clause} \
             ORDER BY score DESC \
             LIMIT {limit}",
            cols = NODE_COLUMNS
                .split(", ")
                .map(|c| format!("n.{c}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            DomainError::storage(format!("Failed to prepare node semantic search: {e}"))
        })?;
        let rows = stmt
            .query_map([], |row| {
                let node = Self::node_from_row(row)?;
                // Score is the column after NODE_COLUMNS (now 9 columns, 0-8).
                let score: f32 = row.get(9)?;
                Ok((node, score))
            })
            .map_err(|e| DomainError::storage(format!("Node semantic search failed: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read node search row: {e}")))
    }

    async fn search_nodes_keyword(
        &self,
        query: &str,
        kind: Option<NodeKind>,
        limit: usize,
    ) -> Result<Vec<(MemoryNode, f32)>, DomainError> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .take(16)
            .collect();
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let escape = |t: &str| {
            t.replace('\\', "\\\\")
                .replace('\'', "''")
                .replace('%', "\\%")
                .replace('_', "\\_")
        };
        // Score = fraction of query terms found in abstract or overview.
        let match_cases: Vec<String> = terms
            .iter()
            .map(|t| {
                let e = escape(t);
                format!(
                    "(CASE WHEN lower(abstract) LIKE '%{e}%' ESCAPE '\\' \
                       OR lower(overview) LIKE '%{e}%' ESCAPE '\\' THEN 1 ELSE 0 END)"
                )
            })
            .collect();
        let score_expr = format!("({}) / {}.0", match_cases.join(" + "), terms.len());
        let kind_clause = match kind {
            Some(k) => format!("AND kind = '{}'", k.as_str()),
            None => String::new(),
        };
        let sql = format!(
            "SELECT {NODE_COLUMNS}, {score_expr} AS score \
             FROM memory_nodes \
             WHERE {score_expr} > 0 {kind_clause} \
             ORDER BY score DESC, updated_at DESC \
             LIMIT {limit}"
        );
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            DomainError::storage(format!("Failed to prepare node keyword search: {e}"))
        })?;
        let rows = stmt
            .query_map([], |row| {
                let node = Self::node_from_row(row)?;
                // Score is the column after NODE_COLUMNS (now 9 columns, 0-8).
                let score: f64 = row.get(9)?;
                Ok((node, score as f32))
            })
            .map_err(|e| DomainError::storage(format!("Node keyword search failed: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read node search row: {e}")))
    }

    async fn record_dream_run(&self, run: &DreamRun) -> Result<(), DomainError> {
        let run = run.clone();
        self.query_blocking(move |conn| {
            conn.execute(
                "INSERT INTO memory_dream_runs \
                 (id, started_at, finished_at, sessions_imported, clusters_found, \
                  operations_applied, operations_skipped, status) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                 ON CONFLICT (id) DO UPDATE SET \
                     started_at = excluded.started_at, \
                     finished_at = excluded.finished_at, \
                     sessions_imported = excluded.sessions_imported, \
                     clusters_found = excluded.clusters_found, \
                     operations_applied = excluded.operations_applied, \
                     operations_skipped = excluded.operations_skipped, \
                     status = excluded.status",
                params![
                    run.id,
                    run.started_at,
                    run.finished_at,
                    run.sessions_imported as i64,
                    run.clusters_found as i64,
                    run.operations_applied as i64,
                    run.operations_skipped as i64,
                    run.status,
                ],
            )
            .map_err(|e| DomainError::storage(format!("Failed to record dream run: {e}")))?;
            Ok(())
        })
        .await
    }

    async fn last_dream_run(&self) -> Result<Option<DreamRun>, DomainError> {
        self.query_blocking(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, started_at, finished_at, sessions_imported, clusters_found, \
                            operations_applied, operations_skipped, status \
                     FROM memory_dream_runs ORDER BY finished_at DESC LIMIT 1",
                )
                .map_err(|e| {
                    DomainError::storage(format!("Failed to prepare last_dream_run: {e}"))
                })?;
            match stmt.query_row([], dream_run_from_row) {
                Ok(run) => Ok(Some(run)),
                Err(duckdb::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(DomainError::storage(format!(
                    "Failed to query dream run: {e}"
                ))),
            }
        })
        .await
    }

    async fn stats(&self) -> Result<MemoryStats, DomainError> {
        self.query_blocking(|conn| {
            // Count items by kind
            let mut items_by_kind: Vec<(String, u64)> = Vec::new();
            for kind in MemoryKind::ALL {
                let kind_str = kind.as_str();
                let sql = format!("SELECT COUNT(*) FROM memory_items WHERE kind = '{kind_str}'");
                let count: u64 = conn.query_row(&sql, [], |row| row.get(0)).unwrap_or(0);
                items_by_kind.push((kind_str.to_string(), count));
            }
            let total_items: u64 = items_by_kind.iter().map(|(_, c)| c).sum();

            // Count sessions
            let total_sessions: u64 = conn
                .query_row("SELECT COUNT(*) FROM memory_sessions", [], |row| row.get(0))
                .unwrap_or(0);

            // Count nodes by kind
            let mut nodes_by_kind: Vec<(String, u64)> = Vec::new();
            for kind in NodeKind::ALL {
                let kind_str = kind.as_str();
                let sql = format!("SELECT COUNT(*) FROM memory_nodes WHERE kind = '{kind_str}'");
                let count: u64 = conn.query_row(&sql, [], |row| row.get(0)).unwrap_or(0);
                nodes_by_kind.push((kind_str.to_string(), count));
            }
            let total_nodes: u64 = nodes_by_kind.iter().map(|(_, c)| c).sum();

            Ok(MemoryStats {
                total_items,
                items_by_kind,
                total_sessions,
                total_nodes,
                nodes_by_kind,
            })
        })
        .await
    }

    async fn embedding_model(&self) -> Result<String, DomainError> {
        self.query_blocking(
            |conn| Ok(Self::read_meta(conn, "embedding_model")?.unwrap_or_default()),
        )
        .await
    }

    async fn create_namespace(&self, name: &str) -> Result<bool, DomainError> {
        let name = validate_namespace(name)?;
        let conn = self.conn.lock().await;
        // An empty namespace is recorded with a placeholder row (project = '')
        // so it can exist before any project is assigned. The placeholder is
        // never returned by `namespace_projects` (globals are implicit).
        let existing: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memory_namespaces WHERE namespace = ?1",
                params![name],
                |row| row.get(0),
            )
            .map_err(|e| DomainError::storage(format!("Failed to check namespace: {e}")))?;
        if existing > 0 {
            return Ok(false);
        }
        conn.execute(
            "INSERT INTO memory_namespaces (namespace, project) VALUES (?1, '')",
            params![name],
        )
        .map_err(|e| DomainError::storage(format!("Failed to create namespace: {e}")))?;
        Ok(true)
    }

    async fn delete_namespace(&self, name: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let deleted = conn
            .execute(
                "DELETE FROM memory_namespaces WHERE namespace = ?1",
                params![name],
            )
            .map_err(|e| DomainError::storage(format!("Failed to delete namespace: {e}")))?;
        Ok(deleted > 0)
    }

    async fn assign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError> {
        let namespace = validate_namespace(namespace)?;
        if project.is_empty() {
            return Err(DomainError::invalid_input("project must not be empty"));
        }
        let conn = self.conn.lock().await;
        let existing: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memory_namespaces WHERE namespace = ?1 AND project = ?2",
                params![namespace, project],
                |row| row.get(0),
            )
            .map_err(|e| DomainError::storage(format!("Failed to check membership: {e}")))?;
        if existing > 0 {
            return Ok(false);
        }
        conn.execute(
            "INSERT INTO memory_namespaces (namespace, project) VALUES (?1, ?2)",
            params![namespace, project],
        )
        .map_err(|e| DomainError::storage(format!("Failed to assign project: {e}")))?;
        Ok(true)
    }

    async fn unassign_project(&self, namespace: &str, project: &str) -> Result<bool, DomainError> {
        let conn = self.conn.lock().await;
        let deleted = conn
            .execute(
                "DELETE FROM memory_namespaces WHERE namespace = ?1 AND project = ?2",
                params![namespace, project],
            )
            .map_err(|e| DomainError::storage(format!("Failed to unassign project: {e}")))?;
        Ok(deleted > 0)
    }

    async fn list_namespaces(&self) -> Result<Vec<(String, u64)>, DomainError> {
        let conn = self.conn.lock().await;
        // Count only real project memberships, not the empty-string placeholder.
        let mut stmt = conn
            .prepare(
                "SELECT namespace, SUM(CASE WHEN project = '' THEN 0 ELSE 1 END) \
                 FROM memory_namespaces GROUP BY namespace ORDER BY namespace",
            )
            .map_err(|e| DomainError::storage(format!("Failed to prepare list_namespaces: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
            })
            .map_err(|e| DomainError::storage(format!("Failed to list namespaces: {e}")))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read namespace row: {e}")))
    }

    async fn namespace_projects(&self, namespace: &str) -> Result<Vec<String>, DomainError> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                "SELECT project FROM memory_namespaces \
                 WHERE namespace = ?1 AND project <> '' ORDER BY project",
            )
            .map_err(|e| {
                DomainError::storage(format!("Failed to prepare namespace_projects: {e}"))
            })?;
        let rows = stmt
            .query_map(params![namespace], |row| row.get::<_, String>(0))
            .map_err(|e| {
                DomainError::storage(format!("Failed to query namespace projects: {e}"))
            })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| DomainError::storage(format!("Failed to read project row: {e}")))
    }
}

/// Validate a user-supplied namespace name: non-empty after trimming. Returns
/// the trimmed name.
fn validate_namespace(name: &str) -> Result<&str, DomainError> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(DomainError::invalid_input("namespace must not be empty"));
    }
    Ok(trimmed)
}

/// Escape a string for interpolation into a single-quoted SQL literal.
fn sql_quote(s: &str) -> String {
    s.replace('\'', "''")
}

/// Build the project-scoping `WHERE` fragment for a search over `column`
/// (`i.project` or `project`). `None` → no restriction (search everything).
/// `Some(list)` → global items (`= ''`) plus items in any listed project; an
/// empty list restricts to globals only.
fn project_scope_clause(column: &str, projects: Option<&[String]>) -> Option<String> {
    let projects = projects?;
    if projects.is_empty() {
        return Some(format!("{column} = ''"));
    }
    let in_list = projects
        .iter()
        .map(|p| format!("'{}'", sql_quote(p)))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!("({column} = '' OR {column} IN ({in_list}))"))
}

fn dream_run_from_row(row: &Row<'_>) -> Result<DreamRun, duckdb::Error> {
    Ok(DreamRun {
        id: row.get(0)?,
        started_at: row.get(1)?,
        finished_at: row.get(2)?,
        sessions_imported: row.get::<_, i64>(3)? as usize,
        clusters_found: row.get::<_, i64>(4)? as usize,
        operations_applied: row.get::<_, i64>(5)? as usize,
        operations_skipped: row.get::<_, i64>(6)? as usize,
        status: row.get(7)?,
    })
}

fn session_from_row(row: &Row<'_>) -> Result<ImportedSession, duckdb::Error> {
    Ok(ImportedSession {
        id: row.get(0)?,
        source: row.get(1)?,
        imported_at: row.get(2)?,
        message_count: row.get::<_, i64>(3)? as usize,
        items_written: row.get::<_, i64>(4)? as usize,
        // Nullable on databases migrated from before the column existed; the
        // backfill covers those, and a stray NULL still reads as `Imported`.
        status: row
            .get::<_, Option<String>>(5)?
            .map(|s| SessionStatus::parse(&s))
            .unwrap_or(SessionStatus::Imported),
        last_error: row.get::<_, Option<String>>(6)?,
    })
}
