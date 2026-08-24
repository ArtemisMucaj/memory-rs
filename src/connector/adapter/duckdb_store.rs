//! DuckDB-backed store for the memory system.
//!
//! Memory lives in its own database file (`memory.duckdb`), so the store can
//! be inspected, backed up, or wiped independently. Vectors are stored as
//! fixed-width `FLOAT[dimensions]` columns and recalled with DuckDB's native
//! `array_cosine_distance`. `dimensions` is pinned on first open — a later
//! open at a different width is rejected, since vectors would be
//! incomparable. The embedding *model* is also recorded on first open, but an
//! existing store keeps its original: it is authoritative for retrieval, so
//! queries are embedded with it (see [`MemoryRepository::embedding_model`])
//! rather than being rejected.
//!
//! Schema is deliberately small: `memories`, `entities`, `entity_names`,
//! `memory_embeddings`, `memory_sessions`, `memory_resources`,
//! `memory_resource_embeddings`, `memory_namespaces`, `memory_meta`. There is
//! no migration from older databases — delete the file and re-import.

use std::path::Path;
use std::sync::Arc;

use duckdb::Connection;
use tokio::sync::Mutex;
use tracing::debug;

use crate::domain::DomainError;

/// File name of the memory database inside the data directory.
pub const MEMORY_DB_FILE: &str = "memory.duckdb";

pub struct DuckdbStore {
    /// `pub` so integration tests can introspect the schema (e.g. `SHOW
    /// TABLES`); production code goes through the `MemoryRepository` trait.
    pub conn: Arc<Mutex<Connection>>,
    pub(crate) dimensions: usize,
    /// The embedding model that wrote the stored vectors — the value seeded
    /// on first open (an existing store keeps its original, ignoring the
    /// argument). Authoritative for retrieval; exposed via
    /// [`Self::stored_embedding_model`].
    stored_embedding_model: String,
}

impl DuckdbStore {
    /// Open (or create) the memory database at `db_path`.
    ///
    /// `dimensions` and `embedding_model` describe the embedding setup. Both
    /// are persisted on first open. A later open at a different `dimensions`
    /// is rejected (incomparable vector widths); a different `embedding_model`
    /// is *not* — the stored model wins and is returned by
    /// [`Self::stored_embedding_model`], since it must be used to embed
    /// queries.
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
        let conn = Connection::open_in_memory()
            .map_err(|e| DomainError::storage(format!("Failed to open in-memory database: {e}")))?;
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

        // Fail fast on a pre-simplification database. `CREATE TABLE IF NOT
        // EXISTS` would silently keep the old 14-column `memories` table
        // (with `predicate`, `subject_*`, `object_*`) — SELECTs on the new
        // explicit column list still work, but every INSERT fails with a
        // confusing storage error. Detecting the shape here lets the error
        // name the fix.
        reject_legacy_schema(&conn)?;

        conn.execute_batch(&format!(
            r#"
            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                statement TEXT NOT NULL,
                project TEXT NOT NULL DEFAULT '',
                recorded_at BIGINT NOT NULL,
                source_session_id TEXT,
                source_message_index BIGINT,
                source_kind TEXT NOT NULL,
                confidence DOUBLE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                vector FLOAT[{dimensions}] NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_names (
                -- As written, for display.
                name TEXT NOT NULL,
                -- Lowercased, for lookup. A separate column because the query
                -- used to be `WHERE lower(name) = lower(?)`, and wrapping the
                -- column in a function makes the index unusable.
                name_key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (name_key, entity_id)
            );
            CREATE INDEX IF NOT EXISTS entity_names_key_idx ON entity_names (name_key);
            -- The memory↔entity link: a memory mentions an entity. Many-to-many.
            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS memory_entities_entity_idx ON memory_entities (entity_id);
            CREATE TABLE IF NOT EXISTS memory_sessions (
                -- Composite identity: `id` alone is not enough, because each
                -- discovery source (Claude, OpenCode, Zed) has an independent
                -- ID space and two sources can mint the same value.
                id TEXT NOT NULL,
                source TEXT NOT NULL,
                imported_at BIGINT NOT NULL,
                message_count BIGINT NOT NULL,
                items_written BIGINT NOT NULL,
                status TEXT NOT NULL DEFAULT 'imported',
                last_error TEXT,
                project TEXT,
                PRIMARY KEY (source, id)
            );
            CREATE TABLE IF NOT EXISTS memory_resources (
                uri TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                name TEXT NOT NULL,
                abstract TEXT NOT NULL,
                overview TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at BIGINT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_resource_embeddings (
                uri TEXT PRIMARY KEY,
                vector FLOAT[{dimensions}] NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_namespaces (
                namespace TEXT NOT NULL,
                project TEXT NOT NULL,
                created_at BIGINT,
                UNIQUE (namespace, project)
            );
            "#
        ))
        .map_err(|e| DomainError::storage(format!("Failed to initialize memory schema: {e}")))?;

        // `dimensions` is a hard pin: vectors of different widths cannot be
        // compared, so a mismatch is a genuine incompatibility and is rejected.
        Self::check_meta(&conn, "dimensions", &dimensions.to_string())?;
        // The embedding *model* is seeded on a fresh store but NOT rejected
        // on reopen: the stored model is authoritative for retrieval (queries
        // must be embedded with whatever wrote the vectors).
        Self::seed_meta(&conn, "embedding_model", embedding_model)?;
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
    pub fn stored_embedding_model(&self) -> &str {
        &self.stored_embedding_model
    }

    /// Pin `key = expected`: set on first open, must match on later opens.
    pub(crate) fn check_meta(
        conn: &Connection,
        key: &str,
        expected: &str,
    ) -> Result<(), DomainError> {
        match Self::read_meta(conn, key)? {
            None => {
                conn.execute(
                    "INSERT INTO memory_meta (key, value) VALUES (?, ?)",
                    duckdb::params![key, expected],
                )
                .map_err(|e| DomainError::storage(format!("Failed to pin memory_meta: {e}")))?;
                Ok(())
            }
            Some(existing) if existing == expected => Ok(()),
            Some(existing) => Err(DomainError::storage(format!(
                "memory database was opened with incompatible {key}: stored '{existing}', requested '{expected}'"
            ))),
        }
    }

    /// Seed `key = value` if not already set, returning the stored value.
    pub(crate) fn seed_meta(conn: &Connection, key: &str, value: &str) -> Result<(), DomainError> {
        if Self::read_meta(conn, key)?.is_none() {
            conn.execute(
                "INSERT INTO memory_meta (key, value) VALUES (?, ?)",
                duckdb::params![key, value],
            )
            .map_err(|e| DomainError::storage(format!("Failed to seed memory_meta: {e}")))?;
        }
        Ok(())
    }

    pub(crate) fn read_meta(conn: &Connection, key: &str) -> Result<Option<String>, DomainError> {
        let mut stmt = conn
            .prepare("SELECT value FROM memory_meta WHERE key = ?")
            .map_err(|e| DomainError::storage(format!("Failed to read memory_meta: {e}")))?;
        let mut rows = stmt
            .query(duckdb::params![key])
            .map_err(|e| DomainError::storage(format!("Failed to read memory_meta: {e}")))?;
        let value = rows
            .next()
            .map_err(|e| DomainError::storage(format!("Failed to read memory_meta: {e}")))?
            .map(|row| row.get::<_, String>(0));
        match value {
            Some(Ok(v)) => Ok(Some(v)),
            Some(Err(e)) => Err(DomainError::storage(format!(
                "Failed to decode memory_meta: {e}"
            ))),
            None => Ok(None),
        }
    }
}

/// Storage helper: `project` is stored as the empty string for global rows,
/// because SQL treats `NULL`s as distinct inside a `UNIQUE` and would let
/// duplicate global rows through. Read side of the convention.
pub(crate) fn project_from_column(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Storage helper, write side of the same convention.
pub(crate) fn project_to_column(project: Option<&str>) -> &str {
    project.unwrap_or("")
}

/// Build the `WHERE` clause fragment for project scoping. `None` means every
/// scope (no filter); `Some(&[])` means globals only; `Some(&["a","b"])`
/// means globals plus those projects.
///
/// Appends bind parameters to `params_out` in the order they appear in the
/// returned SQL.
pub(crate) fn project_scope_clause(
    projects: Option<&[String]>,
    column: &str,
    params_out: &mut Vec<String>,
) -> String {
    match projects {
        None => String::new(),
        Some([]) => format!(" AND {column} = ''"),
        Some(projects) => {
            let placeholders = std::iter::repeat_n("?", projects.len())
                .collect::<Vec<_>>()
                .join(", ");
            for p in projects {
                params_out.push(p.clone());
            }
            format!(" AND ({column} = '' OR {column} IN ({placeholders}))")
        }
    }
}

/// Reject a database written by the pre-simplification schema (the one with
/// `predicate`, `subject_*`, `object_*` columns on `memories`). The old
/// store cannot be upgraded in place — there is no meaningful mapping of
/// `subject/predicate/object` onto a mention list — so the fix is to wipe
/// the file and re-import. Saying that here beats failing at first write
/// with a confusing type error.
fn reject_legacy_schema(conn: &Connection) -> Result<(), DomainError> {
    // `PRAGMA table_info` errors when the table does not exist, which is
    // exactly the fresh-database case — treat that as fine.
    let mut stmt = match conn.prepare("PRAGMA table_info(memories)") {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let mut rows = match stmt.query([]) {
        Ok(r) => r,
        Err(_) => return Ok(()),
    };
    let mut columns: Vec<String> = Vec::new();
    while let Ok(Some(row)) = rows.next() {
        if let Ok(name) = row.get::<_, String>(1) {
            columns.push(name);
        }
    }
    if columns.is_empty() {
        return Ok(());
    }
    // The legacy schema carries `predicate` and `subject_entity_id`; the
    // simplified one does not. Either is proof of a pre-simplification file.
    if columns
        .iter()
        .any(|c| c == "predicate" || c == "subject_entity_id")
    {
        return Err(DomainError::storage(
            "this memory database was written by an older version of memory-rs \
             and cannot be upgraded in place. Delete `memory.duckdb` and \
             re-import your sessions (e.g. `memory-rs import <transcript>` or \
             `memory-rs dream` to harvest)."
                .to_string(),
        ));
    }
    Ok(())
}
