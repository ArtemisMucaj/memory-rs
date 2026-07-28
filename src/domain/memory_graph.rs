//! The append-only memory graph model.
//!
//! This is the storage-facing vocabulary for the memory-graph memory model: an
//! immutable log of [`Memory`]s linked by typed [`MemoryEdge`]s over resolved
//! [`Entity`] nodes. Unlike the [`MemoryItem`](super::MemoryItem) store, a
//! memory is never rewritten in place — an "update" is a new memory plus a
//! `supersedes` edge, and the current-truth view is the set of `active` memories.
//!
//! The model deliberately keeps a **single event timeline** (`recorded_at`,
//! with `valid_to` closed on supersession) rather than an independent
//! world-valid time, and a coarse `(session_id, message_index)` provenance.
//! Both are simplifications: a bitemporal model and per-span provenance buy
//! precision nothing downstream currently reads, at the cost of a write path
//! that runs on every ingested message.
//!
//! Memories share the [`MemoryKind`] taxonomy and the same `memory.duckdb` file
//! as [`MemoryItem`](super::MemoryItem) — they are a second projection over one
//! store, not a second store.

use serde::{Deserialize, Serialize};

use crate::domain::memory::MemoryKind;

/// The subject or object of a [`Memory`]: either a resolved canonical entity
/// (referenced by id) or a literal value rendered as text.
///
/// A memory's subject is normally an [`EntityRef::Entity`]; its object may be
/// either (e.g. `has_pet -> Entity("dog_rex")` vs `prefers -> Literal("tabs")`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
pub enum EntityRef {
    /// Reference to an [`Entity`] by its id.
    Entity(String),
    /// A literal value (string, number, date — stored as text).
    Literal(String),
}

impl EntityRef {
    /// The referenced entity id, if this is an [`EntityRef::Entity`].
    pub fn entity_id(&self) -> Option<&str> {
        match self {
            EntityRef::Entity(id) => Some(id),
            EntityRef::Literal(_) => None,
        }
    }

    /// The literal value, if this is an [`EntityRef::Literal`].
    pub fn literal(&self) -> Option<&str> {
        match self {
            EntityRef::Literal(v) => Some(v),
            EntityRef::Entity(_) => None,
        }
    }

    /// Reconstruct from the `(entity_id, literal)` column pair used in storage.
    /// Prefers the entity id when both are somehow present.
    pub fn from_columns(entity_id: Option<String>, literal: Option<String>) -> Self {
        match entity_id {
            Some(id) => EntityRef::Entity(id),
            None => EntityRef::Literal(literal.unwrap_or_default()),
        }
    }
}

/// Where a memory came from, used to arbitrate contradictions. The ordering
/// `user_stated > assistant_inferred > derived` is the primary arbiter (see
/// [`arbitrate`]); `confidence` is only a tiebreak within one source kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// The user asserted it directly — the most trusted.
    UserStated,
    /// The assistant inferred it from context.
    AssistantInferred,
    /// Produced by the consolidation pass, not primary observation.
    Derived,
}

impl SourceKind {
    /// Stable identifier used in storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceKind::UserStated => "user_stated",
            SourceKind::AssistantInferred => "assistant_inferred",
            SourceKind::Derived => "derived",
        }
    }

    pub fn parse(s: &str) -> Option<SourceKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "user_stated" => Some(SourceKind::UserStated),
            "assistant_inferred" => Some(SourceKind::AssistantInferred),
            "derived" => Some(SourceKind::Derived),
            _ => None,
        }
    }

    /// Trust rank, higher being more trusted.
    ///
    /// Deliberately **not** consulted on the ingestion path. Ingestion honours
    /// a `supersedes` relation unconditionally, because supersession is a
    /// temporal chain and the newest link is the current answer — and because a
    /// wrong link there is now visible (it comes back in a result's provenance)
    /// and correctable (consolidation can supersede it in turn), rather than
    /// silent and permanent as it was when supersession hid a memory forever.
    ///
    /// It exists for consolidation, which reconciles `contradicts` pairs with
    /// the whole neighbourhood in view and can tell an ingestion-authored edge
    /// from its own via [`EdgeOrigin`]. Nothing calls it until that pass lands.
    pub fn trust_rank(&self) -> u8 {
        match self {
            SourceKind::UserStated => 2,
            SourceKind::AssistantInferred => 1,
            SourceKind::Derived => 0,
        }
    }
}

/// Lifecycle state of a memory. The current-truth projection is the set of
/// [`MemoryStatus::Active`] memories.
///
/// There is no "unresolved conflict" state, on purpose. Two memories that
/// contradict each other both stay `Active` and both keep answering queries,
/// with the `contradicts` edge between them reported alongside. Hiding both
/// would be the more cautious-looking choice and the less honest one: the true
/// answer to a contested question is *these two things are on record and they
/// disagree*, not silence. The conflict queue is therefore derived — the
/// contradictions whose endpoints are still active — rather than a status a
/// memory can get stuck in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    /// Current and believed true.
    Active,
    /// Replaced by a newer memory via a `supersedes` edge; kept for history.
    Superseded,
    /// Marked as never having been true (a bad extraction).
    Retracted,
}

impl MemoryStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryStatus::Active => "active",
            MemoryStatus::Superseded => "superseded",
            MemoryStatus::Retracted => "retracted",
        }
    }

    pub fn parse(s: &str) -> Option<MemoryStatus> {
        match s.trim().to_ascii_lowercase().as_str() {
            "active" => Some(MemoryStatus::Active),
            "superseded" => Some(MemoryStatus::Superseded),
            "retracted" => Some(MemoryStatus::Retracted),
            _ => None,
        }
    }
}

/// The typed relationship between two memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// Temporal replacement — old was true, new is true now.
    Supersedes,
    /// Genuine conflict, no temporal ordering.
    Contradicts,
    /// Enrichment / specialization of a broader memory.
    Refines,
    /// The target memory was never true (bad extraction).
    Retracts,
    /// An independent source confirms the target memory.
    Corroborates,
    /// Generic association discovered later; navigational only.
    RelatesTo,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::Supersedes => "supersedes",
            EdgeType::Contradicts => "contradicts",
            EdgeType::Refines => "refines",
            EdgeType::Retracts => "retracts",
            EdgeType::Corroborates => "corroborates",
            EdgeType::RelatesTo => "relates_to",
        }
    }

    pub fn parse(s: &str) -> Option<EdgeType> {
        match s.trim().to_ascii_lowercase().as_str() {
            "supersedes" => Some(EdgeType::Supersedes),
            "contradicts" => Some(EdgeType::Contradicts),
            "refines" => Some(EdgeType::Refines),
            "retracts" => Some(EdgeType::Retracts),
            "corroborates" => Some(EdgeType::Corroborates),
            "relates_to" => Some(EdgeType::RelatesTo),
            _ => None,
        }
    }
}

/// Who created an edge — provenance for the graph itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeOrigin {
    /// Written on the online ingestion path.
    Ingestion,
    /// Written by the offline consolidation ("dream") pass.
    Consolidation,
    /// Written by an explicit user/manual action.
    Manual,
}

impl EdgeOrigin {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeOrigin::Ingestion => "ingestion",
            EdgeOrigin::Consolidation => "consolidation",
            EdgeOrigin::Manual => "manual",
        }
    }

    pub fn parse(s: &str) -> Option<EdgeOrigin> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ingestion" => Some(EdgeOrigin::Ingestion),
            "consolidation" => Some(EdgeOrigin::Consolidation),
            "manual" => Some(EdgeOrigin::Manual),
            _ => None,
        }
    }
}

/// A resolved, canonical entity: the anchor a memory's subject/object points at.
///
/// `aliases` are the surface forms that resolve to this entity (e.g. `"Alice"`,
/// `"my coworker Alice"`), kept separate from `canonical_name`. Entities are
/// global (not project-scoped); the project scope lives on the [`Memory`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    /// Coarse type: `person`, `place`, `project`, `tool`, …
    pub entity_type: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A single immutable memory in the append-only log.
///
/// Fields are public because a memory is a record-like value (as with
/// [`ImportedSession`](super::ImportedSession) / [`DreamRun`](super::DreamRun)),
/// constructed by the ingestion path and read back by retrieval; it carries no
/// invariants beyond those the store enforces.
///
/// Field order is load-bearing for the storage adapter, which builds its
/// `INSERT`/`SELECT` column lists in this order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    /// Taxonomy shared with [`MemoryItem`](super::MemoryItem) rather than a
    /// memory-only vocabulary, so one recall path can filter both projections by
    /// the same kind instead of maintaining a mapping between two enums.
    pub kind: MemoryKind,
    /// Resolved subject — normally an [`EntityRef::Entity`].
    pub subject: EntityRef,
    /// Predicate from a small controlled-ish vocabulary (e.g. `prefers`,
    /// `lives_in`, `uses`).
    pub predicate: String,
    pub object: EntityRef,
    /// Human-readable rendering of the triple, used for embedding and display.
    pub statement: String,
    /// Project/namespace scope, or `None` for a global memory. Resolved at
    /// ingestion, mirroring `MemoryItem::project`. Storage flattens `None` to
    /// the empty string rather than `NULL`, because SQL treats `NULL`s as
    /// distinct inside a `UNIQUE` and would let duplicate global rows through.
    pub project: Option<String>,

    /// When this memory entered the log (the single event timeline).
    pub recorded_at: i64,
    /// Defaults to `recorded_at`; only distinct when an explicit date was lifted
    /// from the source text.
    pub valid_from: i64,
    /// Set to the recording time of the memory that supersedes this one; `None`
    /// while the memory is still current.
    pub valid_to: Option<i64>,

    /// Session this memory was extracted from (provenance, half 1).
    pub source_session_id: Option<String>,
    /// Index of the transcript message it came from (provenance, half 2).
    pub source_message_index: Option<i64>,
    pub source_kind: SourceKind,
    /// Best-effort confidence in `[0, 1]`; advisory only — see [`arbitrate`]
    /// for exactly how little weight it carries.
    pub confidence: f32,

    pub status: MemoryStatus,
    /// True when produced by consolidation rather than ingestion.
    pub derived: bool,
    /// Source memory ids this was derived from (empty for primary memories).
    pub derived_from: Vec<String>,
}

/// A typed, directed edge between two memories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub from_memory: String,
    pub to_memory: String,
    pub edge_type: EdgeType,
    pub created_at: i64,
    pub created_by: EdgeOrigin,
    pub confidence: f32,
}

/// Aggregate statistics about the memory store.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryStoreStats {
    pub total_memories: u64,
    /// Memory counts by `status` string. Everything not `active` is history:
    /// superseded links in a chain, or retractions.
    pub memories_by_status: Vec<(String, u64)>,
    /// Memory counts by [`MemoryKind`] string, mirroring the item store's
    /// per-kind breakdown so the same UI can render it.
    pub memories_by_kind: Vec<(String, u64)>,
    pub total_entities: u64,
    pub total_edges: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [SourceKind; 3] = [
        SourceKind::UserStated,
        SourceKind::AssistantInferred,
        SourceKind::Derived,
    ];

    /// `trust_rank` gates nothing today — consolidation is its only future
    /// caller — but the ordering it encodes is the thing that pass will lean
    /// on, so it is pinned here rather than left to be rediscovered.
    #[test]
    fn trust_rank_orders_user_above_inferred_above_derived() {
        assert!(SourceKind::UserStated.trust_rank() > SourceKind::AssistantInferred.trust_rank());
        assert!(SourceKind::AssistantInferred.trust_rank() > SourceKind::Derived.trust_rank());
    }

    #[test]
    fn memory_enums_round_trip_through_storage_strings() {
        for kind in KINDS {
            assert_eq!(SourceKind::parse(kind.as_str()), Some(kind));
        }
        for status in [
            MemoryStatus::Active,
            MemoryStatus::Superseded,
            MemoryStatus::Retracted,
        ] {
            assert_eq!(MemoryStatus::parse(status.as_str()), Some(status));
        }
        for edge_type in [
            EdgeType::Supersedes,
            EdgeType::Contradicts,
            EdgeType::Refines,
            EdgeType::Retracts,
            EdgeType::Corroborates,
            EdgeType::RelatesTo,
        ] {
            assert_eq!(EdgeType::parse(edge_type.as_str()), Some(edge_type));
        }
        for origin in [
            EdgeOrigin::Ingestion,
            EdgeOrigin::Consolidation,
            EdgeOrigin::Manual,
        ] {
            assert_eq!(EdgeOrigin::parse(origin.as_str()), Some(origin));
        }
        assert_eq!(
            SourceKind::parse("  USER_STATED "),
            Some(SourceKind::UserStated)
        );
        assert_eq!(MemoryStatus::parse("nonsense"), None);
    }

    /// The status a memory could once get stuck in is gone. A store written by
    /// an older build may still hold the string, and it must not resurrect the
    /// variant or silently parse as something else — the adapter's
    /// `unwrap_or(Active)` is what puts such a row back in circulation.
    #[test]
    fn the_retired_needs_resolution_status_no_longer_parses() {
        assert_eq!(MemoryStatus::parse("needs_resolution"), None);
    }

    #[test]
    fn memory_entity_ref_from_columns_prefers_the_entity_id() {
        assert_eq!(
            EntityRef::from_columns(Some("entity-1".to_string()), Some("stale".to_string())),
            EntityRef::Entity("entity-1".to_string()),
        );
        assert_eq!(
            EntityRef::from_columns(None, Some("tabs".to_string())),
            EntityRef::Literal("tabs".to_string()),
        );
        // A row with neither column set degrades to an empty literal rather
        // than failing the whole read.
        assert_eq!(
            EntityRef::from_columns(None, None),
            EntityRef::Literal(String::new()),
        );
    }

    #[test]
    fn memory_entity_ref_accessors_are_exclusive() {
        let entity = EntityRef::Entity("entity-1".to_string());
        assert_eq!(entity.entity_id(), Some("entity-1"));
        assert_eq!(entity.literal(), None);

        let literal = EntityRef::Literal("tabs".to_string());
        assert_eq!(literal.literal(), Some("tabs"));
        assert_eq!(literal.entity_id(), None);
    }
}
