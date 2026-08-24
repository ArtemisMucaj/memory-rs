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

/// The relation in a memory's subject–predicate–object triple.
///
/// Closed vocabulary, deliberately small. Anything that does not fit one of
/// these seven should be expressed with [`Predicate::RelatesTo`] and carried
/// by the `statement` text — the statement is what recall reads; the
/// predicate is a coarse filter at best.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
    /// A durable taste or habit of the user.
    Prefers,
    /// A durable dislike, or something deliberately not done.
    Avoids,
    /// Depends on, is built with, employs.
    Uses,
    /// Resolves a problem.
    Fixes,
    /// A choice that was made, with its rationale in the statement.
    Decided,
    /// Type or category membership.
    IsA,
    /// Escape hatch — a genuine relation none of the above expresses.
    RelatesTo,
}

impl Predicate {
    pub const ALL: [Predicate; 7] = [
        Predicate::Prefers,
        Predicate::Avoids,
        Predicate::Uses,
        Predicate::Fixes,
        Predicate::Decided,
        Predicate::IsA,
        Predicate::RelatesTo,
    ];

    /// Stable identifier used in storage and in the extraction JSON protocol.
    pub fn as_str(&self) -> &'static str {
        match self {
            Predicate::Prefers => "prefers",
            Predicate::Avoids => "avoids",
            Predicate::Uses => "uses",
            Predicate::Fixes => "fixes",
            Predicate::Decided => "decided",
            Predicate::IsA => "is_a",
            Predicate::RelatesTo => "relates_to",
        }
    }

    /// One-line meaning, used in the extraction prompt so the model picks on
    /// semantics rather than on which word looks closest.
    pub fn meaning(&self) -> &'static str {
        match self {
            Predicate::Prefers => "a durable taste or habit of the user",
            Predicate::Avoids => "a durable dislike, or something deliberately not done",
            Predicate::Uses => "depends on, is built with, employs",
            Predicate::Fixes => "resolves a problem",
            Predicate::Decided => "a choice that was made (put the rationale in the statement)",
            Predicate::IsA => "type or category membership",
            Predicate::RelatesTo => "none of the above fits — use only as a last resort",
        }
    }

    /// Parse a stored or model-supplied value. Tolerant of case and surrounding
    /// whitespace; a few near-misses the extraction model reaches for are
    /// folded in rather than lost to the escape hatch.
    pub fn parse(s: &str) -> Option<Predicate> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace([' ', '-'], "_")
            .as_str()
        {
            "prefers" | "prefer" | "likes" => Some(Predicate::Prefers),
            "avoids" | "avoid" | "dislikes" => Some(Predicate::Avoids),
            "uses" | "use" | "used" | "utilises" | "utilizes" | "depends_on" => {
                Some(Predicate::Uses)
            }
            "fixes" | "fix" | "fixed" | "resolves" => Some(Predicate::Fixes),
            "decided" | "decides" | "chose" | "chooses" => Some(Predicate::Decided),
            "is_a" | "isa" | "is" | "type_of" => Some(Predicate::IsA),
            "relates_to" | "related_to" | "relates" => Some(Predicate::RelatesTo),
            _ => None,
        }
    }
}

/// A single fact in the memory store.
///
/// Updates are hard delete + insert at the repository layer; there is no
/// lifecycle status and no validity window. Newest write wins.
///
/// Fields are public because a memory is a record-like value, constructed by
/// the ingestion path and read back by retrieval; it carries no invariants
/// beyond those the store enforces.
///
/// Field order is load-bearing for the storage adapter, which builds its
/// `INSERT`/`SELECT` column lists in this order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    /// Taxonomy kept as a single-variant enum so the storage column is
    /// forward-compatible — see [`MemoryKind`].
    pub kind: MemoryKind,
    /// Resolved subject — normally an [`EntityRef::Entity`].
    pub subject: EntityRef,
    /// The relation, from a closed vocabulary — see [`Predicate`].
    pub predicate: Predicate,
    pub object: EntityRef,
    /// Human-readable rendering of the triple, used for embedding and display.
    pub statement: String,
    /// Project/namespace scope, or `None` for a global memory. Storage
    /// flattens `None` to the empty string rather than `NULL`, because SQL
    /// treats `NULL`s as distinct inside a `UNIQUE` and would let duplicate
    /// global rows through.
    pub project: Option<String>,

    /// When this memory entered the store.
    pub recorded_at: i64,

    /// Session this memory was extracted from (provenance, half 1).
    pub source_session_id: Option<String>,
    /// Index of the transcript message it came from (provenance, half 2).
    pub source_message_index: Option<i64>,
    pub source_kind: SourceKind,
    /// Best-effort confidence in `[0, 1]`; advisory only.
    pub confidence: f32,
}

/// Entity types the extraction model may use. `project` is deliberately not
/// on the list — projects live on `Memory.project`, not as entities.
pub const VALID_ENTITY_TYPES: &[&str] = &["person", "tool", "service", "library", "concept"];

/// A resolved, canonical entity: the anchor a memory's subject/object points at.
///
/// (superseded doc — see the struct doc above)
/// `"my coworker Alice"`), kept separate from `canonical_name`. Entities are
/// global (not project-scoped); the project scope lives on the [`Memory`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    /// Coarse type: `person`, `place`, `project`, `tool`, …
    pub entity_type: String,
    pub canonical_name: String,
    pub names: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Words that say what an entity *is* rather than name it. A name differing
/// only by one of these is the same thing described two ways: "orders-events",
/// "the orders-events service" and "orders-events package" are one anchor.
///
/// Only trailing occurrences are stripped, and never the whole name — "service"
/// alone is a name, not a role word.
const ROLE_WORDS: &[&str] = &[
    "service",
    "services",
    "package",
    "packages",
    "repo",
    "repos",
    "repository",
    "library",
    "lib",
    "crate",
    "module",
    "app",
    "application",
    "project",
    "server",
    "daemon",
    "binary",
    "cli",
    "sdk",
    "api",
];

/// The lookup key two surface forms share when they name the same entity.
///
/// Entity resolution has three tiers — exact name, embedding similarity, then a
/// model adjudication — and only the first is free, deterministic and available
/// with embeddings switched off. Matching on the raw lowercased name made it
/// the weakest tier instead of the strongest: every spelling variant fell
/// through to a cosine score, and the variants that matter score *below* the
/// attribute threshold ("orders-events" vs "orders-events service" measures
/// 0.95 on nomic-embed-text; "orders-events package" vs "…service", 0.91), so
/// the decision landed on a small local model that reliably called them
/// different things. Two anchors for one service, permanently.
///
/// Normalizing here moves that whole class of variant back to the exact tier:
/// lowercase, drop a leading article, unify separators, and strip trailing role
/// words.
///
/// This deliberately makes the key *broader* than the name. Two genuinely
/// distinct things whose names differ only by a role word — a `foo` package and
/// a separate `foo` service — now share a key. The type guard in entity
/// resolution is what keeps those apart when their types differ; when they are
/// both `project`, they merge, which is the trade this key is making on
/// purpose.
pub fn entity_name_key(name: &str) -> String {
    let lowered = name.trim().to_lowercase();
    // Punctuation that only ever wraps a name in prose or markdown.
    let stripped = lowered.trim_matches(|c: char| {
        matches!(
            c,
            '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '.' | ',' | ':'
        )
    });
    let stripped = stripped.strip_prefix("the ").unwrap_or(stripped);

    // Separators unify: "orders events", "orders_events" and "orders-events"
    // are the same name typed three ways.
    let mut words: Vec<&str> = stripped
        .split(|c: char| c.is_whitespace() || c == '-' || c == '_' || c == '/')
        .filter(|w| !w.is_empty())
        .collect();

    // Strip trailing role words, but never all of them — a name made only of
    // role words ("the service") is a name.
    while words.len() > 1 && ROLE_WORDS.contains(words.last().unwrap()) {
        words.pop();
    }

    let key = words.join("-");
    if key.is_empty() {
        lowered
    } else {
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [SourceKind; 3] = [
        SourceKind::UserStated,
        SourceKind::AssistantInferred,
        SourceKind::Derived,
    ];

    /// The failure this guards, observed in a real store: one session produced
    /// both a "orders-events package" entity and a "orders-events service"
    /// entity for the same service, because neither the name tier (raw string
    /// compare) nor the similarity tier (0.91, under the attribute threshold)
    /// could see they were the same.
    #[test]
    fn one_name_key_covers_a_name_and_its_role_word_variants() {
        let key = entity_name_key("orders-events");
        for variant in [
            "orders-events",
            "Orders-Events",
            "the orders-events service",
            "orders-events package",
            "orders events",
            "orders_events",
            "`orders-events`",
            "orders-events repository",
            "the orders-events service.",
        ] {
            assert_eq!(entity_name_key(variant), key, "variant {variant:?}");
        }
    }

    #[test]
    fn distinct_names_keep_distinct_keys() {
        assert_ne!(
            entity_name_key("payments-core"),
            entity_name_key("orders-events")
        );
        assert_ne!(entity_name_key("auth-api"), entity_name_key("auth-gateway"));
    }

    /// A name made only of role words is a name: stripping it to nothing would
    /// make every such entity collide with every other.
    #[test]
    fn a_name_that_is_only_role_words_survives() {
        assert_eq!(entity_name_key("service"), "service");
        assert_eq!(entity_name_key("the API"), "api");
        // "package service" is two role words, so only the last one goes.
        assert_eq!(entity_name_key("package service"), "package");
    }

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
        assert_eq!(
            SourceKind::parse("  USER_STATED "),
            Some(SourceKind::UserStated)
        );
    }

    #[test]
    fn project_is_not_an_entity_type() {
        assert!(!VALID_ENTITY_TYPES.contains(&"project"));
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

    #[test]
    fn shrunken_predicate_vocabulary_round_trips() {
        for predicate in Predicate::ALL {
            assert_eq!(Predicate::parse(predicate.as_str()), Some(predicate));
        }
    }

    #[test]
    fn retired_predicates_parse_to_none() {
        for retired in [
            "requires",
            "provides",
            "implements",
            "contains",
            "derived_from",
            "configures",
            "causes",
            "prevents",
            "has",
            "works_on",
        ] {
            assert_eq!(Predicate::parse(retired), None, "{retired}");
        }
    }

    #[test]
    fn predicate_vocabulary_is_seven() {
        assert_eq!(Predicate::ALL.len(), 7);
    }
}
