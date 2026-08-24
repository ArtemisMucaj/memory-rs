//! The memory store model.
//!
//! A memory is a self-contained English statement plus a set of entity
//! references — the durable things the statement is *about*. There is no
//! predicate and no subject/object split: the verb is in the statement,
//! where a reader looks for it. Entities anchor a memory to the things it
//! mentions (people, tools, services, libraries, concepts), so "what do we
//! know about X" is a join rather than a text search.

use serde::{Deserialize, Serialize};

use crate::domain::memory::MemoryKind;

/// Where a memory came from. `user_stated` is what the user actually said;
/// `extracted` is anything the model produced from a transcript — whether
/// it read the fact straight off the page or pieced it together from
/// context. The distinction between those two is not stable enough to be
/// worth a third variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// The user asserted it directly — the most trusted.
    UserStated,
    /// The extraction model produced it from the transcript.
    Extracted,
}

impl SourceKind {
    /// Stable identifier used in storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceKind::UserStated => "user_stated",
            SourceKind::Extracted => "extracted",
        }
    }

    /// Parse a stored value. The retired `assistant_inferred` and `derived`
    /// both fold into `Extracted` — they were the same fact seen through
    /// two names.
    pub fn parse(s: &str) -> Option<SourceKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "user_stated" => Some(SourceKind::UserStated),
            "extracted" | "assistant_inferred" | "derived" => Some(SourceKind::Extracted),
            _ => None,
        }
    }

    /// Trust rank, higher being more trusted.
    pub fn trust_rank(&self) -> u8 {
        match self {
            SourceKind::UserStated => 1,
            SourceKind::Extracted => 0,
        }
    }
}

/// Entity types the extraction model may use. `project` is deliberately not
/// on the list — projects live on `Memory.project`, not as entities.
pub const VALID_ENTITY_TYPES: &[&str] = &["person", "tool", "service", "library", "concept"];

/// A resolved, canonical entity: an anchor a memory points at.
///
/// Entities are global (not project-scoped); the project scope lives on the
/// [`Memory`]. `names` holds every surface form the entity is known by —
/// the canonical name plus aliases learned on write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    /// Coarse type: `person`, `tool`, `service`, `library`, `concept`.
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
/// Only trailing occurrences are stripped, and never the whole name —
/// "service" alone is a name, not a role word.
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
/// Normalizing moves spelling variants to the exact-match tier: lowercase,
/// drop a leading article, unify separators, and strip trailing role words.
/// "orders-events", "the orders-events service" and "orders-events package"
/// all land on the same key.
///
/// This deliberately makes the key *broader* than the name. Two genuinely
/// distinct things whose names differ only by a role word — a `foo` package
/// and a separate `foo` service — now share a key. The type guard in
/// entity resolution is what keeps those apart when their types differ;
/// when they are both `service`, they merge, which is the trade this key
/// is making on purpose.
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

    // Strip trailing role words, but never all of them — a name made only
    // of role words ("the service") is a name.
    while words.len() > 1 && ROLE_WORDS.contains(words.last().unwrap()) {
        words.pop();
    }

    let key = words.join("-");
    if key.is_empty() { lowered } else { key }
}

/// A single fact in the memory store.
///
/// Fields are public because a memory is a record-like value, constructed
/// by the ingestion path and read back by retrieval.
///
/// Field order is load-bearing for the storage adapter, which builds its
/// `INSERT`/`SELECT` column lists in this order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    /// Taxonomy kept as a single-variant enum so the storage column stays
    /// forward-compatible — see [`MemoryKind`].
    pub kind: MemoryKind,
    /// The fact itself, written to read on its own.
    pub statement: String,
    /// Ids of the entities this fact mentions. May be empty (a fact about
    /// the user, or about nothing durable).
    pub entity_ids: Vec<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [SourceKind; 2] = [SourceKind::UserStated, SourceKind::Extracted];

    /// The failure this guards, observed in a real store: one session
    /// produced both a "orders-events package" entity and a "orders-events
    /// service" entity for the same service, because neither the name tier
    /// (raw string compare) nor the similarity tier (0.91, under the
    /// attribute threshold) could see they were the same.
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

    /// A name made only of role words is a name: stripping it to nothing
    /// would make every such entity collide with every other.
    #[test]
    fn a_name_that_is_only_role_words_survives() {
        assert_eq!(entity_name_key("service"), "service");
        assert_eq!(entity_name_key("the API"), "api");
        // "package service" is two role words, so only the last one goes.
        assert_eq!(entity_name_key("package service"), "package");
    }

    #[test]
    fn trust_rank_orders_user_above_extracted() {
        assert!(SourceKind::UserStated.trust_rank() > SourceKind::Extracted.trust_rank());
    }

    #[test]
    fn source_kind_round_trips_through_storage_strings() {
        for kind in KINDS {
            assert_eq!(SourceKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(
            SourceKind::parse("  USER_STATED "),
            Some(SourceKind::UserStated)
        );
    }

    #[test]
    fn retired_source_kinds_fold_into_extracted() {
        // Rows written by an older build stay readable.
        assert_eq!(
            SourceKind::parse("assistant_inferred"),
            Some(SourceKind::Extracted)
        );
        assert_eq!(SourceKind::parse("derived"), Some(SourceKind::Extracted));
    }

    #[test]
    fn project_is_not_an_entity_type() {
        assert!(!VALID_ENTITY_TYPES.contains(&"project"));
    }
}
