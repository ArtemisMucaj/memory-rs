//! A file or URL added explicitly via `memory-rs add`. The LLM writes a
//! one-line abstract (L0) and a longer overview (L1) at ingest time; the full
//! content is kept alongside. What is gone is the `memory://` *tree* and the
//! `MemoryNode` type — resources are their own table, not tree nodes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryResource {
    /// `memory://resources/<slug>` — primary key.
    pub uri: String,
    /// Original source (file path or URL) the content was fetched from.
    pub source: String,
    /// Display name (slug).
    pub name: String,
    /// L0 — one-line summary, what recall ranks and listings display.
    pub abstract_: String,
    /// L1 — a paragraph orienting the reader before opening `content`.
    pub overview: String,
    /// Full text.
    pub content: String,
    pub created_at: i64,
}

impl MemoryResource {
    /// Text used for the embedding — abstract plus overview, mirroring how
    /// the old `MemoryNode::embedding_text` combined the two levels.
    pub fn embedding_text(&self) -> String {
        if self.overview.trim().is_empty() {
            self.abstract_.clone()
        } else {
            format!("{}\n\n{}", self.abstract_, self.overview)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_text_combines_abstract_and_overview() {
        let r = MemoryResource {
            uri: "memory://resources/x".into(),
            source: "/tmp/x.md".into(),
            name: "x".into(),
            abstract_: "A note about x".into(),
            overview: "Longer context about x.".into(),
            content: "hello world".into(),
            created_at: 0,
        };
        assert_eq!(
            r.embedding_text(),
            "A note about x\n\nLonger context about x."
        );
    }

    #[test]
    fn embedding_text_falls_back_to_abstract_when_overview_empty() {
        let r = MemoryResource {
            uri: "memory://resources/x".into(),
            source: "/tmp/x.md".into(),
            name: "x".into(),
            abstract_: "A note about x".into(),
            overview: String::new(),
            content: "hello world".into(),
            created_at: 0,
        };
        assert_eq!(r.embedding_text(), "A note about x");
    }
}
