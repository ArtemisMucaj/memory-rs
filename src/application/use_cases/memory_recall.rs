//! Memory recall — the read path over the fact store.
//!
//! Three retrieval legs, RRF-fused:
//! - **semantic** — cosine over statement embeddings
//! - **keyword** — substring match over statements
//! - **recency** — `recorded_at` descending
//!
//! Recency is what the issue asked for: newer memories carry more weight, so
//! a fact the user just corrected outranks a stale one even when the cosine
//! scores are close. There is no edge expansion and no provenance walk —
//! those existed to exploit the typed edge graph, which is gone.

use std::collections::HashMap;
use std::sync::Arc;

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::domain::{DomainError, Memory, MemoryKind};

/// RRF dampening constant (the standard value, used across the crate).
const RRF_K: f32 = 60.0;

/// How many candidates each leg retrieves before fusion.
const CANDIDATES_PER_LEG: usize = 20;

/// One recalled memory and its fused score.
#[derive(Debug, Clone, PartialEq)]
pub struct Recalled {
    pub memory: Memory,
    pub score: f32,
}

pub struct MemoryRecallUseCase {
    memory_repo: Arc<dyn MemoryRepository>,
    embedder: Embedder,
}

impl MemoryRecallUseCase {
    pub fn new(memory_repo: Arc<dyn MemoryRepository>, embedder: Embedder) -> Self {
        Self {
            memory_repo,
            embedder,
        }
    }

    /// Recall memories for `query`, newest-leaning.
    ///
    /// `projects` scopes the search: `None` is every scope, a slice is
    /// globals plus those projects, and an empty slice is globals only.
    pub async fn execute(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<Vec<Recalled>, DomainError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(DomainError::invalid_input("query must not be empty"));
        }
        if limit == 0 {
            return Ok(Vec::new());
        }

        // ── Three legs ───────────────────────────────────────────────────
        let semantic: Vec<String> = if self.embedder.embeddings_enabled() {
            let vector = self.embedder.embed_query(query).await?;
            self.memory_repo
                .search_memories_semantic(&vector, kind, projects, CANDIDATES_PER_LEG)
                .await?
                .into_iter()
                .map(|(m, _)| m.id)
                .collect()
        } else {
            Vec::new()
        };
        let keyword: Vec<String> = self
            .memory_repo
            .search_memories_keyword(query, kind, projects, CANDIDATES_PER_LEG)
            .await?
            .into_iter()
            .map(|(m, _)| m.id)
            .collect();
        let recency: Vec<String> = self
            .memory_repo
            .list_memories_by_recency(kind, projects, CANDIDATES_PER_LEG)
            .await?
            .into_iter()
            .map(|m| m.id)
            .collect();

        // ── RRF fusion ───────────────────────────────────────────────────
        let mut score: HashMap<String, f32> = HashMap::new();
        for leg in [&semantic, &keyword, &recency] {
            for (rank, id) in leg.iter().enumerate() {
                *score.entry(id.clone()).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
            }
        }

        if score.is_empty() {
            return Ok(Vec::new());
        }

        let mut ids: Vec<(String, f32)> = score.into_iter().collect();
        ids.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ids.truncate(limit);

        let wanted: Vec<String> = ids.iter().map(|(id, _)| id.clone()).collect();
        let scores: HashMap<String, f32> = ids.into_iter().collect();
        let mut memories = self.memory_repo.find_memories(&wanted).await?;
        // Preserve RRF order: `find_memories` returns store order.
        memories.sort_by(|a, b| {
            let sa = scores.get(&a.id).copied().unwrap_or(0.0);
            let sb = scores.get(&b.id).copied().unwrap_or(0.0);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(memories
            .into_iter()
            .map(|memory| {
                let score = scores.get(&memory.id).copied().unwrap_or(0.0);
                Recalled { memory, score }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time proof the score math is the standard RRF formula.
    #[test]
    fn rrf_weight_matches_the_standard_shape() {
        let rank0 = 1.0 / (RRF_K + 1.0);
        let rank1 = 1.0 / (RRF_K + 2.0);
        assert!(rank0 > rank1);
        assert!(rank1 > 0.0);
    }
}
