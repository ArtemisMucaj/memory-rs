//! Hybrid (semantic + keyword) search over the memory store, fused with
//! Reciprocal Rank Fusion — the same retrieval shape as code search, applied
//! to memory items.

use std::collections::HashMap;
use std::sync::Arc;

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::domain::{DomainError, MemoryItem, MemoryKind};

/// RRF dampening constant (standard value used across the codebase).
const RRF_K: f32 = 60.0;

/// How many candidates each leg retrieves before fusion.
const CANDIDATES_PER_LEG: usize = 20;

pub struct MemorySearchUseCase {
    memory_repo: Arc<dyn MemoryRepository>,
    embedding_service: Arc<Embedder>,
}

impl MemorySearchUseCase {
    pub fn new(memory_repo: Arc<dyn MemoryRepository>, embedding_service: Arc<Embedder>) -> Self {
        Self {
            memory_repo,
            embedding_service,
        }
    }

    /// Search memories by natural-language query.
    /// Returns `(item, fused_score)` pairs, best first.
    ///
    /// `project` restricts results to global items plus items belonging to that
    /// project/namespace; `None` searches the whole store.
    pub async fn execute(
        &self,
        query: &str,
        kind: Option<MemoryKind>,
        project: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(MemoryItem, f32)>, DomainError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(DomainError::invalid_input("query must not be empty"));
        }

        let semantic = if self.embedding_service.embeddings_enabled() {
            let vector = self.embedding_service.embed_query(query).await?;
            self.memory_repo
                .search_semantic(&vector, kind, project, CANDIDATES_PER_LEG)
                .await?
        } else {
            Vec::new()
        };
        let keyword = self
            .memory_repo
            .search_keyword(query, kind, project, CANDIDATES_PER_LEG)
            .await?;

        // Reciprocal Rank Fusion over the two ranked lists.
        let mut fused: HashMap<String, (MemoryItem, f32)> = HashMap::new();
        for results in [semantic, keyword] {
            for (rank, (item, _score)) in results.into_iter().enumerate() {
                let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
                fused
                    .entry(item.id().to_string())
                    .and_modify(|(_, score)| *score += contribution)
                    .or_insert((item, contribution));
            }
        }

        let mut results: Vec<(MemoryItem, f32)> = fused.into_values().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }
}
