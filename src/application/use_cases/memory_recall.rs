//! Memory recall — the read path over the memory graph.
//!
//! Graph-anchored, not vector-first: a hybrid (semantic + keyword) search finds
//! *entry-point* memories, then a bounded 1-hop walk over enrichment edges
//! (`refines` / `corroborates` / `relates_to`) pulls in their neighbours.
//!
//! The distinction that matters is which edges are navigational. `refines` and
//! `corroborates` say something about the *content* of a neighbouring memory, so
//! following them adds information a keyword match missed. `supersedes`,
//! `contradicts` and `retracts` say something about a memory's *status* — they
//! exist to explain why something is not current, and following them would drag
//! exactly the stale memories the store went to the trouble of retiring back into
//! the answer.
//!
//! No conflict resolution happens here, and none is needed: contradicting
//! memories are both active by design and both belong in the results, with the
//! disagreement reported alongside them. What the storage legs do filter out is
//! superseded and retracted memories — in SQL, in the adapter, so that no
//! future caller can forget to.

use std::collections::HashMap;
use std::sync::Arc;

use crate::application::interfaces::{Embedder, MemoryRepository};
use crate::domain::{DomainError, EdgeType, Memory, MemoryKind, MemoryStatus};

/// RRF dampening constant (the standard value, used across the crate).
const RRF_K: f32 = 60.0;

/// How many candidates each hybrid leg retrieves before fusion.
const CANDIDATES_PER_LEG: usize = 20;

/// How many top anchors are expanded across enrichment edges.
const MAX_EXPANSION_SEEDS: usize = 5;

/// Score multiplier applied to a memory pulled in only by graph expansion, so
/// enrichment neighbours rank below the anchors that found them.
const EXPANSION_DECAY: f32 = 0.3;

// A decay of 1.0 or more would let a neighbour outrank the anchor that found
// it, and 0.0 would make expansion pointless. Checked at compile time rather
// than in a test, because a test for a constant can only fail after someone has
// already built and shipped it.
const _: () = assert!(EXPANSION_DECAY > 0.0 && EXPANSION_DECAY < 1.0);

/// Enrichment edge types walked during expansion. See the module doc for why
/// `supersedes` / `contradicts` / `retracts` are deliberately absent.
const EXPANSION_EDGES: [EdgeType; 3] = [
    EdgeType::Refines,
    EdgeType::Corroborates,
    EdgeType::RelatesTo,
];

/// How far back a supersession chain is walked when building provenance.
///
/// A long-lived preference can accumulate a deep chain, and dumping all of it
/// into every result would bury the answer in its own history. The walk is
/// batched — one query per hop, not per memory — so the cost is bounded by this
/// constant rather than by the result count.
const MAX_CHAIN_HOPS: usize = 5;

/// A memory referenced from another memory's provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRef {
    pub id: String,
    pub statement: String,
    pub recorded_at: i64,
}

/// How a recalled memory came to be the answer.
///
/// This is what the graph buys over a flat store. "The user prefers spaces" is
/// a fact; "the user prefers spaces, which replaced 'prefers tabs' recorded in
/// March, corroborated twice since, and currently contradicted by nothing" is
/// an answer you can act on and, more importantly, argue with.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Provenance {
    /// The supersession chain behind this memory, most recently replaced first.
    pub supersedes: Vec<MemoryRef>,
    /// `true` when the chain was deeper than [`MAX_CHAIN_HOPS`], so the tail is
    /// missing. Reported rather than silently dropped — a truncated history
    /// that looks complete is worse than one that says it is not.
    pub chain_truncated: bool,
    /// Still-active memories that contradict this one. A non-empty list means
    /// the store is knowingly holding two irreconcilable answers.
    pub contradicted_by: Vec<MemoryRef>,
    /// How many independent memories corroborate this one.
    pub corroborations: usize,
    /// Memories that refine this one, or that it refines.
    pub refinements: Vec<MemoryRef>,
}

impl Provenance {
    /// Nothing to say about this memory beyond the memory itself.
    pub fn is_empty(&self) -> bool {
        self.supersedes.is_empty()
            && self.contradicted_by.is_empty()
            && self.refinements.is_empty()
            && self.corroborations == 0
    }
}

/// One recalled memory, its score, and the path that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct Recalled {
    pub memory: Memory,
    pub score: f32,
    pub provenance: Provenance,
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

    /// Recall memories for `query`.
    ///
    /// `projects` scopes the search the same way the item search does: `None`
    /// is every scope, a slice is globals plus those projects, and an empty
    /// slice is globals only. It is a slice rather than a single project
    /// because a namespace resolves to many projects and namespace-wide recall
    /// is a shipped feature.
    ///
    /// Returns [`Recalled`] best first — active memories only, each with the
    /// provenance that explains why it is the answer.
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

        // ── Entry points: hybrid semantic + keyword, RRF-fused ────────────
        let semantic = if self.embedder.embeddings_enabled() {
            let vector = self.embedder.embed_query(query).await?;
            self.memory_repo
                .search_memories_semantic(&vector, kind, projects, CANDIDATES_PER_LEG)
                .await?
        } else {
            Vec::new()
        };
        let keyword = self
            .memory_repo
            .search_memories_keyword(query, kind, projects, CANDIDATES_PER_LEG)
            .await?;

        let mut fused: HashMap<String, (Memory, f32)> = HashMap::new();
        for leg in [semantic, keyword] {
            for (rank, (memory, _)) in leg.into_iter().enumerate() {
                let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
                fused
                    .entry(memory.id.clone())
                    .and_modify(|(_, score)| *score += contribution)
                    .or_insert((memory, contribution));
            }
        }

        // ── Expansion: one hop over enrichment edges from the top anchors ──
        let mut seeds: Vec<(String, f32)> = fused
            .iter()
            .map(|(id, (_, score))| (id.clone(), *score))
            .collect();
        // Break score ties on id so a run over the same store is reproducible;
        // HashMap iteration order is not.
        seeds.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        seeds.truncate(MAX_EXPANSION_SEEDS);

        // Collect every neighbour id first, then fetch them in ONE query.
        // Resolving them one at a time is an N+1 against a connection shared
        // with nodes, sessions and digests — and expansion is the one part of
        // recall whose cost scales with how well-connected the graph is.
        let mut wanted: Vec<String> = Vec::new();
        let mut best_inherited: HashMap<String, f32> = HashMap::new();
        for (seed_id, seed_score) in seeds {
            let inherited = seed_score * EXPANSION_DECAY;
            for neighbour_id in self.enrichment_neighbours(&seed_id).await? {
                if fused.contains_key(&neighbour_id) {
                    continue;
                }
                // A memory reachable from two anchors keeps the better score
                // rather than whichever seed happened to be visited last.
                let slot = best_inherited.entry(neighbour_id.clone()).or_insert(0.0);
                if inherited > *slot {
                    *slot = inherited;
                }
                if !wanted.contains(&neighbour_id) {
                    wanted.push(neighbour_id);
                }
            }
        }
        if !wanted.is_empty() {
            for memory in self.memory_repo.find_memories(&wanted).await? {
                // Only surface neighbours that are themselves current. An edge
                // may legitimately point at a superseded memory; the edge is
                // history, the memory is not an answer.
                if memory.status != MemoryStatus::Active {
                    continue;
                }
                let score = best_inherited.get(&memory.id).copied().unwrap_or(0.0);
                fused.insert(memory.id.clone(), (memory, score));
            }
        }

        let mut results: Vec<(Memory, f32)> = fused.into_values().collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.id.cmp(&b.0.id))
        });
        results.truncate(limit);

        // Provenance is built only for what survived the truncation, so its
        // cost scales with `limit` rather than with how many candidates the two
        // legs happened to turn up.
        let ids: Vec<String> = results.iter().map(|(m, _)| m.id.clone()).collect();
        let mut provenance = self.provenance_for(&ids).await?;
        Ok(results
            .into_iter()
            .map(|(memory, score)| {
                let provenance = provenance.remove(&memory.id).unwrap_or_default();
                Recalled {
                    memory,
                    score,
                    provenance,
                }
            })
            .collect())
    }

    /// Build provenance for a whole result set at once.
    ///
    /// Two batched phases. The immediate neighbourhood is one `edges_for` call
    /// covering contradictions, corroborations and refinements; the supersession
    /// chain is then walked backwards a hop at a time, one query per hop, which
    /// bounds the total at `1 + MAX_CHAIN_HOPS` queries however many results
    /// there are.
    async fn provenance_for(
        &self,
        ids: &[String],
    ) -> Result<HashMap<String, Provenance>, DomainError> {
        let mut out: HashMap<String, Provenance> = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        for id in ids {
            out.insert(id.clone(), Provenance::default());
        }

        // ── Immediate neighbourhood ──────────────────────────────────────
        let edges = self.memory_repo.edges_for(ids).await?;
        let owned: std::collections::HashSet<&String> = ids.iter().collect();
        let mut wanted: Vec<String> = Vec::new();
        // (owner, edge kind, other id) collected first so every referenced
        // statement can be fetched in a single query below.
        let mut pending: Vec<(String, EdgeType, String)> = Vec::new();
        // Each endpoint is considered independently, because both sides of an
        // edge can be in the same result set and an `if/else` would silently
        // credit only whichever happened to be checked first.
        for edge in &edges {
            let from_owned = owned.contains(&edge.from_memory);
            let to_owned = owned.contains(&edge.to_memory);
            let mut note = |owner: &String, other: &String| {
                pending.push((owner.clone(), edge.edge_type, other.clone()));
                wanted.push(other.clone());
            };
            match edge.edge_type {
                // Direction is load-bearing: "X corroborates Y" is evidence
                // *for Y*. Crediting X — the restatement — would let a memory
                // inflate its own support just by being repeated back.
                EdgeType::Corroborates => {
                    if to_owned {
                        if let Some(p) = out.get_mut(&edge.to_memory) {
                            p.corroborations += 1;
                        }
                    }
                }
                // Symmetric: each side is contradicted by the other, and both
                // deserve to carry the disagreement.
                EdgeType::Contradicts => {
                    if from_owned {
                        note(&edge.from_memory, &edge.to_memory);
                    }
                    if to_owned {
                        note(&edge.to_memory, &edge.from_memory);
                    }
                }
                // Relevant both ways: the specialisation and what it
                // specialises are each context for the other.
                EdgeType::Refines => {
                    if from_owned {
                        note(&edge.from_memory, &edge.to_memory);
                    }
                    if to_owned {
                        note(&edge.to_memory, &edge.from_memory);
                    }
                }
                // Supersession is handled by the chain walk below, which needs
                // direction: only edges *from* a result point at what it
                // replaced.
                _ => {}
            }
        }

        // ── Supersession chain, one batched hop at a time ────────────────
        let mut chain: HashMap<String, Vec<String>> = HashMap::new();
        let mut truncated: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Frontier maps the id being walked to the result it belongs to.
        let mut frontier: HashMap<String, String> =
            ids.iter().map(|id| (id.clone(), id.clone())).collect();
        let mut visited: std::collections::HashSet<String> = ids.iter().cloned().collect();
        let mut hop_edges = edges;
        for hop in 0..MAX_CHAIN_HOPS {
            if frontier.is_empty() {
                break;
            }
            if hop > 0 {
                let frontier_ids: Vec<String> = frontier.keys().cloned().collect();
                hop_edges = self.memory_repo.edges_for(&frontier_ids).await?;
            }
            let mut next: HashMap<String, String> = HashMap::new();
            for edge in &hop_edges {
                if edge.edge_type != EdgeType::Supersedes {
                    continue;
                }
                let Some(owner) = frontier.get(&edge.from_memory) else {
                    continue;
                };
                // A cycle (A supersedes B, B supersedes A) would otherwise walk
                // forever; a memory is only ever visited once.
                if !visited.insert(edge.to_memory.clone()) {
                    continue;
                }
                chain
                    .entry(owner.clone())
                    .or_default()
                    .push(edge.to_memory.clone());
                wanted.push(edge.to_memory.clone());
                next.insert(edge.to_memory.clone(), owner.clone());
            }
            // Anything still extending on the final hop has a deeper history
            // than was fetched, and the result must say so.
            if hop + 1 == MAX_CHAIN_HOPS {
                for owner in next.values() {
                    truncated.insert(owner.clone());
                }
            }
            frontier = next;
        }

        // ── Resolve every referenced statement in one query ──────────────
        wanted.sort();
        wanted.dedup();
        let by_id: HashMap<String, Memory> = if wanted.is_empty() {
            HashMap::new()
        } else {
            self.memory_repo
                .find_memories(&wanted)
                .await?
                .into_iter()
                .map(|m| (m.id.clone(), m))
                .collect()
        };
        let as_ref = |id: &String| -> Option<MemoryRef> {
            by_id.get(id).map(|m| MemoryRef {
                id: m.id.clone(),
                statement: m.statement.clone(),
                recorded_at: m.recorded_at,
            })
        };

        for (owner, edge_type, other) in pending {
            let Some(reference) = as_ref(&other) else {
                continue;
            };
            let Some(p) = out.get_mut(&owner) else {
                continue;
            };
            match edge_type {
                // A contradiction only counts while the other side is still
                // current: once consolidation supersedes it, the disagreement
                // is settled and reporting it would be misleading.
                EdgeType::Contradicts => {
                    if by_id.get(&other).map(|m| m.status) == Some(MemoryStatus::Active) {
                        p.contradicted_by.push(reference);
                    }
                }
                EdgeType::Refines => p.refinements.push(reference),
                _ => {}
            }
        }
        for (owner, replaced) in chain {
            let Some(p) = out.get_mut(&owner) else {
                continue;
            };
            p.supersedes = replaced.iter().filter_map(as_ref).collect();
            p.chain_truncated = truncated.contains(&owner);
        }
        Ok(out)
    }

    /// Ids of memories linked to `memory_id` by an enrichment edge, in either
    /// direction — a `refines` child and its parent are both relevant to
    /// whoever found one of them.
    async fn enrichment_neighbours(&self, memory_id: &str) -> Result<Vec<String>, DomainError> {
        let mut ids = Vec::new();
        for edge in self.memory_repo.edges_from(memory_id).await? {
            if EXPANSION_EDGES.contains(&edge.edge_type) {
                ids.push(edge.to_memory);
            }
        }
        for edge in self.memory_repo.edges_to(memory_id).await? {
            if EXPANSION_EDGES.contains(&edge.edge_type) {
                ids.push(edge.from_memory);
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_edges_exclude_every_status_bearing_edge() {
        // Stated as an assertion rather than a comment: adding `supersedes`
        // here would resurrect exactly the memories supersession retired.
        for status_edge in [
            EdgeType::Supersedes,
            EdgeType::Contradicts,
            EdgeType::Retracts,
        ] {
            assert!(
                !EXPANSION_EDGES.contains(&status_edge),
                "{} is navigational — recall would surface stale memories",
                status_edge.as_str(),
            );
        }
        for enrichment in [
            EdgeType::Refines,
            EdgeType::Corroborates,
            EdgeType::RelatesTo,
        ] {
            assert!(EXPANSION_EDGES.contains(&enrichment));
        }
    }
}
