//! "What was I working on" — the briefing that lets a session pick up where the
//! last one stopped.
//!
//! Everything here is already in the store; what was missing was a view that
//! answers the question in one call. Recall answers "what do I know about X",
//! which needs you to know X — precisely what you have lost when you come back
//! to a project after a week. This assembles the other direction: the most
//! recent sessions in a scope, each with the summary written when it was
//! imported and the durable memories it left behind.
//!
//! No LLM call. The abstracts and overviews were generated at import time, so a
//! briefing is a handful of reads and stays cheap enough to open every session
//! with.

use std::collections::HashMap;
use std::sync::Arc;

use crate::application::interfaces::{MemoryRepository, NodeRepository};
use crate::application::SESSIONS_ROOT_URI;
use crate::domain::{DomainError, ImportedSession, Memory, MemoryStatus, SessionStatus};

/// Default number of sessions a briefing covers.
pub const DEFAULT_SESSION_LIMIT: usize = 5;

/// Hard cap, so a caller cannot ask for the whole history in one payload.
pub const MAX_SESSION_LIMIT: usize = 50;

/// One session in a briefing: what it was about, and what it left behind.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionRecap {
    pub session: ImportedSession,
    /// L0 from the session node — one sentence, written at import.
    pub summary: String,
    /// L1 from the session node — the arc of the session as bullets.
    pub overview: String,
    /// The still-current memories this session produced, newest first.
    pub memories: Vec<Memory>,
}

/// The answer to "what was I working on": recent sessions, newest first.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeBriefing {
    /// The projects the briefing was scoped to, empty when it covers everything.
    pub projects: Vec<String>,
    pub sessions: Vec<SessionRecap>,
    /// Sessions in scope that were not included by `limit`.
    pub more: usize,
}

impl ResumeBriefing {
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

pub struct MemoryResumeUseCase {
    node_repo: Arc<dyn NodeRepository>,
    memory_repo: Arc<dyn MemoryRepository>,
}

impl MemoryResumeUseCase {
    pub fn new(node_repo: Arc<dyn NodeRepository>, memory_repo: Arc<dyn MemoryRepository>) -> Self {
        Self {
            node_repo,
            memory_repo,
        }
    }

    /// Build a briefing over the `limit` most recent sessions.
    ///
    /// `projects` scopes it the way recall is scoped — `None` covers every
    /// session. A session whose project was never recorded (imported before the
    /// column existed, or a transcript that named none) is only ever returned
    /// unscoped: guessing it into a project would put someone else's work in
    /// your briefing.
    pub async fn execute(
        &self,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<ResumeBriefing, DomainError> {
        let limit = limit.clamp(1, MAX_SESSION_LIMIT);

        // Failed harvests are markers, not work — they carry no transcript, no
        // summary and no memories, and would crowd out real sessions.
        let in_scope: Vec<ImportedSession> = self
            .node_repo
            .list_sessions()
            .await?
            .into_iter()
            .filter(|s| s.status == SessionStatus::Imported)
            .filter(|s| match (projects, s.project.as_deref()) {
                (None, _) => true,
                (Some(scope), Some(project)) => scope.iter().any(|p| p == project),
                (Some(_), None) => false,
            })
            .collect();

        let more = in_scope.len().saturating_sub(limit);
        let selected: Vec<ImportedSession> = in_scope.into_iter().take(limit).collect();
        if selected.is_empty() {
            return Ok(ResumeBriefing {
                projects: projects.map(|p| p.to_vec()).unwrap_or_default(),
                sessions: Vec::new(),
                more: 0,
            });
        }

        // One pass over the active memories in scope, bucketed by the session
        // that wrote them — rather than a query per session, which is an N+1
        // against the same connection the nodes are read from.
        let mut by_session: HashMap<String, Vec<Memory>> = HashMap::new();
        for memory in self
            .memory_repo
            .list_memories(None, Some(MemoryStatus::Active), projects)
            .await?
        {
            if let Some(session_id) = memory.source_session_id.clone() {
                by_session.entry(session_id).or_default().push(memory);
            }
        }

        let mut sessions = Vec::with_capacity(selected.len());
        for session in selected {
            let node = self
                .node_repo
                .find_node(&format!("{SESSIONS_ROOT_URI}/{}", session.id))
                .await?;
            let memories = by_session.remove(&session.id).unwrap_or_default();
            sessions.push(SessionRecap {
                summary: node
                    .as_ref()
                    .map(|n| n.abstract_().to_string())
                    .unwrap_or_default(),
                overview: node
                    .as_ref()
                    .map(|n| n.overview().to_string())
                    .unwrap_or_default(),
                memories,
                session,
            });
        }

        Ok(ResumeBriefing {
            projects: projects.map(|p| p.to_vec()).unwrap_or_default(),
            sessions,
            more,
        })
    }
}
