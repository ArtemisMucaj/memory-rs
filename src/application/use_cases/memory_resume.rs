//! "What was I working on" — the briefing that lets a session pick up where
//! the last one stopped.
//!
//! The most recent sessions in scope, each with the durable memories it left
//! behind. No LLM call: the session rows were written at import time, and the
//! memories are listed straight from the store.
//!
//! The L0/L1 abstract/overview the old version carried came from the session
//! node in the `memory://` tree. With the tree gone, a session recap is the
//! session row plus its facts.

use std::collections::HashMap;
use std::sync::Arc;

use crate::application::interfaces::MemoryRepository;
use crate::domain::{DomainError, ImportedSession, Memory, SessionStatus};

/// Default number of sessions a briefing covers.
pub const DEFAULT_SESSION_LIMIT: usize = 5;

/// Hard cap, so a caller cannot ask for the whole history in one payload.
pub const MAX_SESSION_LIMIT: usize = 50;

/// One session in a briefing: its row, and the facts it left behind.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionRecap {
    pub session: ImportedSession,
    /// The memories this session produced, newest first.
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
    memory_repo: Arc<dyn MemoryRepository>,
}

impl MemoryResumeUseCase {
    pub fn new(memory_repo: Arc<dyn MemoryRepository>) -> Self {
        Self { memory_repo }
    }

    /// Build a briefing over the `limit` most recent sessions.
    ///
    /// `projects` scopes it the way recall is scoped — `None` covers every
    /// session. A session whose project was never recorded is only ever
    /// returned unscoped: guessing it into a project would put someone
    /// else's work in your briefing.
    pub async fn execute(
        &self,
        projects: Option<&[String]>,
        limit: usize,
    ) -> Result<ResumeBriefing, DomainError> {
        let limit = limit.clamp(1, MAX_SESSION_LIMIT);

        // Failed harvests are markers, not work — they carry no transcript
        // and no memories, and would crowd out real sessions. The filter
        // lives in the SQL, not in Rust, so a flood of failed rows cannot
        // shrink the window the limit then applies to.
        let in_scope: Vec<ImportedSession> = self
            .memory_repo
            .list_sessions_by_status(SessionStatus::Imported, projects, MAX_SESSION_LIMIT)
            .await?;

        let more = in_scope.len().saturating_sub(limit);
        let selected: Vec<ImportedSession> = in_scope.into_iter().take(limit).collect();
        if selected.is_empty() {
            return Ok(ResumeBriefing {
                projects: projects.map(|p| p.to_vec()).unwrap_or_default(),
                sessions: Vec::new(),
                more: 0,
            });
        }

        // One pass over the memories in scope, bucketed by the session that
        // wrote them — rather than a query per session.
        let mut by_session: HashMap<String, Vec<Memory>> = HashMap::new();
        for memory in self.memory_repo.list_memories(None, projects).await? {
            if let Some(session_id) = memory.source_session_id.clone() {
                by_session.entry(session_id).or_default().push(memory);
            }
        }

        let sessions = selected
            .into_iter()
            .map(|session| {
                let memories = by_session.remove(&session.id).unwrap_or_default();
                SessionRecap { session, memories }
            })
            .collect();

        Ok(ResumeBriefing {
            projects: projects.map(|p| p.to_vec()).unwrap_or_default(),
            sessions,
            more,
        })
    }
}
