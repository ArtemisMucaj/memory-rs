//! Import a finished session transcript into the memory store.
//!
//! Orchestrates the session-commit flow: idempotence check, memory
//! ingestion, and recording of the imported-session marker.

use std::sync::Arc;

use tracing::info;

use crate::application::interfaces::MemoryRepository;
use crate::application::use_cases::memory_ingestion::{IngestionOutcome, MemoryIngestionUseCase};
use crate::domain::{DomainError, ImportedSession, SessionStatus, SessionTranscript};

/// Minimum number of non-empty messages a transcript must contain for
/// extraction to be worthwhile.
const MIN_MESSAGES: usize = 2;

/// Outcome of an import request.
pub enum ImportOutcome {
    /// Ingestion ran; the report describes what was written.
    Imported {
        session: ImportedSession,
        report: crate::application::use_cases::memory_ingestion::IngestionReport,
    },
    /// The session was already imported and `force` was not set.
    AlreadyImported { session: ImportedSession },
}

pub struct ImportSessionUseCase {
    memory_repo: Arc<dyn MemoryRepository>,
    ingestion: MemoryIngestionUseCase,
}

impl ImportSessionUseCase {
    pub fn new(memory_repo: Arc<dyn MemoryRepository>, ingestion: MemoryIngestionUseCase) -> Self {
        Self {
            memory_repo,
            ingestion,
        }
    }

    /// Import `transcript`, running memory extraction over it.
    ///
    /// Imports are idempotent per transcript ID: a session that has already
    /// been imported is skipped unless `force` is set.
    pub async fn execute(
        &self,
        transcript: &SessionTranscript,
        force: bool,
    ) -> Result<ImportOutcome, DomainError> {
        let non_empty = transcript
            .messages
            .iter()
            .filter(|m| !m.content.trim().is_empty())
            .count();
        if non_empty < MIN_MESSAGES {
            return Err(DomainError::invalid_input(format!(
                "transcript '{}' has only {} non-empty messages (minimum {})",
                transcript.id, non_empty, MIN_MESSAGES
            )));
        }

        if !force {
            if let Some(session) = self
                .memory_repo
                .find_session(&transcript.source, &transcript.id)
                .await?
            {
                return Ok(ImportOutcome::AlreadyImported { session });
            }
        }

        // `force` is threaded through: a forced re-import must clear the
        // session's prior memories, or the second run appends a near-
        // duplicate set alongside the first instead of replacing it.
        let report = match self.ingestion.execute(transcript, force).await? {
            IngestionOutcome::Ingested(report) => report,
            // The session marker said "not imported" but the memory store
            // disagrees. Trust the store and report nothing written rather
            // than double-ingesting.
            IngestionOutcome::AlreadyIngested => Default::default(),
        };
        info!(
            "session '{}': {} memories written, {} deduped",
            transcript.id, report.memories_written, report.memories_deduped
        );

        let session = ImportedSession {
            id: transcript.id.clone(),
            source: transcript.source.clone(),
            imported_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            message_count: transcript.messages.len(),
            project: transcript.project.clone(),
            items_written: report.memories_written,
            status: SessionStatus::Imported,
            last_error: None,
        };
        self.memory_repo.record_session(&session).await?;

        Ok(ImportOutcome::Imported { session, report })
    }
}
