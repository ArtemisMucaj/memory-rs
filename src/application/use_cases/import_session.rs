//! Import a finished session transcript into the memory store.
//!
//! Orchestrates the session-commit flow: idempotence check, memory ingestion,
//! and recording of the imported-session marker.
//!
//! Import writes **memories**, not items. The item extractor still exists and
//! still compiles (it is deleted in its own step), but nothing calls it — so
//! `memory_items` stops growing and the memory log is the projection every
//! surface reads. Anything that reads items after this point is reading a table
//! that no longer receives writes.

use std::sync::Arc;

use tracing::{info, warn};

use crate::application::interfaces::NodeRepository;
use crate::application::use_cases::memory_ingestion::{IngestionOutcome, MemoryIngestionUseCase};
use crate::application::use_cases::memory_summary::SummarizeMemoryUseCase;
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
    node_repo: Arc<dyn NodeRepository>,
    ingestion: MemoryIngestionUseCase,
    summary: SummarizeMemoryUseCase,
}

impl ImportSessionUseCase {
    pub fn new(
        node_repo: Arc<dyn NodeRepository>,
        ingestion: MemoryIngestionUseCase,
        summary: SummarizeMemoryUseCase,
    ) -> Self {
        Self {
            node_repo,
            ingestion,
            summary,
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
            if let Some(session) = self.node_repo.find_session(&transcript.id).await? {
                return Ok(ImportOutcome::AlreadyImported { session });
            }
        }

        // `force` is threaded through: a forced re-import must clear the
        // session's prior memories, or the second run appends a near-duplicate
        // set alongside the first instead of replacing it.
        let report = match self.ingestion.execute(transcript, force).await? {
            IngestionOutcome::Ingested(report) => report,
            // The session marker said "not imported" but the memory log
            // disagrees. Trust the memory log and report nothing written rather
            // than double-ingesting.
            IngestionOutcome::AlreadyIngested => Default::default(),
        };
        info!(
            "session '{}': {} memories written, {} corroborated, {} conflicts recorded",
            transcript.id,
            report.memories_written,
            report.memories_corroborated,
            report.conflicts_recorded
        );

        // Build the virtual-filesystem layer over the flat items:
        //   1. store this session as a node (transcript L2 + generated L0/L1),
        //   2. regenerate the whole-memory digest so it reflects the new items.
        // Both are best-effort — extraction already succeeded, so a summary
        // failure must not fail the import. Errors are logged and swallowed.
        if let Err(e) = self.summary.summarize_session(transcript).await {
            warn!(
                "session '{}': failed to store session node: {e}",
                transcript.id
            );
        }
        if let Err(e) = self.summary.regenerate_digest().await {
            warn!(
                "session '{}': failed to regenerate memory digest: {e}",
                transcript.id
            );
        }
        // Per-project digests check their own staleness, so this typically
        // regenerates only the project this session's items landed in.
        if let Err(e) = self.summary.regenerate_project_digests().await {
            warn!(
                "session '{}': failed to regenerate project digests: {e}",
                transcript.id
            );
        }

        let session = ImportedSession {
            id: transcript.id.clone(),
            source: transcript.source.clone(),
            imported_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            message_count: transcript.messages.len(),
            // Column name predates memories; it is the count of memories this
            // session wrote, which is now memories. Renaming it would rewrite the
            // sessions table for no behavioural gain.
            items_written: report.memories_written,
            status: SessionStatus::Imported,
            last_error: None,
        };
        self.node_repo.record_session(&session).await?;

        Ok(ImportOutcome::Imported { session, report })
    }
}
