use async_trait::async_trait;

use crate::domain::{DiscoveredSession, DomainError, SessionTranscript};

/// Port for discovering finished assistant sessions on this machine and
/// materializing their transcripts.
///
/// The connector layer implements this over the local session stores (Claude
/// Code JSONL logs, OpenCode/Zed SQLite databases). The dream use case depends
/// on this trait — not on the concrete discovery code — so it can harvest
/// finished sessions without the application layer knowing where they live.
#[async_trait]
pub trait SessionDiscovery: Send + Sync {
    /// List sessions from every available source, newest first. A missing or
    /// broken source contributes nothing rather than failing discovery.
    async fn discover(&self) -> Result<Vec<DiscoveredSession>, DomainError>;

    /// Materialize the full transcript for one discovered session.
    async fn load_transcript(
        &self,
        session: &DiscoveredSession,
    ) -> Result<SessionTranscript, DomainError>;
}
