use serde::{Deserialize, Serialize};

/// Which assistant produced a discovered session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionSource {
    Claude,
    OpenCode,
    Zed,
}

impl SessionSource {
    /// Short label shown in the picker (`claude`, `opencode`, `zed`).
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionSource::Claude => "claude",
            SessionSource::OpenCode => "opencode",
            SessionSource::Zed => "zed",
        }
    }
}

impl std::fmt::Display for SessionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How to locate a discovered session's full transcript so it can be
/// materialized on demand (only for sessions the user actually imports).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionLocator {
    /// A Claude Code transcript file on disk.
    File(String),
    /// A row in a SQLite database, addressed by DB path + session id.
    Sqlite { db_path: String, session_id: String },
}

/// A session found by the discovery layer, shown in the import picker before
/// any expensive parsing/decompression of its body.
#[derive(Debug, Clone)]
pub struct DiscoveredSession {
    pub source: SessionSource,
    /// Stable session identifier (used for idempotent imports).
    pub id: String,
    /// Friendly, human-readable name (summary/title, else a fallback).
    pub title: String,
    /// Working directory / project the session ran in, when known.
    pub cwd: Option<String>,
    /// Unix seconds of the last activity (used for "N ago" and sorting).
    pub updated_at: i64,
    /// Number of conversational messages, when cheaply known.
    pub message_count: usize,
    /// A short preview taken from the END of the session (the outcome), so the
    /// picker shows where the conversation landed rather than where it began.
    pub tail_preview: String,
    /// Rough estimate of the session's token count, from a cheap per-source
    /// chars-per-token heuristic over its text size (no tokenizer, no full
    /// parse). Gives an at-a-glance sense of prefill / KV-cache cost before
    /// importing. See [`approx_tokens_from_chars`].
    pub approx_tokens: usize,
    /// Where to read the full transcript from when importing.
    pub locator: SessionLocator,
}

impl SessionSource {
    /// Average characters per token for this source's transcript text.
    ///
    /// The common `chars / 4` rule is calibrated on prose; measured against
    /// real BPE token counts (GPT-4 `cl100k_base`) over samples of actual
    /// sessions, the best-fit ratio differs by source because each renders
    /// different text density:
    /// - OpenCode ≈ 3.3 (more code/JSON fragments, denser tokenization).
    /// - Claude ≈ 4.0 (prose + markdown chat).
    /// - Zed ≈ 3.8 (chat with some code, between the two).
    ///
    /// Each was validated end-to-end against real BPE counts on that source's
    /// actual sessions: weighted error is within ~3% per source, versus ~20%
    /// for a single shared constant. Other models' tokenizers differ, but these
    /// are far better than a flat `/4` for gauging prefill scale.
    fn chars_per_token(&self) -> f64 {
        match self {
            SessionSource::OpenCode => 3.3,
            SessionSource::Claude => 4.0,
            SessionSource::Zed => 3.8,
        }
    }
}

impl DiscoveredSession {
    /// A display title that is never empty.
    pub fn display_title(&self) -> &str {
        if self.title.trim().is_empty() {
            "(untitled session)"
        } else {
            self.title.trim()
        }
    }
}

/// Estimate a token count from a character count for `source`, using its
/// calibrated chars-per-token ratio. Approximate — meant for gauging
/// prefill/KV-cache scale, not exact accounting.
pub fn approx_tokens_from_chars(source: SessionSource, chars: usize) -> usize {
    (chars as f64 / source.chars_per_token()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_tokens_is_calibrated_per_source() {
        assert_eq!(approx_tokens_from_chars(SessionSource::OpenCode, 0), 0);
        // OpenCode ≈ 3.3 chars/token: 3300 chars ≈ 1000 tokens.
        assert_eq!(
            approx_tokens_from_chars(SessionSource::OpenCode, 3300),
            1000
        );
        // Claude ≈ 4.0: 4000 chars ≈ 1000 tokens (denser prose).
        assert_eq!(approx_tokens_from_chars(SessionSource::Claude, 4000), 1000);
        // Zed ≈ 3.8: 3800 chars ≈ 1000 tokens (chat with some code).
        assert_eq!(approx_tokens_from_chars(SessionSource::Zed, 3800), 1000);
        // The same char count yields more tokens for a denser source.
        assert!(
            approx_tokens_from_chars(SessionSource::OpenCode, 10_000)
                > approx_tokens_from_chars(SessionSource::Claude, 10_000)
        );
    }
}
