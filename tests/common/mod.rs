//! Shared test doubles for the LLM and embedding backends.
//!
//! `#![allow(dead_code)]` because this module is compiled *into each test
//! binary separately* — anything one file does not happen to use is dead code
//! from that binary's point of view, even though another file relies on it.
//!
//! These implement the [`openai_rs`] ports directly, so the use cases under
//! test are wired exactly as production wires them — only the far side of the
//! network boundary is replaced. Everything below the use case (the DuckDB
//! store, the SQL, the vector round-trip) stays real.
//!
//! Lives in `tests/common/` because both the item pipeline and the memory
//! pipeline need the same two doubles, and a second copy would drift.

#![allow(dead_code)]

use async_trait::async_trait;
use tokio::sync::Mutex;

use openai_rs::{ChatClient, ChatRequest, EmbeddingClient, OpenAiError};

/// Embedding width used by the test doubles and the in-memory stores they feed.
pub const DIMS: usize = 16;

/// A deterministic embedding client: hashes each input into a fixed-width
/// vector so identical text embeds identically and similar text lands nearby,
/// without any network. Enough to exercise the semantic-search SQL path.
pub struct MockEmbeddingClient;

#[async_trait]
impl EmbeddingClient for MockEmbeddingClient {
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, OpenAiError> {
        Ok(inputs.iter().map(|s| embed_text(s)).collect())
    }

    fn model(&self) -> &str {
        "mock-embedding"
    }
}

/// Map text to a unit-ish vector by bucketing byte values into [`DIMS`] slots.
pub fn embed_text(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIMS];
    for (i, b) in text.bytes().enumerate() {
        v[i % DIMS] += (b as f32) / 255.0;
    }
    // Normalize so cosine distance behaves.
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// An embedding client that returns the *same* vector for every input.
///
/// Every pair therefore scores 1.0, which is the only way to exercise the
/// attribution path deterministically: [`MockEmbeddingClient`] hashes bytes, so
/// the cosine between two arbitrary strings is whatever the hash happens to
/// produce and cannot be aimed at a threshold. The threshold logic itself is
/// unit-tested separately; this double exists to drive the *wiring*.
pub struct ConstantEmbeddingClient;

#[async_trait]
impl EmbeddingClient for ConstantEmbeddingClient {
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, OpenAiError> {
        let mut unit = vec![0.0f32; DIMS];
        unit[0] = 1.0;
        Ok(inputs.iter().map(|_| unit.clone()).collect())
    }

    fn model(&self) -> &str {
        "constant-embedding"
    }
}

/// An embedding client whose vectors land every pair inside the *ambiguous*
/// band (0.85–0.95) — the only region where the adjudication tier runs.
///
/// It returns a fixed unit vector tilted slightly off-axis, so the cosine
/// against [`ambiguous_seed_vector`] is ~0.906: close to the real 0.907 that
/// `orders-events package` and `orders-events service` produced, and a value
/// no hashing embedder could be aimed at.
pub struct AmbiguousEmbeddingClient;

/// The vector to seed a candidate entity with so it sits in the band relative
/// to whatever [`AmbiguousEmbeddingClient`] produces.
pub fn ambiguous_seed_vector() -> Vec<f32> {
    let mut v = vec![0.0f32; DIMS];
    v[0] = 1.0;
    v
}

#[async_trait]
impl EmbeddingClient for AmbiguousEmbeddingClient {
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, OpenAiError> {
        // cos = 1/sqrt(1 + 0.44^2) ≈ 0.915 against the unit seed vector.
        let mut v = vec![0.0f32; DIMS];
        v[0] = 1.0;
        v[1] = 0.44;
        Ok(inputs.iter().map(|_| v.clone()).collect())
    }

    fn model(&self) -> &str {
        "ambiguous-embedding"
    }
}

const SUMMARY_REPLY: &str = r#"{"abstract": "Test session summary.", "overview": "- did a thing"}"#;

/// A chat client that replays scripted responses for *extraction* calls and
/// answers *summarization* calls with a canned valid reply, so summary calls
/// never drain the extraction script. Extraction calls are recorded.
pub struct ScriptedChatClient {
    responses: Mutex<Vec<String>>,
    calls: Mutex<Vec<(String, String)>>,
}

impl ScriptedChatClient {
    pub fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// The `(system, user)` pairs the code under test sent, in order.
    pub async fn recorded_calls(&self) -> Vec<(String, String)> {
        self.calls.lock().await.clone()
    }
}

/// A summarization system prompt, as opposed to an extraction one.
fn is_summary_call(system: &str) -> bool {
    system.contains("summarize a finished")
        || system.contains("summarize a document")
        || system.contains("top-level index")
        || system.contains("about ONE project")
}

#[async_trait]
impl ChatClient for ScriptedChatClient {
    async fn chat(&self, _request: &ChatRequest) -> Result<String, OpenAiError> {
        // Extraction goes through `complete_json` and summary through
        // `complete`; both are overridden, so the raw `chat` path is unused.
        Err(OpenAiError::decode("chat() not used by these tests"))
    }

    /// Extraction path: record the call and return the next scripted response.
    async fn complete_json(
        &self,
        system: &str,
        user: &str,
        _schema_name: &str,
        _schema: &serde_json::Value,
    ) -> Result<String, OpenAiError> {
        self.calls
            .lock()
            .await
            .push((system.to_string(), user.to_string()));
        let mut responses = self.responses.lock().await;
        if responses.is_empty() {
            return Err(OpenAiError::decode("no scripted response left"));
        }
        Ok(responses.remove(0))
    }

    /// Summarization path: answer with a canned valid reply so summary calls
    /// never drain the extraction script.
    async fn complete(&self, system: &str, _user: &str) -> Result<String, OpenAiError> {
        if is_summary_call(system) {
            return Ok(SUMMARY_REPLY.to_string());
        }
        Err(OpenAiError::decode(
            "unexpected non-summary complete() call",
        ))
    }
}
