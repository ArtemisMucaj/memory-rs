//! Use-case-level integration tests for the import → extract → store → search
//! pipeline, driven by test doubles for the LLM and embedding backends.
//!
//! These exercise the same use cases the CLI router calls, but wire them up
//! with a scripted chat client and a deterministic mock embedder instead of a
//! live model — so the pipeline runs offline and reproducibly. The DuckDB store
//! is real (in-memory).

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use memory_rs::application::interfaces::Embedder;
use memory_rs::{
    DuckdbMemoryRepository, ImportOutcome, ImportSessionUseCase, MemoryExtractionUseCase,
    MemoryKind, MemoryRepository, MemorySearchUseCase, SessionMessage, SessionTranscript,
    SummarizeMemoryUseCase,
};
use openai_rs::{ChatClient, ChatRequest, EmbeddingClient, OpenAiError};

const DIMS: usize = 16;

/// A deterministic embedding client: hashes each input into a fixed-width
/// vector so identical text embeds identically and similar text lands nearby,
/// without any network. Enough to exercise the semantic-search SQL path.
struct MockEmbeddingClient;

#[async_trait]
impl EmbeddingClient for MockEmbeddingClient {
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, OpenAiError> {
        Ok(inputs.iter().map(|s| embed_text(s)).collect())
    }

    fn model(&self) -> &str {
        "mock-embedding"
    }
}

/// Map text to a unit-ish vector by bucketing byte values into `DIMS` slots.
fn embed_text(text: &str) -> Vec<f32> {
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

/// A chat client that replays scripted responses for *extraction* calls and
/// answers *summarization* calls with a canned valid reply, so summary calls
/// never drain the extraction script. Extraction calls are recorded.
struct ScriptedChatClient {
    responses: Mutex<Vec<String>>,
    calls: Mutex<Vec<(String, String)>>,
}

const SUMMARY_REPLY: &str = r#"{"abstract": "Test session summary.", "overview": "- did a thing"}"#;

impl ScriptedChatClient {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().map(String::from).collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    async fn recorded_calls(&self) -> Vec<(String, String)> {
        self.calls.lock().await.clone()
    }
}

/// A summarization system prompt, as opposed to the extraction one.
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

struct Harness {
    memory_repo: Arc<dyn MemoryRepository>,
    embedder: Arc<Embedder>,
}

impl Harness {
    fn new() -> Self {
        Self {
            memory_repo: Arc::new(
                DuckdbMemoryRepository::in_memory(DIMS, "mock-embedding").unwrap(),
            ),
            embedder: Arc::new(Embedder::new(Arc::new(MockEmbeddingClient))),
        }
    }

    fn import_use_case(&self, chat: Arc<ScriptedChatClient>) -> ImportSessionUseCase {
        let extraction = MemoryExtractionUseCase::new(
            Arc::clone(&chat) as Arc<dyn ChatClient>,
            Arc::clone(&self.memory_repo),
            Arc::clone(&self.embedder),
        );
        let summary = SummarizeMemoryUseCase::new(
            chat as Arc<dyn ChatClient>,
            Arc::clone(&self.memory_repo),
            Arc::clone(&self.embedder),
        );
        ImportSessionUseCase::new(Arc::clone(&self.memory_repo), extraction, summary)
    }

    fn search_use_case(&self) -> MemorySearchUseCase {
        MemorySearchUseCase::new(Arc::clone(&self.memory_repo), Arc::clone(&self.embedder))
    }
}

fn transcript(id: &str, messages: &[(&str, &str)]) -> SessionTranscript {
    SessionTranscript {
        id: id.to_string(),
        source: format!("{id}.jsonl"),
        project: None,
        messages: messages
            .iter()
            .map(|(role, content)| SessionMessage {
                role: role.to_string(),
                content: content.to_string(),
                timestamp: Some("2026-07-01T10:00:00Z".to_string()),
            })
            .collect(),
    }
}

fn extraction_json(preference: (&str, &str)) -> String {
    format!(
        r#"{{"preferences": [{{"name": "{}", "content": "{}"}}],
            "experiences": [], "skills": [], "facts": [], "delete": []}}"#,
        preference.0, preference.1
    )
}

#[tokio::test]
async fn import_extracts_stores_and_records_session() {
    let harness = Harness::new();
    let chat = Arc::new(ScriptedChatClient::new(vec![
        r###"{"preferences": [{"name": "rust_error_handling", "content": "Prefers ? over unwrap in library code"}],
            "experiences": [], "skills": [],
            "facts": [{"name": "project_uses_duckdb", "content": "The project stores indexed data in DuckDB"}],
            "delete": []}"###,
    ]));
    let use_case = harness.import_use_case(Arc::clone(&chat));

    let transcript = transcript(
        "session-1",
        &[
            (
                "user",
                "Please never use unwrap in library code, use ? instead",
            ),
            ("assistant", "Understood, refactored to use ? everywhere."),
        ],
    );
    let outcome = use_case.execute(&transcript, false).await.unwrap();

    let ImportOutcome::Imported { session, report } = outcome else {
        panic!("expected Imported outcome");
    };
    assert_eq!(session.id, "session-1");
    assert_eq!(session.items_written, 2);
    assert_eq!(report.applied.len(), 2);

    let prefs = harness
        .memory_repo
        .list_items(Some(MemoryKind::Preference))
        .await
        .unwrap();
    assert_eq!(prefs.len(), 1);
    assert_eq!(prefs[0].name(), "rust_error_handling");

    // The session marker is recorded and the extraction prompt carried the chat.
    let recorded = harness
        .memory_repo
        .find_session("session-1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(recorded.message_count, 2);
    let calls = chat.recorded_calls().await;
    assert_eq!(calls.len(), 1);
    assert!(calls[0].0.contains("memory extraction agent"));
    assert!(calls[0].1.contains("never use unwrap"));
}

#[tokio::test]
async fn import_is_idempotent_unless_forced() {
    let harness = Harness::new();
    let chat = Arc::new(ScriptedChatClient::new(vec![
        &extraction_json(("tabs_vs_spaces", "Prefers tabs")),
        &extraction_json(("tabs_vs_spaces", "Prefers tabs, strongly")),
    ]));
    let use_case = harness.import_use_case(chat);
    let transcript = transcript(
        "session-2",
        &[("user", "I prefer tabs"), ("assistant", "Noted.")],
    );

    assert!(matches!(
        use_case.execute(&transcript, false).await.unwrap(),
        ImportOutcome::Imported { .. }
    ));
    // Second import without force is skipped (no scripted response consumed).
    assert!(matches!(
        use_case.execute(&transcript, false).await.unwrap(),
        ImportOutcome::AlreadyImported { .. }
    ));
    // Forced re-import rewrites the item.
    assert!(matches!(
        use_case.execute(&transcript, true).await.unwrap(),
        ImportOutcome::Imported { .. }
    ));
    let item = harness
        .memory_repo
        .find_item(MemoryKind::Preference, "tabs_vs_spaces", None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(item.content(), "Prefers tabs, strongly");
    assert_eq!(item.update_count(), 1);
}

#[tokio::test]
async fn imported_memories_are_searchable() {
    let harness = Harness::new();
    let chat = Arc::new(ScriptedChatClient::new(vec![
        r###"{"preferences": [],
            "experiences": [], "skills": [],
            "facts": [{"name": "network_timeout_policy", "content": "retry network timeouts with exponential backoff"}],
            "delete": []}"###,
    ]));
    let use_case = harness.import_use_case(chat);
    let transcript = transcript(
        "session-3",
        &[
            ("user", "how should we handle network timeouts?"),
            ("assistant", "retry with backoff"),
        ],
    );
    use_case.execute(&transcript, false).await.unwrap();

    // Hybrid search (semantic via the mock embedder + keyword) finds it.
    let results = harness
        .search_use_case()
        .execute("network timeout", None, None, 10)
        .await
        .unwrap();
    assert!(!results.is_empty(), "expected a hit for the stored fact");
    assert_eq!(results[0].0.name(), "network_timeout_policy");
}
