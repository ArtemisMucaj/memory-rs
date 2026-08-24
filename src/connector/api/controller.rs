//! Shared controllers — the single source of truth for every memory operation.
//!
//! Each function takes the [`Container`] plus typed parameters and returns
//! **domain data** (memories, entities, sessions, outcome enums), performing
//! the logic common to all surfaces — scope resolution for a namespace,
//! mostly — but *no* presentation. The CLI router, the HTTP management API,
//! and the MCP server all call these, then render the result in their own
//! format. This keeps the three surfaces in lockstep with zero duplicated
//! logic.

use crate::application::{
    DreamReport, ImportOutcome, IngestionOutcome, Recalled, ResumeBriefing,
};
use crate::connector::adapter::{fetch_resource, parse_transcript_file};
use crate::connector::api::Container;
use crate::domain::{
    DomainError, Entity, ImportedSession, Memory, MemoryKind, MemoryResource,
};

/// How a memory should be scoped for a search.
#[derive(Debug, Clone, Default)]
pub enum SearchScope {
    /// Everything (globals + all projects).
    #[default]
    All,
    /// Globals plus one project.
    Project(String),
    /// Globals plus a namespace's member projects.
    Namespace(String),
}

/// Resolve a [`SearchScope`] into the concrete project list the repository
/// expects: `None` = all; `Some(list)` = globals + `list`. A namespace with
/// no members resolves to [`ScopeResolution::EmptyNamespace`] so the caller
/// can tell the user rather than silently searching only globals.
pub enum ScopeResolution {
    All,
    Projects(Vec<String>),
    EmptyNamespace(String),
}

pub async fn resolve_scope(
    container: &Container,
    scope: &SearchScope,
) -> Result<ScopeResolution, DomainError> {
    Ok(match scope {
        SearchScope::All => ScopeResolution::All,
        SearchScope::Project(p) => ScopeResolution::Projects(vec![p.clone()]),
        SearchScope::Namespace(ns) => {
            let projects = container.memory_repository()?.namespace_projects(ns).await?;
            if projects.is_empty() {
                ScopeResolution::EmptyNamespace(ns.clone())
            } else {
                ScopeResolution::Projects(projects)
            }
        }
    })
}

/// The result of a scoped recall: the ranked hits, or a note that the
/// requested namespace has no member projects.
pub enum MemorySearchOutcome {
    Hits(Vec<Recalled>),
    EmptyNamespace(String),
}

/// Ingest a transcript file into the memory store.
pub async fn ingest_memories(
    container: &Container,
    path: &str,
    force: bool,
) -> Result<IngestionOutcome, DomainError> {
    let transcript = parse_transcript_file(std::path::Path::new(path))?;
    container
        .memory_ingestion_use_case()?
        .execute(&transcript, force)
        .await
}

/// Recall memories for `query` within `scope`.
pub async fn recall_memories(
    container: &Container,
    query: &str,
    kind: Option<MemoryKind>,
    scope: &SearchScope,
    limit: usize,
) -> Result<MemorySearchOutcome, DomainError> {
    let projects = match resolve_scope(container, scope).await? {
        ScopeResolution::All => None,
        ScopeResolution::Projects(p) => Some(p),
        ScopeResolution::EmptyNamespace(ns) => return Ok(MemorySearchOutcome::EmptyNamespace(ns)),
    };
    let hits = container
        .memory_recall_use_case()?
        .execute(query, kind, projects.as_deref(), limit)
        .await?;
    Ok(MemorySearchOutcome::Hits(hits))
}

/// Canonical names for every entity referenced by `memories`, keyed by id.
///
/// A memory stores its subject and object as entity *ids*, which are UUIDs
/// and mean nothing to a reader. This resolves them in one query so any
/// surface can render "orders-events deployment" where the row would
/// otherwise say `@c95de38f-…`.
pub async fn entity_labels(
    container: &Container,
    memories: &[Memory],
) -> Result<std::collections::HashMap<String, String>, DomainError> {
    let mut ids: Vec<String> = memories
        .iter()
        .flat_map(|m| [m.subject.entity_id(), m.object.entity_id()])
        .flatten()
        .map(str::to_string)
        .collect();
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    Ok(container
        .memory_repository()?
        .find_entities(&ids)
        .await?
        .into_iter()
        .map(|e| (e.id, e.canonical_name))
        .collect())
}

/// A memory's subject/object rendered for display: the entity's canonical
/// name when it resolves, the literal value otherwise.
pub fn entity_ref_label(
    r: &crate::domain::EntityRef,
    labels: &std::collections::HashMap<String, String>,
) -> Option<String> {
    match r {
        crate::domain::EntityRef::Entity(id) => {
            Some(labels.get(id).cloned().unwrap_or_else(|| id.clone()))
        }
        crate::domain::EntityRef::Literal(v) if v.is_empty() => None,
        crate::domain::EntityRef::Literal(v) => Some(v.clone()),
    }
}

/// One entity plus how many memories hang off it.
pub struct EntitySummary {
    pub entity: Entity,
    /// How many memories reference it as subject or object.
    pub memory_count: usize,
}

/// Every entity, most-referenced first.
pub async fn list_entities(container: &Container) -> Result<Vec<EntitySummary>, DomainError> {
    let repo = container.memory_repository()?;
    let entities = repo.list_entities().await?;
    let mut out = Vec::with_capacity(entities.len());
    for entity in entities {
        let memory_count = repo.memories_for_entity(&entity.id).await?.len();
        out.push(EntitySummary {
            entity,
            memory_count,
        });
    }
    out.sort_by(|a, b| {
        b.memory_count
            .cmp(&a.memory_count)
            .then_with(|| a.entity.canonical_name.cmp(&b.entity.canonical_name))
    });
    Ok(out)
}

/// One entity with the memories that reference it.
pub async fn show_entity(
    container: &Container,
    id: &str,
) -> Result<Option<(Entity, Vec<Memory>)>, DomainError> {
    let repo = container.memory_repository()?;
    let Some(entity) = repo.find_entity(id).await? else {
        return Ok(None);
    };
    let memories = repo.memories_for_entity(id).await?;
    Ok(Some((entity, memories)))
}

/// List memories, newest first, optionally restricted by project scope.
pub async fn list_memories(
    container: &Container,
    projects: Option<&[String]>,
) -> Result<Vec<Memory>, DomainError> {
    container.memory_repository()?.list_memories(None, projects).await
}

/// What a `show <id>` against the memory store resolves to.
pub enum MemoryShowOutcome {
    /// A stored resource (`memory://resources/<name>`).
    Resource(MemoryResource),
    /// A memory.
    Memory(Box<Memory>),
    NotFound,
}

/// Resolve a reference: a `memory://resources/...` URI → a resource;
/// otherwise a memory id.
pub async fn show_memory(
    container: &Container,
    id: &str,
) -> Result<MemoryShowOutcome, DomainError> {
    if let Some(_slug) = id.strip_prefix("memory://resources/") {
        return Ok(
            match container.memory_repository()?.find_resource(id).await? {
                Some(resource) => MemoryShowOutcome::Resource(resource),
                None => MemoryShowOutcome::NotFound,
            },
        );
    }
    let repo = container.memory_repository()?;
    let Some(memory) = repo.find_memory(id).await? else {
        return Ok(MemoryShowOutcome::NotFound);
    };
    Ok(MemoryShowOutcome::Memory(Box::new(memory)))
}

/// What `forget <id>` did.
pub enum ForgetOutcome {
    Deleted,
    NotFound,
}

/// Hard-delete a memory. There is no tombstone; the row is gone.
pub async fn forget_memory(container: &Container, id: &str) -> Result<ForgetOutcome, DomainError> {
    let deleted = container.memory_repository()?.delete_memory(id).await?;
    Ok(if deleted {
        ForgetOutcome::Deleted
    } else {
        ForgetOutcome::NotFound
    })
}

/// List imported sessions, newest first.
pub async fn sessions(container: &Container) -> Result<Vec<ImportedSession>, DomainError> {
    container
        .memory_repository()?
        .list_sessions(None, usize::MAX)
        .await
}

/// The result of a scoped resume briefing.
pub enum ResumeOutcome {
    Briefing(ResumeBriefing),
    EmptyNamespace(String),
}

/// "What was I working on" — the last `limit` sessions in `scope`, each
/// with the memories it produced.
pub async fn resume(
    container: &Container,
    scope: &SearchScope,
    limit: usize,
) -> Result<ResumeOutcome, DomainError> {
    let projects = match resolve_scope(container, scope).await? {
        ScopeResolution::All => None,
        ScopeResolution::Projects(p) => Some(p),
        ScopeResolution::EmptyNamespace(ns) => return Ok(ResumeOutcome::EmptyNamespace(ns)),
    };
    let briefing = container
        .memory_resume_use_case()?
        .execute(projects.as_deref(), limit)
        .await?;
    Ok(ResumeOutcome::Briefing(briefing))
}

/// A stub listing for `browse_memory`: the resources in the store plus the
/// recent sessions. Replaces the old L0/L1/L2 `memory://` tree with the
/// minimum that still answers "what is in here".
pub async fn tree(
    container: &Container,
    uri: Option<&str>,
) -> Result<Vec<serde_json::Value>, DomainError> {
    let repo = container.memory_repository()?;
    let mut out = Vec::new();
    match uri {
        None | Some("memory://") | Some("memory://resources") => {
            for r in repo.list_resources().await? {
                out.push(serde_json::json!({
                    "uri": r.uri,
                    "kind": "resource",
                    "abstract": r.abstract_,
                }));
            }
        }
        Some("memory://sessions") => {
            for s in repo.list_sessions(None, 50).await? {
                out.push(serde_json::json!({
                    "uri": format!("memory://sessions/{}", s.id),
                    "kind": "session",
                    "abstract": format!("{} messages, {} memories", s.message_count, s.items_written),
                }));
            }
        }
        Some(_) => {}
    }
    Ok(out)
}

/// Import a transcript file, running extraction over it.
pub async fn import(
    container: &Container,
    path: &str,
    force: bool,
) -> Result<ImportOutcome, DomainError> {
    let transcript = parse_transcript_file(std::path::Path::new(path))?;
    container
        .memory_import_use_case()?
        .execute(&transcript, force)
        .await
}

/// A resource added to the store.
pub struct AddedResource {
    pub resource: MemoryResource,
    pub source: String,
    pub chars: usize,
}

/// Fetch a file/URL, summarize it, and store it.
pub async fn add_resource(
    container: &Container,
    source: &str,
    name: Option<&str>,
) -> Result<AddedResource, DomainError> {
    use crate::application::use_cases::llm_json::unix_now;
    use crate::connector::adapter::resource_fetch::resource_slug;

    // Fetch first — a bad path/URL should fail before we spin up the LLM.
    let fetched = fetch_resource(source)
        .await
        .map_err(|e| DomainError::internal(format!("failed to fetch resource '{source}': {e}")))?;
    let slug = resource_slug(name.unwrap_or(&fetched.title));
    let uri = format!("memory://resources/{slug}");

    // Summarize: one LLM call returns the abstract and overview. Failure is
    // not fatal — a resource with an empty abstract is still findable by
    // name and still keyword-searchable.
    let chat = container.chat_client_for(crate::connector::adapter::LlmUsage::Summarize)?;
    let system = "You summarize a document for later retrieval. Output ONLY a JSON object.";
    let user = format!(
        "Document (source: {}):\n\n{}\n\n\
         Respond with ONLY a JSON object:\n\
         {{\"abstract\": \"<one-line summary>\", \"overview\": \"<a paragraph orienting a reader before opening the content>\"}}",
        fetched.source,
        fetched.text.chars().take(20_000).collect::<String>(),
    );
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "abstract": { "type": "string" },
            "overview": { "type": "string" }
        },
        "required": ["abstract", "overview"],
        "additionalProperties": false
    });
    let (abstract_, overview) = match chat.complete_json(system, &user, "add_resource", &schema).await {
        Ok(response) => {
            let v: serde_json::Value = serde_json::from_str(&response).unwrap_or_default();
            (
                v.get("abstract").and_then(|x| x.as_str()).unwrap_or("").to_string(),
                v.get("overview").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            )
        }
        Err(e) => {
            tracing::warn!("resource summarization failed for '{source}': {e}");
            (String::new(), String::new())
        }
    };

    let resource = MemoryResource {
        uri: uri.clone(),
        source: fetched.source.clone(),
        name: slug,
        abstract_,
        overview,
        content: fetched.text.clone(),
        created_at: unix_now(),
    };

    // Embed the abstract + overview. Embeddings are best-effort: a failure
    // leaves the resource keyword-searchable by name and abstract.
    let vector = match container.embedder_for_resources() {
        Ok(embedder) if embedder.embeddings_enabled() => {
            match embedder.embed_query(&resource.embedding_text()).await {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("failed to embed resource '{source}': {e}");
                    None
                }
            }
        }
        _ => None,
    };
    container
        .memory_repository()?
        .upsert_resource(&resource, vector.as_deref())
        .await?;

    Ok(AddedResource {
        resource,
        source: fetched.source,
        chars: fetched.text.len(),
    })
}

/// Run one harvest cycle. `idle_minutes` is the finished-session inactivity
/// window.
pub async fn dream(container: &Container, idle_minutes: u64) -> Result<DreamReport, DomainError> {
    let idle_secs = i64::try_from(idle_minutes.saturating_mul(60)).unwrap_or(i64::MAX);
    container.memory_dream_use_case()?.harvest(idle_secs).await
}

// ── Namespaces ───────────────────────────────────────────────────────────────

pub async fn create_namespace(container: &Container, name: &str) -> Result<bool, DomainError> {
    container.memory_repository()?.create_namespace(name).await
}

pub async fn delete_namespace(container: &Container, name: &str) -> Result<bool, DomainError> {
    container.memory_repository()?.delete_namespace(name).await
}

pub async fn assign_project(
    container: &Container,
    namespace: &str,
    project: &str,
) -> Result<bool, DomainError> {
    container
        .memory_repository()?
        .assign_project(namespace, project)
        .await
}

pub async fn unassign_project(
    container: &Container,
    namespace: &str,
    project: &str,
) -> Result<bool, DomainError> {
    container
        .memory_repository()?
        .unassign_project(namespace, project)
        .await
}

/// All namespaces with their member-project counts.
pub async fn list_namespaces(container: &Container) -> Result<Vec<(String, u64)>, DomainError> {
    container.memory_repository()?.list_namespaces().await
}

/// A namespace's member projects.
pub async fn namespace_projects(
    container: &Container,
    name: &str,
) -> Result<Vec<String>, DomainError> {
    container.memory_repository()?.namespace_projects(name).await
}
