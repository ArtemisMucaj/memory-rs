//! Maps parsed CLI commands to use cases and formats their output.
//!
//! The router owns all human-facing text and JSON rendering; the use cases
//! return domain values. Every handler returns a `String` for `main` to
//! print, or a [`DomainError`] to report.

use crate::application::{DreamReport, ImportOutcome, Recalled, ResumeBriefing};
use crate::cli::{Cli, Command, NamespaceCommand, OutputFormat};
use crate::connector::api::controller::{
    self, ForgetOutcome, MemorySearchOutcome, MemoryShowOutcome, ResumeOutcome, SearchScope,
};
use crate::connector::api::Container;
use crate::domain::{DomainError, Memory, SessionStatus};

/// Dispatch a parsed CLI to the matching handler, returning the text to print.
pub async fn run(cli: Cli, container: &Container) -> Result<String, DomainError> {
    match cli.command {
        Command::Import { path, force } => import(container, path, force).await,
        Command::Search {
            query,
            num,
            project,
            namespace,
            format,
        } => search(container, query, num, project, namespace, format).await,
        Command::List {
            project,
            namespace,
            format,
        } => list(container, project, namespace, format).await,
        Command::Show { id } => show(container, id).await,
        Command::Delete { id } => delete(container, id).await,
        Command::Sessions { format } => sessions(container, format).await,
        Command::Resume {
            project,
            namespace,
            limit,
            format,
        } => resume(container, project, namespace, limit, format).await,
        Command::Add { source, name } => add_resource(container, source, name).await,
        Command::Dream { idle_minutes } => dream(container, idle_minutes).await,
        Command::Entities { format } => entities(container, format).await,
        Command::Entity { name } => entity(container, name).await,
        Command::Namespace { command } => namespace(container, command).await,
        // `tui` / `serve` / `mcp` are long-running commands dispatched by
        // `main` before the (text-returning) router runs; they never reach
        // here.
        Command::Tui | Command::Serve { .. } | Command::Mcp => Err(DomainError::internal(
            "this command is handled by main, not the router",
        )),
    }
}

async fn import(container: &Container, path: String, force: bool) -> Result<String, DomainError> {
    let outcome = controller::import(container, &path, force).await?;
    Ok(render_import_outcome(&outcome))
}

async fn search(
    container: &Container,
    query: String,
    num: usize,
    project: Option<String>,
    namespace: Option<String>,
    format: OutputFormat,
) -> Result<String, DomainError> {
    // The CLI makes --project and --namespace mutually exclusive.
    let scope = match (namespace, project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };

    let results = match controller::recall_memories(container, &query, None, &scope, num).await? {
        MemorySearchOutcome::Hits(hits) => hits,
        MemorySearchOutcome::EmptyNamespace(ns) => {
            return Ok(format!(
                "Namespace '{ns}' has no projects (assign one with: memory-rs namespace assign \
                 {ns} <project>). Only global memories would match."
            ));
        }
    };

    match format {
        OutputFormat::Json => to_json(&memories_with_scores(&results)),
        OutputFormat::Text => {
            if results.is_empty() {
                return Ok("No memories matched.".to_string());
            }
            Ok(render_memory_list(&results))
        }
    }
}

async fn namespace(
    container: &Container,
    command: NamespaceCommand,
) -> Result<String, DomainError> {
    match command {
        NamespaceCommand::Create { name } => {
            if controller::create_namespace(container, &name).await? {
                Ok(format!("Created namespace '{name}'."))
            } else {
                Ok(format!("Namespace '{name}' already exists."))
            }
        }
        NamespaceCommand::Delete { name } => {
            if controller::delete_namespace(container, &name).await? {
                Ok(format!("Deleted namespace '{name}'."))
            } else {
                Ok(format!("No namespace '{name}'."))
            }
        }
        NamespaceCommand::Assign { namespace, project } => {
            if controller::assign_project(container, &namespace, &project).await? {
                Ok(format!("Assigned '{project}' to namespace '{namespace}'."))
            } else {
                Ok(format!(
                    "'{project}' is already in namespace '{namespace}'."
                ))
            }
        }
        NamespaceCommand::Unassign { namespace, project } => {
            if controller::unassign_project(container, &namespace, &project).await? {
                Ok(format!("Removed '{project}' from namespace '{namespace}'."))
            } else {
                Ok(format!("'{project}' is not in namespace '{namespace}'."))
            }
        }
        NamespaceCommand::List { format } => {
            let namespaces = controller::list_namespaces(container).await?;
            match format {
                OutputFormat::Json => to_json(&namespaces),
                OutputFormat::Text => {
                    if namespaces.is_empty() {
                        return Ok(
                            "No namespaces. Create one with: memory-rs namespace create <name>"
                                .to_string(),
                        );
                    }
                    let mut out = format!("{} namespace(s):\n\n", namespaces.len());
                    for (name, count) in &namespaces {
                        out.push_str(&format!("{name}  ({count} project(s))\n"));
                    }
                    Ok(out)
                }
            }
        }
        NamespaceCommand::Show { name, format } => {
            let projects = controller::namespace_projects(container, &name).await?;
            match format {
                OutputFormat::Json => to_json(&projects),
                OutputFormat::Text => {
                    if projects.is_empty() {
                        return Ok(format!(
                            "Namespace '{name}' has no projects (assign one with: \
                             memory-rs namespace assign {name} <project>)."
                        ));
                    }
                    let mut out =
                        format!("Namespace '{name}' ({} project(s)):\n\n", projects.len());
                    for p in &projects {
                        out.push_str(&format!("  {p}\n"));
                    }
                    Ok(out)
                }
            }
        }
    }
}

async fn list(
    container: &Container,
    project: Option<String>,
    namespace: Option<String>,
    format: OutputFormat,
) -> Result<String, DomainError> {
    let scope = match (namespace, project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };
    let projects = match controller::resolve_scope(container, &scope).await? {
        controller::ScopeResolution::All => None,
        controller::ScopeResolution::Projects(p) => Some(p),
        controller::ScopeResolution::EmptyNamespace(ns) => {
            return Ok(format!("Namespace '{ns}' has no projects."));
        }
    };
    let memories = controller::list_memories(container, projects.as_deref()).await?;

    match format {
        OutputFormat::Json => to_json(&memories),
        OutputFormat::Text => {
            if memories.is_empty() {
                return Ok(
                    "No memories stored. Import a session with: memory-rs import <transcript.jsonl>"
                        .to_string(),
                );
            }
            let mut output = format!("{} memories:\n\n", memories.len());
            for memory in &memories {
                output.push_str(&render_memory(memory, None));
            }
            Ok(output)
        }
    }
}

async fn entities(container: &Container, format: OutputFormat) -> Result<String, DomainError> {
    let summaries = controller::list_entities(container).await?;
    match format {
        OutputFormat::Json => to_json(&serde_json::json!(summaries
            .iter()
            .map(|s| serde_json::json!({
                "id": s.entity.id,
                "type": s.entity.entity_type,
                "name": s.entity.canonical_name,
                "aliases": s.entity.names,
                "memory_count": s.memory_count,
            }))
            .collect::<Vec<_>>())),
        OutputFormat::Text => {
            if summaries.is_empty() {
                return Ok("No entities stored.".to_string());
            }
            let mut out = format!("{} entities:\n\n", summaries.len());
            for s in &summaries {
                out.push_str(&format!(
                    "  [{}] {} ({} memories)\n",
                    s.entity.entity_type, s.entity.canonical_name, s.memory_count
                ));
            }
            Ok(out)
        }
    }
}

async fn entity(container: &Container, name: String) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;
    let found = repo.find_entities_by_name(&name).await?;
    if found.is_empty() {
        return Ok(format!("No entity named '{name}'."));
    }
    if found.len() > 1 {
        let mut out = format!("'{name}' matches {} entities:\n\n", found.len());
        for e in &found {
            out.push_str(&format!(
                "  [{}] {} (id: {})\n",
                e.entity_type, e.canonical_name, e.id
            ));
        }
        out.push_str("\nDisambiguate by id with `memory-rs show <id>`.\n");
        return Ok(out);
    }
    let entity = found.into_iter().next().unwrap();
    let memories = repo.memories_for_entity(&entity.id).await?;
    let mut out = format!(
        "[{}] {} ({} memories)\n",
        entity.entity_type,
        entity.canonical_name,
        memories.len()
    );
    if !entity.names.is_empty() {
        out.push_str(&format!("  also known as: {}\n", entity.names.join(", ")));
    }
    out.push('\n');
    for m in &memories {
        out.push_str(&format!("  - {}\n", m.statement));
    }
    Ok(out)
}

async fn show(container: &Container, id: String) -> Result<String, DomainError> {
    match controller::show_memory(container, &id).await? {
        MemoryShowOutcome::Resource(resource) => {
            let mut out = format!("[resource] {}\n\n", resource.uri);
            out.push_str(&format!("## Abstract\n{}\n\n", resource.abstract_));
            if !resource.overview.trim().is_empty() {
                out.push_str(&format!("## Overview\n{}\n\n", resource.overview));
            }
            if !resource.content.trim().is_empty() {
                out.push_str(&format!("## Content\n{}\n", resource.content));
            }
            Ok(out)
        }
        MemoryShowOutcome::Memory(memory) => Ok(render_memory(&memory, Some(&id))),
        MemoryShowOutcome::NotFound => {
            if id.starts_with("memory://") {
                Ok(format!("No resource found at '{id}'."))
            } else {
                Ok(format!("No memory found with ID '{id}'."))
            }
        }
    }
}

async fn delete(container: &Container, id: String) -> Result<String, DomainError> {
    match controller::forget_memory(container, &id).await? {
        ForgetOutcome::Deleted => Ok(format!("Deleted memory '{id}'.")),
        ForgetOutcome::NotFound => Ok(format!("No memory found with ID '{id}'.")),
    }
}

/// `resume` — the briefing, rendered so it can be read top-to-bottom as the
/// answer to "where was I".
async fn resume(
    container: &Container,
    project: Option<String>,
    namespace: Option<String>,
    limit: usize,
    format: OutputFormat,
) -> Result<String, DomainError> {
    // The CLI makes --project and --namespace mutually exclusive.
    let scope = match (namespace, project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };

    let briefing = match controller::resume(container, &scope, limit).await? {
        ResumeOutcome::Briefing(b) => b,
        ResumeOutcome::EmptyNamespace(ns) => {
            return Ok(format!(
                "Namespace '{ns}' has no projects (assign one with: memory-rs namespace assign \
                 {ns} <project>)."
            ));
        }
    };

    match format {
        OutputFormat::Json => to_json(&resume_json(&briefing)),
        OutputFormat::Text => Ok(render_briefing(&briefing)),
    }
}

fn render_briefing(briefing: &ResumeBriefing) -> String {
    if briefing.is_empty() {
        // Say which of the two reasons it is: an empty store and a scope
        // with no sessions look identical otherwise, and the fixes are
        // different.
        return if briefing.projects.is_empty() {
            "No sessions imported yet. Import one with: memory-rs import <transcript>".to_string()
        } else {
            format!(
                "No sessions recorded for {}. Sessions imported before projects were \
                 tracked have none, and only appear unscoped.",
                briefing.projects.join(", ")
            )
        };
    }

    let mut out = String::new();
    let scope = if briefing.projects.is_empty() {
        "all projects".to_string()
    } else {
        briefing.projects.join(", ")
    };
    out.push_str(&format!(
        "Recent work — {} session(s), {scope}\n\n",
        briefing.sessions.len()
    ));

    for recap in &briefing.sessions {
        let session = &recap.session;
        out.push_str(&format!(
            "{} · {}\n",
            format_unix(session.imported_at),
            session.project.as_deref().unwrap_or("no project"),
        ));
        if recap.memories.is_empty() {
            out.push_str("  (no durable memories from this session)\n");
        } else {
            out.push_str(&format!("  remembered ({}):\n", recap.memories.len()));
            for memory in &recap.memories {
                out.push_str(&format!("    - {}\n", memory.statement));
            }
        }
        out.push_str(&format!(
            "  session: {} · {}\n\n",
            session.id, session.source
        ));
    }

    if briefing.more > 0 {
        out.push_str(&format!(
            "{} older session(s) not shown — raise --limit to include them.\n",
            briefing.more
        ));
    }
    out
}

fn resume_json(briefing: &ResumeBriefing) -> serde_json::Value {
    serde_json::json!({
        "projects": briefing.projects,
        "more": briefing.more,
        "sessions": briefing
            .sessions
            .iter()
            .map(|recap| serde_json::json!({
                "id": recap.session.id,
                "source": recap.session.source,
                "project": recap.session.project,
                "imported_at": recap.session.imported_at,
                "message_count": recap.session.message_count,
                "memories": recap
                    .memories
                    .iter()
                    .map(|m| serde_json::json!({
                        "id": m.id,
                        "statement": m.statement,
                        "project": m.project,
                    }))
                    .collect::<Vec<_>>(),
            }))
            .collect::<Vec<_>>(),
    })
}

/// A unix timestamp as `YYYY-MM-DD HH:MM` UTC — enough to place a session in
/// time without pulling in a date library.
fn format_unix(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let secs_of_day = seconds.rem_euclid(86_400);
    // Civil-from-days (Howard Hinnant's algorithm), epoch-shifted to 0000-03-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        secs_of_day / 3_600,
        (secs_of_day % 3_600) / 60
    )
}

async fn sessions(container: &Container, format: OutputFormat) -> Result<String, DomainError> {
    let sessions = controller::sessions(container).await?;

    match format {
        OutputFormat::Json => to_json(&sessions),
        OutputFormat::Text => {
            if sessions.is_empty() {
                return Ok("No sessions imported yet.".to_string());
            }
            let failed = sessions
                .iter()
                .filter(|s| s.status == SessionStatus::Failed)
                .count();
            let mut output = format!("{} sessions ({failed} failed):\n\n", sessions.len());
            for session in &sessions {
                output.push_str(&format!("{}\n    source: {}\n", session.id, session.source));
                match session.status {
                    SessionStatus::Imported => output.push_str(&format!(
                        "    messages: {}, memories written: {}\n\n",
                        session.message_count, session.items_written
                    )),
                    SessionStatus::Failed => output.push_str(&format!(
                        "    FAILED: {}\n    retry with: memory-rs import <path> --force\n\n",
                        session.last_error.as_deref().unwrap_or("unknown error")
                    )),
                }
            }
            Ok(output)
        }
    }
}

async fn add_resource(
    container: &Container,
    source: String,
    name: Option<String>,
) -> Result<String, DomainError> {
    let added = controller::add_resource(container, &source, name.as_deref()).await?;
    Ok(format!(
        "Added resource '{}' ({} chars) at {}\n\n{}",
        added.source, added.chars, added.resource.uri, added.resource.abstract_
    ))
}

async fn dream(container: &Container, idle_minutes: u64) -> Result<String, DomainError> {
    let report = controller::dream(container, idle_minutes).await?;
    Ok(render_dream_report(&report))
}

// ── Rendering helpers ────────────────────────────────────────────────────────

fn render_import_outcome(outcome: &ImportOutcome) -> String {
    match outcome {
        ImportOutcome::AlreadyImported { session } => format!(
            "Session '{}' was already imported ({} messages, {} memories written). \
             Use --force to re-import.\n",
            session.id, session.message_count, session.items_written
        ),
        ImportOutcome::Imported { session, report } => {
            let mut output = format!(
                "Imported session '{}' ({} messages).\n",
                session.id, session.message_count
            );
            if report.memories_written == 0 && report.memories_deduped == 0 {
                output.push_str("No memories extracted — nothing durable in this session.\n");
            } else {
                output.push_str(&format!(
                    "  {} memory(s) written, {} entit(ies) created, {} deduped.\n",
                    report.memories_written, report.entities_created, report.memories_deduped
                ));
            }
            output
        }
    }
}

fn render_dream_report(report: &DreamReport) -> String {
    let mut out = String::new();
    out.push_str("Harvest finished\n");
    out.push_str(&format!(
        "  sessions: {} finished and not yet imported, {} imported\n",
        report.sessions_eligible, report.sessions_imported
    ));
    if report.sessions_failed > 0 {
        out.push_str(&format!(
            "  {} failed and marked (not retried automatically) — see: memory-rs sessions\n",
            report.sessions_failed
        ));
    }
    out
}

/// A memory rendered for the terminal. `highlight_id` is passed when the
/// reader asked for this specific memory.
fn render_memory(memory: &Memory, highlight_id: Option<&str>) -> String {
    let mut out = format!("{}\n", memory.statement);
    out.push_str(&format!(
        "    {} · confidence {:.2} · {}\n",
        memory.source_kind.as_str(),
        memory.confidence,
        memory.project.as_deref().unwrap_or("global"),
    ));
    if highlight_id.is_some() {
        out.push_str(&format!("    id: {}\n", memory.id));
    }
    out.push('\n');
    out
}

fn render_memory_list(hits: &[Recalled]) -> String {
    let mut out = String::new();
    for hit in hits {
        out.push_str(&format!("[{:.3}] ", hit.score));
        out.push_str(&render_memory(&hit.memory, None));
    }
    out
}

fn memories_with_scores(hits: &[Recalled]) -> Vec<serde_json::Value> {
    hits.iter()
        .map(|hit| {
            let mut value = serde_json::to_value(&hit.memory).unwrap_or_default();
            if let Some(obj) = value.as_object_mut() {
                obj.insert("score".to_string(), serde_json::json!(hit.score));
            }
            value
        })
        .collect()
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String, DomainError> {
    serde_json::to_string_pretty(value)
        .map_err(|e| DomainError::internal(format!("failed to serialize JSON: {e}")))
}
