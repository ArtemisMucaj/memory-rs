//! Maps parsed CLI commands to use cases and formats their output.
//!
//! The router owns all human-facing text and JSON rendering; the use cases
//! return domain values. Every handler returns a `String` for `main` to print,
//! or a [`DomainError`] to report.

use crate::application::{
    resource_slug, DreamReport, ExtractionReport, ImportOutcome, MEMORY_ROOT_URI,
    RESOURCES_ROOT_URI, SESSIONS_ROOT_URI,
};
use crate::cli::{Cli, Command, MemoryKindArg, NamespaceCommand, OutputFormat};
use crate::connector::adapter::{fetch_resource, parse_transcript_file};
use crate::connector::api::Container;
use crate::domain::{
    DomainError, MemoryItem, MemoryKind, MemoryNode, MemoryOperation, SessionStatus,
};

/// Characters of item/node content shown in list/search previews.
const CONTENT_PREVIEW_CHARS: usize = 160;

/// Dispatch a parsed CLI to the matching handler, returning the text to print.
pub async fn run(cli: Cli, container: &Container) -> Result<String, DomainError> {
    match cli.command {
        Command::Import { path, force } => import(container, path, force).await,
        Command::Search {
            query,
            num,
            kind,
            project,
            namespace,
            format,
        } => search(container, query, num, kind, project, namespace, format).await,
        Command::List { kind, format } => list(container, kind, format).await,
        Command::Show { id } => show(container, id).await,
        Command::Delete { id } => delete(container, id).await,
        Command::Sessions { format } => sessions(container, format).await,
        Command::Add { source, name } => add_resource(container, source, name).await,
        Command::Dream { idle_minutes } => dream(container, idle_minutes).await,
        Command::Tree { uri, format } => tree(container, uri, format).await,
        Command::Stats { format } => stats(container, format).await,
        Command::Namespace { command } => namespace(container, command).await,
        // `tui` takes over the terminal and is launched by `main` before the
        // router runs; it should never reach here.
        Command::Tui => Err(DomainError::internal(
            "the `tui` command is handled by main, not the router",
        )),
    }
}

async fn import(container: &Container, path: String, force: bool) -> Result<String, DomainError> {
    let transcript = parse_transcript_file(std::path::Path::new(&path))?;
    let use_case = container.memory_import_use_case()?;
    let outcome = use_case.execute(&transcript, force).await?;
    Ok(render_import_outcome(&outcome))
}

#[allow(clippy::too_many_arguments)]
async fn search(
    container: &Container,
    query: String,
    num: usize,
    kind: Option<MemoryKindArg>,
    project: Option<String>,
    namespace: Option<String>,
    format: OutputFormat,
) -> Result<String, DomainError> {
    let use_case = container.memory_search_use_case()?;
    let kind = kind.map(MemoryKind::from);

    // Scope: a --namespace expands to its member projects (globals always
    // included); a single --project is a one-element scope; neither searches
    // everything. The two flags are mutually exclusive at the CLI layer.
    let scope: Option<Vec<String>> = match (namespace, project) {
        (Some(ns), _) => {
            let projects = container
                .memory_repository()?
                .namespace_projects(&ns)
                .await?;
            if projects.is_empty() {
                return Ok(format!(
                    "Namespace '{ns}' has no projects (assign one with: memory-rs namespace assign {ns} <project>). \
                     Only global memories would match."
                ));
            }
            Some(projects)
        }
        (None, Some(p)) => Some(vec![p]),
        (None, None) => None,
    };

    let results = use_case
        .execute(&query, kind, scope.as_deref(), num)
        .await?;

    match format {
        OutputFormat::Json => {
            let items: Vec<serde_json::Value> = results
                .iter()
                .map(|(item, score)| {
                    let mut value = serde_json::to_value(item).unwrap_or_default();
                    if let Some(obj) = value.as_object_mut() {
                        obj.insert("score".to_string(), serde_json::json!(score));
                    }
                    value
                })
                .collect();
            to_json(&items)
        }
        OutputFormat::Text => {
            if results.is_empty() {
                return Ok("No memories found.".to_string());
            }
            let mut output = String::new();
            for (item, score) in &results {
                output.push_str(&format!(
                    "[{:.3}] [{}] {}{} ({})\n",
                    score,
                    item.kind(),
                    item.name(),
                    project_tag(item),
                    item.id()
                ));
                output.push_str(&format!(
                    "    {}\n\n",
                    preview(item.content(), CONTENT_PREVIEW_CHARS)
                ));
            }
            Ok(output)
        }
    }
}

async fn namespace(
    container: &Container,
    command: NamespaceCommand,
) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;
    match command {
        NamespaceCommand::Create { name } => {
            if repo.create_namespace(&name).await? {
                Ok(format!("Created namespace '{name}'."))
            } else {
                Ok(format!("Namespace '{name}' already exists."))
            }
        }
        NamespaceCommand::Delete { name } => {
            if repo.delete_namespace(&name).await? {
                Ok(format!("Deleted namespace '{name}'."))
            } else {
                Ok(format!("No namespace '{name}'."))
            }
        }
        NamespaceCommand::Assign { namespace, project } => {
            if repo.assign_project(&namespace, &project).await? {
                Ok(format!("Assigned '{project}' to namespace '{namespace}'."))
            } else {
                Ok(format!(
                    "'{project}' is already in namespace '{namespace}'."
                ))
            }
        }
        NamespaceCommand::Unassign { namespace, project } => {
            if repo.unassign_project(&namespace, &project).await? {
                Ok(format!("Removed '{project}' from namespace '{namespace}'."))
            } else {
                Ok(format!("'{project}' is not in namespace '{namespace}'."))
            }
        }
        NamespaceCommand::List { format } => {
            let namespaces = repo.list_namespaces().await?;
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
            let projects = repo.namespace_projects(&name).await?;
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
    kind: Option<MemoryKindArg>,
    format: OutputFormat,
) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;
    let items = repo.list_items(kind.map(MemoryKind::from)).await?;

    match format {
        OutputFormat::Json => to_json(&items),
        OutputFormat::Text => {
            if items.is_empty() {
                return Ok(
                    "No memories stored. Import a session with: memory-rs import <transcript.jsonl>"
                        .to_string(),
                );
            }
            let mut output = format!("{} memories:\n\n", items.len());
            for item in &items {
                output.push_str(&format!(
                    "[{}] {}{} ({})\n",
                    item.kind(),
                    item.name(),
                    project_tag(item),
                    item.id()
                ));
                output.push_str(&format!(
                    "    {}\n\n",
                    preview(item.content(), CONTENT_PREVIEW_CHARS)
                ));
            }
            Ok(output)
        }
    }
}

async fn show(container: &Container, id: String) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;

    // A 'memory://' URI addresses a virtual-filesystem node.
    if id.starts_with("memory://") {
        return match repo.find_node(&id).await? {
            Some(node) => Ok(render_node(&node)),
            None => Ok(format!("No memory node found at '{id}'.")),
        };
    }

    // Accept '<kind>/<name>' as an alternative to the item ID. The same name can
    // exist in several projects, so show them all rather than guessing one.
    if let Some((kind_str, name)) = id.split_once('/') {
        if let Some(kind) = MemoryKind::parse(kind_str) {
            let items = repo.find_items_named(kind, name).await?;
            match items.as_slice() {
                [item] => return Ok(render_item(item)),
                [] => {}
                many => {
                    return Ok(render_many_matches(
                        &id,
                        many,
                        "Show one with: memory-rs show",
                    ))
                }
            }
        }
    }

    match repo.find_item_by_id(&id).await? {
        Some(item) => Ok(render_item(&item)),
        None => Ok(format!("No memory item found with ID '{id}'.")),
    }
}

async fn delete(container: &Container, id: String) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;

    // Accept '<kind>/<name>' as an alternative to the item ID. When the name
    // exists in several projects it is refused rather than guessed — deleting
    // the wrong project's memory is unrecoverable.
    if let Some((kind_str, name)) = id.split_once('/') {
        if let Some(kind) = MemoryKind::parse(kind_str) {
            let items = repo.find_items_named(kind, name).await?;
            match items.as_slice() {
                [item] => {
                    repo.delete_item_by_id(item.id()).await?;
                    return Ok(format!("Deleted memory item '{id}'."));
                }
                [] => {}
                many => {
                    return Ok(render_many_matches(
                        &id,
                        many,
                        "Delete one by ID: memory-rs delete",
                    ))
                }
            }
        }
    }

    if repo.delete_item_by_id(&id).await? {
        Ok(format!("Deleted memory item '{id}'."))
    } else {
        Ok(format!("No memory item found with ID '{id}'."))
    }
}

async fn sessions(container: &Container, format: OutputFormat) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;
    let sessions = repo.list_sessions().await?;

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
                        "    messages: {}, items written: {}\n\n",
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
    // Fetch first — a bad path/URL should fail before we spin up the LLM.
    let fetched = fetch_resource(&source)
        .await
        .map_err(|e| DomainError::internal(format!("failed to fetch resource '{source}': {e}")))?;
    let slug = resource_slug(name.as_deref().unwrap_or(&fetched.title));

    let summary = container.memory_summary_use_case()?;
    let node = summary
        .summarize_resource(&slug, &fetched.source, &fetched.text)
        .await?;
    // Keep the whole-memory digest in sync (best-effort).
    if let Err(e) = summary.regenerate_digest().await {
        tracing::warn!("failed to regenerate memory digest after `add`: {e}");
    }

    Ok(format!(
        "Added resource '{}' ({} chars) at {}\n\n{}",
        fetched.source,
        fetched.text.len(),
        node.uri(),
        node.abstract_()
    ))
}

async fn dream(container: &Container, idle_minutes: u64) -> Result<String, DomainError> {
    let use_case = container.memory_dream_use_case()?;
    // Clamp instead of wrapping: an absurd --idle-minutes must not become a
    // negative threshold that makes still-active sessions eligible.
    let idle_secs = i64::try_from(idle_minutes.saturating_mul(60)).unwrap_or(i64::MAX);
    // A manual `dream` always harvests.
    let report = use_case.execute(idle_secs, true).await?;
    Ok(render_dream_report(&report))
}

async fn tree(
    container: &Container,
    uri: Option<String>,
    format: OutputFormat,
) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;

    let (children, header) = match uri.as_deref() {
        None => {
            let mut nodes = Vec::new();
            if let Some(digest) = repo.find_node(MEMORY_ROOT_URI).await? {
                nodes.push(digest);
            }
            nodes.extend(repo.list_child_nodes(SESSIONS_ROOT_URI).await?);
            nodes.extend(repo.list_child_nodes(RESOURCES_ROOT_URI).await?);
            (nodes, "Memory filesystem".to_string())
        }
        Some(dir) => (
            repo.list_child_nodes(dir).await?,
            format!("Children of {dir}"),
        ),
    };

    match format {
        OutputFormat::Json => to_json(&children),
        OutputFormat::Text => {
            if children.is_empty() {
                return Ok(
                    "Nothing here yet. Import a session with: memory-rs import <transcript.jsonl>"
                        .to_string(),
                );
            }
            let mut output = format!("{header}:\n\n");
            for node in &children {
                output.push_str(&format!("[{}] {}\n", node.kind(), node.uri()));
                output.push_str(&format!(
                    "    {}\n\n",
                    preview(node.abstract_(), CONTENT_PREVIEW_CHARS)
                ));
            }
            output.push_str("Drill in with: memory-rs show <uri>\n");
            Ok(output)
        }
    }
}

async fn stats(container: &Container, format: OutputFormat) -> Result<String, DomainError> {
    let repo = container.memory_repository()?;
    let stats = repo.stats().await?;

    match format {
        OutputFormat::Json => {
            let value = serde_json::json!({
                "total_items": stats.total_items,
                "items_by_kind": stats.items_by_kind,
                "total_sessions": stats.total_sessions,
                "total_nodes": stats.total_nodes,
                "nodes_by_kind": stats.nodes_by_kind,
            });
            to_json(&value)
        }
        OutputFormat::Text => {
            let mut output = String::new();
            output.push_str(&format!("Items:    {}\n", stats.total_items));
            for (kind, count) in &stats.items_by_kind {
                output.push_str(&format!("    {kind}: {count}\n"));
            }
            output.push_str(&format!("Sessions: {}\n", stats.total_sessions));
            output.push_str(&format!("Nodes:    {}\n", stats.total_nodes));
            for (kind, count) in &stats.nodes_by_kind {
                output.push_str(&format!("    {kind}: {count}\n"));
            }
            Ok(output)
        }
    }
}

// ── Rendering helpers ────────────────────────────────────────────────────────

fn render_import_outcome(outcome: &ImportOutcome) -> String {
    match outcome {
        ImportOutcome::AlreadyImported { session } => format!(
            "Session '{}' was already imported ({} messages, {} items written). \
             Use --force to re-import.\n",
            session.id, session.message_count, session.items_written
        ),
        ImportOutcome::Imported { session, report } => {
            let mut output = format!(
                "Imported session '{}' ({} messages).\n",
                session.id, session.message_count
            );
            if report.applied.is_empty() {
                output.push_str("No memories extracted — nothing durable in this session.\n");
            }
            output.push_str(&render_operations(report, "  "));
            output
        }
    }
}

/// Render applied and skipped operation lines (`+`/`-`/`~ … skipped:`), shared
/// by the import and dream reports.
fn render_operations_lists(
    applied: &[MemoryOperation],
    skipped: &[(MemoryOperation, String)],
    indent: &str,
) -> String {
    let mut out = String::new();
    for op in applied {
        match op {
            MemoryOperation::Upsert { kind, name, .. } => {
                out.push_str(&format!("{indent}+ [{kind}] {name}\n"));
            }
            MemoryOperation::Delete { kind, name } => {
                out.push_str(&format!("{indent}- [{kind}] {name}\n"));
            }
        }
    }
    for (op, reason) in skipped {
        let (kind, name) = match op {
            MemoryOperation::Upsert { kind, name, .. } | MemoryOperation::Delete { kind, name } => {
                (kind, name)
            }
        };
        out.push_str(&format!("{indent}~ [{kind}] {name} skipped: {reason}\n"));
    }
    out
}

fn render_operations(report: &ExtractionReport, indent: &str) -> String {
    render_operations_lists(&report.applied, &report.skipped, indent)
}

fn render_dream_report(report: &DreamReport) -> String {
    let mut out = String::new();
    out.push_str("Dream cycle finished\n");
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
    out.push_str(&format!(
        "  consolidation clusters examined: {}\n",
        report.clusters_found
    ));
    if report.applied.is_empty() {
        out.push_str("  memory already consolidated — no operations\n");
    } else {
        out.push_str(&format!(
            "  {} operation(s) applied:\n",
            report.applied.len()
        ));
    }
    out.push_str(&render_operations_lists(
        &report.applied,
        &report.skipped,
        "    ",
    ));
    out
}

fn render_item(item: &MemoryItem) -> String {
    let project = match item.project() {
        Some(project) => format!(", project: {project}"),
        None => ", project: global".to_string(),
    };
    format!(
        "[{}] {} ({})\nupdated {} time(s), source session: {}{}\n\n{}\n",
        item.kind(),
        item.name(),
        item.id(),
        item.update_count(),
        item.source_session_id().unwrap_or("(unknown)"),
        project,
        item.content()
    )
}

/// Render a virtual-filesystem node with its L0 abstract, L1 overview, and L2
/// detail (present only for nodes that store content).
fn render_node(node: &MemoryNode) -> String {
    let mut out = format!("[{}] {}\n\n", node.kind(), node.uri());
    out.push_str(&format!("## Abstract (L0)\n{}\n\n", node.abstract_()));
    if !node.overview().trim().is_empty() {
        out.push_str(&format!("## Overview (L1)\n{}\n\n", node.overview()));
    }
    // Project digest nodes carry an internal manifest in `content`; mask it.
    let content = if node.kind() == crate::domain::NodeKind::Project {
        ""
    } else {
        node.content()
    };
    if !content.trim().is_empty() {
        out.push_str(&format!("## Detail (L2)\n{content}\n"));
    }
    out
}

/// One `<id> @project` line per match, with a hint on how to pick one.
fn render_many_matches(reference: &str, items: &[MemoryItem], hint: &str) -> String {
    let mut out = format!(
        "'{reference}' matches {} memories in different projects:\n\n",
        items.len()
    );
    for item in items {
        out.push_str(&format!(
            "  {} @{}\n",
            item.id(),
            item.project().unwrap_or("(global)")
        ));
    }
    out.push_str(&format!("\n{hint} <id>\n"));
    out
}

/// A compact ` @project` suffix for a project-specific memory, or empty.
fn project_tag(item: &MemoryItem) -> String {
    match item.project() {
        Some(project) => format!(" @{project}"),
        None => String::new(),
    }
}

fn preview(content: &str, max_chars: usize) -> String {
    let single_line: String = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.chars().count() <= max_chars {
        return single_line;
    }
    let truncated: String = single_line
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect();
    format!("{truncated}...")
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String, DomainError> {
    serde_json::to_string_pretty(value)
        .map_err(|e| DomainError::internal(format!("failed to serialize JSON: {e}")))
}
