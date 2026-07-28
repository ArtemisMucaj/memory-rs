//! Maps parsed CLI commands to use cases and formats their output.
//!
//! The router owns all human-facing text and JSON rendering; the use cases
//! return domain values. Every handler returns a `String` for `main` to print,
//! or a [`DomainError`] to report.

use crate::application::{DreamReport, ImportOutcome, Recalled};
use crate::cli::{Cli, Command, MemoryKindArg, NamespaceCommand, OutputFormat};
use crate::connector::api::controller::{
    self, ForgetOutcome, MemorySearchOutcome, MemoryShowOutcome, SearchScope,
};
use crate::connector::api::Container;
use crate::domain::{
    DomainError, Memory, MemoryKind, MemoryNode, MemoryOperation, MemoryStatus, SessionStatus,
};

/// Characters of node content shown in tree previews.
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
        Command::List {
            kind,
            status,
            format,
        } => list(container, kind, status, format).await,
        Command::Show { id } => show(container, id).await,
        Command::Delete { id } => delete(container, id).await,
        Command::Sessions { format } => sessions(container, format).await,
        Command::Add { source, name } => add_resource(container, source, name).await,
        Command::Dream { idle_minutes } => dream(container, idle_minutes).await,
        Command::Tree { uri, format } => tree(container, uri, format).await,
        Command::Stats { format } => stats(container, format).await,
        Command::Conflicts { format } => conflicts(container, format).await,
        Command::Namespace { command } => namespace(container, command).await,
        // `tui` / `serve` / `mcp` are long-running commands dispatched by `main`
        // before the (text-returning) router runs; they never reach here.
        Command::Tui | Command::Serve { .. } | Command::Mcp => Err(DomainError::internal(
            "this command is handled by main, not the router",
        )),
    }
}

async fn import(container: &Container, path: String, force: bool) -> Result<String, DomainError> {
    let outcome = controller::import(container, &path, force).await?;
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
    let kind = kind.map(MemoryKind::from);
    // The CLI makes --project and --namespace mutually exclusive.
    let scope = match (namespace, project) {
        (Some(ns), _) => SearchScope::Namespace(ns),
        (None, Some(p)) => SearchScope::Project(p),
        (None, None) => SearchScope::All,
    };

    let results = match controller::recall_memories(container, &query, kind, &scope, num).await? {
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

async fn conflicts(container: &Container, format: OutputFormat) -> Result<String, DomainError> {
    let conflicts = controller::memory_conflicts(container).await?;
    match format {
        OutputFormat::Json => to_json(
            &conflicts
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "recorded_at": c.recorded_at,
                        "a": c.a,
                        "b": c.b,
                    })
                })
                .collect::<Vec<_>>(),
        ),
        OutputFormat::Text => {
            if conflicts.is_empty() {
                return Ok("No unresolved disagreements.".to_string());
            }
            let mut out = format!(
                "{} unresolved disagreement(s). Both sides still answer queries.\n\n",
                conflicts.len()
            );
            for c in &conflicts {
                out.push_str(&format!("  {}\n    {}\n", c.a.statement, c.a.id));
                out.push_str(&format!("  vs\n  {}\n    {}\n\n", c.b.statement, c.b.id));
            }
            Ok(out.trim_end().to_string())
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
    kind: Option<MemoryKindArg>,
    status: String,
    format: OutputFormat,
) -> Result<String, DomainError> {
    let status = parse_status_arg(&status)?;
    let memories = controller::list_memories(container, kind.map(MemoryKind::from), status).await?;

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

async fn show(container: &Container, id: String) -> Result<String, DomainError> {
    match controller::show_memory(container, &id).await? {
        MemoryShowOutcome::Node(node) => Ok(render_node(&node)),
        MemoryShowOutcome::Memory { memory, edges } => {
            let mut out = render_memory(&memory, Some(&id));
            if edges.is_empty() {
                out.push_str("\nNo edges.\n");
            } else {
                out.push_str(&format!("\nEdges ({}):\n", edges.len()));
                for edge in &edges {
                    // Render direction explicitly: "supersedes X" and
                    // "superseded by X" are opposite facts about this memory,
                    // and an arrow alone leaves the reader to work out which.
                    let (relation, other) = if edge.from_memory == memory.id {
                        (edge.edge_type.as_str(), &edge.to_memory)
                    } else {
                        (edge.edge_type.as_str(), &edge.from_memory)
                    };
                    let direction = if edge.from_memory == memory.id {
                        "->"
                    } else {
                        "<-"
                    };
                    out.push_str(&format!("  {direction} {relation} {other}\n"));
                }
            }
            Ok(out)
        }
        MemoryShowOutcome::NotFound => {
            if id.starts_with("memory://") {
                Ok(format!("No memory node found at '{id}'."))
            } else {
                Ok(format!("No memory found with ID '{id}'."))
            }
        }
    }
}

async fn delete(container: &Container, id: String) -> Result<String, DomainError> {
    match controller::forget_memory(container, &id).await? {
        ForgetOutcome::Retracted => Ok(format!(
            "Retracted memory '{id}'. It stays in the log for provenance and \
             will no longer be recalled."
        )),
        ForgetOutcome::NotFound => Ok(format!("No memory found with ID '{id}'.")),
    }
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
    let added = controller::add_resource(container, &source, name.as_deref()).await?;
    Ok(format!(
        "Added resource '{}' ({} chars) at {}\n\n{}",
        added.source,
        added.chars,
        added.node.uri(),
        added.node.abstract_()
    ))
}

async fn dream(container: &Container, idle_minutes: u64) -> Result<String, DomainError> {
    let report = controller::dream(container, idle_minutes).await?;
    Ok(render_dream_report(&report))
}

async fn tree(
    container: &Container,
    uri: Option<String>,
    format: OutputFormat,
) -> Result<String, DomainError> {
    let children = controller::tree(container, uri.as_deref()).await?;
    let header = match uri.as_deref() {
        None => "Memory filesystem".to_string(),
        Some(dir) => format!("Children of {dir}"),
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
    let stats = controller::stats(container).await?;
    let memories = controller::memory_stats(container).await?;

    match format {
        OutputFormat::Json => {
            let value = serde_json::json!({
                "total_memories": memories.total_memories,
                "memories_by_kind": memories.memories_by_kind,
                "memories_by_status": memories.memories_by_status,
                "total_entities": memories.total_entities,
                "total_edges": memories.total_edges,
                "total_sessions": stats.total_sessions,
                "total_nodes": stats.total_nodes,
                "nodes_by_kind": stats.nodes_by_kind,
            });
            to_json(&value)
        }
        OutputFormat::Text => {
            let mut output = String::new();
            output.push_str(&format!("Memories:   {}\n", memories.total_memories));
            for (kind, count) in &memories.memories_by_kind {
                output.push_str(&format!("    {kind}: {count}\n"));
            }
            for (status, count) in &memories.memories_by_status {
                if status != "active" {
                    output.push_str(&format!("    ({status}): {count}\n"));
                }
            }
            output.push_str(&format!("Entities: {}\n", memories.total_entities));
            output.push_str(&format!("Edges:    {}\n", memories.total_edges));
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
            if report.memories_written == 0 && report.memories_corroborated == 0 {
                output.push_str("No memories extracted — nothing durable in this session.\n");
            } else {
                output.push_str(&format!(
                    "  {} memory(s), {} entity(ies), {} edge(s).\n",
                    report.memories_written, report.entities_created, report.edges_added
                ));
                if report.memories_corroborated > 0 {
                    output.push_str(&format!(
                        "  {} prior memory(s) corroborated instead of duplicated.\n",
                        report.memories_corroborated
                    ));
                }
                if report.memories_superseded > 0 {
                    output.push_str(&format!(
                        "  {} prior memory(s) superseded.\n",
                        report.memories_superseded
                    ));
                }
                if report.conflicts_recorded > 0 {
                    output.push_str(&format!(
                        "  {} contradiction(s) recorded — both sides stay recallable; \
                         review with: memory-rs conflicts\n",
                        report.conflicts_recorded
                    ));
                }
            }
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

/// A memory rendered for the terminal. `highlight_id` is passed when the reader
/// asked for this specific memory, in which case the id is worth repeating back.
fn render_memory(memory: &Memory, highlight_id: Option<&str>) -> String {
    let mut out = format!("[{}] {}\n", memory.kind.as_str(), memory.statement);
    out.push_str(&format!(
        "    {} · confidence {:.2} · {} · {}\n",
        memory.source_kind.as_str(),
        memory.confidence,
        memory.status.as_str(),
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
        // One line, only when there is something to say: a bare result should
        // not grow a blank provenance line.
        let p = &hit.provenance;
        if !p.is_empty() {
            let mut parts = Vec::new();
            if !p.supersedes.is_empty() {
                parts.push(format!(
                    "replaced {}{}",
                    p.supersedes.len(),
                    if p.chain_truncated { "+" } else { "" }
                ));
            }
            if p.corroborations > 0 {
                parts.push(format!("corroborated {}x", p.corroborations));
            }
            if !p.contradicted_by.is_empty() {
                parts.push(format!("contradicted by {}", p.contradicted_by.len()));
            }
            if !p.refinements.is_empty() {
                parts.push(format!("refines {}", p.refinements.len()));
            }
            out.push_str(&format!("    ({})\n\n", parts.join(" · ")));
        }
    }
    out
}

fn memories_with_scores(hits: &[Recalled]) -> Vec<serde_json::Value> {
    hits.iter()
        .map(|hit| {
            let mut value = serde_json::to_value(&hit.memory).unwrap_or_default();
            if let Some(obj) = value.as_object_mut() {
                obj.insert("score".to_string(), serde_json::json!(hit.score));
                obj.insert(
                    "provenance".to_string(),
                    serde_json::json!({
                        "supersedes_count": hit.provenance.supersedes.len(),
                        "chain_truncated": hit.provenance.chain_truncated,
                        "corroborations": hit.provenance.corroborations,
                        "contradicted_by": hit.provenance.contradicted_by.len(),
                        "refinements_count": hit.provenance.refinements.len(),
                    }),
                );
            }
            value
        })
        .collect()
}

/// Absent/`active` is the default; `all` is the only way to reach history.
fn parse_status_arg(status: &str) -> Result<Option<MemoryStatus>, DomainError> {
    if status.eq_ignore_ascii_case("all") {
        return Ok(None);
    }
    MemoryStatus::parse(status).map(Some).ok_or_else(|| {
        DomainError::invalid_input(format!(
            "unknown status '{status}' (expected active, superseded, retracted, or all)"
        ))
    })
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String, DomainError> {
    serde_json::to_string_pretty(value)
        .map_err(|e| DomainError::internal(format!("failed to serialize JSON: {e}")))
}
