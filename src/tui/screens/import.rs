//! The Import screen: discovered sessions on the left, the highlighted
//! session's transcript on the right, and an Import action.
//!
//! Discovery runs once in a background task and streams its result back over a
//! channel, so the screen opens immediately. Each import also runs in the
//! background, reporting queued → importing → done/failed via a second channel,
//! so the list stays responsive and every row shows live status. Transcripts
//! are loaded lazily the first time a session is highlighted and cached.

use std::collections::HashMap;
use std::sync::Arc;

use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use tokio::sync::mpsc;

use crate::application::{ImportOutcome, SessionDiscovery};
use crate::connector::api::Container;
use crate::domain::{DiscoveredSession, SessionMessage};
use crate::tui::{markdown, theme};

/// Stable identity of a discovered session: `(source, id)`.
type SessionKey = (String, String);

/// Per-session import status shown as a row marker.
#[derive(Debug, Clone)]
enum Status {
    None,
    AlreadyImported,
    Queued,
    Importing,
    Done,
    /// Import failed; the error is surfaced in the footer via `last_result`.
    Failed,
}

/// A cached transcript load result.
enum Loaded {
    Ok(Vec<SessionMessage>),
    Failed(String),
}

/// A status update from a background import task.
enum ImportEvent {
    /// A session already present in the store (seeded on open), keyed by id.
    AlreadyImported(String),
    Started(SessionKey),
    Done {
        key: SessionKey,
        summary: String,
    },
    Failed {
        key: SessionKey,
        error: String,
    },
}

pub struct ImportScreen {
    sessions: Vec<DiscoveredSession>,
    selected: usize,
    list_scroll: usize,
    transcript_scroll: u16,
    now_secs: i64,
    status: HashMap<SessionKey, Status>,
    /// Ids of sessions already in the store, from the open-time seed. Kept
    /// separately from `sessions` because the seed can arrive before discovery
    /// populates the list; the ✓ marks are (re)applied whenever sessions load.
    imported_ids: std::collections::HashSet<String>,
    cache: HashMap<usize, Loaded>,
    discovering: bool,
    last_result: Option<String>,

    // Channels: discovery result in, import events in.
    discovery_rx: Option<mpsc::UnboundedReceiver<Vec<DiscoveredSession>>>,
    events_rx: Option<mpsc::UnboundedReceiver<ImportEvent>>,
    events_tx: mpsc::UnboundedSender<ImportEvent>,
    discovery: Option<Arc<dyn SessionDiscovery>>,
}

impl ImportScreen {
    pub fn new() -> Self {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        Self {
            sessions: Vec::new(),
            selected: 0,
            list_scroll: 0,
            transcript_scroll: 0,
            now_secs: 0,
            status: HashMap::new(),
            imported_ids: std::collections::HashSet::new(),
            cache: HashMap::new(),
            discovering: false,
            last_result: None,
            discovery_rx: None,
            events_rx: Some(events_rx),
            events_tx,
            discovery: None,
        }
    }

    /// Kick off discovery in the background (idempotent per launch).
    pub fn start_discovery(&mut self, container: &Container) {
        if self.discovery.is_some() {
            return;
        }
        let discovery = container.session_discovery();
        self.discovery = Some(Arc::clone(&discovery));
        self.now_secs = unix_now();
        self.discovering = true;

        let (tx, rx) = mpsc::unbounded_channel();
        self.discovery_rx = Some(rx);
        tokio::spawn(async move {
            let found = discovery.discover().await.unwrap_or_default();
            let _ = tx.send(found);
        });

        // Seed the ✓ marks for sessions already imported into the store.
        self.seed_imported(container);
    }

    fn seed_imported(&mut self, container: &Container) {
        let Ok(repo) = container.memory_repository() else {
            return;
        };
        let events_tx = self.events_tx.clone();
        // Read the imported set off-thread and feed each id back as an
        // AlreadyImported marker. The stored session id matches the discovery
        // id; source is not stored, so marks are keyed by id (applied to
        // whichever discovered session carries that id, whenever it loads).
        tokio::spawn(async move {
            if let Ok(sessions) = repo.list_sessions().await {
                for s in sessions {
                    let _ = events_tx.send(ImportEvent::AlreadyImported(s.id));
                }
            }
        });
    }

    /// (Re)apply the seeded "already imported" marks to the current session
    /// list. Called after discovery loads and after each new seed id, so the
    /// marks survive whichever of the two arrives first.
    fn apply_imported_marks(&mut self) {
        for s in &self.sessions {
            if self.imported_ids.contains(&s.id) {
                self.status
                    .entry(session_key(s))
                    // Don't clobber a live import result (Done/Failed/Importing).
                    .or_insert(Status::AlreadyImported);
            }
        }
    }

    /// Drain the discovery and import-event channels into state. Called each
    /// tick from the app loop so streamed data appears without a keypress.
    pub fn pump(&mut self) {
        // Drain each channel into a local first, then mutate `self`, so the
        // borrow of the receiver doesn't overlap the `&mut self` state updates.
        let mut latest_found = None;
        if let Some(rx) = self.discovery_rx.as_mut() {
            while let Ok(found) = rx.try_recv() {
                latest_found = Some(found);
            }
        }
        if let Some(found) = latest_found {
            self.sessions = found;
            self.discovering = false;
            self.clamp_selection();
            self.apply_imported_marks();
        }
        if let Some(rx) = self.events_rx.as_mut() {
            let mut drained = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                drained.push(ev);
            }
            for ev in drained {
                self.apply_event(ev);
            }
        }
    }

    fn apply_event(&mut self, ev: ImportEvent) {
        match ev {
            ImportEvent::AlreadyImported(id) => {
                self.imported_ids.insert(id);
                // Sessions may or may not have loaded yet; apply against whatever
                // is present now, and `pump` reapplies when discovery loads.
                self.apply_imported_marks();
            }
            ImportEvent::Started(key) => {
                self.status.insert(key, Status::Importing);
            }
            ImportEvent::Done { key, summary } => {
                self.status.insert(key, Status::Done);
                self.last_result = Some(summary);
            }
            ImportEvent::Failed { key, error } => {
                self.last_result = Some(format!("Import failed: {error}"));
                self.status.insert(key, Status::Failed);
            }
        }
    }

    pub async fn handle_key(
        &mut self,
        code: KeyCode,
        _modifiers: KeyModifiers,
        container: &Container,
    ) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::PageUp => {
                self.transcript_scroll = self.transcript_scroll.saturating_sub(8);
            }
            KeyCode::PageDown => {
                self.transcript_scroll = self.transcript_scroll.saturating_add(8);
            }
            KeyCode::Char('r') => self.rescan(container),
            KeyCode::Enter | KeyCode::Char('i') => self.import_selected(container),
            _ => {}
        }
        // Ensure the highlighted transcript is loaded for the detail pane.
        self.ensure_loaded().await;
    }

    fn rescan(&mut self, container: &Container) {
        self.sessions.clear();
        self.cache.clear();
        self.selected = 0;
        self.discovery = None;
        self.discovery_rx = None;
        self.start_discovery(container);
    }

    /// Import the highlighted session in the background, reporting progress.
    fn import_selected(&mut self, container: &Container) {
        let Some(session) = self.sessions.get(self.selected).cloned() else {
            return;
        };
        let key = session_key(&session);
        if matches!(
            self.status.get(&key),
            Some(Status::Importing | Status::Queued)
        ) {
            return;
        }

        // Build the import use case + discovery now; a build error (e.g. no LLM
        // configured) is surfaced immediately rather than in the task.
        let import = match container.memory_import_use_case() {
            Ok(uc) => uc,
            Err(e) => {
                self.last_result = Some(format!("Cannot import: {e}"));
                return;
            }
        };
        let Some(discovery) = self.discovery.clone() else {
            return;
        };
        self.status.insert(key.clone(), Status::Queued);
        let events_tx = self.events_tx.clone();

        tokio::spawn(async move {
            let _ = events_tx.send(ImportEvent::Started(key.clone()));
            let transcript = match discovery.load_transcript(&session).await {
                Ok(t) => t,
                Err(e) => {
                    let _ = events_tx.send(ImportEvent::Failed {
                        key,
                        error: e.to_string(),
                    });
                    return;
                }
            };
            match import.execute(&transcript, false).await {
                Ok(ImportOutcome::Imported { report, .. }) => {
                    let _ = events_tx.send(ImportEvent::Done {
                        key,
                        summary: format!(
                            "Imported '{}' — {} memories",
                            session.display_title(),
                            report.applied.len()
                        ),
                    });
                }
                Ok(ImportOutcome::AlreadyImported { .. }) => {
                    let _ = events_tx.send(ImportEvent::Done {
                        key,
                        summary: format!("'{}' was already imported", session.display_title()),
                    });
                }
                Err(e) => {
                    let _ = events_tx.send(ImportEvent::Failed {
                        key,
                        error: e.to_string(),
                    });
                }
            }
        });
    }

    async fn ensure_loaded(&mut self) {
        let idx = self.selected;
        if self.cache.contains_key(&idx) {
            return;
        }
        let Some((session, discovery)) =
            self.sessions.get(idx).cloned().zip(self.discovery.clone())
        else {
            return;
        };
        let loaded = match discovery.load_transcript(&session).await {
            Ok(t) => Loaded::Ok(t.messages),
            Err(e) => Loaded::Failed(e.to_string()),
        };
        self.cache.insert(idx, loaded);
    }

    fn move_selection(&mut self, delta: i32) {
        if self.sessions.is_empty() {
            return;
        }
        let len = self.sessions.len() as i32;
        let next = (self.selected as i32 + delta).clamp(0, len - 1) as usize;
        if next != self.selected {
            self.selected = next;
            self.transcript_scroll = 0;
        }
    }

    fn clamp_selection(&mut self) {
        if self.selected >= self.sessions.len() {
            self.selected = self.sessions.len().saturating_sub(1);
        }
    }

    pub fn footer_hint(&self) -> &'static str {
        "  ↑↓: move  Enter: import  r: rescan"
    }

    /// The most recent import outcome, shown in the footer until superseded.
    pub fn status_line(&self) -> Option<&str> {
        self.last_result.as_deref()
    }

    // ── Rendering ────────────────────────────────────────────────────────────

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(area);
        self.render_header(frame, rows[0]);

        let panes = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[1]);
        self.render_list(frame, panes[0]);
        self.render_transcript(frame, panes[1]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let imported = self
            .sessions
            .iter()
            .filter(|s| {
                matches!(
                    self.status.get(&session_key(s)),
                    Some(Status::Done | Status::AlreadyImported)
                )
            })
            .count();

        let mut left = vec![
            Span::styled(
                format!("  {} ", self.sessions.len()),
                Style::default()
                    .fg(ratatui::style::Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("found    ", Style::default().fg(theme::MUTED)),
            Span::styled(
                format!("{imported} "),
                Style::default()
                    .fg(ratatui::style::Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("imported", Style::default().fg(theme::MUTED)),
        ];
        if self.discovering {
            left.push(Span::styled(
                "   · discovering…",
                Style::default().fg(theme::ACCENT),
            ));
        }
        let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(12)]).split(inner);
        frame.render_widget(Paragraph::new(Line::from(left)), cols[0]);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "↻ Rescan (r)",
                Style::default().fg(theme::MUTED),
            )))
            .alignment(Alignment::Right),
            cols[1],
        );
    }

    fn render_list(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.sessions.is_empty() {
            let msg = if self.discovering {
                "  Discovering sessions…"
            } else {
                "  No sessions found on this machine."
            };
            frame.render_widget(
                Paragraph::new(msg).style(Style::default().fg(theme::MUTED)),
                inner,
            );
            return;
        }

        let height = inner.height as usize;
        if self.selected < self.list_scroll {
            self.list_scroll = self.selected;
        } else if height > 0 && self.selected >= self.list_scroll + height {
            self.list_scroll = self.selected + 1 - height;
        }

        let width = inner.width as usize;
        let lines: Vec<Line> = self
            .sessions
            .iter()
            .enumerate()
            .skip(self.list_scroll)
            .take(height)
            .map(|(i, s)| self.session_line(i, s, width))
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn session_line(&self, i: usize, s: &DiscoveredSession, width: usize) -> Line<'static> {
        let selected = i == self.selected;
        let bg = if selected {
            theme::SELECTION_BG
        } else {
            ratatui::style::Color::Reset
        };
        let (marker, marker_color) =
            status_marker(self.status.get(&session_key(s)).unwrap_or(&Status::None));
        let title_width = width.saturating_sub(34);
        Line::from(vec![
            Span::styled(
                format!("{marker} "),
                Style::default().fg(marker_color).bg(bg),
            ),
            Span::styled(
                format!("{:<9}", s.source.as_str()),
                Style::default()
                    .fg(theme::source_color(s.source.as_str()))
                    .bg(bg),
            ),
            Span::styled(
                format!("{:>8}  ", theme::relative_time(s.updated_at, self.now_secs)),
                Style::default().fg(theme::MUTED).bg(bg),
            ),
            Span::styled(
                format!("{:>6}  ", theme::fmt_tokens(s.approx_tokens)),
                Style::default().fg(ratatui::style::Color::Yellow).bg(bg),
            ),
            Span::styled(
                theme::truncate(s.display_title(), title_width),
                Style::default()
                    .fg(if selected {
                        ratatui::style::Color::White
                    } else {
                        ratatui::style::Color::Gray
                    })
                    .bg(bg)
                    .add_modifier(if selected {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
        ])
    }

    fn render_transcript(&self, frame: &mut Frame, area: Rect) {
        let title = self
            .sessions
            .get(self.selected)
            .map(|s| format!(" {} ", theme::truncate(s.display_title(), 60)))
            .unwrap_or_else(|| " Transcript ".to_string());
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(theme::MUTED));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let lines: Vec<Line> = if self.sessions.is_empty() {
            vec![Line::from(Span::styled(
                if self.discovering {
                    "Discovering…"
                } else {
                    "No session selected."
                },
                Style::default().fg(theme::MUTED),
            ))]
        } else {
            match self.cache.get(&self.selected) {
                Some(Loaded::Ok(messages)) => render_conversation(messages),
                Some(Loaded::Failed(e)) => vec![Line::from(Span::styled(
                    format!("Could not load transcript: {e}"),
                    Style::default().fg(ratatui::style::Color::Red),
                ))],
                None => vec![Line::from(Span::styled(
                    "Loading…",
                    Style::default().fg(theme::MUTED),
                ))],
            }
        };
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((self.transcript_scroll, 0)),
            inner,
        );
    }
}

/// Render a per-turn conversation: a coloured role header per message followed
/// by its Markdown-rendered content.
fn render_conversation(messages: &[SessionMessage]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for msg in messages {
        if msg.content.trim().is_empty() {
            continue;
        }
        let (label, color) = match msg.role.as_str() {
            "user" => ("▌ User", theme::ACCENT),
            "assistant" => ("▌ Assistant", ratatui::style::Color::Green),
            "system" => ("▌ System", ratatui::style::Color::Yellow),
            _ => ("▌ Message", ratatui::style::Color::Yellow),
        };
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            label.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        lines.extend(markdown::render(&msg.content));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "(no textual content)",
            Style::default().fg(theme::MUTED),
        )));
    }
    lines
}

fn status_marker(status: &Status) -> (&'static str, ratatui::style::Color) {
    match status {
        Status::None => ("○", theme::MUTED),
        Status::AlreadyImported => ("✓", ratatui::style::Color::Green),
        Status::Queued => ("…", ratatui::style::Color::Yellow),
        Status::Importing => ("⟳", theme::ACCENT),
        Status::Done => ("✓", ratatui::style::Color::Green),
        Status::Failed => ("✗", ratatui::style::Color::Red),
    }
}

fn session_key(s: &DiscoveredSession) -> SessionKey {
    (s.source.as_str().to_string(), s.id.clone())
}

/// Current Unix time in seconds, for relative-time labels.
fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

impl Default for ImportScreen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{SessionLocator, SessionSource};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn session(source: SessionSource, id: &str, title: &str, tokens: usize) -> DiscoveredSession {
        DiscoveredSession {
            source,
            id: id.into(),
            title: title.into(),
            cwd: None,
            updated_at: 900,
            message_count: 3,
            approx_tokens: tokens,
            tail_preview: String::new(),
            locator: SessionLocator::File(format!("{id}.jsonl")),
        }
    }

    fn render_to_text(screen: &mut ImportScreen, w: u16, h: u16) -> String {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| screen.render(f, f.area())).unwrap();
        let buffer = terminal.backend().buffer().clone();
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    fn screen_with(sessions: Vec<DiscoveredSession>) -> ImportScreen {
        let mut s = ImportScreen::new();
        s.sessions = sessions;
        s.now_secs = 1000;
        s
    }

    #[test]
    fn lists_sessions_with_source_and_metadata() {
        let mut s = screen_with(vec![
            session(
                SessionSource::Claude,
                "a",
                "Extract memory features",
                15_000,
            ),
            session(SessionSource::OpenCode, "b", "Fix the CI pipeline", 875),
        ]);
        let text = render_to_text(&mut s, 120, 20);
        assert!(text.contains("claude"), "source badge label");
        assert!(text.contains("opencode"), "second source label");
        assert!(text.contains("Extract memory features"), "session title");
        assert!(text.contains("~15k"), "token estimate");
        assert!(text.contains("found"), "header found-count label");
    }

    #[test]
    fn empty_state_when_nothing_found() {
        let mut s = ImportScreen::new();
        // Not discovering (default), no sessions.
        let text = render_to_text(&mut s, 100, 16);
        assert!(text.contains("No sessions found"), "empty-list hint");
    }

    #[test]
    fn imported_sessions_are_marked() {
        let sess = session(SessionSource::Zed, "z", "Some session", 2000);
        let mut s = screen_with(vec![sess.clone()]);
        s.status.insert(session_key(&sess), Status::Done);
        let text = render_to_text(&mut s, 120, 12);
        assert!(text.contains('✓'), "imported marker present");
        assert!(text.contains('1'), "header counts the import");
    }

    #[test]
    fn seed_before_discovery_still_marks_imported() {
        // Regression: the "already imported" seed can arrive before discovery
        // populates the session list. The mark must survive and apply once the
        // sessions load, not be dropped against an empty list.
        let mut s = ImportScreen::new();
        // Seed arrives first (no sessions yet).
        s.apply_event(ImportEvent::AlreadyImported("z".to_string()));
        assert!(s.status.is_empty(), "nothing to mark yet");

        // Discovery loads the session with that id.
        let sess = session(SessionSource::Zed, "z", "Some session", 2000);
        s.sessions = vec![sess.clone()];
        s.apply_imported_marks();

        assert!(
            matches!(
                s.status.get(&session_key(&sess)),
                Some(Status::AlreadyImported)
            ),
            "the seeded mark is applied once the session loads"
        );
    }

    #[test]
    fn conversation_has_role_headers() {
        let msgs = vec![
            SessionMessage {
                role: "user".into(),
                content: "hello".into(),
                timestamp: None,
            },
            SessionMessage {
                role: "assistant".into(),
                content: "hi back".into(),
                timestamp: None,
            },
        ];
        let lines = render_conversation(&msgs);
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|sp| sp.content.to_string()))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("User"));
        assert!(text.contains("Assistant"));
        assert!(text.contains("hi back"));
    }
}
