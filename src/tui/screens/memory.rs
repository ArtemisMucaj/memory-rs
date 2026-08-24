//! The Memory browser screen: a flat list of facts on the left, a detail
//! pane on the right.
//!
//! With an empty search box the list is every memory, newest first. With a
//! query the list is the RRF-fused recall hits. Selecting a fact shows its
//! statement, subject, predicate, object, and provenance on the right.

use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::connector::api::controller::{self, SearchScope};
use crate::connector::api::Container;
use crate::domain::Memory;
use crate::tui::{markdown, theme};

/// How many ranked hits a search shows.
const SEARCH_LIMIT: usize = 50;

pub struct MemoryScreen {
    /// The memories being shown, newest first (or best-match first when
    /// searching).
    memories: Vec<Memory>,
    /// Cursor over `memories`.
    selected: usize,
    /// Vertical scroll offset for the list pane.
    scroll: usize,
    /// Scroll offset for the detail pane.
    detail_scroll: u16,
    /// Search box contents; empty means browse the store.
    query: String,
    /// Whether the search box has focus (keys type into it).
    searching: bool,
    /// Total memory count for the header.
    total_memories: usize,
    error: Option<String>,
}

impl MemoryScreen {
    pub fn new() -> Self {
        Self {
            memories: Vec::new(),
            selected: 0,
            scroll: 0,
            detail_scroll: 0,
            query: String::new(),
            searching: false,
            total_memories: 0,
            error: None,
        }
    }

    /// Load (or reload) the memory list.
    pub async fn refresh(&mut self, container: &Container) {
        let result = if self.query.trim().is_empty() {
            controller::list_memories(container, None).await
        } else {
            match controller::recall_memories(
                container,
                self.query.trim(),
                None,
                &SearchScope::All,
                SEARCH_LIMIT,
            )
            .await
            {
                Ok(controller::MemorySearchOutcome::Hits(hits)) => {
                    Ok(hits.into_iter().map(|h| h.memory).collect())
                }
                Ok(controller::MemorySearchOutcome::EmptyNamespace(_)) => Ok(Vec::new()),
                Err(e) => Err(e),
            }
        };
        match result {
            Ok(memories) => {
                self.total_memories = memories.len();
                self.memories = memories;
                if self.selected >= self.memories.len() {
                    self.selected = self.memories.len().saturating_sub(1);
                }
                self.error = None;
            }
            Err(e) => {
                self.error = Some(e.to_string());
            }
        }
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

        // Search box.
        let search = Paragraph::new(self.query.as_str()).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    "Search ({} memories){}",
                    self.total_memories,
                    if self.searching { " — typing" } else { "" }
                )),
        );
        frame.render_widget(search, chunks[0]);

        let body = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        self.render_list(frame, body[0]);
        self.render_detail(frame, body[1]);
    }

    fn render_list(&mut self, frame: &mut Frame, area: Rect) {
        let visible_height = area.height.saturating_sub(2) as usize;
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + visible_height {
            self.scroll = self.selected.saturating_sub(visible_height.saturating_sub(1));
        }

        let mut lines: Vec<Line> = Vec::new();
        for (idx, memory) in self
            .memories
            .iter()
            .enumerate()
            .skip(self.scroll)
            .take(visible_height)
        {
            let is_selected = idx == self.selected;
            let marker = if is_selected { "▶" } else { " " };
            let style = if is_selected {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(
                format!("{marker} {}", truncate(&memory.statement, 80)),
                style,
            )));
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "No memories. Import a session to get started.",
                Style::default().fg(theme::MUTED),
            )));
        }
        let list = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Memories"));
        frame.render_widget(list, area);
    }

    fn render_detail(&mut self, frame: &mut Frame, area: Rect) {
        let content = if let Some(error) = &self.error {
            format!("Error: {error}")
        } else if let Some(memory) = self.memories.get(self.selected) {
            let mut s = String::new();
            s.push_str(&memory.statement);
            s.push_str("\n\n");
            s.push_str(&format!("kind: {}\n", memory.kind.as_str()));
            s.push_str(&format!("predicate: {}\n", memory.predicate.as_str()));
            s.push_str(&format!(
                "project: {}\n",
                memory.project.as_deref().unwrap_or("global")
            ));
            s.push_str(&format!(
                "source: {} (confidence {:.2})\n",
                memory.source_kind.as_str(),
                memory.confidence
            ));
            if let Some(session_id) = &memory.source_session_id {
                s.push_str(&format!("session: {session_id}\n"));
            }
            s.push_str(&format!("id: {}", memory.id));
            s
        } else {
            String::new()
        };
        let paragraph = Paragraph::new(markdown::render(&content))
            .block(Block::default().borders(Borders::ALL).title("Detail"))
            .wrap(Wrap { trim: false })
            .scroll((self.detail_scroll, 0));
        frame.render_widget(paragraph, area);
    }

    /// Whether the search box is capturing input (so global keys route here
    /// rather than quitting).
    pub fn is_searching(&self) -> bool {
        self.searching
    }

    /// The footer status line for this screen, if any.
    pub fn status_line(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// The key hints shown in the footer when this screen is active.
    pub fn footer_hint(&self) -> &'static str {
        "  /: search  j/k: move  d/u: scroll detail  r: refresh"
    }

    /// Handle a key event.
    pub async fn handle_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
        container: &Container,
    ) {
        let key = crossterm::event::KeyEvent::new(code, modifiers);
        if self.searching {
            match key.code {
                KeyCode::Esc => {
                    self.searching = false;
                    self.query.clear();
                    self.refresh(container).await;
                }
                KeyCode::Enter => {
                    self.searching = false;
                    self.refresh(container).await;
                }
                KeyCode::Backspace => {
                    self.query.pop();
                }
                KeyCode::Char(c) => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                        return;
                    }
                    self.query.push(c);
                }
                _ => {}
            }
            return;
        }

        match key.code {
            KeyCode::Char('/') => {
                self.searching = true;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                if self.selected + 1 < self.memories.len() {
                    self.selected += 1;
                    self.detail_scroll = 0;
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.selected > 0 {
                    self.selected -= 1;
                    self.detail_scroll = 0;
                }
            }
            KeyCode::Char('d') => {
                self.detail_scroll = self.detail_scroll.saturating_add(1);
            }
            KeyCode::Char('u') => {
                self.detail_scroll = self.detail_scroll.saturating_sub(1);
            }
            KeyCode::Char('r') => {
                self.refresh(container).await;
            }
            _ => {}
        }
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars.saturating_sub(3)).collect();
    format!("{truncated}...")
}
