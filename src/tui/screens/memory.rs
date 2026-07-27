//! The Memory browser screen: a grouped, collapsible hierarchy on the left and
//! a detail pane on the right.
//!
//! The tree mirrors the companion app: `Memories` with per-kind category
//! subgroups (Preferences / Skills / Facts / Experiences), then `Projects`,
//! then `Sessions` — each group header carrying a count and a chevron. Leaves
//! are memory items and virtual-filesystem nodes; selecting one shows its
//! L0/L1/L2 detail on the right. A non-empty search box replaces the tree with
//! a ranked flat hit list.

use std::collections::HashSet;

use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::application::{MemoryRow, RowTarget};
use crate::connector::api::Container;
use crate::domain::NodeKind;
use crate::tui::{markdown, theme};

/// How many ranked hits a search shows.
const SEARCH_LIMIT: usize = 50;

pub struct MemoryScreen {
    /// The full grouped tree (or ranked hits when searching), as returned by
    /// the use case — unfiltered by collapse state.
    rows: Vec<MemoryRow>,
    /// Group keys the user has collapsed. Groups default to expanded, so an
    /// absent key means "expanded".
    collapsed: HashSet<String>,
    /// Cursor over the *visible* rows (after collapse filtering).
    selected: usize,
    /// Vertical scroll offset for the tree pane.
    scroll: usize,
    /// Scroll offset for the detail pane.
    detail_scroll: u16,
    /// Search box contents; empty means browse the tree.
    query: String,
    /// Whether the search box has focus (keys type into it).
    searching: bool,
    /// Total item / session counts for the header, from the last refresh.
    total_items: usize,
    total_sessions: usize,
    error: Option<String>,
}

impl MemoryScreen {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            collapsed: HashSet::new(),
            selected: 0,
            scroll: 0,
            detail_scroll: 0,
            query: String::new(),
            searching: false,
            total_items: 0,
            total_sessions: 0,
            error: None,
        }
    }

    pub fn is_searching(&self) -> bool {
        self.searching
    }

    /// Reload the tree (or search results) from the store.
    pub async fn refresh(&mut self, container: &Container) {
        match container.memory_browse_use_case() {
            Ok(use_case) => match use_case.grouped_tree(&self.query, SEARCH_LIMIT).await {
                Ok(rows) => {
                    self.rows = rows;
                    self.error = None;
                    self.recount();
                    self.clamp_selection();
                }
                Err(e) => self.error = Some(e.to_string()),
            },
            Err(e) => self.error = Some(e.to_string()),
        }
    }

    fn recount(&mut self) {
        self.total_items = self
            .rows
            .iter()
            .filter(|r| matches!(r.target, RowTarget::Item(_)))
            .count();
        self.total_sessions = self
            .rows
            .iter()
            .filter(|r| matches!(&r.target, RowTarget::Node(n) if n.kind() == NodeKind::Session))
            .count();
    }

    // ── Visible-row computation (collapse handling) ──────────────────────────

    /// Indices into `self.rows` that are currently visible: a row is hidden
    /// when any ancestor group (a shallower `Group` row above it) is collapsed.
    fn visible_indices(&self) -> Vec<usize> {
        let mut visible = Vec::new();
        // `hide_below_depth`: once a group at depth D collapses, every following
        // row deeper than D is hidden until a row at depth ≤ D appears.
        let mut hide_below_depth: Option<u8> = None;
        for (i, row) in self.rows.iter().enumerate() {
            if let Some(limit) = hide_below_depth {
                if row.depth > limit {
                    continue;
                }
                hide_below_depth = None;
            }
            visible.push(i);
            if let RowTarget::Group { key, .. } = &row.target {
                if self.collapsed.contains(key) {
                    hide_below_depth = Some(row.depth);
                }
            }
        }
        visible
    }

    fn clamp_selection(&mut self) {
        let visible = self.visible_indices();
        if visible.is_empty() {
            self.selected = 0;
        } else if self.selected >= visible.len() {
            self.selected = visible.len() - 1;
        }
    }

    /// The `self.rows` index currently under the cursor, if any.
    fn selected_row_index(&self) -> Option<usize> {
        self.visible_indices().get(self.selected).copied()
    }

    fn selected_row(&self) -> Option<&MemoryRow> {
        self.selected_row_index().and_then(|i| self.rows.get(i))
    }

    // ── Input ────────────────────────────────────────────────────────────────

    pub async fn handle_key(
        &mut self,
        code: KeyCode,
        _modifiers: KeyModifiers,
        container: &Container,
    ) {
        if self.searching {
            self.handle_search_key(code, container).await;
            return;
        }
        match code {
            KeyCode::Char('/') => {
                self.searching = true;
            }
            KeyCode::Up | KeyCode::Char('k') => self.move_cursor(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_cursor(1),
            KeyCode::PageUp => self.detail_scroll = self.detail_scroll.saturating_sub(8),
            KeyCode::PageDown => self.detail_scroll = self.detail_scroll.saturating_add(8),
            KeyCode::Enter | KeyCode::Char(' ') => self.toggle_selected_group(),
            KeyCode::Left => self.collapse_selected(),
            KeyCode::Right => self.expand_selected(),
            _ => {}
        }
    }

    async fn handle_search_key(&mut self, code: KeyCode, container: &Container) {
        match code {
            KeyCode::Esc => {
                self.searching = false;
                if !self.query.is_empty() {
                    self.query.clear();
                    self.refresh(container).await;
                }
            }
            KeyCode::Enter => {
                self.searching = false;
            }
            KeyCode::Backspace => {
                self.query.pop();
                self.selected = 0;
                self.refresh(container).await;
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                self.selected = 0;
                self.refresh(container).await;
            }
            _ => {}
        }
    }

    fn move_cursor(&mut self, delta: i32) {
        let len = self.visible_indices().len();
        if len == 0 {
            return;
        }
        let next = (self.selected as i32 + delta).clamp(0, len as i32 - 1) as usize;
        if next != self.selected {
            self.selected = next;
            self.detail_scroll = 0;
        }
    }

    fn toggle_selected_group(&mut self) {
        if let Some(RowTarget::Group { key, .. }) = self.selected_row().map(|r| &r.target) {
            let key = key.clone();
            if !self.collapsed.remove(&key) {
                self.collapsed.insert(key);
            }
            self.clamp_selection();
        }
    }

    fn collapse_selected(&mut self) {
        if let Some(RowTarget::Group { key, .. }) = self.selected_row().map(|r| &r.target) {
            self.collapsed.insert(key.clone());
            self.clamp_selection();
        }
    }

    fn expand_selected(&mut self) {
        if let Some(RowTarget::Group { key, .. }) = self.selected_row().map(|r| &r.target) {
            self.collapsed.remove(&key.clone());
        }
    }

    pub fn footer_hint(&self) -> &'static str {
        if self.searching {
            "  type to search  Enter: keep  Esc: clear"
        } else {
            "  ↑↓: move  Enter: fold  /: search"
        }
    }

    // ── Rendering ────────────────────────────────────────────────────────────

    pub fn render(&mut self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([
            Constraint::Length(3), // search + counts header
            Constraint::Min(0),    // tree | detail
        ])
        .split(area);
        self.render_header(frame, rows[0]);

        let panes = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)])
            .split(rows[1]);
        self.render_tree(frame, panes[0]);
        self.render_detail(frame, panes[1]);
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if self.searching {
                theme::ACCENT
            } else {
                theme::MUTED
            }));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(24)]).split(inner);

        // Search field.
        let search = if self.query.is_empty() && !self.searching {
            Span::styled("  Search memories…", Style::default().fg(theme::MUTED))
        } else {
            let caret = if self.searching { "▎" } else { "" };
            Span::styled(
                format!("  {}{caret}", self.query),
                Style::default().fg(ratatui::style::Color::White),
            )
        };
        frame.render_widget(Paragraph::new(Line::from(search)), cols[0]);

        // Counts, right-aligned.
        let counts = Line::from(vec![
            Span::styled(
                format!("{} ", self.total_items),
                Style::default()
                    .fg(ratatui::style::Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("items   ", Style::default().fg(theme::MUTED)),
            Span::styled(
                format!("{} ", self.total_sessions),
                Style::default()
                    .fg(ratatui::style::Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("sessions", Style::default().fg(theme::MUTED)),
        ]);
        frame.render_widget(
            Paragraph::new(counts).alignment(ratatui::layout::Alignment::Right),
            cols[1],
        );
    }

    fn render_tree(&mut self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if let Some(err) = &self.error {
            frame.render_widget(
                Paragraph::new(format!("  Error: {err}"))
                    .style(Style::default().fg(ratatui::style::Color::Red))
                    .wrap(Wrap { trim: false }),
                inner,
            );
            return;
        }

        let visible = self.visible_indices();
        if visible.is_empty() {
            let msg = if self.query.is_empty() {
                "  No memories yet. Import a session to get started."
            } else {
                "  No matches."
            };
            frame.render_widget(
                Paragraph::new(msg).style(Style::default().fg(theme::MUTED)),
                inner,
            );
            return;
        }

        // Keep the cursor within the scroll window.
        let height = inner.height as usize;
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if height > 0 && self.selected >= self.scroll + height {
            self.scroll = self.selected + 1 - height;
        }

        let searching = !self.query.is_empty();
        let lines: Vec<Line> = visible
            .iter()
            .enumerate()
            .skip(self.scroll)
            .take(height)
            .map(|(vis_i, &row_i)| {
                self.row_line(&self.rows[row_i], vis_i == self.selected, searching)
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }

    /// Render one tree row: indent + chevron/glyph + label (+ count / score).
    fn row_line(&self, row: &MemoryRow, selected: bool, searching: bool) -> Line<'static> {
        let indent = "  ".repeat(row.depth as usize);
        let bg = if selected {
            theme::SELECTION_BG
        } else {
            ratatui::style::Color::Reset
        };

        let mut spans = vec![Span::styled(indent, Style::default().bg(bg))];

        match &row.target {
            RowTarget::Group { key, count } => {
                let chevron = if self.collapsed.contains(key) {
                    "▸ "
                } else {
                    "▾ "
                };
                spans.push(Span::styled(
                    chevron,
                    Style::default().fg(theme::MUTED).bg(bg),
                ));
                spans.push(Span::styled(
                    row.label.clone(),
                    Style::default()
                        .fg(ratatui::style::Color::White)
                        .bg(bg)
                        .add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled(
                    format!("  {count}"),
                    Style::default().fg(theme::MUTED).bg(bg),
                ));
            }
            RowTarget::Item(item) => {
                spans.push(Span::styled(
                    "• ",
                    Style::default()
                        .fg(theme::kind_color(item.kind().as_str()))
                        .bg(bg),
                ));
                spans.push(Span::styled(row.label.clone(), label_style(selected, bg)));
            }
            RowTarget::Node(node) => {
                let glyph = if node.kind() == NodeKind::Session {
                    "◆ "
                } else {
                    "★ "
                };
                spans.push(Span::styled(
                    glyph,
                    Style::default().fg(theme::ACCENT).bg(bg),
                ));
                spans.push(Span::styled(
                    theme::truncate(&row.label, 60),
                    label_style(selected, bg),
                ));
            }
            RowTarget::Directory | RowTarget::NodeLevel { .. } => {
                spans.push(Span::styled(row.label.clone(), label_style(selected, bg)));
            }
        }

        if searching {
            if let Some(score) = row.score {
                spans.push(Span::styled(
                    format!("  {score:.2}"),
                    Style::default().fg(theme::MUTED).bg(bg),
                ));
            }
        }
        Line::from(spans)
    }

    fn render_detail(&self, frame: &mut Frame, area: Rect) {
        let selected = if self.error.is_some() {
            None
        } else {
            self.selected_row()
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED));
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let Some(row) = selected else {
            self.render_empty_detail(frame, inner);
            return;
        };

        let body = detail_body(row);
        frame.render_widget(
            Paragraph::new(body)
                .wrap(Wrap { trim: false })
                .scroll((self.detail_scroll, 0)),
            inner,
        );
    }

    fn render_empty_detail(&self, frame: &mut Frame, area: Rect) {
        let lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No memory selected",
                Style::default()
                    .fg(theme::MUTED)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "  Select a row to read it here.",
                Style::default().fg(theme::MUTED),
            )),
        ];
        frame.render_widget(Paragraph::new(lines), area);
    }
}

fn label_style(selected: bool, bg: ratatui::style::Color) -> Style {
    let base = Style::default().fg(ratatui::style::Color::Gray).bg(bg);
    if selected {
        base.fg(ratatui::style::Color::White)
            .add_modifier(Modifier::BOLD)
    } else {
        base
    }
}

/// Build the styled detail for the selected row. Items show metadata + content;
/// nodes show their L0/L1 and, when present, L2; groups/directories show a hint.
fn detail_body(row: &MemoryRow) -> Vec<Line<'static>> {
    match &row.target {
        RowTarget::Group { count, .. } => vec![
            section_header(&row.label),
            Line::from(""),
            meta_line(&format!("{count} item(s). Select a child to read it.")),
        ],
        RowTarget::Directory => vec![meta_line("Directory — select a child to view it.")],
        RowTarget::Item(item) => {
            let mut lines = vec![
                Line::from(Span::styled(
                    format!("[{}] {}", item.kind(), item.name()),
                    Style::default()
                        .fg(theme::kind_color(item.kind().as_str()))
                        .add_modifier(Modifier::BOLD),
                )),
                meta_line(&format!(
                    "updated {}×  ·  source: {}",
                    item.update_count(),
                    item.source_session_id().unwrap_or("(unknown)")
                )),
                Line::from(""),
            ];
            lines.extend(markdown::render(item.content()));
            lines
        }
        RowTarget::NodeLevel { node, .. } => {
            let mut lines = vec![section_header(node.uri()), Line::from("")];
            lines.extend(markdown::render(node.abstract_()));
            lines
        }
        RowTarget::Node(node) => {
            let mut lines = vec![
                Line::from(Span::styled(
                    node.uri().to_string(),
                    Style::default()
                        .fg(theme::ACCENT)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                section_header("L0 · Abstract"),
            ];
            lines.extend(markdown::render(node.abstract_()));
            if !node.overview().trim().is_empty() {
                lines.push(Line::from(""));
                lines.push(section_header("L1 · Overview"));
                lines.extend(markdown::render(node.overview()));
            }
            let has_content = node.kind() != NodeKind::Project && !node.content().trim().is_empty();
            if has_content {
                lines.push(Line::from(""));
                lines.push(section_header("L2 · Detail"));
                lines.extend(markdown::render(node.content()));
            }
            lines
        }
    }
}

fn section_header(label: &str) -> Line<'static> {
    Line::from(Span::styled(
        format!("▍ {label}"),
        Style::default()
            .fg(theme::ACCENT)
            .add_modifier(Modifier::BOLD),
    ))
}

fn meta_line(text: &str) -> Line<'static> {
    Line::from(Span::styled(
        text.to_string(),
        Style::default().fg(theme::MUTED),
    ))
}

impl Default for MemoryScreen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{MemoryItem, MemoryKind};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn group(key: &str, label: &str, count: usize, depth: u8) -> MemoryRow {
        MemoryRow {
            depth,
            kind_label: String::new(),
            label: label.into(),
            preview: None,
            score: None,
            target: RowTarget::Group {
                key: key.into(),
                count,
            },
        }
    }

    fn item_row_test(name: &str, depth: u8) -> MemoryRow {
        MemoryRow {
            depth,
            kind_label: "fact".into(),
            label: name.into(),
            preview: None,
            score: None,
            target: RowTarget::Item(MemoryItem::new(
                name.into(),
                MemoryKind::Fact,
                name.into(),
                "content body".into(),
                None,
                None,
                0,
                0,
                0,
            )),
        }
    }

    fn render_to_text(screen: &mut MemoryScreen, w: u16, h: u16) -> String {
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

    fn screen_with(rows: Vec<MemoryRow>) -> MemoryScreen {
        let mut s = MemoryScreen::new();
        s.rows = rows;
        s.recount();
        s
    }

    #[test]
    fn renders_groups_with_counts_and_chevrons() {
        let mut s = screen_with(vec![
            group("memories", "Memories", 2, 0),
            group("memories/fact", "Facts", 2, 1),
            item_row_test("duckdb_locks", 2),
            item_row_test("storage_engine", 2),
        ]);
        let text = render_to_text(&mut s, 100, 20);
        assert!(text.contains("Memories"), "top group header");
        assert!(text.contains("Facts"), "category subgroup");
        assert!(text.contains("duckdb_locks"), "leaf item");
        assert!(text.contains('▾'), "expanded chevron");
    }

    #[test]
    fn collapsing_a_group_hides_its_children() {
        let mut s = screen_with(vec![
            group("memories", "Memories", 1, 0),
            group("memories/fact", "Facts", 1, 1),
            item_row_test("duckdb_locks", 2),
        ]);
        // Cursor on the top "Memories" group, collapse it.
        s.selected = 0;
        s.collapse_selected();
        let text = render_to_text(&mut s, 100, 20);
        assert!(text.contains("Memories"), "collapsed group still shown");
        assert!(text.contains('▸'), "collapsed chevron");
        assert!(
            !text.contains("duckdb_locks"),
            "children hidden when the ancestor group is collapsed"
        );
    }

    #[test]
    fn empty_store_shows_no_memory_selected() {
        let mut s = MemoryScreen::new();
        let text = render_to_text(&mut s, 100, 20);
        assert!(text.contains("No memories yet"), "empty tree hint");
        assert!(text.contains("No memory selected"), "empty detail state");
    }

    #[test]
    fn selecting_an_item_shows_its_content() {
        let mut s = screen_with(vec![
            group("memories", "Memories", 1, 0),
            item_row_test("duckdb_locks", 1),
        ]);
        s.selected = 1; // the item
        let text = render_to_text(&mut s, 100, 20);
        assert!(
            text.contains("content body"),
            "item content in the detail pane"
        );
    }
}
