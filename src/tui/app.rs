//! The two-screen TUI shell.
//!
//! A slim top tab bar switches between the **Memory** browser and the
//! **Import** picker; each screen owns its own state and rendering. The app
//! holds the [`Container`] so screens can load data (the grouped memory tree,
//! discovered sessions) and run actions (import) against the real store.
//!
//! Rendering is synchronous (ratatui); data loading is async. Screens load
//! eagerly on entry / refresh and cache the result in their state, so the draw
//! path never awaits.

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::connector::api::Container;
use crate::domain::DomainError;
use crate::tui::screens::{ImportScreen, MemoryScreen};
use crate::tui::theme;

/// How long the event loop waits for a key before redrawing, so streamed
/// discovery and background import progress stay live without a keypress.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Which screen is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Memory,
    Import,
}

impl Tab {
    fn title(self) -> &'static str {
        match self {
            Tab::Memory => "Memory",
            Tab::Import => "Import",
        }
    }

    const ALL: [Tab; 2] = [Tab::Memory, Tab::Import];
}

/// The running TUI application.
pub struct App {
    container: Container,
    tab: Tab,
    memory: MemoryScreen,
    import: ImportScreen,
    should_quit: bool,
}

impl App {
    pub fn new(container: Container) -> Self {
        Self {
            container,
            tab: Tab::Memory,
            memory: MemoryScreen::new(),
            import: ImportScreen::new(),
            should_quit: false,
        }
    }

    /// Set up the terminal, run the loop, and always restore the terminal —
    /// even on error — so a panic or `?` never leaves the user in raw mode.
    pub async fn run(mut self) -> Result<(), DomainError> {
        let mut terminal = ratatui::init();
        // Load the initial screen's data before the first paint.
        self.memory.refresh(&self.container).await;
        self.import.start_discovery(&self.container);
        let result = self.run_loop(&mut terminal).await;
        ratatui::restore();
        result
    }

    async fn run_loop(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
    ) -> Result<(), DomainError> {
        while !self.should_quit {
            // Let the import screen ingest any streamed sessions / import events.
            self.import.pump();

            terminal
                .draw(|frame| self.render(frame))
                .map_err(|e| DomainError::internal(format!("draw failed: {e}")))?;

            if !event::poll(POLL_INTERVAL)
                .map_err(|e| DomainError::internal(format!("event poll failed: {e}")))?
            {
                continue;
            }
            let Event::Key(key) = event::read()
                .map_err(|e| DomainError::internal(format!("event read failed: {e}")))?
            else {
                continue;
            };
            if key.kind == KeyEventKind::Release {
                continue;
            }

            // Global keys first (quit, tab switch); then screen-local keys.
            match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.should_quit = true;
                }
                KeyCode::Char('q') if !self.memory.is_searching() || self.tab != Tab::Memory => {
                    // `q` quits unless the Memory search box is capturing input.
                    self.should_quit = true;
                }
                KeyCode::Tab => self.switch_tab().await,
                KeyCode::Char('1') if !self.capturing_text() => self.set_tab(Tab::Memory).await,
                KeyCode::Char('2') if !self.capturing_text() => self.set_tab(Tab::Import).await,
                _ => self.handle_screen_key(key.code, key.modifiers).await,
            }
        }
        Ok(())
    }

    /// Whether the active screen is currently capturing free text (so number
    /// keys type into a field rather than switching tabs).
    fn capturing_text(&self) -> bool {
        self.tab == Tab::Memory && self.memory.is_searching()
    }

    async fn switch_tab(&mut self) {
        let next = match self.tab {
            Tab::Memory => Tab::Import,
            Tab::Import => Tab::Memory,
        };
        self.set_tab(next).await;
    }

    async fn set_tab(&mut self, tab: Tab) {
        if self.tab == tab {
            return;
        }
        self.tab = tab;
        // Refresh the memory tree when returning to it, so newly-imported items
        // appear without a manual reload.
        if tab == Tab::Memory {
            self.memory.refresh(&self.container).await;
        }
    }

    async fn handle_screen_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        match self.tab {
            Tab::Memory => {
                self.memory
                    .handle_key(code, modifiers, &self.container)
                    .await
            }
            Tab::Import => {
                self.import
                    .handle_key(code, modifiers, &self.container)
                    .await
            }
        }
    }

    fn render(&mut self, frame: &mut Frame) {
        let rows = Layout::vertical([
            Constraint::Length(1), // tab bar
            Constraint::Min(0),    // active screen
            Constraint::Length(1), // footer hints
        ])
        .split(frame.area());

        self.render_tab_bar(frame, rows[0]);
        match self.tab {
            Tab::Memory => self.memory.render(frame, rows[1]),
            Tab::Import => self.import.render(frame, rows[1]),
        }
        self.render_footer(frame, rows[2]);
    }

    fn render_tab_bar(&self, frame: &mut Frame, area: Rect) {
        let mut spans = vec![Span::styled(
            " memory-rs ",
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )];
        for tab in Tab::ALL {
            let active = tab == self.tab;
            let style = if active {
                Style::default()
                    .fg(ratatui::style::Color::Black)
                    .bg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::MUTED)
            };
            spans.push(Span::raw(" "));
            spans.push(Span::styled(format!(" {} ", tab.title()), style));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        // A screen may surface a transient status here (an import result, or a
        // load error) instead of the key hints. Errors show in red, other
        // statuses in the accent colour; either is truncated to one line.
        let status = match self.tab {
            Tab::Memory => self.memory.status_line().map(|s| (s, true)),
            Tab::Import => self.import.status_line().map(|s| (s, false)),
        };
        if let Some((status, is_error)) = status {
            let color = if is_error {
                ratatui::style::Color::Red
            } else {
                theme::ACCENT
            };
            let text = one_line(status, area.width.saturating_sub(2) as usize);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    format!("  {text}"),
                    Style::default().fg(color),
                ))),
                area,
            );
            return;
        }

        let hint = match self.tab {
            Tab::Memory => self.memory.footer_hint(),
            Tab::Import => self.import.footer_hint(),
        };
        let global = "  Tab: switch  q: quit";
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(hint, Style::default().fg(theme::MUTED)),
                Span::styled(global, Style::default().fg(theme::MUTED)),
            ])),
            area,
        );
    }
}

/// Collapse whitespace (newlines included) to a single line and truncate to
/// `max` display columns, so a multi-line error fits the one-row footer.
fn one_line(text: &str, max: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    theme::truncate(&flat, max)
}

/// Launch the TUI against `container`, blocking until the user quits.
pub async fn run(container: Container) -> Result<(), DomainError> {
    // Guard: a terminal is required.
    if !io::IsTerminal::is_terminal(&io::stdout()) {
        return Err(DomainError::invalid_input(
            "the `tui` command needs an interactive terminal (stdout is not a TTY)",
        ));
    }
    App::new(container).run().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::api::ContainerConfig;
    use crate::domain::{MemoryItem, MemoryKind};
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_to_text(app: &mut App, w: u16, h: u16) -> String {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.render(f)).unwrap();
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

    fn temp_container(dir: &std::path::Path) -> Container {
        Container::new(ContainerConfig {
            data_dir: dir.to_str().unwrap().to_string(),
            embedding_dimensions: 4,
            openai_endpoint: None,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn tab_bar_shows_both_screens_memory_active() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(temp_container(dir.path()));
        let text = render_to_text(&mut app, 120, 24);
        assert!(text.contains("memory-rs"), "app title");
        assert!(text.contains("Memory"), "Memory tab");
        assert!(text.contains("Import"), "Import tab");
    }

    #[tokio::test]
    async fn memory_tab_renders_grouped_tree_from_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let container = temp_container(dir.path());
        // Seed a couple of items through the real repository.
        let repo = container.memory_repository().unwrap();
        for name in ["duckdb_locks", "storage_engine"] {
            let item = MemoryItem::new(
                name.into(),
                MemoryKind::Fact,
                name.into(),
                "content".into(),
                None,
                None,
                0,
                0,
                0,
            );
            repo.upsert_item(&item, None).await.unwrap();
        }

        let mut app = App::new(container);
        app.memory.refresh(&app.container).await;
        let text = render_to_text(&mut app, 120, 24);
        assert!(text.contains("Memories"), "top group renders");
        assert!(text.contains("Facts"), "category subgroup renders");
        assert!(text.contains("duckdb_locks"), "leaf item renders");
    }

    #[tokio::test]
    async fn tab_switches_to_import() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(temp_container(dir.path()));
        app.set_tab(Tab::Import).await;
        let text = render_to_text(&mut app, 120, 24);
        // Import header shows the found/imported counters.
        assert!(text.contains("found"), "import header rendered");
        assert!(text.contains("imported"), "import header rendered");
    }

    #[tokio::test]
    async fn db_open_error_goes_to_the_footer_not_the_tree() {
        // Create a store pinned at 4 dims, then open the SAME data dir at 8 dims:
        // the container's DuckDB open fails with a dimension-mismatch error.
        let dir = tempfile::tempdir().unwrap();
        {
            let seed = temp_container(dir.path()); // 4 dims (see helper)
            seed.memory_repository().unwrap(); // creates memory.duckdb at dim 4
        }
        let mismatched = Container::new(ContainerConfig {
            data_dir: dir.path().to_str().unwrap().to_string(),
            embedding_dimensions: 8, // ← different width → open error
            openai_endpoint: None,
        })
        .unwrap();

        let mut app = App::new(mismatched);
        app.memory.refresh(&app.container).await; // sets the error
        let h = 24;
        let text = render_to_text(&mut app, 120, h);
        let rows: Vec<&str> = text.lines().collect();
        // The last row is the footer; everything above the footer is the
        // tab-bar + the two panes (the "tree region").
        let footer = rows.last().copied().unwrap_or_default();
        let body = rows[..rows.len().saturating_sub(1)].join("\n");

        // The tree/body shows the calm placeholder, not the raw error.
        assert!(
            body.contains("Memory unavailable"),
            "calm pane placeholder in the body"
        );
        assert!(
            !body.contains("1536") && !body.contains("dimensions"),
            "the raw mismatch error must NOT appear in the body:\n{body}"
        );
        // The footer surfaces the flattened error.
        assert!(
            footer.contains("dimensions") || footer.contains("768"),
            "the error is on the footer status line: {footer:?}"
        );
    }
}
