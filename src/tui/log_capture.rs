//! In-memory capture of `tracing` log lines for display inside the TUI.
//!
//! While the TUI owns the terminal, logs must not be written to stderr — they
//! would print on top of the ratatui render and corrupt it. Instead the `tui`
//! command installs a subscriber whose writer is a [`LogCapture`]; the app
//! drains it each tick and surfaces the most recent warning on the footer.

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

/// How many recent log lines to retain. Only the latest is shown, but a small
/// ring avoids unbounded growth over a long TUI session.
const MAX_LINES: usize = 64;

/// A shared, bounded buffer of recent log lines. Cloneable: the writer side is
/// handed to the subscriber, the reader side to the app.
#[derive(Clone, Default)]
pub struct LogCapture {
    lines: Arc<Mutex<Vec<String>>>,
}

impl LogCapture {
    pub fn new() -> Self {
        Self::default()
    }

    /// The most recent captured log line, trimmed, if any.
    pub fn latest(&self) -> Option<String> {
        let lines = self.lines.lock().ok()?;
        lines.last().map(|l| l.trim().to_string())
    }

    fn push(&self, line: String) {
        if let Ok(mut lines) = self.lines.lock() {
            lines.push(line);
            let overflow = lines.len().saturating_sub(MAX_LINES);
            if overflow > 0 {
                lines.drain(0..overflow);
            }
        }
    }
}

/// The `Write` end handed to the subscriber. Buffers bytes until a newline,
/// then commits each complete line to the shared [`LogCapture`].
pub struct LogWriter {
    capture: LogCapture,
    buf: Vec<u8>,
}

impl Write for LogWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(data);
        // Split out complete lines, keeping any trailing partial in `buf`.
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let text = String::from_utf8_lossy(&line).trim_end().to_string();
            if !text.is_empty() {
                self.capture.push(text);
            }
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for LogCapture {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogWriter {
            capture: self.clone(),
            buf: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_complete_lines() {
        let capture = LogCapture::new();
        let mut w = capture.make_writer();
        w.write_all(b"first line\nsecond ").unwrap();
        // Only the completed line is captured; the partial waits for its newline.
        assert_eq!(capture.latest().as_deref(), Some("first line"));
        w.write_all(b"line\n").unwrap();
        assert_eq!(capture.latest().as_deref(), Some("second line"));
    }

    #[test]
    fn ring_is_bounded() {
        let capture = LogCapture::new();
        let mut w = capture.make_writer();
        for i in 0..(MAX_LINES + 10) {
            w.write_all(format!("line {i}\n").as_bytes()).unwrap();
        }
        assert_eq!(capture.latest().as_deref(), Some("line 73"));
        assert!(capture.lines.lock().unwrap().len() <= MAX_LINES);
    }
}
