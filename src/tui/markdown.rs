//! A minimal Markdown-to-styled-lines renderer for the terminal.
//!
//! Full Markdown looks poor when dumped as raw text into a `Paragraph`
//! (literal `##`, `**`, `-` markers). This renderer converts the common
//! constructs — ATX headings, bullet/numbered lists, blockquotes, fenced code,
//! and inline `**bold**` / `` `code` `` — into styled ratatui [`Line`]s so
//! memory content reads cleanly in the detail pane.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Render Markdown `source` into styled lines.
pub fn render(source: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_fence = false;

    for raw in source.lines() {
        let trimmed = raw.trim_start();

        // Fenced code blocks: toggle on ``` and render contents verbatim.
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            lines.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::Rgb(180, 200, 160)),
            )));
            continue;
        }

        // Headings: # … ###### → bold, brighter for higher levels.
        if let Some((level, text)) = heading(trimmed) {
            let color = match level {
                1 => Color::Cyan,
                2 => Color::LightCyan,
                _ => Color::Blue,
            };
            lines.push(Line::from(Span::styled(
                text.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        // Blockquotes: > … → dim, quote bar.
        if let Some(rest) = trimmed.strip_prefix("> ") {
            let mut spans = vec![Span::styled("▏ ", Style::default().fg(Color::DarkGray))];
            spans.extend(inline(rest, Style::default().fg(Color::Gray)));
            lines.push(Line::from(spans));
            continue;
        }

        // Bullet list items: -, *, + → • bullet.
        if let Some(rest) = bullet(trimmed) {
            let mut spans = vec![Span::styled("  • ", Style::default().fg(Color::Yellow))];
            spans.extend(inline(rest, Style::default().fg(Color::White)));
            lines.push(Line::from(spans));
            continue;
        }

        // Numbered list items: keep the number, style the marker.
        if let Some((marker, rest)) = numbered(trimmed) {
            let mut spans = vec![Span::styled(
                format!("  {marker} "),
                Style::default().fg(Color::Yellow),
            )];
            spans.extend(inline(rest, Style::default().fg(Color::White)));
            lines.push(Line::from(spans));
            continue;
        }

        // Plain paragraph line (or blank).
        if trimmed.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(inline(raw, Style::default().fg(Color::White))));
        }
    }

    lines
}

/// Parse an ATX heading, returning `(level, text)`.
fn heading(line: &str) -> Option<(usize, &str)> {
    if !line.starts_with('#') {
        return None;
    }
    let hashes = line.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = line[hashes..].strip_prefix(' ')?;
    Some((hashes, rest))
}

/// Parse a bullet list item marker, returning the item text.
fn bullet(line: &str) -> Option<&str> {
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(marker) {
            return Some(rest);
        }
    }
    None
}

/// Parse a numbered list item, returning `(marker, text)` e.g. `("1.", "…")`.
fn numbered(line: &str) -> Option<(String, &str)> {
    let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let after = &line[digits.len()..];
    let rest = after
        .strip_prefix(". ")
        .or_else(|| after.strip_prefix(") "))?;
    Some((format!("{digits}."), rest))
}

/// Render inline `**bold**` and `` `code` `` spans within a line, over `base`.
fn inline(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    let mut chars = text.chars().peekable();

    let flush = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), base));
        }
    };

    while let Some(c) = chars.next() {
        match c {
            // `code`
            '`' => {
                flush(&mut buf, &mut spans);
                let mut code = String::new();
                for cc in chars.by_ref() {
                    if cc == '`' {
                        break;
                    }
                    code.push(cc);
                }
                spans.push(Span::styled(
                    code,
                    Style::default().fg(Color::Rgb(180, 200, 160)),
                ));
            }
            // **bold**
            '*' if chars.peek() == Some(&'*') => {
                chars.next(); // consume second '*'
                flush(&mut buf, &mut spans);
                let mut bold = String::new();
                while let Some(cc) = chars.next() {
                    if cc == '*' && chars.peek() == Some(&'*') {
                        chars.next();
                        break;
                    }
                    bold.push(cc);
                }
                spans.push(Span::styled(bold, base.add_modifier(Modifier::BOLD)));
            }
            _ => buf.push(c),
        }
    }
    flush(&mut buf, &mut spans);
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_heading_and_bullets() {
        let lines = render("# Title\n\n- one\n- two");
        assert_eq!(lines.len(), 4);
        // Heading text is stripped of '#'.
        assert_eq!(lines[0].spans[0].content, "Title");
        // Bullet marker replaced with •.
        assert!(lines[2].spans[0].content.contains('•'));
    }

    #[test]
    fn parses_numbered_items() {
        assert_eq!(numbered("1. hi"), Some(("1.".to_string(), "hi")));
        assert_eq!(numbered("12) yo"), Some(("12.".to_string(), "yo")));
        assert_eq!(numbered("nope"), None);
    }

    #[test]
    fn inline_bold_and_code() {
        let spans = inline("a **b** `c`", Style::default());
        // "a ", "b", " ", "c"
        assert!(spans.iter().any(|s| s.content == "b"));
        assert!(spans.iter().any(|s| s.content == "c"));
    }
}
