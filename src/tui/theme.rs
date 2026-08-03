//! Shared palette and text formatters for the TUI.
//!
//! Colours and the small `relative_time` / `fmt_tokens` / `truncate` helpers
//! live here so both screens render with one consistent visual language,
//! matching the companion macOS app: per-source badges, a soft accent, and a
//! dim metadata grey.

use ratatui::style::Color;

/// Primary accent — used for the active tab, focused borders, and headings.
pub const ACCENT: Color = Color::Cyan;

/// Selection background for the highlighted row.
pub const SELECTION_BG: Color = Color::Rgb(38, 40, 46);

/// Dim grey for metadata (timestamps, counts, hints).
pub const MUTED: Color = Color::DarkGray;

/// Colour for something the store is knowingly unsure about — currently a live
/// contradiction between two memories. Deliberately the only warm colour in the
/// palette, so an unresolved disagreement reads as one at a glance.
pub const WARN: Color = Color::Yellow;

/// Colour for a session's source badge — matching the app:
/// claude = magenta, opencode = green, zed (and anything else) = blue.
pub fn source_color(source: &str) -> Color {
    match source {
        "claude" => Color::Magenta,
        "opencode" => Color::Green,
        _ => Color::Blue,
    }
}

/// Accent colour for a memory kind label (`preference`, `skill`, …).
pub fn kind_color(kind: &str) -> Color {
    match kind {
        "preference" => Color::Blue,
        "experience" => Color::Magenta,
        "skill" => Color::Green,
        "fact" => Color::Yellow,
        _ => Color::Cyan,
    }
}

/// A compact "N ago" label from two Unix timestamps.
pub fn relative_time(then_secs: i64, now_secs: i64) -> String {
    let d = (now_secs - then_secs).max(0);
    if d < 60 {
        "just now".to_string()
    } else if d < 3600 {
        format!("{}m ago", d / 60)
    } else if d < 86400 {
        format!("{}h ago", d / 3600)
    } else if d < 86400 * 30 {
        format!("{}d ago", d / 86400)
    } else if d < 86400 * 365 {
        format!("{}mo ago", d / (86400 * 30))
    } else {
        format!("{}y ago", d / (86400 * 365))
    }
}

/// Compact token estimate: `~450`, `~1.2k`, `~48k`, `~1.5M`. Empty for zero.
pub fn fmt_tokens(tokens: usize) -> String {
    match tokens {
        0 => String::new(),
        n if n < 1_000 => format!("~{n}"),
        n if n < 1_000_000 => {
            if n < 10_000 {
                format!("~{:.1}k", n as f64 / 1_000.0)
            } else {
                format!("~{}k", n / 1_000)
            }
        }
        n => format!("~{:.1}M", n as f64 / 1_000_000.0),
    }
}

/// Truncate to `max` characters, appending `…` when shortened.
pub fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_time_buckets() {
        let now = 1_000_000;
        assert_eq!(relative_time(now - 30, now), "just now");
        assert_eq!(relative_time(now - 120, now), "2m ago");
        assert_eq!(relative_time(now - 7200, now), "2h ago");
        assert_eq!(relative_time(now - 86400 * 3, now), "3d ago");
    }

    #[test]
    fn fmt_tokens_buckets() {
        assert_eq!(fmt_tokens(0), "");
        assert_eq!(fmt_tokens(450), "~450");
        assert_eq!(fmt_tokens(1_200), "~1.2k");
        assert_eq!(fmt_tokens(48_000), "~48k");
        assert_eq!(fmt_tokens(1_500_000), "~1.5M");
    }

    #[test]
    fn truncate_adds_ellipsis() {
        assert_eq!(truncate("hello", 10), "hello");
        assert!(truncate("hello world", 5).ends_with('…'));
    }
}
