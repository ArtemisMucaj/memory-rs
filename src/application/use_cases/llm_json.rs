//! Small primitives shared by the LLM-driven use cases.
//!
//! Chat models do not reliably return the JSON they were asked for. They wrap
//! it in prose or a markdown fence, and small local models emit escapes that
//! are valid Markdown but illegal JSON (`\_`, `\(`) plus raw newlines inside
//! string literals — all of which strict `serde_json` rejects outright.
//! [`extract_json_object`] and [`repair_json_string_escapes`] are the
//! tolerant recovery pass every one of those call sites needs; defining them
//! once means a model quirk gets fixed in one place instead of in each
//! parser separately.
//!
//! [`unix_now`] is not JSON and does not belong here on merit. It is here
//! because every use case needs a clock and four lines do not justify a
//! module of their own.

/// Repair invalid backslash escapes inside JSON string literals.
///
/// Small local models frequently emit markdown content with escapes that are
/// valid in Markdown but invalid in JSON — `\_`, `\(`, `\<`, a trailing `\`,
/// or raw control characters (a literal newline/tab) inside a string. Strict
/// `serde_json` rejects all of these. This walks the text tracking string
/// context and, inside strings, passes valid JSON escapes through untouched
/// while escaping anything else so the result parses. Text outside strings is
/// left exactly as-is.
pub(crate) fn repair_json_string_escapes(json: &str) -> String {
    let mut out = String::with_capacity(json.len() + json.len() / 16);
    let mut in_string = false;
    let mut chars = json.chars().peekable();
    while let Some(ch) = chars.next() {
        if !in_string {
            if ch == '"' {
                in_string = true;
            }
            out.push(ch);
            continue;
        }
        match ch {
            '"' => {
                in_string = false;
                out.push(ch);
            }
            '\\' => match chars.peek() {
                // Valid JSON escape — copy the pair through verbatim.
                Some(&next @ ('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')) => {
                    out.push('\\');
                    out.push(next);
                    chars.next();
                }
                // Invalid escape (`\_`, `\(`, …) or a trailing backslash:
                // escape the backslash itself so it becomes a literal.
                _ => out.push_str("\\\\"),
            },
            // Raw control characters are illegal inside a JSON string; escape
            // the common ones and drop anything else unrepresentable.
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out
}

/// Extract the first balanced `{ ... }` object from mixed model output.
pub(crate) fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..start + offset + ch.len_utf8()]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Current Unix time in seconds.
pub(crate) fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_leaves_valid_escapes_untouched() {
        let valid = r#"{"a": "tab\there \"quoted\" and \\ slash and \n newline"}"#;
        assert_eq!(repair_json_string_escapes(valid), valid);
    }

    #[test]
    fn repair_rescues_markdown_escapes() {
        // `\_` and `\(` are valid Markdown but invalid JSON escapes — exactly
        // what small local models emit when they quote prose back at you.
        let broken = r#"{"content": "use my\_var and call foo\(bar\)"}"#;
        let repaired = repair_json_string_escapes(broken);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["content"], "use my\\_var and call foo\\(bar\\)");
    }

    #[test]
    fn repair_escapes_raw_control_characters() {
        // A literal newline or tab inside a string value is illegal JSON;
        // the repair pass must preserve the character, not drop the value.
        let broken = "{\"content\": \"line one\nline two\tindented\"}";
        let repaired = repair_json_string_escapes(broken);
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["content"], "line one\nline two\tindented");
    }

    #[test]
    fn repair_leaves_text_outside_strings_alone() {
        // Structural whitespace is not inside a string literal, so it must not
        // be turned into `\n` — that would corrupt the document itself.
        let json = "{\n  \"a\": 1\n}";
        assert_eq!(repair_json_string_escapes(json), json);
    }

    #[test]
    fn extracts_object_from_fenced_output_with_prose() {
        let response = "Here are the memories:\n```json\n{\"a\": 1}\n```";
        assert_eq!(extract_json_object(response), Some(r#"{"a": 1}"#));
    }

    #[test]
    fn extracts_object_with_braces_inside_strings() {
        // A brace inside a string literal must not be counted towards depth,
        // or the object gets cut short at the first `}` in the content.
        let response = r#"{"content": "code: fn x() { y() }"}"#;
        assert_eq!(extract_json_object(response), Some(response));
    }

    #[test]
    fn extracts_object_ignoring_escaped_quotes() {
        // An escaped quote inside a string must not be read as the string's
        // end, which would flip brace tracking back on mid-content.
        let response = r#"{"content": "he said \"{\" once"}"#;
        assert_eq!(extract_json_object(response), Some(response));
    }

    #[test]
    fn extracts_first_object_when_several_follow() {
        let response = r#"{"a": {"nested": 1}} trailing {"b": 2}"#;
        assert_eq!(
            extract_json_object(response),
            Some(r#"{"a": {"nested": 1}}"#)
        );
    }

    #[test]
    fn rejects_output_without_an_object() {
        assert_eq!(extract_json_object("I cannot help with that"), None);
        // An opening brace that never closes is not a usable object either.
        assert_eq!(extract_json_object(r#"{"a": 1"#), None);
    }

    #[test]
    fn unix_now_returns_a_plausible_epoch_second() {
        // Guards the fallback arm: a `0` here means the clock read failed and
        // every timestamp written this run is wrong.
        assert!(unix_now() > 1_700_000_000);
    }
}
