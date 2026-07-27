//! Fetch the text of a resource (a URL or a local file) so it can be stored
//! as a node in the memory virtual filesystem.
//!
//! Web pages and local HTML are cleaned with the [`defuddle`] CLI (`defuddle
//! parse <src> --md`), which strips navigation/ads/boilerplate and emits
//! readable Markdown — much better summary fodder than a raw HTML dump. Plain
//! local files (Markdown, text, source, …) are read as-is; there is nothing to
//! declutter and shelling out would only add failure modes.
//!
//! [`defuddle`]: https://github.com/kepano/defuddle-cli

use std::path::Path;

use anyhow::{anyhow, Result};
use tracing::{debug, info};

/// External CLI used to declutter HTML into Markdown.
const DEFUDDLE_BIN: &str = "defuddle";

/// Install hint surfaced when `defuddle` is not on `PATH`.
const DEFUDDLE_INSTALL_HINT: &str =
    "Install it with `npm install -g defuddle` (it must be on PATH). \
     If it is installed under nvm, ensure the node bin directory is exported.";

/// File extensions treated as already-readable text — read directly instead of
/// running them through defuddle.
const TEXT_EXTENSIONS: &[&str] = &[
    "md", "markdown", "txt", "text", "rst", "org", "json", "yaml", "yml", "toml", "csv", "log",
    "rs", "py", "js", "ts", "go", "java", "c", "cpp", "h", "hpp", "sh", "sql",
];

/// A fetched resource: the source it came from and its extracted text.
pub struct FetchedResource {
    /// The original source string (URL or file path) as provided.
    pub source: String,
    /// A human-readable title, when one could be derived (URL last segment or
    /// file stem). Used to name the resource node when the caller gives none.
    pub title: String,
    /// The extracted text content (Markdown for HTML, raw text otherwise).
    pub text: String,
}

/// Fetch and extract the text of `source`, which may be an `http(s)://` URL or
/// a local filesystem path.
pub async fn fetch_resource(source: &str) -> Result<FetchedResource> {
    if is_url(source) {
        fetch_url(source).await
    } else {
        fetch_file(source).await
    }
}

fn is_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

/// Fetch a URL via defuddle, returning cleaned Markdown.
async fn fetch_url(url: &str) -> Result<FetchedResource> {
    let text = run_defuddle(url).await?;
    Ok(FetchedResource {
        source: url.to_string(),
        title: url_title(url),
        text,
    })
}

/// Read a local file. HTML files go through defuddle; everything else is read
/// as text.
async fn fetch_file(path_str: &str) -> Result<FetchedResource> {
    let path = Path::new(path_str);
    if !path.exists() {
        return Err(anyhow!(
            "resource '{path_str}' is neither an http(s) URL nor an existing file"
        ));
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    let text = if ext == "html" || ext == "htm" {
        run_defuddle(path_str).await?
    } else if ext.is_empty() || TEXT_EXTENSIONS.contains(&ext.as_str()) {
        tokio::fs::read_to_string(path)
            .await
            .map_err(|e| anyhow!("failed to read '{path_str}': {e}"))?
    } else {
        // Unknown/binary extension: attempt a UTF-8 read, but fail clearly
        // rather than storing mojibake.
        tokio::fs::read_to_string(path).await.map_err(|e| {
            anyhow!("'{path_str}' is not readable as UTF-8 text ({e}); only text and HTML files are supported")
        })?
    };

    Ok(FetchedResource {
        source: path_str.to_string(),
        title: file_title(path),
        text,
    })
}

/// Run `defuddle parse <src> --md` and return its Markdown output.
async fn run_defuddle(src: &str) -> Result<String> {
    if !defuddle_available().await {
        return Err(anyhow!(
            "'{DEFUDDLE_BIN}' was not found on PATH.\n  {DEFUDDLE_INSTALL_HINT}"
        ));
    }

    info!("Fetching resource with defuddle: {src}");
    let result = tokio::process::Command::new(DEFUDDLE_BIN)
        .arg("parse")
        .arg(src)
        .arg("--md")
        .output()
        .await;

    match result {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if text.is_empty() {
                return Err(anyhow!("defuddle returned no content for '{src}'"));
            }
            debug!("defuddle extracted {} chars from {src}", text.len());
            Ok(text)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow!(
                "defuddle failed for '{src}' (exit {:?}): {}",
                output.status.code(),
                stderr.trim()
            ))
        }
        Err(e) => Err(anyhow!("failed to spawn '{DEFUDDLE_BIN}': {e}")),
    }
}

/// Returns `true` if defuddle is present and responds to `--version`.
async fn defuddle_available() -> bool {
    tokio::process::Command::new(DEFUDDLE_BIN)
        .arg("--version")
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Derive a display title from a URL (last non-empty path segment, or host).
fn url_title(url: &str) -> String {
    let without_scheme = url.trim_end_matches('/').split("://").nth(1).unwrap_or(url);
    // Drop any query/fragment.
    let path = without_scheme
        .split(['?', '#'])
        .next()
        .unwrap_or(without_scheme);
    path.rsplit('/')
        .find(|seg| !seg.is_empty())
        .unwrap_or(path)
        .to_string()
}

/// Derive a display title from a file path (file stem).
fn file_title(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("resource")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_urls() {
        assert!(is_url("https://example.com/a"));
        assert!(is_url("http://example.com"));
        assert!(!is_url("/tmp/file.md"));
        assert!(!is_url("./notes.txt"));
    }

    #[test]
    fn derives_url_title() {
        assert_eq!(url_title("https://example.com/docs/guide"), "guide");
        assert_eq!(url_title("https://example.com/docs/guide/"), "guide");
        assert_eq!(url_title("https://example.com/page?x=1#frag"), "page");
        assert_eq!(url_title("https://example.com"), "example.com");
    }

    #[test]
    fn derives_file_title() {
        assert_eq!(file_title(Path::new("/a/b/notes.md")), "notes");
        assert_eq!(file_title(Path::new("README")), "README");
    }
}
