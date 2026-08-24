//! Deriving a stable project name from a session's working directory.
//!
//! Two callers need the same answer and must not disagree: session discovery
//! (which decides what a harvested session is scoped to) and the transcript
//! parser (which scopes a session imported by path, bypassing discovery). A
//! session that lands under one project name when harvested and another when
//! imported is worse than one that lands under neither — recall would split in
//! two. So the derivation lives here, once.

/// Derive a stable project name from a session's working directory:
/// 1. the git remote `owner/repo` (from `.git/config`, walking up), else
/// 2. the directory's basename, else
/// 3. `None` (a global memory) when nothing usable can be derived.
pub(crate) fn project_from_cwd(cwd: &str) -> Option<String> {
    let cwd = cwd.trim();
    if cwd.is_empty() {
        return None;
    }
    let path = std::path::Path::new(cwd);
    if let Some(remote) = git_remote_url(path) {
        if let Some(name) = repo_name_from_remote(&remote) {
            return Some(name);
        }
    }
    // Fallback: the basename of the cwd (e.g. `/home/me/svc-billing` → `svc-billing`).
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

/// Read `remote.origin.url` from the `.git/config` of the repo containing
/// `start`, walking up parent directories to find the `.git` directory. Reads
/// the config file directly — no `git` binary required.
fn git_remote_url(start: &std::path::Path) -> Option<String> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let git_config = d.join(".git").join("config");
        if let Ok(contents) = std::fs::read_to_string(&git_config) {
            if let Some(url) = parse_origin_url(&contents) {
                return Some(url);
            }
        }
        dir = d.parent();
    }
    None
}

/// Parse `url = …` from the `[remote "origin"]` section of a git config file.
/// A minimal INI walk: track the current section header, and inside
/// `[remote "origin"]` return the first `url` value.
fn parse_origin_url(config: &str) -> Option<String> {
    let mut in_origin = false;
    for line in config.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            // Section header, e.g. `[remote "origin"]`.
            in_origin = line == "[remote \"origin\"]";
            continue;
        }
        if in_origin {
            if let Some(rest) = line.strip_prefix("url") {
                let rest = rest.trim_start();
                if let Some(value) = rest.strip_prefix('=') {
                    let url = value.trim();
                    if !url.is_empty() {
                        return Some(url.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Derive `owner/repo` from a git remote URL, across the common forms:
/// - SSH scp-like: `git@github.com:owner/repo.git`
/// - HTTPS: `https://github.com/owner/repo.git`
/// - SSH URL: `ssh://git@github.com/owner/repo.git`
///
/// Returns `owner/repo` (the last two path segments), with any `.git` suffix
/// stripped. `None` when no usable `owner/repo` can be extracted.
fn repo_name_from_remote(url: &str) -> Option<String> {
    let url = url.trim().trim_end_matches('/');
    // Take the part after the host: everything after the last ':' (scp-like) or
    // after the host in a scheme URL. Normalizing both to a '/'-joined path.
    let path = if let Some(idx) = url.find("://") {
        // scheme://[user@]host/owner/repo(.git) — drop scheme+host.
        let after_scheme = &url[idx + 3..];
        after_scheme.split_once('/').map(|(_, p)| p)?.to_string()
    } else if let Some((_, after_colon)) = url.split_once(':') {
        // scp-like git@host:owner/repo(.git)
        after_colon.to_string()
    } else {
        url.to_string()
    };

    let path = path.trim_matches('/');
    let path = path.strip_suffix(".git").unwrap_or(path);
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    match segments.as_slice() {
        [.., owner, repo] => Some(format!("{owner}/{repo}")),
        [repo] => Some(repo.to_string()),
        [] => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_name_from_remote_forms() {
        // SSH scp-like.
        assert_eq!(
            repo_name_from_remote("git@github.com:acme/svc-billing.git").as_deref(),
            Some("acme/svc-billing")
        );
        // HTTPS with .git.
        assert_eq!(
            repo_name_from_remote("https://github.com/acme/svc-billing.git").as_deref(),
            Some("acme/svc-billing")
        );
        // HTTPS without .git.
        assert_eq!(
            repo_name_from_remote("https://gitlab.corp.example.com/team/proj").as_deref(),
            Some("team/proj")
        );
        // ssh:// URL form.
        assert_eq!(
            repo_name_from_remote("ssh://git@github.com/acme/svc-ledger.git").as_deref(),
            Some("acme/svc-ledger")
        );
        // Nested group path keeps only the last two segments (owner/repo).
        assert_eq!(
            repo_name_from_remote("https://gitlab.com/group/subgroup/proj.git").as_deref(),
            Some("subgroup/proj")
        );
        // A trailing slash is tolerated.
        assert_eq!(
            repo_name_from_remote("https://github.com/acme/proj/").as_deref(),
            Some("acme/proj")
        );
        // A bare repo name (no owner) round-trips.
        assert_eq!(
            repo_name_from_remote("git@host:proj.git").as_deref(),
            Some("proj")
        );
    }

    #[test]
    fn parse_origin_url_reads_only_origin() {
        let config = "\
[core]
\trepositoryformatversion = 0
[remote \"upstream\"]
\turl = git@github.com:someone/fork.git
[remote \"origin\"]
\turl = git@github.com:acme/svc-billing.git
\tfetch = +refs/heads/*:refs/remotes/origin/*
";
        assert_eq!(
            parse_origin_url(config).as_deref(),
            Some("git@github.com:acme/svc-billing.git")
        );
    }

    #[test]
    fn project_from_cwd_uses_git_remote_then_falls_back_to_basename() {
        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("svc-billing");
        std::fs::create_dir_all(repo.join(".git")).unwrap();

        // With a remote, the derived project is owner/repo.
        std::fs::write(
            repo.join(".git").join("config"),
            "[remote \"origin\"]\n\turl = git@github.com:acme/svc-billing.git\n",
        )
        .unwrap();
        assert_eq!(
            project_from_cwd(repo.to_str().unwrap()).as_deref(),
            Some("acme/svc-billing")
        );

        // A subdirectory of the repo resolves to the same project (walks up).
        let sub = repo.join("crates").join("core");
        std::fs::create_dir_all(&sub).unwrap();
        assert_eq!(
            project_from_cwd(sub.to_str().unwrap()).as_deref(),
            Some("acme/svc-billing")
        );

        // No .git at all → fall back to the directory basename.
        let plain = dir.path().join("scratchpad");
        std::fs::create_dir_all(&plain).unwrap();
        assert_eq!(
            project_from_cwd(plain.to_str().unwrap()).as_deref(),
            Some("scratchpad")
        );

        // A hyphenated repo name survives intact — the character that the
        // Claude project-directory encoding cannot round-trip.
        let hyphenated = dir.path().join("memory-rs");
        std::fs::create_dir_all(&hyphenated).unwrap();
        assert_eq!(
            project_from_cwd(hyphenated.to_str().unwrap()).as_deref(),
            Some("memory-rs")
        );

        // Empty cwd → global (None).
        assert_eq!(project_from_cwd(""), None);
    }
}
