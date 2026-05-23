//! Keyword extraction and grep utilities shared between the deterministic
//! baseline (B0) and the hybrid candidate source (B4/B5).
//!
//! Extracted from `DeterministicBaselineRunner` so both the baseline runner
//! and the retrieval pipeline can reuse the same logic without duplication.

use regex::Regex;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// File extensions considered "code" by the keyword grep.
pub fn is_code_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "py" | "pyx"
            | "rs"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "mjs"
            | "go"
            | "java"
            | "kt"
            | "scala"
            | "rb"
            | "c"
            | "cc"
            | "cpp"
            | "cxx"
            | "h"
            | "hpp"
            | "cs"
            | "swift"
            | "m"
            | "mm"
    )
}

/// Path components that almost certainly do not contain bug-target source.
pub fn is_skip_component(c: &str) -> bool {
    matches!(
        c,
        ".git"
            | "node_modules"
            | "vendor"
            | "third_party"
            | "build"
            | "dist"
            | "target"
            | "__pycache__"
            | ".pytest_cache"
            | ".mypy_cache"
            | ".tox"
            | "site-packages"
            | "venv"
            | ".venv"
            | "env"
            | ".env"
    )
}

/// Extract identifier-shaped keywords from a problem statement.
///
/// Prefers CamelCase or snake_case tokens (length ≥ `min_len`), sorted
/// longest-first (more specific), deduplicated, capped at `max_keywords`.
pub fn extract_keywords(statement: &str, max_keywords: usize, min_len: usize) -> Vec<String> {
    let kw = Regex::new(r"[A-Za-z_][A-Za-z0-9_]+").expect("regex compiles");
    let mut out: Vec<String> = kw
        .find_iter(statement)
        .map(|m| m.as_str().to_string())
        .filter(|s| s.len() >= min_len)
        .filter(|s| {
            s.chars().any(char::is_uppercase) || s.contains('_')
        })
        .collect();
    out.sort_by(|a, b| b.len().cmp(&a.len()));
    out.dedup();
    out.truncate(max_keywords);
    out
}

/// Collect candidate code-file paths under `root`.
///
/// Filters by extension and skip-listed path components. Files larger than
/// `max_file_size_bytes` are skipped.
pub fn collect_code_files(root: &Path, max_file_size_bytes: u64) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let walker = WalkDir::new(root)
        .max_depth(10)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let Some(name) = e.file_name().to_str() else {
                return false;
            };
            !is_skip_component(name)
        });
    for entry in walker.flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !is_code_extension(path) {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.len() > max_file_size_bytes {
            continue;
        }
        files.push(path.to_path_buf());
    }
    files
}

/// Grep candidate files for `pattern` (exact substring match via regex escape).
/// Returns up to `max_hits` matching paths.
pub fn grep_files<'a>(
    candidates: &'a [PathBuf],
    pattern: &str,
    max_hits: usize,
) -> Vec<&'a PathBuf> {
    let escaped = regex::escape(pattern);
    let Ok(re) = Regex::new(&escaped) else {
        return Vec::new();
    };
    let mut hits = Vec::new();
    for path in candidates {
        if hits.len() >= max_hits {
            break;
        }
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        if re.is_match(&content) {
            hits.push(path);
        }
    }
    hits
}

/// Run all keywords against candidate files, returning deduplicated hit file
/// paths (relative to `repo_root`). This is the "shadow" mode used by the
/// coverage fallback: it collects file paths without reading file contents.
pub fn keyword_grep_hit_paths(
    statement: &str,
    repo_root: &Path,
    max_keywords: usize,
    min_keyword_len: usize,
    max_files_per_keyword: usize,
    max_file_size_bytes: u64,
) -> Vec<String> {
    let keywords = extract_keywords(statement, max_keywords, min_keyword_len);
    let candidates = collect_code_files(repo_root, max_file_size_bytes);
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for kw in &keywords {
        let hits = grep_files(&candidates, kw, max_files_per_keyword);
        for p in hits {
            if let Ok(rel) = p.strip_prefix(repo_root) {
                let s = rel.to_string_lossy().into_owned();
                if seen.insert(s.clone()) {
                    out.push(s);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_keywords_prefers_identifiers() {
        let stmt = "ASCIIUsernameValidator allows trailing newline in validate_password";
        let kws = extract_keywords(stmt, 8, 3);
        assert!(kws.contains(&"ASCIIUsernameValidator".to_string()));
        assert!(kws.contains(&"validate_password".to_string()));
        // "allows" / "trailing" / "newline" are lowercase-only, should be filtered
        assert!(!kws.iter().any(|k| k == "allows"));
    }

    #[test]
    fn extract_keywords_respects_max() {
        let stmt = "FooBar BazQux CorgeGrault GarplyWaldo FredPlugh XyzzyThud";
        let kws = extract_keywords(stmt, 3, 3);
        assert_eq!(kws.len(), 3);
    }

    #[test]
    fn is_code_extension_accepts_python_rust() {
        assert!(is_code_extension(Path::new("foo.py")));
        assert!(is_code_extension(Path::new("foo.rs")));
        assert!(!is_code_extension(Path::new("foo.md")));
        assert!(!is_code_extension(Path::new("foo.txt")));
    }

    #[test]
    fn is_skip_component_rejects_git_and_vendor() {
        assert!(is_skip_component(".git"));
        assert!(is_skip_component("node_modules"));
        assert!(is_skip_component("target"));
        assert!(!is_skip_component("src"));
    }

    #[test]
    fn keyword_grep_finds_matching_files() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/validators.py"),
            "class ASCIIUsernameValidator:\n    regex = r'^[\\w.@+-]+$'\n",
        )
        .unwrap();
        std::fs::write(root.join("src/util.py"), "def util():\n    pass\n").unwrap();

        let hits =
            keyword_grep_hit_paths("ASCIIUsernameValidator regex", root, 8, 3, 5, 256 * 1024);
        assert!(hits.iter().any(|p| p.ends_with("validators.py")));
    }
}
