use std::collections::HashSet;

/// Extract the set of file paths modified by `patch` (unified diff format).
///
/// Strips the `b/` prefix from `+++ b/path`. Skips `/dev/null` (deletions).
#[must_use]
pub fn modified_files(patch: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            if path == "/dev/null" {
                continue;
            }
            let stripped = path.strip_prefix("b/").unwrap_or(path);
            out.insert(stripped.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_file_diff() {
        let diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n";
        let files = modified_files(diff);
        assert_eq!(files.len(), 1);
        assert!(files.contains("x.py"));
    }

    #[test]
    fn parses_multi_file_diff() {
        let diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/y/z.py b/y/z.py\n--- a/y/z.py\n+++ b/y/z.py\n@@ -1 +1 @@\n-c\n+d\n";
        let files = modified_files(diff);
        assert_eq!(files.len(), 2);
        assert!(files.contains("x.py"));
        assert!(files.contains("y/z.py"));
    }

    #[test]
    fn skips_dev_null_for_deletions() {
        let diff = "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n";
        let files = modified_files(diff);
        assert!(files.is_empty());
    }
}
