//! Hook execution result caching (§6.7.5).
//!
//! Executes error-referenced files (e.g., test files mentioned in tracebacks)
//! and caches stdout/stderr. The execution output is then parsed for additional
//! file paths, error types, and keywords that feed back into candidate generation.
//!
//! This provides a "runtime signal" that complements static error-message
//! extraction — actually running the failing code surfaces import chains,
//! dynamically-referenced modules, and concrete error locations.

use crate::retrieval::error_extract::ErrorSignalExtractor;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

/// Cached result of executing a file.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub file_path: String,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub wall_time_ms: u64,
}

/// In-memory cache of file execution results.
///
/// Executes Python files (scripts or test files) referenced in error
/// messages, captures their output, and parses it for additional retrieval
/// signals.
pub struct HookExecutionCache {
    pub timeout_secs: u64,
    pub max_output_bytes: usize,
    cache: HashMap<String, ExecutionResult>,
}

impl Default for HookExecutionCache {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_output_bytes: 64 * 1024,
            cache: HashMap::new(),
        }
    }
}

impl HookExecutionCache {
    /// Execute a single file and cache the result. Returns the cached result
    /// on subsequent calls for the same path.
    pub fn execute(&mut self, repo_root: &Path, file_path: &str) -> &ExecutionResult {
        if !self.cache.contains_key(file_path) {
            let result = Self::run_file(repo_root, file_path, self.timeout_secs, self.max_output_bytes);
            self.cache.insert(file_path.to_string(), result);
        }
        self.cache.get(file_path).expect("just inserted")
    }

    /// Execute all files referenced in error messages within `statement`.
    /// Returns cloned results for successfully identified and executed files.
    pub fn execute_error_files(
        &mut self,
        repo_root: &Path,
        statement: &str,
    ) -> Vec<ExecutionResult> {
        let extractor = ErrorSignalExtractor::default();
        let paths = extractor.extract_file_paths(statement);
        // Collect paths to avoid borrowing conflicts
        let to_execute: Vec<String> = paths
            .iter()
            .filter(|p| {
                let full = repo_root.join(p);
                full.exists() && full.is_file()
            })
            .cloned()
            .collect();
        let mut results = Vec::new();
        for path in &to_execute {
            let result = self.execute(repo_root, path).clone();
            results.push(result);
        }
        results
    }

    /// Parse all cached execution outputs for additional file paths
    /// (tracebacks, import errors, module references).
    pub fn extract_output_file_paths(&self) -> Vec<String> {
        let extractor = ErrorSignalExtractor::default();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out = Vec::new();

        for result in self.cache.values() {
            let combined = format!("{}\n{}", result.stdout, result.stderr);
            for path in extractor.extract_file_paths(&combined) {
                if seen.insert(path.clone()) {
                    out.push(path);
                }
            }
        }
        out
    }

    /// Number of cached results.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    // ── private ────────────────────────────────────────────────────────

    fn run_file(
        repo_root: &Path,
        file_path: &str,
        _timeout_secs: u64,
        max_output: usize,
    ) -> ExecutionResult {
        let full_path = repo_root.join(file_path);
        let start = Instant::now();

        // Try running as a Python script
        let mut cmd = Command::new("python3");
        cmd.arg(&full_path)
            .current_dir(repo_root)
            .stdin(std::process::Stdio::null());

        let output = cmd.output();

        let wall_time_ms = start.elapsed().as_millis() as u64;

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout)
                    .chars()
                    .take(max_output)
                    .collect::<String>();
                let stderr = String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(max_output)
                    .collect::<String>();
                ExecutionResult {
                    file_path: file_path.to_string(),
                    exit_code: Some(out.status.code().unwrap_or(-1)),
                    stdout,
                    stderr,
                    timed_out: false,
                    wall_time_ms,
                }
            }
            Err(e) => {
                let msg = format!("execution error: {e}");
                ExecutionResult {
                    file_path: file_path.to_string(),
                    exit_code: None,
                    stdout: String::new(),
                    stderr: msg,
                    timed_out: false,
                    wall_time_ms,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_repo_with_error_script() -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        // Script that raises an error mentioning another file
        std::fs::write(
            root.join("src/test_validators.py"),
            r#"
import sys
print("running test...", file=sys.stderr)
print("FAIL: test_validators.py:42 - AssertionError", file=sys.stderr)
print('Traceback (most recent call last):', file=sys.stderr)
print('  File "src/validators.py", line 42, in validate', file=sys.stderr)
print('AssertionError: invalid input', file=sys.stderr)
sys.exit(1)
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("src/validators.py"),
            "def validate(x):\n    assert x, 'invalid input'\n",
        )
        .unwrap();
        td
    }

    #[test]
    fn execute_caches_result() {
        let repo = fake_repo_with_error_script();
        let mut cache = HookExecutionCache::default();
        let exit_code = {
            let result = cache.execute(repo.path(), "src/test_validators.py");
            assert_eq!(result.file_path, "src/test_validators.py");
            result.exit_code
        };
        assert_eq!(cache.len(), 1);
        // Second call should return cached
        let exit_code2;
        {
            let result2 = cache.execute(repo.path(), "src/test_validators.py");
            exit_code2 = result2.exit_code;
        }
        assert_eq!(cache.len(), 1);
        assert_eq!(exit_code2, exit_code);
    }

    #[test]
    fn execute_error_files_finds_executable_paths() {
        let repo = fake_repo_with_error_script();
        let mut cache = HookExecutionCache::default();
        let stmt = r#"Traceback:
  File "src/test_validators.py", line 42, in test_validate
AssertionError: invalid input"#;
        let results = cache.execute_error_files(repo.path(), stmt);
        assert!(!results.is_empty(), "should execute found file");
        assert!(results[0].file_path.contains("test_validators.py"));
    }

    #[test]
    fn extract_output_file_paths_finds_traceback_refs() {
        let repo = fake_repo_with_error_script();
        let mut cache = HookExecutionCache::default();
        cache.execute(repo.path(), "src/test_validators.py");
        let paths = cache.extract_output_file_paths();
        assert!(
            paths.iter().any(|p| p.contains("validators.py")),
            "should extract validators.py from stderr traceback, got: {paths:?}"
        );
    }

    #[test]
    fn empty_cache_returns_empty_paths() {
        let cache = HookExecutionCache::default();
        assert!(cache.is_empty());
        assert!(cache.extract_output_file_paths().is_empty());
    }
}
