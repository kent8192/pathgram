//! Extract error-relevant signals from problem statements.
//!
//! Parses Python-style tracebacks and error messages to recover file paths and
//! line numbers that keyword extraction alone would miss. These signals feed
//! into the hybrid candidate source as `CandidateSource::ErrorMessage`.

use regex::Regex;

/// A single error-relevant signal extracted from a problem statement.
#[derive(Debug, Clone)]
pub struct ErrorSnippet {
    pub file_path: Option<String>,
    pub line_number: Option<usize>,
    pub error_type: Option<String>,
    pub message: String,
}

pub struct ErrorSignalExtractor {
    pub max_snippets: usize,
}

impl Default for ErrorSignalExtractor {
    fn default() -> Self {
        Self { max_snippets: 5 }
    }
}

impl ErrorSignalExtractor {
    /// Parse a problem statement for Python-style tracebacks and error
    /// messages. Extracts file paths and line numbers from traceback frames,
    /// error types from `XxxError:` patterns, and assertion messages.
    pub fn extract(&self, statement: &str) -> Vec<ErrorSnippet> {
        let mut out = Vec::new();

        let tb_frame =
            Regex::new(r#"File\s+"([^"]+)",\s*line\s+(\d+)"#).expect("traceback regex compiles");
        let error_type_re =
            Regex::new(r"(\w+(?:Error|Exception|Warning))(?::|\.|\s)").expect("error-type regex compiles");
        let assertion =
            Regex::new(r"AssertionError:\s*(.+?)(?:\n|$)").expect("assertion regex compiles");

        // 1. Traceback frames
        for cap in tb_frame.captures_iter(statement) {
            if out.len() >= self.max_snippets {
                break;
            }
            let path = cap.get(1).map(|m| m.as_str().to_string());
            let line: Option<usize> = cap.get(2).and_then(|m| m.as_str().parse().ok());
            let message = cap
                .get(0)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            out.push(ErrorSnippet {
                file_path: path,
                line_number: line,
                error_type: None,
                message,
            });
        }

        // 2. General error-type mentions (not traceback frames)
        if out.len() < self.max_snippets {
            for cap in error_type_re.captures_iter(statement) {
                if out.len() >= self.max_snippets {
                    break;
                }
                let err_type = cap.get(1).map(|m| m.as_str().to_string());
                let message = cap
                    .get(0)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let already = out.iter().any(|s| s.message == message);
                if already {
                    continue;
                }
                out.push(ErrorSnippet {
                    file_path: None,
                    line_number: None,
                    error_type: err_type,
                    message,
                });
            }
        }

        // 3. AssertionError messages
        if out.len() < self.max_snippets {
            for cap in assertion.captures_iter(statement) {
                if out.len() >= self.max_snippets {
                    break;
                }
                let msg = cap.get(1).map(|m| m.as_str().trim().to_string());
                let message = cap
                    .get(0)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                let already = out.iter().any(|s| s.message == message);
                if already {
                    continue;
                }
                out.push(ErrorSnippet {
                    file_path: None,
                    line_number: None,
                    error_type: Some("AssertionError".to_string()),
                    message: msg.unwrap_or_default(),
                });
            }
        }

        out
    }

    /// Convenience: extract unique file paths mentioned in tracebacks.
    pub fn extract_file_paths(&self, statement: &str) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for s in self.extract(statement) {
            if let Some(p) = s.file_path {
                if seen.insert(p.clone()) {
                    out.push(p);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_traceback_file_and_line() {
        let stmt = r#"Traceback (most recent call last):
  File "/app/src/validators.py", line 42, in validate
    raise AssertionError("invalid input")
AssertionError: invalid input"#;
        let ex = ErrorSignalExtractor::default();
        let snippets = ex.extract(stmt);
        let has_validators = snippets.iter().any(|s| {
            s.file_path
                .as_ref()
                .is_some_and(|p| p.contains("validators.py"))
                && s.line_number == Some(42)
        });
        assert!(has_validators, "should extract file+line from traceback");
    }

    #[test]
    fn extract_error_type() {
        let stmt = "ValueError: invalid literal for int() with base 10";
        let ex = ErrorSignalExtractor::default();
        let snippets = ex.extract(stmt);
        assert!(snippets
            .iter()
            .any(|s| s.error_type.as_deref() == Some("ValueError")));
    }

    #[test]
    fn extract_assertion_message() {
        let stmt = "AssertionError: expected True but got False\nmore text";
        let ex = ErrorSignalExtractor::default();
        let snippets = ex.extract(stmt);
        let has_assertion = snippets.iter().any(|s| {
            s.error_type.as_deref() == Some("AssertionError")
                && s.message.contains("expected True")
        });
        assert!(has_assertion);
    }

    #[test]
    fn extract_file_paths_convenience() {
        let stmt = r#"Traceback:
  File "/app/src/validators.py", line 42, in validate
  File "/app/src/utils.py", line 10, in helper
ValueError: something went wrong"#;
        let ex = ErrorSignalExtractor::default();
        let paths = ex.extract_file_paths(stmt);
        assert!(paths.iter().any(|p| p.contains("validators.py")));
        assert!(paths.iter().any(|p| p.contains("utils.py")));
    }

    #[test]
    fn respects_max_snippets() {
        let stmt = r#"File "a.py", line 1
File "b.py", line 2
File "c.py", line 3
File "d.py", line 4
File "e.py", line 5
File "f.py", line 6"#;
        let ex = ErrorSignalExtractor { max_snippets: 3 };
        let snippets = ex.extract(stmt);
        assert_eq!(snippets.len(), 3);
    }
}
