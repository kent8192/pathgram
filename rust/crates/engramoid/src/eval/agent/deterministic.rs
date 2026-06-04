use super::runner::{AgentRunner, RunError, ToolCall, ToolKind, Trace};
use crate::eval::instance::SweInstance;
use crate::retrieval::keyword;
use std::path::Path;

/// A reproducible, LLM-free baseline retrieval agent.
///
/// 1. Extract identifier-shaped keywords from `instance.problem_statement`
///    (CamelCase or snake_case, length ≥ `min_keyword_len`, total capped at
///    `max_keywords` to bound work on long problem statements).
/// 2. For each keyword, grep over **code files only** (extension filter)
///    rooted at `repo_root`, collecting the first `max_files_per_keyword`
///    hits. Vendor / test / generated paths are skipped.
/// 3. For each hit, record a Read tool call.
/// 4. Final reading context = last 5 distinct file reads.
///
/// Keyword extraction and file grepping are delegated to
/// `crate::retrieval::keyword` so the hybrid pipeline can reuse them.
pub struct DeterministicBaselineRunner {
    pub tool_call_budget: usize,
    pub max_files_per_keyword: usize,
    pub min_keyword_len: usize,
    pub max_keywords: usize,
    pub max_file_size_bytes: u64,
}

impl Default for DeterministicBaselineRunner {
    fn default() -> Self {
        Self {
            tool_call_budget: 30,
            max_files_per_keyword: 5,
            min_keyword_len: 3,
            max_keywords: 8,
            max_file_size_bytes: 256 * 1024,
        }
    }
}

impl AgentRunner for DeterministicBaselineRunner {
    fn name(&self) -> &'static str {
        "DeterministicBaseline"
    }

    fn run(&self, instance: &SweInstance, repo_root: &Path) -> Result<Trace, RunError> {
        let keywords = keyword::extract_keywords(
            &instance.problem_statement,
            self.max_keywords,
            self.min_keyword_len,
        );
        let candidates = keyword::collect_code_files(repo_root, self.max_file_size_bytes);
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut accessed_order: Vec<String> = Vec::new();

        'outer: for kw in keywords {
            if tool_calls.len() >= self.tool_call_budget {
                break;
            }
            let hits = keyword::grep_files(&candidates, &kw, self.max_files_per_keyword);
            let hit_strs: Vec<String> = hits
                .iter()
                .filter_map(|p| p.strip_prefix(repo_root).ok())
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            tool_calls.push(ToolCall {
                kind: ToolKind::Grep,
                input: kw.clone(),
                accessed_files: hit_strs.clone(),
            });
            for f in hit_strs {
                if tool_calls.len() >= self.tool_call_budget {
                    break 'outer;
                }
                tool_calls.push(ToolCall {
                    kind: ToolKind::Read,
                    input: f.clone(),
                    accessed_files: vec![f.clone()],
                });
                accessed_order.push(f);
            }
        }

        let final_reading_context: Vec<String> = accessed_order
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        Ok(Trace {
            instance_id: instance.instance_id.clone(),
            tool_calls,
            final_reading_context,
            completed: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_repo() -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/validators.py"),
            "class ASCIIUsernameValidator:\n    regex = r'^[\\w.@+-]+$'\n",
        )
        .unwrap();
        std::fs::write(root.join("src/util.py"), "def util():\n    pass\n").unwrap();
        std::fs::write(root.join("README.md"), "# project\n").unwrap();
        td
    }

    fn synth_inst(stmt: &str) -> SweInstance {
        SweInstance {
            instance_id: "fake-1".into(),
            repo: "x/y".into(),
            base_commit: "0".repeat(40),
            problem_statement: stmt.into(),
            patch: String::new(),
            test_patch: String::new(),
            fail_to_pass: vec![],
            pass_to_pass: vec![],
            hints_text: None,
            version: None,
        }
    }

    #[test]
    fn finds_keyword_match() {
        let repo = fake_repo();
        let inst = synth_inst("ASCIIUsernameValidator allows trailing newline");
        let runner = DeterministicBaselineRunner::default();
        let trace = runner.run(&inst, repo.path()).expect("run");
        assert!(trace.completed);
        let files = trace.distinct_accessed_files();
        assert!(
            files.iter().any(|p| p.ends_with("validators.py")),
            "expected validators.py in {files:?}"
        );
    }

    #[test]
    fn respects_budget() {
        let repo = fake_repo();
        let inst = synth_inst("Foo Bar Baz Quux Corge");
        let runner = DeterministicBaselineRunner {
            tool_call_budget: 3,
            ..Default::default()
        };
        let trace = runner.run(&inst, repo.path()).unwrap();
        assert!(trace.step_count() <= 3);
    }
}
