use crate::eval::instance::SweInstance;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub mod swe_bench_lite;
pub mod swe_gym;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub trait Loader {
    /// Load all instances from the given path.
    fn load(&self, path: &Path) -> Result<Vec<SweInstance>, LoadError>;

    /// Logical name of this dataset, used for reporting.
    fn dataset_name(&self) -> &'static str;
}

/// Shared JSONL parser used by every Loader implementation.
pub(crate) fn read_jsonl(path: &Path) -> Result<Vec<SweInstance>, LoadError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line)?);
    }
    Ok(out)
}

/// Return the subset of `train` whose `repo` does NOT appear in `eval`.
/// Enforces the repo-disjoint invariant required by the design doc §8.3.
#[must_use]
pub fn repo_disjoint_split(
    train: &[SweInstance],
    eval: &[SweInstance],
) -> Vec<SweInstance> {
    let eval_repos: HashSet<&str> = eval.iter().map(|i| i.repo.as_str()).collect();
    train
        .iter()
        .filter(|i| !eval_repos.contains(i.repo.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(id: &str, repo: &str) -> SweInstance {
        SweInstance {
            instance_id: id.into(),
            repo: repo.into(),
            base_commit: "0".repeat(40),
            problem_statement: String::new(),
            patch: String::new(),
            test_patch: String::new(),
            fail_to_pass: vec![],
            pass_to_pass: vec![],
            hints_text: None,
            version: None,
        }
    }

    #[test]
    fn repo_disjoint_split_excludes_eval_repos() {
        let train = vec![synth("a-1", "django/django"), synth("a-2", "flask/flask")];
        let eval = vec![synth("b-1", "django/django")];
        let disjoint = repo_disjoint_split(&train, &eval);
        assert_eq!(disjoint.len(), 1);
        assert_eq!(disjoint[0].instance_id, "a-2");
    }
}
