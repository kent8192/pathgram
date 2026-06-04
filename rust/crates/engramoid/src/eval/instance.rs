use serde::{Deserialize, Serialize};

/// One SWE-bench (or SWE-Gym) instance.
///
/// The schema mirrors the canonical
/// [`princeton-nlp/SWE-bench_Lite`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)
/// columns. Optional fields stay `Option`/`Vec` so SWE-Gym's slightly
/// different shape parses without rejection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweInstance {
    pub instance_id: String,
    pub repo: String,
    pub base_commit: String,
    pub problem_statement: String,
    pub patch: String,
    #[serde(default)]
    pub test_patch: String,
    #[serde(rename = "FAIL_TO_PASS", default)]
    pub fail_to_pass: Vec<String>,
    #[serde(rename = "PASS_TO_PASS", default)]
    pub pass_to_pass: Vec<String>,
    #[serde(default)]
    pub hints_text: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
}

impl SweInstance {
    /// Synthetic test fixtures are tagged with `-test-` in their id so the
    /// eval runner can route them around git checkout.
    #[must_use]
    pub fn is_synthetic_fixture(&self) -> bool {
        self.instance_id.contains("-test-")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_instance_from_json() {
        let json = r#"{
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "base_commit": "d26b2424437dabeeca94d7900b37d2df4410da0c",
            "problem_statement": "stmt",
            "patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
            "test_patch": "",
            "FAIL_TO_PASS": ["test_x"],
            "PASS_TO_PASS": []
        }"#;
        let inst: SweInstance = serde_json::from_str(json).unwrap();
        assert_eq!(inst.instance_id, "django__django-11099");
        assert_eq!(inst.repo, "django/django");
        assert_eq!(inst.base_commit.len(), 40);
        assert_eq!(inst.fail_to_pass, vec!["test_x"]);
        assert!(inst.pass_to_pass.is_empty());
        assert!(!inst.is_synthetic_fixture());
    }

    #[test]
    fn detects_synthetic_fixture() {
        let json = r#"{"instance_id":"flask-test-1","repo":"x/y","base_commit":"0000000000000000000000000000000000000000","problem_statement":"p","patch":"","test_patch":"","FAIL_TO_PASS":[],"PASS_TO_PASS":[]}"#;
        let inst: SweInstance = serde_json::from_str(json).unwrap();
        assert!(inst.is_synthetic_fixture());
    }
}
