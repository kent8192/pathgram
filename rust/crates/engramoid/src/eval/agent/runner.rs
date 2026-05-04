use crate::eval::instance::SweInstance;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolKind {
    Grep,
    Glob,
    Read,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub kind: ToolKind,
    pub input: String,
    pub accessed_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub instance_id: String,
    pub tool_calls: Vec<ToolCall>,
    pub final_reading_context: Vec<String>,
    pub completed: bool,
}

impl Trace {
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.tool_calls.len()
    }

    #[must_use]
    pub fn distinct_accessed_files(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for call in &self.tool_calls {
            for f in &call.accessed_files {
                if seen.insert(f.clone()) {
                    out.push(f.clone());
                }
            }
        }
        out
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("agent budget exceeded")]
    BudgetExceeded,
    #[error("custom: {0}")]
    Custom(String),
}

pub trait AgentRunner {
    fn name(&self) -> &'static str;

    /// Execute the agent against `instance` rooted at `repo_root`, return its trace.
    fn run(&self, instance: &SweInstance, repo_root: &Path) -> Result<Trace, RunError>;
}
