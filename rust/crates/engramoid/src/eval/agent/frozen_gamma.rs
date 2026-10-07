use super::runner::{AgentRunner, RunError, ToolCall, ToolKind, Trace};
use crate::eval::instance::SweInstance;
use crate::retrieval::pipeline::FrozenGammaPipeline;
use crate::scorers::{Embedder, Reranker};
use std::path::Path;

/// Wraps the frozen γ retrieval pipeline as an `AgentRunner` so it slots
/// into `EvalRunner.gram_runner` for paired B0 vs B4 measurement.
///
/// From the eval framework's perspective, frozen γ produces exactly one
/// "tool call" (the single-step `find_relevant_context`). The accessed
/// files are the distinct paths cited in the resulting blob.
pub struct FrozenGammaRunner<'a> {
    pub pipeline: FrozenGammaPipeline<'a>,
    name: &'static str,
}

impl<'a> FrozenGammaRunner<'a> {
    pub fn new(embedder: &'a dyn Embedder, reranker: &'a dyn Reranker) -> Self {
        Self {
            pipeline: FrozenGammaPipeline::new(embedder, reranker),
            name: "FrozenGamma",
        }
    }

    pub fn named(
        embedder: &'a dyn Embedder,
        reranker: &'a dyn Reranker,
        name: &'static str,
    ) -> Self {
        Self {
            pipeline: FrozenGammaPipeline::new(embedder, reranker),
            name,
        }
    }
}

impl<'a> AgentRunner for FrozenGammaRunner<'a> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, instance: &SweInstance, repo_root: &Path) -> Result<Trace, RunError> {
        let blob = self
            .pipeline
            .retrieve(&instance.problem_statement, repo_root)
            .map_err(|e| RunError::Custom(format!("frozen γ pipeline: {e}")))?;
        let accessed = blob.distinct_paths();
        let final_reading_context: Vec<String> = accessed
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let single_call = ToolCall {
            kind: ToolKind::Other("FrozenGamma".to_string()),
            input: instance.problem_statement.clone(),
            accessed_files: accessed,
        };
        Ok(Trace {
            instance_id: instance.instance_id.clone(),
            tool_calls: vec![single_call],
            final_reading_context,
            completed: true,
        })
    }
}
