use super::runner::{AgentRunner, RunError, ToolCall, ToolKind, Trace};
use crate::eval::instance::SweInstance;
use crate::graph::engine::GraphEngine;
use crate::graph::models::EdgeKind;
use crate::learning::gram_score::{BlobMode, GramScoreTracker};
use crate::learning::session::process_session;
use crate::retrieval::pipeline::FrozenGammaPipeline;
use crate::scorers::{Embedder, Reranker};
use std::cell::RefCell;
use std::path::Path;

/// Hebbian-augmented pipeline runner (B5).
///
/// Wraps the frozen γ retrieval pipeline with:
/// - Per-session Hebbian updates on `CoAccessed` edges between retrieved files
/// - Rolling gram-score tracking with shadow/canary/mature mode gating
///
/// In **shadow** mode the runner returns an empty trace (silent).
/// In **canary** or **mature** mode it returns the full retrieval result.
///
/// The `GraphEngine` persists across `run()` calls, accumulating `File` nodes
/// and `CoAccessed` edges. The `GramScoreTracker` rolling window gates
/// whether results are returned.
pub struct HebbianGammaRunner<'a> {
    pub pipeline: FrozenGammaPipeline<'a>,
    graph: RefCell<GraphEngine>,
    gram_tracker: RefCell<GramScoreTracker>,
    /// Learning rate for Hebbian updates (default 0.1).
    pub eta: f64,
    /// Gram-score threshold for canary → mature transition (default 0.3).
    pub threshold_gate: f64,
}

impl<'a> HebbianGammaRunner<'a> {
    pub fn new(embedder: &'a dyn Embedder, reranker: &'a dyn Reranker) -> Self {
        Self {
            pipeline: FrozenGammaPipeline::new(embedder, reranker),
            graph: RefCell::new(GraphEngine::new()),
            gram_tracker: RefCell::new(GramScoreTracker::default()),
            eta: 0.1,
            threshold_gate: 0.3,
        }
    }

    /// Set the gram-tracker window size (default 20). Smaller values accelerate
    /// Shadow → Canary → Mature transitions; useful for smoke testing.
    pub fn set_window_size(&self, size: usize) {
        self.gram_tracker.borrow_mut().window_size = size;
    }

    /// Current operational mode derived from the rolling gram score.
    pub fn current_mode(&self) -> BlobMode {
        self.gram_tracker.borrow().mode(self.threshold_gate)
    }
}

impl AgentRunner for HebbianGammaRunner<'_> {
    fn name(&self) -> &'static str {
        "HebbianGamma"
    }

    fn run(&self, instance: &SweInstance, repo_root: &Path) -> Result<Trace, RunError> {
        let blob = self
            .pipeline
            .retrieve(&instance.problem_statement, repo_root)
            .map_err(|e| RunError::Custom(format!("hebbian γ pipeline: {e}")))?;

        let accessed = blob.distinct_paths();

        // Process session: create/update CoAccessed edges between retrieved files.
        // Treat each retrieval as a nominal success; real success/failure feedback
        // would come from an external process_session call after patch evaluation.
        if accessed.len() >= 2 {
            process_session(
                &mut self.graph.borrow_mut(),
                &accessed,
                true,
                self.eta,
            );
        }

        // Compute gram score from the CoAccessed edges now present in the graph.
        let gram_score = {
            let graph = self.graph.borrow();
            let edges = graph.all_edges();
            let coaccessed: Vec<&crate::graph::models::Edge> = edges
                .iter()
                .filter(|e| e.kind == EdgeKind::CoAccessed)
                .copied()
                .collect();
            self.gram_tracker.borrow().compute(&coaccessed)
        };

        self.gram_tracker.borrow_mut().record(gram_score);
        let mode = self.gram_tracker.borrow().mode(self.threshold_gate);

        match mode {
            BlobMode::Shadow => Ok(Trace {
                instance_id: instance.instance_id.clone(),
                tool_calls: vec![],
                final_reading_context: vec![],
                completed: true,
            }),
            BlobMode::Canary | BlobMode::Mature => {
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
                    kind: ToolKind::Other("HebbianGamma".to_string()),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::instance::SweInstance;
    use crate::scorers::cohere_rerank::MockReranker;
    use crate::scorers::openai_embed::MockEmbedder;
    use tempfile::TempDir;

    fn fake_repo() -> TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/validators.py"),
            "class ASCIIUsernameValidator:\n    regex = r'^[\\w.@+-]+$'\n",
        )
        .unwrap();
        std::fs::write(root.join("src/util.py"), "def util():\n    pass\n").unwrap();
        std::fs::write(root.join("src/other.py"), "def other():\n    pass\n").unwrap();
        td
    }

    fn synth_inst(stmt: &str) -> SweInstance {
        SweInstance {
            instance_id: "fake-h1".into(),
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
    fn shadow_mode_returns_empty_trace() {
        let repo = fake_repo();
        let inst = synth_inst("ASCIIUsernameValidator regex");
        let embedder = MockEmbedder::new(64);
        let reranker = MockReranker;
        let mut runner = HebbianGammaRunner::new(&embedder, &reranker);
        runner.pipeline.embedding_dim = 64;

        // First run: gram_tracker has 0 scores → Shadow mode
        let trace = runner.run(&inst, repo.path()).expect("run");
        assert!(trace.tool_calls.is_empty());
        assert!(trace.final_reading_context.is_empty());
    }

    #[test]
    fn graph_accumulates_file_nodes_across_runs() {
        let repo = fake_repo();
        let embedder = MockEmbedder::new(64);
        let reranker = MockReranker;
        let mut runner = HebbianGammaRunner::new(&embedder, &reranker);
        runner.pipeline.embedding_dim = 64;

        let inst1 = synth_inst("ASCIIUsernameValidator regex");
        let inst2 = synth_inst("util function");

        runner.run(&inst1, repo.path()).expect("run 1");
        runner.run(&inst2, repo.path()).expect("run 2");

        let graph = runner.graph.borrow();
        let all = graph.all_nodes();
        let file_nodes: Vec<_> = all
            .iter()
            .filter(|n| n.kind == crate::graph::models::NodeKind::File)
            .collect();
        // Each run creates File nodes for the files in its blob
        assert!(!file_nodes.is_empty(), "graph should have File nodes");
    }

    #[test]
    fn multiple_runs_transition_from_shadow() {
        let _repo = fake_repo();
        let embedder = MockEmbedder::new(64);
        let reranker = MockReranker;
        let runner = HebbianGammaRunner::new(&embedder, &reranker);
        // Override threshold very low so we can see the transition quickly
        runner.gram_tracker.borrow_mut().record(0.9);
        runner.gram_tracker.borrow_mut().record(0.9);
        runner.gram_tracker.borrow_mut().record(0.9);
        // With window_size=20, half=10, 3 scores < 10 → still Shadow
        assert_eq!(runner.current_mode(), BlobMode::Shadow);

        // Fill the window with high scores
        for _ in 0..20 {
            runner.gram_tracker.borrow_mut().record(0.9);
        }
        // Now rolling mean = 0.9, half_window=10 satisfied → Mature
        assert_eq!(runner.current_mode(), BlobMode::Mature);
    }
}
