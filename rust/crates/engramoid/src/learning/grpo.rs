//! Group Relative Policy Optimization for batch edge-weight updates (§6.4).
//!
//! GRPO processes a batch of session traces and optimizes graph edge weights
//! using group-relative advantage scores with clipped policy gradients.
//! This is the offline complement to the online Hebbian layer:
//!
//! 1. Compute per-edge reward = success rate of sessions that traversed it
//! 2. Group edges by kind, compute z-score advantages within each group
//! 3. Apply clipped exponential gradient update: w *= exp(clip(η·A, -ε, +ε))
//! 4. Track confidence based on update count

use crate::eval::agent::runner::Trace;
use crate::graph::engine::GraphEngine;
use crate::graph::models::EdgeKind;
use std::collections::HashMap;

/// Configuration for one GRPO optimization step.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Learning rate for weight updates.
    pub learning_rate: f64,
    /// PPO-style clipping epsilon for the log-weight ratio.
    pub clip_epsilon: f64,
    /// Minimum number of sessions that must traverse an edge before its
    /// weight is eligible for GRPO updates.
    pub min_traversals: usize,
    /// Minimum number of updates before confidence saturates at 1.0.
    pub min_updates_for_confidence: u64,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            clip_epsilon: 0.2,
            min_traversals: 1,
            min_updates_for_confidence: 5,
        }
    }
}

/// Stores per-edge statistics accumulated across a batch of sessions.
struct EdgeBatchStats {
    kind: EdgeKind,
    success_count: usize,
    total_traversals: usize,
}

impl EdgeBatchStats {
    fn reward(&self) -> f64 {
        if self.total_traversals == 0 {
            return 0.5; // neutral prior for unseen edges
        }
        self.success_count as f64 / self.total_traversals as f64
    }
}

/// Group Relative Policy Optimization for batch edge-weight updates.
pub struct GrpoOptimizer {
    config: GrpoConfig,
}

impl GrpoOptimizer {
    #[must_use]
    pub fn new(config: GrpoConfig) -> Self {
        Self { config }
    }

    /// Run one GRPO optimization step over a batch of `(trace, success)` pairs.
    ///
    /// Edges are identified as "traversed" when both endpoint files appear
    /// in the trace's accessed files. The reward for each edge is the fraction
    /// of traversing sessions that were successful.
    ///
    /// Returns the number of edges whose weights were updated.
    pub fn optimize(
        &self,
        graph: &mut GraphEngine,
        sessions: &[(Trace, bool)],
    ) -> usize {
        if sessions.is_empty() {
            return 0;
        }

        // 1. Build per-edge traversal statistics
        let all_edges = graph.all_edges();
        let mut stats: Vec<EdgeBatchStats> = all_edges
            .iter()
            .map(|e| EdgeBatchStats {
                kind: e.kind.clone(),
                success_count: 0,
                total_traversals: 0,
            })
            .collect();

        for (trace, success) in sessions {
            let distinct = trace.distinct_accessed_files();
            let files: Vec<&str> = distinct.iter().map(String::as_str).collect();
            for (i, edge) in all_edges.iter().enumerate() {
                if edge_traversed(edge, graph, &files) {
                    stats[i].total_traversals += 1;
                    if *success {
                        stats[i].success_count += 1;
                    }
                }
            }
        }

        // 2. Group edges by kind and compute advantages
        let mut kind_groups: HashMap<EdgeKind, Vec<usize>> = HashMap::new();
        for (i, s) in stats.iter().enumerate() {
            if s.total_traversals >= self.config.min_traversals {
                kind_groups.entry(s.kind.clone()).or_default().push(i);
            }
        }

        let mut updated = 0;

        for (_kind, indices) in &kind_groups {
            if indices.len() < 2 {
                // Need at least 2 edges in a group for meaningful z-score
                for &idx in indices {
                    let s = &stats[idx];
                    let advantage = if s.reward() > 0.5 { 0.5 } else { -0.5 };
                    apply_update(graph, idx, advantage, &self.config);
                    updated += 1;
                }
                continue;
            }

            let rewards: Vec<f64> = indices.iter().map(|&i| stats[i].reward()).collect();
            let n = rewards.len() as f64;
            let mean: f64 = rewards.iter().sum::<f64>() / n;
            let variance: f64 = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt().max(1e-8);

            for &idx in indices {
                let advantage = (stats[idx].reward() - mean) / std;
                apply_update(graph, idx, advantage, &self.config);
                updated += 1;
            }
        }

        updated
    }
}

/// Check whether an edge was traversed during a session.
///
/// An edge is traversed when both its source and target file paths
/// appear in the session's accessed files. File paths are retrieved
/// from node metadata (key "file").
fn edge_traversed(
    edge: &crate::graph::models::Edge,
    graph: &GraphEngine,
    accessed_files: &[&str],
) -> bool {
    let source_file = graph
        .get_node(&edge.source)
        .and_then(|n| n.get_meta("file"))
        .and_then(|m| match m {
            crate::graph::models::MetaValue::Text(s) => Some(s.as_str()),
            _ => None,
        });

    let target_file = graph
        .get_node(&edge.target)
        .and_then(|n| n.get_meta("file"))
        .and_then(|m| match m {
            crate::graph::models::MetaValue::Text(s) => Some(s.as_str()),
            _ => None,
        });

    match (source_file, target_file) {
        (Some(src), Some(tgt)) => {
            let src_hit = accessed_files.iter().any(|f| *f == src);
            let tgt_hit = accessed_files.iter().any(|f| *f == tgt);
            src_hit && tgt_hit
        }
        _ => false,
    }
}

/// Apply a clipped exponential gradient update to one edge.
///
/// ```text
/// g = clip(η · A, -ε, +ε)
/// w_new = w_old · exp(g)
/// confidence = min(1.0, update_count / min_updates_for_confidence)
/// ```
fn apply_update(graph: &mut GraphEngine, edge_index: usize, advantage: f64, config: &GrpoConfig) {
    let mut edges = graph.all_edges_mut();
    let Some(edge) = edges.get_mut(edge_index) else {
        return;
    };

    let gradient = config.learning_rate * advantage;
    let clipped = gradient.clamp(-config.clip_epsilon, config.clip_epsilon);
    edge.weight *= clipped.exp();
    edge.weight = edge.weight.clamp(0.01, 100.0);
    edge.update_count += 1;
    #[allow(clippy::cast_precision_loss)]
    {
        edge.confidence = (edge.update_count as f64 / config.min_updates_for_confidence as f64)
            .min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::agent::runner::{ToolCall, ToolKind};
    use crate::graph::models::{Edge, MetaValue, Node, NodeKind};

    fn make_trace(instance_id: &str, files: Vec<&str>, completed: bool) -> Trace {
        Trace {
            instance_id: instance_id.to_string(),
            tool_calls: vec![ToolCall {
                kind: ToolKind::Read,
                input: String::new(),
                accessed_files: files.into_iter().map(String::from).collect(),
            }],
            final_reading_context: vec![],
            completed,
        }
    }

    fn build_toy_graph() -> GraphEngine {
        let mut engine = GraphEngine::new();
        // Create 3 File nodes with metadata
        let f1 = {
            let mut n = Node::new(NodeKind::File, "file_a");
            n.set_meta("file", MetaValue::Text("a.py".to_string()));
            let id = n.id;
            engine.add_node(n);
            id
        };
        let f2 = {
            let mut n = Node::new(NodeKind::File, "file_b");
            n.set_meta("file", MetaValue::Text("b.py".to_string()));
            let id = n.id;
            engine.add_node(n);
            id
        };
        let f3 = {
            let mut n = Node::new(NodeKind::File, "file_c");
            n.set_meta("file", MetaValue::Text("c.py".to_string()));
            let id = n.id;
            engine.add_node(n);
            id
        };
        engine.add_edge(Edge::new(f1, f2, EdgeKind::CoAccessed));
        engine.add_edge(Edge::new(f2, f3, EdgeKind::CoAccessed));
        engine
    }

    #[test]
    fn grpo_updates_edges_from_batch() {
        let mut graph = build_toy_graph();
        let original_weight = {
            let edges = graph.all_edges();
            edges[0].weight
        };
        assert!((original_weight - 1.0).abs() < 1e-10, "initial weight should be 1.0");

        let sessions = vec![
            (make_trace("inst1", vec!["a.py", "b.py"], true), true),
            (make_trace("inst2", vec!["a.py", "b.py"], true), true),
            (make_trace("inst3", vec!["b.py", "c.py"], false), false),
        ];

        let optimizer = GrpoOptimizer::new(GrpoConfig {
            min_traversals: 1,
            ..Default::default()
        });
        let updated = optimizer.optimize(&mut graph, &sessions);
        assert!(updated > 0, "should update at least one edge");

        let edges = graph.all_edges();
        // Edge a.py ↔ b.py was traversed in 2 successful sessions → positive advantage
        assert!(edges[0].weight > 1.0, "successful edge should have increased weight, got {}", edges[0].weight);
        // Edge b.py ↔ c.py was traversed in 1 failed session → negative advantage
        assert!(edges[1].weight < 1.0, "failed edge should have decreased weight, got {}", edges[1].weight);
    }

    #[test]
    fn empty_batch_does_nothing() {
        let mut graph = build_toy_graph();
        let optimizer = GrpoOptimizer::new(GrpoConfig::default());
        let updated = optimizer.optimize(&mut graph, &[]);
        assert_eq!(updated, 0);
    }

    #[test]
    fn edge_weights_stay_in_bounds() {
        let mut graph = build_toy_graph();
        // 100 successful sessions for a.py↔b.py
        let sessions: Vec<(Trace, bool)> = (0..100)
            .map(|i| {
                (
                    make_trace(&format!("inst{i}"), vec!["a.py", "b.py"], true),
                    true,
                )
            })
            .collect();

        let optimizer = GrpoOptimizer::new(GrpoConfig {
            learning_rate: 0.5, // large LR to test bounds
            clip_epsilon: 0.2,
            min_traversals: 1,
            min_updates_for_confidence: 5,
        });
        optimizer.optimize(&mut graph, &sessions);
        let edges = graph.all_edges();
        assert!(edges[0].weight <= 100.0, "weight should not exceed upper bound");
        assert!(edges[0].weight >= 0.01, "weight should not go below lower bound");
    }

    #[test]
    fn grpo_config_defaults() {
        let config = GrpoConfig::default();
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
        assert!((config.clip_epsilon - 0.2).abs() < 1e-10);
        assert_eq!(config.min_traversals, 1);
        assert_eq!(config.min_updates_for_confidence, 5);
    }
}
