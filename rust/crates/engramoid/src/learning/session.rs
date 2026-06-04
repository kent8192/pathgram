//! Per-session Hebbian update processing.
//!
//! After each coding session, the set of files that were useful (retrieved
//! and contributed to a successful outcome) is used to update `CoAccessed`
//! edges in the graph. On success, pairwise α is incremented; on failure,
//! β is incremented. Edge weight and confidence are recomputed from the
//! posterior.

use crate::graph::engine::GraphEngine;
use crate::graph::models::{Edge, EdgeKind, NodeId, NodeKind};
use crate::learning::hebbian::BetaPosterior;

/// Process one session trace into the graph via Hebbian updates.
///
/// For every unordered pair of useful file paths, the corresponding
/// `File` node is found (or created), and a `CoAccessed` edge is
/// created (or updated) between them:
///
/// ```text
/// weight_factor = ln(1 + N / (update_count + 1))
/// if success: α += η · weight_factor
/// else:       β += η · weight_factor
/// weight ← α / (α + β)
/// confidence ← 1 − 4·Var[Beta(α, β)]
/// ```
///
/// `useful_file_paths` should be relative paths rooted at the repo
/// (e.g. `"src/validators.py"`).
pub fn process_session(
    graph: &mut GraphEngine,
    useful_file_paths: &[String],
    success: bool,
    eta: f64,
) {
    if useful_file_paths.len() < 2 {
        return;
    }

    // Resolve file paths → NodeIds for File-kind nodes
    let file_nodes: Vec<NodeId> = useful_file_paths
        .iter()
        .map(|path| find_or_create_file_node(graph, path))
        .collect();

    let n = file_nodes.len();
    #[allow(clippy::cast_precision_loss)]
    let weight_factor = (1.0 + n as f64 / 2.0).ln().max(0.0);

    for i in 0..n {
        for j in (i + 1)..n {
            let a = file_nodes[i];
            let b = file_nodes[j];
            update_coaccessed_edge(graph, a, b, success, weight_factor, eta);
        }
    }
}

/// Find a `File` node by path name, or create one if missing.
fn find_or_create_file_node(graph: &mut GraphEngine, path: &str) -> NodeId {
    for node in graph.all_nodes() {
        if node.kind == NodeKind::File && node.name == path {
            return node.id;
        }
    }
    let mut node = crate::graph::models::Node::new(NodeKind::File, &format!("File: {path}"));
    node.name = path.to_string();
    let id = node.id;
    graph.add_node(node);
    id
}

/// Update (or create) a `CoAccessed` edge between two `File` nodes.
fn update_coaccessed_edge(
    graph: &mut GraphEngine,
    a: NodeId,
    b: NodeId,
    success: bool,
    weight_factor: f64,
    eta: f64,
) {
    let found = graph.update_edge(&a, &b, &EdgeKind::CoAccessed, |edge| {
        let mut posterior = posterior_from_edge(edge);
        posterior.update(success, weight_factor, eta);
        edge.weight = posterior.weight();
        edge.weight = edge.weight.clamp(0.001, 0.999);
        edge.confidence = posterior.confidence();
        edge.alpha = posterior.alpha;
        edge.beta = posterior.beta;
        edge.update_count += 1;
    });

    if !found {
        // Create new edge with pessimistic prior, then apply this update
        let mut posterior = BetaPosterior::default();
        posterior.update(success, weight_factor, eta);
        let mut edge = Edge::new(a, b, EdgeKind::CoAccessed);
        edge.weight = posterior.weight().clamp(0.001, 0.999);
        edge.confidence = posterior.confidence();
        edge.alpha = posterior.alpha;
        edge.beta = posterior.beta;
        edge.base_weight = edge.weight;
        edge.update_count = 1;
        graph.add_edge(edge);
    }
}

/// Reconstruct a `BetaPosterior` from an existing edge's alpha/beta fields.
fn posterior_from_edge(edge: &Edge) -> BetaPosterior {
    BetaPosterior::new(edge.alpha, edge.beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::Node;

    fn build_test_graph(file_paths: &[&str]) -> GraphEngine {
        let mut engine = GraphEngine::new();
        for path in file_paths {
            let mut node = Node::new(NodeKind::File, &format!("File: {path}"));
            node.name = (*path).to_string();
            engine.add_node(node);
        }
        engine
    }

    #[test]
    fn session_creates_coaccessed_edges() {
        let mut engine = build_test_graph(&["a.py", "b.py", "c.py"]);
        let paths: Vec<String> = vec!["a.py".into(), "b.py".into(), "c.py".into()];
        process_session(&mut engine, &paths, true, 0.1);

        let coaccessed: Vec<&Edge> = engine
            .all_edges()
            .iter()
            .filter(|e| e.kind == EdgeKind::CoAccessed)
            .copied()
            .collect();
        assert_eq!(coaccessed.len(), 3, "3 pairs from 3 files");
        for e in &coaccessed {
            assert!(e.weight > 0.0);
            assert!(e.update_count > 0);
        }
    }

    #[test]
    fn session_less_than_two_files_is_noop() {
        let mut engine = build_test_graph(&["a.py"]);
        let paths: Vec<String> = vec!["a.py".into()];
        process_session(&mut engine, &paths, true, 0.1);
        assert_eq!(engine.edge_count(), 0);
    }

    #[test]
    fn failure_increases_beta_decreases_weight() {
        let mut engine = build_test_graph(&["x.py", "y.py"]);
        let paths: Vec<String> = vec!["x.py".into(), "y.py".into()];

        // First, a success to establish an edge
        process_session(&mut engine, &paths, true, 0.1);
        let weight_after_success = engine
            .all_edges()
            .iter()
            .find(|e| e.kind == EdgeKind::CoAccessed)
            .map(|e| e.weight)
            .unwrap_or(0.0);

        // Then failures
        for _ in 0..5 {
            process_session(&mut engine, &paths, false, 0.1);
        }
        let weight_after_failures = engine
            .all_edges()
            .iter()
            .find(|e| e.kind == EdgeKind::CoAccessed)
            .map(|e| e.weight)
            .unwrap_or(0.0);

        assert!(
            weight_after_failures < weight_after_success,
            "failure should decrease weight: {weight_after_success} → {weight_after_failures}"
        );
    }

    #[test]
    fn repeated_success_increases_weight() {
        let mut engine = build_test_graph(&["p.py", "q.py"]);
        let paths: Vec<String> = vec!["p.py".into(), "q.py".into()];

        let mut last_weight = 0.0;
        for _ in 0..20 {
            process_session(&mut engine, &paths, true, 0.1);
            let current = engine
                .all_edges()
                .iter()
                .find(|e| e.kind == EdgeKind::CoAccessed)
                .map(|e| e.weight)
                .unwrap_or(0.0);
            assert!(current >= last_weight, "weight should be monotonic non-decreasing");
            last_weight = current;
        }
        assert!(last_weight > 0.4, "weight should grow past 0.4 after 20 successes");
    }
}
