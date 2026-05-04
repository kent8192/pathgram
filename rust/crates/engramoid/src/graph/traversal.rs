//! Ported from kent8192/engramoid `core/src/graph/traversal.rs` (Phase 1).

use crate::graph::engine::GraphEngine;
use crate::graph::models::NodeId;
use std::collections::HashMap;

/// Personalized PageRank from seed nodes.
///
/// Computes relevance scores for all nodes in the graph, biased toward
/// the given seed nodes. Uses edge weights and confidence values to
/// determine transition probabilities. Dangling nodes (no outgoing edges)
/// redistribute their score back to the personalization vector.
///
/// Returns `(NodeId, score)` pairs sorted by score descending, filtered
/// to only include nodes with a positive score.
#[must_use]
pub fn personalized_pagerank(
    engine: &GraphEngine,
    seed_nodes: &[NodeId],
    damping: f64,
    iterations: usize,
) -> Vec<(NodeId, f64)> {
    let all_nodes = engine.all_nodes();
    if all_nodes.is_empty() || seed_nodes.is_empty() {
        return Vec::new();
    }

    let n = all_nodes.len();
    let node_ids: Vec<NodeId> = all_nodes.iter().map(|n| n.id).collect();
    let id_to_idx: HashMap<NodeId, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Personalization vector: uniform over seed nodes
    let mut personalization = vec![0.0; n];
    for seed in seed_nodes {
        if let Some(&idx) = id_to_idx.get(seed) {
            #[allow(clippy::cast_precision_loss)]
            let p = 1.0 / seed_nodes.len() as f64;
            personalization[idx] = p;
        }
    }

    let mut scores = personalization.clone();

    for _ in 0..iterations {
        let mut new_scores = vec![0.0; n];

        for (i, node_id) in node_ids.iter().enumerate() {
            let neighbors = engine.get_neighbors(node_id);
            if neighbors.is_empty() {
                // Dangling node: distribute to personalization
                for (j, p) in personalization.iter().enumerate() {
                    new_scores[j] += scores[i] * p;
                }
            } else {
                let total_weight: f64 = neighbors
                    .iter()
                    .map(|(_, e)| e.weight * e.confidence)
                    .sum();
                if total_weight > 0.0 {
                    for (target_id, edge) in &neighbors {
                        if let Some(&j) = id_to_idx.get(target_id) {
                            let prob = (edge.weight * edge.confidence) / total_weight;
                            new_scores[j] += damping * scores[i] * prob;
                        }
                    }
                }
            }
        }

        // Add teleportation
        for (j, score) in new_scores.iter_mut().enumerate() {
            *score += (1.0 - damping) * personalization[j];
        }

        scores = new_scores;
    }

    let mut result: Vec<(NodeId, f64)> = node_ids
        .into_iter()
        .zip(scores)
        .filter(|(_, s)| *s > 0.0)
        .collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::{Edge, EdgeKind, Node, NodeKind};

    #[test]
    fn test_pagerank_basic() {
        let mut engine = GraphEngine::new();
        let a = Node::new(NodeKind::File, "A");
        let b = Node::new(NodeKind::Function, "B");
        let c = Node::new(NodeKind::Function, "C");
        let id_a = a.id;
        let id_b = b.id;
        let id_c = c.id;
        engine.add_node(a);
        engine.add_node(b);
        engine.add_node(c);
        engine.add_edge(Edge::new(id_a, id_b, EdgeKind::Contains));
        engine.add_edge(Edge::new(id_a, id_c, EdgeKind::Contains));
        engine.add_edge(Edge::new(id_b, id_c, EdgeKind::Calls));

        let scores = personalized_pagerank(&engine, &[id_a], 0.85, 20);
        assert!(!scores.is_empty());
        let a_score = scores
            .iter()
            .find(|(id, _)| *id == id_a)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(a_score > 0.0);
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let engine = GraphEngine::new();
        let scores = personalized_pagerank(&engine, &[], 0.85, 20);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_pagerank_connected_nodes_have_higher_score() {
        let mut engine = GraphEngine::new();
        let a = Node::new(NodeKind::File, "A");
        let b = Node::new(NodeKind::Function, "B");
        let c = Node::new(NodeKind::Function, "C"); // disconnected
        let id_a = a.id;
        let id_b = b.id;
        let id_c = c.id;
        engine.add_node(a);
        engine.add_node(b);
        engine.add_node(c);
        engine.add_edge(Edge::new(id_a, id_b, EdgeKind::Contains));

        let scores = personalized_pagerank(&engine, &[id_a], 0.85, 20);
        let b_score = scores
            .iter()
            .find(|(id, _)| *id == id_b)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let c_score = scores
            .iter()
            .find(|(id, _)| *id == id_c)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            b_score > c_score,
            "Connected node B ({b_score}) should score higher than disconnected C ({c_score})"
        );
    }
}
