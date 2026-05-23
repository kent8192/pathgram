//! Ported from kent8192/engramoid `core/src/graph/engine.rs` (Phase 1) into pathgram.

use crate::graph::models::{Edge, EdgeKind, Node, NodeId, NodeKind};
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

pub struct GraphEngine {
    graph: StableGraph<Node, Edge>,
    node_index_map: HashMap<NodeId, NodeIndex>,
}

impl GraphEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            node_index_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) -> NodeIndex {
        let id = node.id;
        let idx = self.graph.add_node(node);
        self.node_index_map.insert(id, idx);
        idx
    }

    #[must_use]
    pub fn get_node(&self, id: &NodeId) -> Option<&Node> {
        self.node_index_map
            .get(id)
            .and_then(|idx| self.graph.node_weight(*idx))
    }

    pub fn get_node_mut(&mut self, id: &NodeId) -> Option<&mut Node> {
        self.node_index_map
            .get(id)
            .copied()
            .and_then(|idx| self.graph.node_weight_mut(idx))
    }

    pub fn add_edge(&mut self, edge: Edge) -> bool {
        let source_idx = self.node_index_map.get(&edge.source).copied();
        let target_idx = self.node_index_map.get(&edge.target).copied();
        if let (Some(s), Some(t)) = (source_idx, target_idx) {
            self.graph.add_edge(s, t, edge);
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn get_neighbors(&self, id: &NodeId) -> Vec<(NodeId, &Edge)> {
        let Some(&idx) = self.node_index_map.get(id) else {
            return Vec::new();
        };
        self.graph
            .edges(idx)
            .map(|e| {
                let target_idx = e.target();
                let target_node = self.graph.node_weight(target_idx).expect("target exists");
                (target_node.id, e.weight())
            })
            .collect()
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    #[must_use]
    pub fn nodes_by_kind(&self, kind: &NodeKind) -> Vec<&Node> {
        self.graph
            .node_weights()
            .filter(|n| &n.kind == kind)
            .collect()
    }

    #[must_use]
    pub fn all_nodes(&self) -> Vec<&Node> {
        self.graph.node_weights().collect()
    }

    #[must_use]
    pub fn all_edges(&self) -> Vec<&Edge> {
        self.graph.edge_weights().collect()
    }

    /// Returns mutable references to all edges. The references are
    /// non-overlapping (guaranteed by petgraph), so multiple `&mut Edge`
    /// can coexist in the returned Vec.
    pub fn all_edges_mut(&mut self) -> Vec<&mut Edge> {
        self.graph.edge_weights_mut().collect()
    }

    pub fn load_from(&mut self, nodes: Vec<Node>, edges: Vec<Edge>) {
        self.graph.clear();
        self.node_index_map.clear();
        for node in nodes {
            self.add_node(node);
        }
        for edge in edges {
            self.add_edge(edge);
        }
    }

    pub fn record_access(&mut self, id: &NodeId) {
        if let Some(node) = self.get_node_mut(id) {
            node.record_access();
        }
    }

    /// Find and mutate an edge between `source` and `target` of the given
    /// `kind`, checking both directions. Returns `true` if an edge was
    /// found and the closure was applied.
    pub fn update_edge<F>(&mut self, source: &NodeId, target: &NodeId, kind: &EdgeKind, f: F) -> bool
    where
        F: FnOnce(&mut Edge),
    {
        let mut found = false;
        for edge in self.graph.edge_weights_mut() {
            if edge.kind == *kind
                && ((&edge.source == source && &edge.target == target)
                    || (&edge.source == target && &edge.target == source))
            {
                f(edge);
                found = true;
                break;
            }
        }
        found
    }
}

impl Default for GraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::EdgeKind;

    #[test]
    fn test_add_and_get_node() {
        let mut engine = GraphEngine::new();
        let node = Node::new(NodeKind::File, "Main entry point");
        let id = node.id;
        engine.add_node(node);
        let retrieved = engine.get_node(&id).expect("node added");
        assert_eq!(retrieved.summary, "Main entry point");
    }

    #[test]
    fn test_add_edge() {
        let mut engine = GraphEngine::new();
        let n1 = Node::new(NodeKind::File, "file A");
        let n2 = Node::new(NodeKind::Function, "function B");
        let id1 = n1.id;
        let id2 = n2.id;
        engine.add_node(n1);
        engine.add_node(n2);
        engine.add_edge(Edge::new(id1, id2, EdgeKind::Contains));
        let neighbors = engine.get_neighbors(&id1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, id2);
    }

    #[test]
    fn test_graph_stats() {
        let mut engine = GraphEngine::new();
        let n1 = Node::new(NodeKind::File, "A");
        let n2 = Node::new(NodeKind::Function, "B");
        let id1 = n1.id;
        let id2 = n2.id;
        engine.add_node(n1);
        engine.add_node(n2);
        engine.add_edge(Edge::new(id1, id2, EdgeKind::Contains));
        assert_eq!(engine.node_count(), 2);
        assert_eq!(engine.edge_count(), 1);
    }

    #[test]
    fn test_nodes_by_kind() {
        let mut engine = GraphEngine::new();
        engine.add_node(Node::new(NodeKind::File, "A"));
        engine.add_node(Node::new(NodeKind::File, "B"));
        engine.add_node(Node::new(NodeKind::Function, "C"));
        let files = engine.nodes_by_kind(&NodeKind::File);
        assert_eq!(files.len(), 2);
        let functions = engine.nodes_by_kind(&NodeKind::Function);
        assert_eq!(functions.len(), 1);
    }

    #[test]
    fn test_load_from() {
        let mut engine = GraphEngine::new();
        let n1 = Node::new(NodeKind::File, "A");
        let n2 = Node::new(NodeKind::Function, "B");
        let id1 = n1.id;
        let id2 = n2.id;
        let edge = Edge::new(id1, id2, EdgeKind::Contains);
        engine.load_from(vec![n1, n2], vec![edge]);
        assert_eq!(engine.node_count(), 2);
        assert_eq!(engine.edge_count(), 1);
    }

    #[test]
    fn test_record_access() {
        let mut engine = GraphEngine::new();
        let node = Node::new(NodeKind::File, "test");
        let id = node.id;
        engine.add_node(node);
        engine.record_access(&id);
        engine.record_access(&id);
        assert_eq!(engine.get_node(&id).expect("present").access_count, 2);
    }
}
