//! Ported verbatim from kent8192/engramoid `core/src/graph/models.rs`
//! (Phase 1 foundation, branch feat/phase1-foundation) into pathgram.
//!
//! Changes vs upstream:
//! - none (data model is identical; in-process semantics depend only on these
//!   structs)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    File,
    Module,
    Class,
    Function,
    Chunk,
    Crate,
    ApiSurface,
    UsagePattern,
    ArchDecision,
    Convention,
    DomainTerm,
    BugPattern,
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetaValue {
    Text(String),
    Number(f64),
    Bool(bool),
    List(Vec<MetaValue>),
    Map(HashMap<String, MetaValue>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub name: String,
    pub summary: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, MetaValue>,
    pub access_count: u64,
    pub last_accessed: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
}

impl Node {
    #[must_use]
    pub fn new(kind: NodeKind, summary: &str) -> Self {
        let now = Utc::now();
        Self {
            id: NodeId::new(),
            kind,
            name: String::new(),
            summary: summary.to_string(),
            embedding: None,
            metadata: HashMap::new(),
            access_count: 0,
            last_accessed: now,
            created_at: now,
        }
    }

    pub fn set_meta(&mut self, key: &str, value: MetaValue) {
        self.metadata.insert(key.to_string(), value);
    }

    #[must_use]
    pub fn get_meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    Contains,
    Imports,
    Calls,
    Inherits,
    Implements,
    References,
    SimilarTo,
    CoAccessed,
    LeadsTo,
    UsedWith,
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
    pub weight: f64,
    pub base_weight: f64,
    pub confidence: f64,
    pub update_count: u64,
    pub alpha: f64,
    pub beta: f64,
}

impl Edge {
    #[must_use]
    pub fn new(source: NodeId, target: NodeId, kind: EdgeKind) -> Self {
        Self {
            source,
            target,
            kind,
            weight: 1.0,
            base_weight: 1.0,
            confidence: 1.0,
            update_count: 0,
            alpha: 0.5,
            beta: 1.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new(NodeKind::File, "Main application entry point");
        assert!(matches!(node.kind, NodeKind::File));
        assert_eq!(node.summary, "Main application entry point");
        assert_eq!(node.access_count, 0);
        assert!(node.embedding.is_none());
    }

    #[test]
    fn test_node_metadata() {
        let mut node = Node::new(NodeKind::Function, "Parses config file");
        node.set_meta("params", MetaValue::Text("path: &str".into()));
        node.set_meta("is_public", MetaValue::Bool(true));
        assert_eq!(
            node.get_meta("params"),
            Some(&MetaValue::Text("path: &str".into()))
        );
        assert_eq!(node.get_meta("is_public"), Some(&MetaValue::Bool(true)));
        assert_eq!(node.get_meta("nonexistent"), None);
    }

    #[test]
    fn test_edge_creation() {
        let source = NodeId::new();
        let target = NodeId::new();
        let edge = Edge::new(source, target, EdgeKind::Contains);
        assert!((edge.weight - 1.0).abs() < f64::EPSILON);
        assert!((edge.base_weight - 1.0).abs() < f64::EPSILON);
        assert!((edge.confidence - 1.0).abs() < f64::EPSILON);
        assert_eq!(edge.update_count, 0);
    }

    #[test]
    fn test_node_record_access() {
        let mut node = Node::new(NodeKind::File, "test file");
        assert_eq!(node.access_count, 0);
        node.record_access();
        assert_eq!(node.access_count, 1);
        node.record_access();
        assert_eq!(node.access_count, 2);
    }
}
