//! engramoid — in-process knowledge-graph + retrieval substrate for pathgram.
//!
//! Ported from the standalone `kent8192/engramoid` Phase 1 foundation. The
//! port drops the PostgreSQL backend, the gRPC server, and the TypeScript
//! plugin layer; it keeps the in-memory graph, Personalized PageRank,
//! tree-sitter-driven node taxonomy, and tracker data model. Phase 2 work
//! (the gram探索 retriever, GRPO learner, eval harness) is being built
//! directly on top of this in-process crate.
//!
//! See `docs/superpowers/specs/2026-05-04-engramoid-gram-search-design.md`
//! for the design.

pub mod graph;
pub mod tracker;

#[cfg(feature = "eval")]
pub mod eval;

#[cfg(feature = "eval")]
pub mod retrieval;

#[cfg(feature = "eval")]
pub mod scorers;
