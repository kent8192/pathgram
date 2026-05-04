//! Phase 2.1 frozen γ retrieval pipeline (chunking, candidate generation,
//! reranking, blob packing). Implementation lands across submodules:
//!
//! - `chunks` — file → chunk extraction (Phase 2.1: line-based with overlap)
//! - `embedding_store` — in-memory cosine top-N
//! - `packer` — token-budgeted blob construction
//! - `pipeline` — wires chunks + embedder + reranker + packer
//!
//! See `crates/engramoid/src/agent/frozen_gamma.rs` for the AgentRunner
//! implementation that exposes this pipeline as a single-step retrieval.

pub mod chunks;
pub mod embedding_store;
pub mod packer;
pub mod pipeline;
