//! Phase 2.0 evaluation harness for the in-process pathgram port.
//!
//! Gated behind `--features eval`. Provides SWE-bench Lite/Gym JSONL
//! loaders, a tempdir-and-git-clone sandbox, an `AgentRunner` trait with a
//! deterministic baseline implementation (no LLM dependency), the three
//! primary Phase 2 metrics, paired-bootstrap CIs, and an end-to-end
//! `EvalRunner` orchestrator.

pub mod agent;
pub mod bootstrap;
pub mod instance;
pub mod loaders;
pub mod metrics;
pub mod runner;
pub mod sandbox;
