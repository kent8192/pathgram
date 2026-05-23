//! Bayesian-Hebbian online learning layer.
//!
//! - `hebbian` — Beta posterior edge-weight model with closed-form updates
//! - `gram_score` — rolling score tracker with shadow/canary/mature gating
//! - `session` — per-session Hebbian update processing on CoAccessed edges

pub mod gram_score;
pub mod grpo;
pub mod hebbian;
pub mod session;
