//! Rolling gram-score tracker with shadow/canary/mature mode gating.
//!
//! The gram score is computed as the mean weight of retrieved edges.
//! A rolling window of recent scores determines whether the Hebbian layer
//! is operating in shadow (silent), canary (sentinel-only), or mature
//! (full-return) mode.

use crate::graph::models::Edge;
use std::collections::VecDeque;

/// Operational mode for the Hebbian retrieval layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobMode {
    /// Run pipeline silently; do not return a blob. Used for cold-start.
    Shadow,
    /// Return a sentinel blob (empty) to signal the layer is warming.
    Canary,
    /// Full return — the Hebbian layer is mature and confident.
    Mature,
}

/// Tracks a rolling window of gram scores and decides the current
/// operational mode based on score stability and level.
pub struct GramScoreTracker {
    pub window_size: usize,
    recent_scores: VecDeque<f64>,
}

impl GramScoreTracker {
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            recent_scores: VecDeque::with_capacity(window_size),
        }
    }

    /// Compute the gram score from a set of retrieved edges.
    ///
    /// The gram score is the mean of `edge.weight * edge.confidence`
    /// across all provided edges. Returns 0.0 if no edges are provided.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, retrieved_edges: &[&Edge]) -> f64 {
        if retrieved_edges.is_empty() {
            return 0.0;
        }
        let sum: f64 = retrieved_edges
            .iter()
            .map(|e| e.weight * e.confidence)
            .sum();
        sum / retrieved_edges.len() as f64
    }

    /// Record a gram score into the rolling window.
    pub fn record(&mut self, score: f64) {
        if self.recent_scores.len() >= self.window_size {
            self.recent_scores.pop_front();
        }
        self.recent_scores.push_back(score);
    }

    /// Rolling mean of recorded scores. Returns 0.0 if no scores recorded.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rolling_mean(&self) -> f64 {
        if self.recent_scores.is_empty() {
            return 0.0;
        }
        self.recent_scores.iter().sum::<f64>() / self.recent_scores.len() as f64
    }

    /// Decide the current operational mode.
    ///
    /// - `Shadow`: fewer than `window_size / 2` scores recorded (warming up).
    /// - `Canary`: rolling mean < `threshold_gate` (not yet reliable).
    /// - `Mature`: rolling mean ≥ `threshold_gate` (confident).
    #[must_use]
    pub fn mode(&self, threshold_gate: f64) -> BlobMode {
        let half_window = self.window_size / 2;
        if self.recent_scores.len() < half_window {
            return BlobMode::Shadow;
        }
        if self.rolling_mean() < threshold_gate {
            return BlobMode::Canary;
        }
        BlobMode::Mature
    }
}

impl Default for GramScoreTracker {
    fn default() -> Self {
        Self::new(20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::models::{Edge, EdgeKind, NodeId};

    #[test]
    fn compute_returns_mean_weighted_score() {
        let tracker = GramScoreTracker::default();
        let a = NodeId::new();
        let b = NodeId::new();
        let mut e1 = Edge::new(a, b, EdgeKind::CoAccessed);
        e1.weight = 0.8;
        e1.confidence = 0.9;
        let mut e2 = Edge::new(a, b, EdgeKind::CoAccessed);
        e2.weight = 0.4;
        e2.confidence = 0.5;

        let score = tracker.compute(&[&e1, &e2]);
        let expected = (0.8 * 0.9 + 0.4 * 0.5) / 2.0;
        assert!((score - expected).abs() < 1e-10);
    }

    #[test]
    fn compute_empty_edges_returns_zero() {
        let tracker = GramScoreTracker::default();
        assert!((tracker.compute(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn record_and_rolling_mean() {
        let mut tracker = GramScoreTracker::new(5);
        tracker.record(0.5);
        tracker.record(0.7);
        tracker.record(0.6);
        assert!((tracker.rolling_mean() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn mode_progression() {
        let mut tracker = GramScoreTracker::new(4);
        // Not enough data yet (1 < half_window=2) → Shadow
        tracker.record(0.05);
        assert_eq!(tracker.mode(0.3), BlobMode::Shadow);

        // Enough data but low mean → Canary
        tracker.record(0.05);
        assert!(tracker.rolling_mean() < 0.3);
        assert_eq!(tracker.mode(0.3), BlobMode::Canary);

        // Above threshold → Mature
        tracker.record(0.9);
        tracker.record(0.9);
        assert!(tracker.rolling_mean() >= 0.3);
        assert_eq!(tracker.mode(0.3), BlobMode::Mature);
    }

    #[test]
    fn window_evicts_old_scores() {
        let mut tracker = GramScoreTracker::new(3);
        tracker.record(1.0);
        tracker.record(1.0);
        tracker.record(1.0);
        tracker.record(0.0);
        tracker.record(0.0);
        // Window has: [1.0, 0.0, 0.0] after 5 records
        assert!((tracker.rolling_mean() - 1.0 / 3.0).abs() < 1e-10);
    }
}
