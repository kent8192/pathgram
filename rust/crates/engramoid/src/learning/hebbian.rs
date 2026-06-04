//! Beta posterior for edge-weight learning.
//!
//! Each edge maintains a Beta(α, β) posterior over its co-occurrence
//! reliability. On a successful session, α is incremented; on failure,
//! β is incremented. The posterior mean = weight, and the variance
//! drives the confidence term used in PPR transition probabilities.

use serde::{Deserialize, Serialize};

/// Beta posterior for one edge's co-occurrence reliability.
///
/// Default prior: Beta(0.5, 1.5) — weakly pessimistic (mean 0.25),
/// meaning an edge is assumed unreliable until evidence accumulates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaPosterior {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaPosterior {
    #[must_use]
    pub fn new(prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            alpha: prior_alpha,
            beta: prior_beta,
        }
    }

    /// Posterior mean: α / (α + β).
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Confidence = 1 − 4·Var[Beta(α,β)], clamped to [0, 1].
    ///
    /// Var = (α·β) / ((α+β)² · (α+β+1))
    #[must_use]
    pub fn confidence(&self) -> f64 {
        let total = self.alpha + self.beta;
        if total <= 0.0 {
            return 0.0;
        }
        let var = (self.alpha * self.beta) / (total * total * (total + 1.0));
        (1.0 - 4.0 * var).clamp(0.0, 1.0)
    }

    /// Closed-form Beta entropy using a digamma approximation.
    ///
    /// H[Beta(α,β)] = ln B(α,β) − (α−1)·ψ(α) − (β−1)·ψ(β) + (α+β−2)·ψ(α+β)
    ///
    /// Uses the asymptotic expansion ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²)
    /// which has < 0.01 % error for x > 2.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        let total = a + b;
        if a <= 0.0 || b <= 0.0 {
            return f64::INFINITY;
        }
        let ln_beta_ab = ln_gamma(a) + ln_gamma(b) - ln_gamma(total);
        ln_beta_ab - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b)
            + (total - 2.0) * digamma(total)
    }

    /// Apply one Hebbian update.
    ///
    /// On success: `α += η · weight_factor`
    /// On failure: `β += η · weight_factor`
    pub fn update(&mut self, success: bool, weight_factor: f64, eta: f64) {
        let delta = eta * weight_factor;
        if success {
            self.alpha += delta;
        } else {
            self.beta += delta;
        }
    }
}

impl Default for BetaPosterior {
    fn default() -> Self {
        Self::new(0.5, 1.5)
    }
}

// ── digamma approximation ─────────────────────────────────────────────

/// Asymptotic expansion of the digamma function ψ(x).
///
/// ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶)
///
/// Error < 0.01 % for x > 2.
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    let inv_x4 = inv_x2 * inv_x2;
    let inv_x6 = inv_x4 * inv_x2;
    x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x4 / 120.0 - inv_x6 / 252.0
}

/// Stirling-based log-Gamma approximation.
///
/// ln Γ(x) ≈ (x − 0.5)·ln(x) − x + 0.5·ln(2π) + 1/(12x)
///
/// Accurate to ~10⁻⁶ for x > 0.5.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    let half_ln_2pi = 0.5 * std::f64::consts::TAU.ln();
    (x - 0.5) * x.ln() - x + half_ln_2pi + 1.0 / (12.0 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_prior_is_pessimistic() {
        let p = BetaPosterior::default();
        assert!(p.weight() < 0.5, "default prior should be pessimistic");
        assert!((p.weight() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn update_success_increases_weight() {
        let mut p = BetaPosterior::default();
        let w_before = p.weight();
        p.update(true, 1.0, 0.1);
        assert!(p.weight() > w_before);
    }

    #[test]
    fn update_failure_decreases_weight() {
        let mut p = BetaPosterior::default();
        let w_before = p.weight();
        p.update(false, 1.0, 0.1);
        assert!(p.weight() < w_before);
    }

    #[test]
    fn repeated_success_converges_toward_one() {
        let mut p = BetaPosterior::default();
        for _ in 0..200 {
            p.update(true, 1.0, 0.1);
        }
        assert!(p.weight() > 0.90, "weight should converge toward 1.0");
    }

    #[test]
    fn repeated_failure_converges_toward_zero() {
        let mut p = BetaPosterior::default();
        for _ in 0..200 {
            p.update(false, 1.0, 0.1);
        }
        assert!(p.weight() < 0.10, "weight should converge toward 0.0");
    }

    #[test]
    fn confidence_increases_with_more_evidence() {
        let mut p = BetaPosterior::default();
        let conf_initial = p.confidence();
        for _ in 0..50 {
            p.update(true, 1.0, 0.1);
        }
        assert!(p.confidence() > conf_initial);
    }

    #[test]
    fn confidence_clamped_to_unit_range() {
        let p = BetaPosterior::new(0.1, 0.1);
        assert!((0.0..=1.0).contains(&p.confidence()));
        let p2 = BetaPosterior::new(100.0, 1.0);
        assert!((0.0..=1.0).contains(&p2.confidence()));
    }

    #[test]
    fn entropy_finite_for_valid_params() {
        let p = BetaPosterior::default();
        assert!(p.entropy().is_finite());
    }

    #[test]
    fn weight_factor_scales_update() {
        let mut p = BetaPosterior::default();
        let w_before = p.weight();
        p.update(true, 2.0, 0.1);
        let delta_small = p.weight() - w_before;

        let mut p2 = BetaPosterior::default();
        p2.update(true, 1.0, 0.1);
        let delta_half = p2.weight() - w_before;

        assert!(delta_small > delta_half, "higher weight_factor → larger change");
    }
}
