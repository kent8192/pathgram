use crate::eval::metrics::MetricRecord;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Paired bootstrap CI for the mean of `(b_i - a_i)` at the requested
/// confidence level. Returns `(lower, upper)`.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn paired_bootstrap_ci(
    a: &[f64],
    b: &[f64],
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "a and b must be paired");
    let n = a.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<usize> = (0..n).collect();
    let mut deltas: Vec<f64> = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let &i = indices.choose(&mut rng).expect("non-empty");
            sum += b[i] - a[i];
        }
        deltas.push(sum / n as f64);
    }
    deltas.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = (1.0 - confidence) / 2.0;
    let lo = deltas[(n_resamples as f64 * alpha) as usize];
    let hi = deltas[(n_resamples as f64 * (1.0 - alpha)) as usize];
    (lo, hi)
}

pub struct PrimaryMetricCi {
    pub step_reduction: (f64, f64),
    pub recall_at_5: (f64, f64),
    /// `None` when any pair has missing coverage on either side (e.g. one
    /// runner is default-only).
    pub coverage: Option<(f64, f64)>,
}

#[must_use]
pub fn paired_primary_ci(
    a: &[MetricRecord],
    b: &[MetricRecord],
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> PrimaryMetricCi {
    // Filter to pairs where both sides have a meaningful step_reduction.
    // Pairs with None on either side (Shadow mode, API failure) are excluded.
    let step_pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(ai, bi)| Some((ai.step_reduction?, bi.step_reduction?)))
        .collect();
    let (a_step, b_step): (Vec<f64>, Vec<f64>) = step_pairs.into_iter().unzip();
    let a_recall: Vec<f64> = a.iter().map(|m| m.recall_at_5).collect();
    let b_recall: Vec<f64> = b.iter().map(|m| m.recall_at_5).collect();

    // Coverage CI requires both sides to have a value at each pair index.
    let cov_pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter_map(|(ai, bi)| Some((ai.coverage?, bi.coverage?)))
        .collect();
    let coverage = if cov_pairs.is_empty() {
        None
    } else {
        let (a_cov, b_cov): (Vec<f64>, Vec<f64>) = cov_pairs.into_iter().unzip();
        Some(paired_bootstrap_ci(&a_cov, &b_cov, n_resamples, confidence, seed))
    };

    PrimaryMetricCi {
        step_reduction: paired_bootstrap_ci(&a_step, &b_step, n_resamples, confidence, seed),
        recall_at_5: paired_bootstrap_ci(&a_recall, &b_recall, n_resamples, confidence, seed),
        coverage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_brackets_zero_when_a_equals_b() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = a.clone();
        let (lo, hi) = paired_bootstrap_ci(&a, &b, 1000, 0.95, 42);
        assert!(lo <= 0.0 && hi >= 0.0, "got ({lo}, {hi})");
    }

    #[test]
    fn ci_strictly_positive_when_b_dominates() {
        let a = vec![0.0; 50];
        let b = vec![1.0; 50];
        let (lo, _hi) = paired_bootstrap_ci(&a, &b, 1000, 0.95, 42);
        assert!(lo > 0.5, "got lo = {lo}");
    }
}
