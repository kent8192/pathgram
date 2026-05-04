/// `(default - gram) / default`. Returns 0 if default == 0 or if gram > default.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn step_reduction_rate(default_steps: usize, gram_steps: usize) -> f64 {
    if default_steps == 0 {
        return 0.0;
    }
    let d = default_steps as f64;
    let g = gram_steps as f64;
    ((d - g) / d).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_reduction() {
        assert!((step_reduction_rate(10, 1) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn no_reduction() {
        assert!((step_reduction_rate(10, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn negative_clamps_to_zero() {
        assert!((step_reduction_rate(5, 8) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn zero_default_is_zero() {
        assert!((step_reduction_rate(0, 1) - 0.0).abs() < 1e-9);
    }
}
