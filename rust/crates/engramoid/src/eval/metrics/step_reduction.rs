/// `(default - gram) / default`. Returns `None` when `gram_steps == 0`
/// (no meaningful gram result — e.g. Shadow mode or API failure) or when
/// `default_steps == 0`. Clamps negative values to 0.0.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn step_reduction_rate(default_steps: usize, gram_steps: usize) -> Option<f64> {
    if default_steps == 0 || gram_steps == 0 {
        return None;
    }
    let d = default_steps as f64;
    let g = gram_steps as f64;
    Some(((d - g) / d).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_reduction() {
        let v = step_reduction_rate(10, 1).expect("some");
        assert!((v - 0.9).abs() < 1e-9);
    }

    #[test]
    fn no_reduction() {
        let v = step_reduction_rate(10, 10).expect("some");
        assert!((v - 0.0).abs() < 1e-9);
    }

    #[test]
    fn negative_clamps_to_zero() {
        let v = step_reduction_rate(5, 8).expect("some");
        assert!((v - 0.0).abs() < 1e-9);
    }

    #[test]
    fn zero_default_is_none() {
        assert!(step_reduction_rate(0, 1).is_none());
    }

    #[test]
    fn zero_gram_steps_is_none() {
        assert!(step_reduction_rate(10, 0).is_none());
    }
}
