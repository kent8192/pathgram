use std::collections::HashSet;

/// `|retrieved[..k] ∩ golden| / |golden|`. Returns 0.0 if golden is empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn recall_at_k(retrieved: &[String], golden: &HashSet<String>, k: usize) -> f64 {
    if golden.is_empty() {
        return 0.0;
    }
    let topk: HashSet<&String> = retrieved.iter().take(k).collect();
    let hit = golden.iter().filter(|g| topk.contains(g)).count();
    hit as f64 / golden.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn full_hit() {
        let g = golden(&["a", "b"]);
        let r = vec!["a".into(), "b".into(), "c".into()];
        assert!((recall_at_k(&r, &g, 5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn partial_hit() {
        let g = golden(&["a", "b", "c"]);
        let r = vec!["a".into(), "x".into(), "b".into()];
        assert!((recall_at_k(&r, &g, 5) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn top_k_truncates() {
        let g = golden(&["a", "b"]);
        let r = vec!["x".into(), "y".into(), "a".into(), "b".into()];
        assert!((recall_at_k(&r, &g, 2) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn empty_golden_is_zero() {
        let r: Vec<String> = vec!["a".into()];
        assert!((recall_at_k(&r, &HashSet::new(), 5) - 0.0).abs() < 1e-9);
    }
}
