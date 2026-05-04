use std::collections::HashSet;

/// `|blob_files ∩ final_reading_context| / |final_reading_context|`.
/// Returns 0.0 when the final context is empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn coverage(blob_files: &[String], final_context: &[String]) -> f64 {
    if final_context.is_empty() {
        return 0.0;
    }
    let blob: HashSet<&String> = blob_files.iter().collect();
    let hit = final_context.iter().filter(|c| blob.contains(c)).count();
    hit as f64 / final_context.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_coverage() {
        let blob = vec!["a".into(), "b".into()];
        let ctx = vec!["a".into(), "b".into()];
        assert!((coverage(&blob, &ctx) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn partial_coverage() {
        let blob = vec!["a".into()];
        let ctx = vec!["a".into(), "b".into()];
        assert!((coverage(&blob, &ctx) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_context_is_zero() {
        let blob = vec!["a".into()];
        assert!((coverage(&blob, &[]) - 0.0).abs() < 1e-9);
    }
}
