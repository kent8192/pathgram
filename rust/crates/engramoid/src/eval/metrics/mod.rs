use crate::eval::agent::Trace;
use crate::eval::instance::SweInstance;
use serde::{Deserialize, Serialize};

pub mod coverage;
pub mod golden;
pub mod recall;
pub mod step_reduction;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricRecord {
    pub instance_id: String,
    pub default_runner: String,
    pub gram_runner: Option<String>,
    pub default_steps: usize,
    pub gram_steps: Option<usize>,
    pub recall_at_5: f64,
    pub step_reduction: f64,
    pub coverage: f64,
}

/// Build a `MetricRecord` from one trace.
///
/// `coverage_reference_trace` supplies the C-label `final_reading_context`.
/// When measuring the default runner, pass `&trace` itself; when measuring a
/// gram runner, pass the default runner's trace.
#[must_use]
pub fn compute_metrics(
    trace: &Trace,
    instance: &SweInstance,
    coverage_reference_trace: &Trace,
    default_runner_name: &str,
    gram_runner_name: Option<String>,
) -> MetricRecord {
    let golden = golden::modified_files(&instance.patch);
    let retrieved = trace.distinct_accessed_files();
    let recall = recall::recall_at_k(&retrieved, &golden, 5);

    let default_steps = coverage_reference_trace.step_count();
    let (gram_steps, step_red) = if gram_runner_name.is_some() {
        let gs = trace.step_count();
        (Some(gs), step_reduction::step_reduction_rate(default_steps, gs))
    } else {
        (None, 0.0)
    };

    let cov = coverage::coverage(&retrieved, &coverage_reference_trace.final_reading_context);

    MetricRecord {
        instance_id: instance.instance_id.clone(),
        default_runner: default_runner_name.to_string(),
        gram_runner: gram_runner_name,
        default_steps,
        gram_steps,
        recall_at_5: recall,
        step_reduction: step_red,
        coverage: cov,
    }
}

/// Compute one MetricRecord per instance.
///
/// If `gram_traces` is `Some`, it must match `default_traces` in length and
/// pair by index; the returned records measure the gram runner against the
/// default runner's coverage reference. If `None`, records measure the
/// default runner self-paired.
#[must_use]
pub fn compute_batch(
    default_traces: &[Trace],
    gram_traces: Option<&[Trace]>,
    instances: &[SweInstance],
    default_runner_name: &str,
    gram_runner_name: Option<&str>,
) -> Vec<MetricRecord> {
    let n = default_traces.len().min(instances.len());
    if let Some(gt) = gram_traces {
        assert_eq!(gt.len(), n, "gram trace count must match default trace count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let default_t = &default_traces[i];
        let inst = &instances[i];
        if let Some(gt) = gram_traces {
            out.push(compute_metrics(
                &gt[i],
                inst,
                default_t,
                default_runner_name,
                gram_runner_name.map(String::from),
            ));
        } else {
            out.push(compute_metrics(
                default_t,
                inst,
                default_t,
                default_runner_name,
                None,
            ));
        }
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stat {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub n: usize,
}

impl Stat {
    #[must_use]
    pub fn of(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Stat {
                mean: 0.0,
                median: 0.0,
                std: 0.0,
                n: 0,
            };
        }
        #[allow(clippy::cast_precision_loss)]
        let nf = n as f64;
        let mean = values.iter().sum::<f64>() / nf;
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nf;
        Stat {
            mean,
            median,
            std: var.sqrt(),
            n,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub recall_at_5: Stat,
    pub step_reduction: Stat,
    pub coverage: Stat,
}

impl MetricSummary {
    #[must_use]
    pub fn of(records: &[MetricRecord]) -> Self {
        let recall: Vec<f64> = records.iter().map(|r| r.recall_at_5).collect();
        let step: Vec<f64> = records.iter().map(|r| r.step_reduction).collect();
        let cov: Vec<f64> = records.iter().map(|r| r.coverage).collect();
        MetricSummary {
            recall_at_5: Stat::of(&recall),
            step_reduction: Stat::of(&step),
            coverage: Stat::of(&cov),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_record_round_trips_json() {
        let r = MetricRecord {
            instance_id: "x".into(),
            default_runner: "DeterministicBaseline".into(),
            gram_runner: None,
            default_steps: 10,
            gram_steps: None,
            recall_at_5: 0.6,
            step_reduction: 0.0,
            coverage: 0.4,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: MetricRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn summary_computes_mean_median_std() {
        let recs: Vec<MetricRecord> = (1..=5)
            .map(|i| MetricRecord {
                instance_id: format!("x{i}"),
                default_runner: "D".into(),
                gram_runner: None,
                default_steps: 0,
                gram_steps: None,
                recall_at_5: f64::from(i) / 10.0,
                step_reduction: 0.0,
                coverage: 0.5,
            })
            .collect();
        let s = MetricSummary::of(&recs);
        assert!((s.recall_at_5.mean - 0.3).abs() < 1e-9);
        assert!((s.recall_at_5.median - 0.3).abs() < 1e-9);
        assert!((s.coverage.mean - 0.5).abs() < 1e-9);
        assert!(s.recall_at_5.std > 0.0);
        assert_eq!(s.coverage.std, 0.0);
    }
}
