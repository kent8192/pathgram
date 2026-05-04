use crate::eval::agent::AgentRunner;
use crate::eval::instance::SweInstance;
use crate::eval::metrics::{compute_metrics, MetricRecord, MetricSummary};
use crate::eval::sandbox::{git_checkout::RepoCache, Workdir};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub type RepoUrlResolver = Box<dyn Fn(&str) -> String + Send + Sync>;

pub struct EvalRunner<'a> {
    pub default_runner: &'a dyn AgentRunner,
    pub gram_runner: Option<&'a dyn AgentRunner>,
    pub repo_cache_dir: PathBuf,
    pub repo_url_resolver: RepoUrlResolver,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub default_metrics: Vec<MetricRecord>,
    pub gram_metrics: Vec<MetricRecord>,
    pub default_summary: MetricSummary,
    pub gram_summary: Option<MetricSummary>,
    pub instance_count: usize,
    pub processed_count: usize,
    pub default_runner_name: String,
    pub gram_runner_name: Option<String>,
}

impl<'a> EvalRunner<'a> {
    pub fn run(&self, instances: &[SweInstance]) -> EvalReport {
        let cache = RepoCache::new(&self.repo_cache_dir);
        let mut default_metrics = Vec::new();
        let mut gram_metrics = Vec::new();
        let mut processed = 0usize;

        for inst in instances {
            if inst.is_synthetic_fixture() {
                continue;
            }
            let Ok(wd) = Workdir::new() else { continue };
            let url = (self.repo_url_resolver)(&inst.repo);
            let root = match cache.checkout(&url, &inst.base_commit, wd.path()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!(
                        "skip {} (checkout failed: {e})",
                        inst.instance_id
                    );
                    continue;
                }
            };

            let Ok(default_trace) = self.default_runner.run(inst, &root) else {
                continue;
            };
            let m_default = compute_metrics(
                &default_trace,
                inst,
                &default_trace,
                self.default_runner.name(),
                None,
            );
            default_metrics.push(m_default);

            if let Some(gram) = self.gram_runner {
                if let Ok(gram_trace) = gram.run(inst, &root) {
                    let m_gram = compute_metrics(
                        &gram_trace,
                        inst,
                        &default_trace,
                        self.default_runner.name(),
                        Some(gram.name().to_string()),
                    );
                    gram_metrics.push(m_gram);
                }
            }
            processed += 1;
        }

        let default_summary = MetricSummary::of(&default_metrics);
        let gram_summary = if gram_metrics.is_empty() {
            None
        } else {
            Some(MetricSummary::of(&gram_metrics))
        };

        EvalReport {
            default_metrics,
            gram_metrics,
            default_summary,
            gram_summary,
            instance_count: instances.len(),
            processed_count: processed,
            default_runner_name: self.default_runner.name().to_string(),
            gram_runner_name: self.gram_runner.map(|g| g.name().to_string()),
        }
    }
}
