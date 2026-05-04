use clap::{Parser, ValueEnum};
use engramoid::eval::{
    agent::deterministic::DeterministicBaselineRunner,
    loaders::{swe_bench_lite::SweBenchLiteLoader, swe_gym::SweGymLoader, Loader},
    runner::EvalRunner,
};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Dataset {
    Lite,
    Gym,
}

#[derive(Parser, Debug)]
#[command(name = "engramoid-eval", about = "Phase 2 evaluation harness (in-process port)")]
struct Args {
    /// Path to the JSONL dataset
    #[arg(long)]
    data: PathBuf,
    /// Dataset selector
    #[arg(long, value_enum, default_value_t = Dataset::Lite)]
    dataset: Dataset,
    /// Where to cache cloned repos
    #[arg(long, default_value = "/tmp/engramoid_repo_cache")]
    repo_cache: PathBuf,
    /// Output JSON report path
    #[arg(long)]
    out: PathBuf,
    /// Maximum instances to evaluate (smoke testing)
    #[arg(long)]
    limit: Option<usize>,
}

fn main() {
    let args = Args::parse();
    let mut instances = match args.dataset {
        Dataset::Lite => SweBenchLiteLoader.load(&args.data).expect("load lite"),
        Dataset::Gym => SweGymLoader.load(&args.data).expect("load gym"),
    };
    if let Some(lim) = args.limit {
        instances.truncate(lim);
    }

    let det = DeterministicBaselineRunner::default();
    let runner = EvalRunner {
        default_runner: &det,
        gram_runner: None,
        repo_cache_dir: args.repo_cache,
        repo_url_resolver: Box::new(|repo: &str| format!("https://github.com/{repo}")),
    };
    let report = runner.run(&instances);
    let json = serde_json::to_string_pretty(&report).expect("serialize");
    std::fs::write(&args.out, json).expect("write report");
    eprintln!(
        "wrote {} default metric records ({} processed) to {}",
        report.default_metrics.len(),
        report.processed_count,
        args.out.display()
    );
    let cov_text = report.default_summary.coverage
        .as_ref()
        .map(|s| format!("{:.3}", s.mean))
        .unwrap_or_else(|| "n/a (no gram runner)".into());
    eprintln!(
        "default summary: recall@5 mean={:.3} median={:.3} std={:.3}, coverage mean={}, golden_files mean={:.1}",
        report.default_summary.recall_at_5.mean,
        report.default_summary.recall_at_5.median,
        report.default_summary.recall_at_5.std,
        cov_text,
        report.default_summary.golden_file_count.mean,
    );
}
