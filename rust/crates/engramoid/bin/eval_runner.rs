use clap::{Parser, ValueEnum};
use engramoid::eval::{
    agent::{deterministic::DeterministicBaselineRunner, frozen_gamma::FrozenGammaRunner},
    bootstrap::paired_primary_ci,
    loaders::{swe_bench_lite::SweBenchLiteLoader, swe_gym::SweGymLoader, Loader},
    runner::EvalRunner,
};
use engramoid::scorers::{
    cohere_rerank::{CohereReranker, MockReranker},
    gemini_embed::GeminiEmbedder,
    openai_embed::MockEmbedder,
    Embedder, Reranker,
};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Dataset {
    Lite,
    Gym,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq)]
enum GramKind {
    /// No gram runner (default-only baseline measurement)
    None,
    /// Frozen γ pipeline: Gemini Embeddings + Cohere Rerank (live, requires API keys)
    FrozenGamma,
    /// Frozen γ pipeline with mock scorers (offline; for smoke-testing the pipeline plumbing)
    FrozenGammaMock,
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
    /// Gram runner to compare against the deterministic baseline.
    #[arg(long, value_enum, default_value_t = GramKind::None)]
    gram_runner: GramKind,
    /// Bootstrap CI confidence level (0..1).
    #[arg(long, default_value_t = 0.95)]
    bootstrap_confidence: f64,
    /// Bootstrap CI resample count.
    #[arg(long, default_value_t = 1000)]
    bootstrap_resamples: usize,
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

    // Build gram runner if requested. Lifetime gymnastics: we keep the
    // scorers alive in `Some` boxes so `&dyn` references stay valid for
    // the duration of the eval run.
    let embedder_box: Box<dyn Embedder>;
    let cohere_box: Box<dyn Reranker>;
    let gram_runner_storage: Option<FrozenGammaRunner>;

    match args.gram_runner {
        GramKind::None => {
            embedder_box = Box::new(MockEmbedder::new(1));
            cohere_box = Box::new(MockReranker);
            gram_runner_storage = None;
        }
        GramKind::FrozenGamma => {
            let gem = GeminiEmbedder::from_env().unwrap_or_else(|e| {
                eprintln!("frozen-gamma requires GEMINI_API_KEY: {e}");
                std::process::exit(2);
            });
            let cr = CohereReranker::from_env().unwrap_or_else(|e| {
                eprintln!("frozen-gamma requires COHERE_API_KEY: {e}");
                std::process::exit(2);
            });
            embedder_box = Box::new(gem);
            cohere_box = Box::new(cr);
            gram_runner_storage = None;  // assigned below from refs
        }
        GramKind::FrozenGammaMock => {
            embedder_box = Box::new(MockEmbedder::new(768));
            cohere_box = Box::new(MockReranker);
            gram_runner_storage = None;
        }
    }

    // Re-borrow into &dyn for the runner construction. We must do this
    // after the `match` so the boxes are alive.
    let _ = gram_runner_storage;
    let gram_owned: Option<FrozenGammaRunner> = if args.gram_runner == GramKind::None {
        None
    } else {
        Some(FrozenGammaRunner::new(&*embedder_box, &*cohere_box))
    };
    let gram_ref: Option<&dyn engramoid::eval::agent::AgentRunner> =
        gram_owned.as_ref().map(|r| r as _);

    let runner = EvalRunner {
        default_runner: &det,
        gram_runner: gram_ref,
        repo_cache_dir: args.repo_cache,
        repo_url_resolver: Box::new(|repo: &str| format!("https://github.com/{repo}")),
    };
    let report = runner.run(&instances);
    let json = serde_json::to_string_pretty(&report).expect("serialize");
    std::fs::write(&args.out, json).expect("write report");

    eprintln!(
        "wrote {} default + {} gram metric records ({} processed) to {}",
        report.default_metrics.len(),
        report.gram_metrics.len(),
        report.processed_count,
        args.out.display()
    );
    let cov_text = report
        .default_summary
        .coverage
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
    if let Some(gs) = &report.gram_summary {
        eprintln!(
            "gram summary:    recall@5 mean={:.3} median={:.3} std={:.3}, step_reduction mean={:.3}, coverage mean={}",
            gs.recall_at_5.mean,
            gs.recall_at_5.median,
            gs.recall_at_5.std,
            gs.step_reduction.mean,
            gs.coverage.as_ref().map(|s| format!("{:.3}", s.mean)).unwrap_or_else(|| "n/a".into()),
        );
        // Paired bootstrap CI (B5 − B4 deltas; here gram vs default)
        if !report.gram_metrics.is_empty()
            && report.gram_metrics.len() == report.default_metrics.len()
        {
            let ci = paired_primary_ci(
                &report.default_metrics,
                &report.gram_metrics,
                args.bootstrap_resamples,
                args.bootstrap_confidence,
                42,
            );
            eprintln!(
                "paired CI ({:.0}%, n_resamples={}):",
                args.bootstrap_confidence * 100.0,
                args.bootstrap_resamples
            );
            eprintln!(
                "  step_reduction (gram − default): [{:.3}, {:.3}]",
                ci.step_reduction.0, ci.step_reduction.1
            );
            eprintln!(
                "  recall@5       (gram − default): [{:.3}, {:.3}]",
                ci.recall_at_5.0, ci.recall_at_5.1
            );
            if let Some((lo, hi)) = ci.coverage {
                eprintln!("  coverage       (gram − default): [{lo:.3}, {hi:.3}]");
            }
        }
    }
}
