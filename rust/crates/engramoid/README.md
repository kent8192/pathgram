# engramoid (in-process port)

This crate is the **in-process Rust port** of [`kent8192/engramoid`][upstream]
into the pathgram (claw-code) workspace. It supplies the knowledge-graph
substrate for Phase 2 of the gram探索 design — a per-project online-learned
single-step retrieval system for coding agents.

## What was ported

From `kent8192/engramoid`'s `feat/phase1-foundation` branch:

| Module | Source | Status |
| --- | --- | --- |
| `graph::models` | `core/src/graph/models.rs` | **verbatim** (Node, Edge, NodeKind ×12, EdgeKind ×10, MetaValue) |
| `graph::engine` | `core/src/graph/engine.rs` | **verbatim** (`GraphEngine` over `petgraph::StableGraph`) |
| `graph::traversal` | `core/src/graph/traversal.rs` | **verbatim** (Personalized PageRank) |
| `tracker::models` | `core/src/tracker/models.rs` | **verbatim** (`TraceAction`, `Session`) |

## What was dropped

| Upstream layer | Reason it doesn't ship in pathgram |
| --- | --- |
| PostgreSQL backend (`core/src/storage/postgres.rs`, `core/migrations/`) | Pathgram is a single-binary CLI; a Postgres dependency would be a major install regression. The port keeps the in-memory graph only. |
| gRPC server (`core/src/server/grpc.rs`) | In-process integration: pathgram calls into engramoid as a Rust crate, no IPC needed. |
| WebSocket dashboard (`core/src/server/ws.rs`) | Out-of-scope for Phase 2 measurement; can be re-introduced as a follow-up if needed. |
| TypeScript MCP plugin (`plugin/`) | The plugin's purpose (MCP tool exposure) will be reimplemented as a native Rust MCP server inside pathgram in a follow-up. |

## What was added (Phase 2.0 eval harness)

A Phase-2-specific evaluation harness, gated behind `--features eval`:

| Module | Purpose |
| --- | --- |
| `eval::instance` | `SweInstance` (SWE-bench Lite / SWE-Gym schema) |
| `eval::loaders` | JSONL loaders + `repo_disjoint_split` train/eval splitter |
| `eval::sandbox` | `Workdir` tempdir + `RepoCache` git2-based clone-and-checkout |
| `eval::agent` | `AgentRunner` trait + `DeterministicBaselineRunner` (LLM-free keyword grep + read) |
| `eval::metrics` | `recall_at_k`, `step_reduction_rate`, `coverage`, `MetricRecord`, `MetricSummary`, golden-diff parser |
| `eval::bootstrap` | `paired_bootstrap_ci` + `paired_primary_ci` (1000 resamples × 95% CI by default) |
| `eval::runner` | `EvalRunner` orchestrator + `EvalReport` |
| `bin/eval_runner.rs` | `engramoid-eval` CLI binary |

## Quick start

```bash
cd rust

# default build (in-process graph + tracker only)
cargo build -p engramoid
cargo test  -p engramoid                                   # 16 tests

# with the eval harness
cargo build -p engramoid --features eval --bin engramoid-eval
cargo test  -p engramoid --features eval                   # 43 tests

# run the deterministic baseline on the bundled fixture
cargo run -p engramoid --features eval --bin engramoid-eval -- \
    --data crates/engramoid/tests/eval/fixtures/swe_bench_lite_subset.jsonl \
    --out  /tmp/report.json
```

## Baseline measurement (2026-05-04)

Two fixtures were executed by the deterministic baseline runner. Repos
were cloned via `git2` (vendored libgit2 + HTTPS) into the
`/tmp/engramoid_repo_cache` per-repo bare-clone cache.

### SWE-bench Lite (single-file fixes)

> **Note:** all 300 instances of the official `princeton-nlp/SWE-bench_Lite`
> dataset modify exactly **1 golden file**. Recall@5 on this dataset is
> mathematically constrained to {0, 1}. Full JSON report:
> [`docs/baseline_lite_2026-05-04.json`](docs/baseline_lite_2026-05-04.json).

| Instance | steps | recall@5 | golden files |
| --- | --- | --- | --- |
| `django__django-11099` | 17 | **1.000** | 1 |
| `sympy__sympy-13647` | 12 | 0.000 | 1 |
| `sphinx-doc__sphinx-8721` | 10 | 0.000 | 1 |
| **mean** | **13.0** | **0.333** | 1.0 |

Coverage is reported as `null` because no gram runner is plugged in yet
(self-paired coverage degenerates to 1.0 trivially).

### SWE-bench Verified (multi-file fixes)

Six instances drawn from `princeton-nlp/SWE-bench_Verified` (which
contains 49 instances with 2 files modified, 12 with 3, 7 with 4, etc.).
Full JSON report:
[`docs/baseline_verified_multifile_2026-05-04.json`](docs/baseline_verified_multifile_2026-05-04.json).

| Instance | steps | recall@5 | golden files |
| --- | --- | --- | --- |
| `astropy__astropy-14369` | 30 | 0.000 | 2 |
| `astropy__astropy-8707` | 30 | 0.000 | 2 |
| `django__django-10554` | 30 | 0.000 | 2 |
| `django__django-11400` | 30 | 0.000 | 3 |
| `django__django-11734` | 30 | 0.000 | 3 |
| `django__django-11532` | 30 | 0.000 | 5 |
| **mean** | **30.0** | **0.000** | 2.8 |

### Interpreting these numbers

- **Step count saturation on Verified (30.0)** — every multi-file instance
  hit the agent's `tool_call_budget` of 30 without converging. The
  identifier-shaped keyword extraction generates more grep candidates than
  the budget allows.
- **Recall@5 collapses on multi-file fixes (0.333 → 0.000)** — the
  deterministic baseline's keyword grep cannot find files that don't
  share identifiers with the bug description. This is a structural
  limitation, not a tuning issue: it sets a strong lower bound for
  Phase 2's gram探索 to beat.
- **Coverage = `null`** — the metric is `Option<f64>` and only populated
  when a `gram_runner` is plugged into `EvalRunner::gram_runner`. Reporting
  a self-paired coverage of `1.0` would be misleading because
  `final_reading_context ⊆ distinct_accessed_files` holds by definition.
- **Step-reduction headroom for gram探索:**
  - vs Lite baseline (mean 13 steps): gram探索 in 1 call → reduction ≈ 0.92
  - vs Verified baseline (mean 30 steps): gram探索 in 1 call → reduction ≈ 0.97

  These numbers become meaningful once gram探索 also clears the recall bar
  (must beat 0.333 on Lite and any non-zero on Verified).

### What's NOT yet measured

Phase 2.0 ships the harness; it does **not** ship the gram探索 system itself.
Real efficacy comparison (B5 = gram探索 vs B4 = frozen γ pipeline, design
doc §8.2) requires:

1. **Phase 2.1**: implement OpenAI Embeddings + Cohere Rerank clients,
   plug into `EvalRunner::gram_runner` slot.
2. **Phase 2.2**: implement Bayesian-Hebbian online edge updates from
   recorded traces.
3. **Phase 2.3**: implement Tabular GRPO offline batch updates.
4. **Phase 2.4**: add Active-Inference augmentation (Ambiguity + IG terms).
5. **Phase 2.5**: dogfood + statistical comparison on full SWE-bench Lite
   (300 instances).

See the [Phase 2 design doc][spec] for the full plan.

## Testing

```bash
cargo test -p engramoid                                # 16 lib tests, no features
cargo test -p engramoid --features eval                # 41 lib tests + 2 integration
cargo test -p engramoid --features "eval eval-offline" # CI-friendly (no network)
```

## Source attribution

This crate is a **port**, not original work for the
`engramoid::graph::*` and `engramoid::tracker::*` modules. The original
implementation lives at <https://github.com/kent8192/engramoid> and remains
the authoritative source for the standalone-service variant.

[upstream]: https://github.com/kent8192/engramoid
[spec]: https://github.com/kent8192/engramoid/blob/docs/phase2-gram-search-design/docs/superpowers/specs/2026-05-04-engramoid-gram-search-design.md
