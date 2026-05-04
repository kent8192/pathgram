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

The deterministic baseline runner was executed against three real
SWE-bench Lite instances (the bundled fixture's first three rows; the two
synthetic `*-test-*` rows are skipped because they don't have real git
checkouts available). Repos are cloned via `git2` with vendored libgit2 +
HTTPS; cache directory at `/tmp/engramoid_repo_cache`.

| Instance | steps | recall@5 | coverage |
| --- | --- | --- | --- |
| `django__django-11099` | **17** | **1.000** | 1.000 |
| `sympy__sympy-13647` | **12** | 0.000 | 1.000 |
| `sphinx-doc__sphinx-8721` | **10** | 0.000 | 1.000 |
| **mean** | **13.0** | **0.333** | 1.000 |

Full JSON report: [`docs/baseline_measurement_2026-05-04.json`](docs/baseline_measurement_2026-05-04.json).

### Interpreting these numbers

- **Step count (mean 13)** — what gram探索 must compress into 1 single tool
  call. If gram探索 achieves comparable recall, the step-reduction-rate
  metric becomes `(13 − 1) / 13 ≈ 0.92` (92% reduction).
- **Recall@5 (mean 0.333)** — the baseline that gram探索 must exceed. The
  baseline finds the right file by identifier-keyword grep in 1/3 cases
  (django's "ASCIIUsernameValidator" name appears verbatim in
  `validators.py`); fails in cases where the bug is described semantically
  rather than by name (sympy's "col_insert", sphinx's "viewcode_enable_epub").
- **Coverage (1.000)** — trivially 1.0 because we're measuring the default
  runner against itself (no gram runner is plugged in yet). Once Phase 2.1
  ships the LLM-backed runner and Phase 2.2/2.3 ship the gram探索 reranker,
  coverage will be the gram blob measured against the LLM agent's final
  reading context.

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
