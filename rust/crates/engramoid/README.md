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

## Baseline measurement (n=29, 2026-05-04)

Two fixtures executed by the deterministic baseline runner; repos were
cloned via `git2` (vendored libgit2 + HTTPS) into the
`/tmp/engramoid_repo_cache` per-repo bare-clone cache. Both fixtures use
**fixed RNG seed 42** for reproducibility.

The deterministic agent is a keyword-grep + iterative-read baseline:
- Up to 8 longest identifier-shaped keywords (CamelCase or snake_case)
  extracted from the problem statement.
- Each keyword greps the repo (code-extension files only, vendor / build
  / cache dirs excluded).
- Up to 5 hits per keyword become Read tool calls.
- `tool_call_budget = 30`.

### SWE-bench Lite — single-file fixes (n=15)

> **Dataset note:** all 300 instances of `princeton-nlp/SWE-bench_Lite` are
> single-file fixes by curation. Recall@5 is therefore mathematically
> constrained to {0, 1} per instance.
> Full JSON: [`docs/baseline_n30_lite_2026-05-04.json`](docs/baseline_n30_lite_2026-05-04.json).

| metric | value |
| --- | --- |
| recall@5 mean | **0.333** |
| recall@5 median | 0.000 |
| recall@5 std | 0.471 |
| step count mean | **25.3** |
| repo mix | django ×11, matplotlib ×2, sympy ×1, sphinx ×1 |
| processed | 15 / 15 |
| wall time | 25 min 32 s |

Per-instance: 5/15 hit recall@5=1.000 (django-13551, matplotlib-25498,
matplotlib-23476, django-14382, django-12915); 10/15 missed entirely.

### SWE-bench Verified — multi-file fixes (n=14, 1 skipped)

15 instances were attempted from `princeton-nlp/SWE-bench_Verified`
(stratified by `n_files`: 8 of n=2, 4 of n=3, 2 of n=4, 1 of n=5).
1 instance (`pydata__xarray-6992`) skipped because its `base_commit` is
not reachable from the public repo.
Full JSON: [`docs/baseline_n30_verified_2026-05-04.json`](docs/baseline_n30_verified_2026-05-04.json).

| metric | value |
| --- | --- |
| recall@5 mean | **0.217** |
| recall@5 median | 0.000 |
| recall@5 std | 0.353 |
| step count mean | 26.9 |
| golden files mean | 2.8 |
| golden files range | 2 – 5 |
| processed | 14 / 15 |
| wall time | 16 min 8 s |

Per-instance: scikit-learn-12682 (recall=1.000, n_files=2) and
matplotlib-25479 (recall=1.000, n_files=2) are direct hits;
astropy-8707 (0.500, n_files=2), django-13344 (0.333, n_files=3),
django-11532 (0.200, n_files=5) hit a fraction. The remaining 9 missed
entirely.

### Combined n=29 summary

| metric | Lite n=15 | Verified n=14 | combined n=29 |
| --- | --- | --- | --- |
| recall@5 mean | 0.333 | 0.217 | **0.277** |
| step count mean | 25.3 | 26.9 | 26.1 |
| golden files mean | 1.0 | 2.8 | 1.9 |

Coverage is `null` across all rows because no gram runner is plugged in
yet — self-paired coverage (`final_reading_context ⊆ distinct_accessed_files`)
trivially evaluates to 1.0 and would mislead the reader.

### Interpreting these numbers

- **Recall@5 ≈ 0.28** is the floor that Phase 2.1+ gram探索 must beat to
  show any retrieval improvement.
- **Step count ≈ 26** sets the step-reduction headroom: gram探索 in a
  single tool call yields `(26 − 1) / 26 ≈ 0.96` step reduction if it
  matches baseline recall.
- **Lite (single-file, 0.333) > Verified (multi-file, 0.217)** confirms
  the qualitative finding: keyword-grep degrades as bug fixes span more
  files. Multi-file is the regime where gram探索 has the most headroom.
- **9/15 Verified instances saturate `tool_call_budget = 30`** — the
  deterministic agent can't even decide where to stop.

### What's NOT yet measured

- **Coverage**: needs a second runner to compare against. Lands when
  Phase 2.1's `gram_runner` slot is filled by an LLM-backed agent.
- **Bootstrap CI on B5 − B4 deltas**: meaningful only when a gram runner
  exists; the harness (`bootstrap::paired_primary_ci`) is ready and tested.
- **Patch success rate**: requires SWE-bench harness execution (Phase
  3+ scope per design doc §13).

### Reproducibility

```bash
cd rust
cargo build -p engramoid --features eval --bin engramoid-eval --release  # release for speed

# Lite n=15 (~25 min on a clean cache; ~5 min warm)
./target/release/engramoid-eval \
    --data /path/to/baseline_n30_lite.jsonl \
    --out  baseline_n30_lite_report.json

# Verified n=15 (~16 min on a clean cache)
./target/release/engramoid-eval \
    --data /path/to/baseline_n30_verified.jsonl \
    --out  baseline_n30_verified_report.json
```

Fixture generation script (Python, requires `datasets` library) is at
`scripts/build_n30_fixtures.py`.

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
