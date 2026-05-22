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

### Sample-size summary

| Cell | n | source | curation |
| --- | --- | --- | --- |
| Lite (single-file) | **n = 15** | `princeton-nlp/SWE-bench_Lite` test split (300 total) | random sample, seed 42 |
| Verified (multi-file) attempted | n = 15 | `princeton-nlp/SWE-bench_Verified` test split (500 total) | stratified by golden `n_files`: 8×2-file, 4×3-file, 2×4-file, 1×5-file, seed 42 |
| Verified processed | **n = 14** | — | 1 skipped (`pydata__xarray-6992`: `base_commit` unreachable) |
| **Combined processed** | **n = 29** | — | Lite n=15 + Verified n=14 |

### SWE-bench Lite — single-file fixes (n = 15)

> **Dataset note:** all 300 instances of `princeton-nlp/SWE-bench_Lite` are
> single-file fixes by curation. Recall@5 on this dataset is therefore
> mathematically constrained to **{0, 1}** per instance.
> Full JSON: [`docs/baseline_n30_lite_2026-05-04.json`](docs/baseline_n30_lite_2026-05-04.json).

| metric | value (n = 15) |
| --- | --- |
| recall@5 mean | **0.333** |
| recall@5 median | 0.000 |
| recall@5 std | 0.471 |
| recall@5 hit count (==1.000) | **5 / 15** |
| recall@5 miss count (==0.000) | **10 / 15** |
| step count mean | **25.3** |
| step count saturated at 30 | 6 / 15 |
| golden_file_count | 1 (all 15 instances) |
| repo mix | django ×11, matplotlib ×2, sympy ×1, sphinx ×1 |
| processed | 15 / 15 |
| wall time | 25 min 32 s |

Per-instance recall = 1.000 (5 / 15): `django-13551`, `matplotlib-25498`,
`matplotlib-23476`, `django-14382`, `django-12915`. The remaining 10
returned 0.000.

### SWE-bench Verified — multi-file fixes (n = 14, 1 skipped of 15 attempted)

> **Dataset note:** of `princeton-nlp/SWE-bench_Verified`'s 500 instances,
> 71 modify ≥ 2 files (49 × 2-file, 12 × 3-file, 7 × 4-file, 2 × 5-file,
> 1 × 6-file). We sampled 15 stratified across the n_files distribution.
> Full JSON: [`docs/baseline_n30_verified_2026-05-04.json`](docs/baseline_n30_verified_2026-05-04.json).

| metric | value (n = 14) |
| --- | --- |
| recall@5 mean | **0.217** |
| recall@5 median | 0.000 |
| recall@5 std | 0.353 |
| recall@5 full hit count (==1.000) | **2 / 14** |
| recall@5 partial hit count (0 < r < 1) | **3 / 14** |
| recall@5 miss count (==0.000) | **9 / 14** |
| step count mean | 26.9 |
| step count saturated at 30 | 10 / 14 |
| golden_file_count distribution (post-skip) | 7 × 2-file, 4 × 3-file, 2 × 4-file, 1 × 5-file |
| golden_file_count mean | 2.8 |
| golden_file_count range | 2 – 5 |
| processed | 14 / 15 (1 checkout failure) |
| wall time | 16 min 8 s |

Per-instance recall (sorted, n = 14):

- 1.000: `scikit-learn-12682` (n=2), `matplotlib-25479` (n=2) — 2 / 14
- 0.500: `astropy-8707` (n=2) — 1 / 14
- 0.333: `django-13344` (n=3) — 1 / 14
- 0.200: `django-11532` (n=5) — 1 / 14
- 0.000: 9 / 14 (`pylint-6528`, `django-14170`, `sphinx-8120`,
  `pylint-4661`, `sphinx-10673`, `sphinx-9461`, `matplotlib-14623`,
  `pylint-6386`, `astropy-13398`)

Skipped (1 / 15): `pydata__xarray-6992` — `base_commit`
`45c0a114e2b7b27b83c9618bc05b36afac82183c` not present in the upstream
xarray repo at clone time.

### Combined n = 29 summary

| metric | Lite (n = 15) | Verified (n = 14) | combined (n = 29) |
| --- | --- | --- | --- |
| recall@5 mean | 0.333 | 0.217 | **0.277** |
| recall@5 std | 0.471 | 0.353 | 0.422 |
| step count mean | 25.3 | 26.9 | 26.1 |
| step count std | — | — | 6.2 |
| step count saturated at 30 | 6 / 15 | 10 / 14 | **16 / 29** |
| golden_file_count mean | 1.0 | 2.8 | 1.86 |
| processed | 15 / 15 | 14 / 15 | **29 / 30** |

Coverage is `null` across all 29 rows because no gram runner is plugged
in yet — self-paired coverage (`final_reading_context ⊆ distinct_accessed_files`)
trivially evaluates to 1.0 and would mislead the reader.

### Interpreting these numbers

- **Recall@5 ≈ 0.28** (combined n = 29) is the floor that Phase 2.1+
  gram探索 must beat to show any retrieval improvement.
- **Step count ≈ 26** (combined n = 29) sets the step-reduction
  headroom: gram探索 in a single tool call yields `(26 − 1) / 26 ≈ 0.96`
  step reduction if it matches baseline recall.
- **Lite (n = 15, recall 0.333) > Verified (n = 14, recall 0.217)**
  confirms the qualitative finding: keyword-grep degrades as bug fixes
  span more files. Multi-file (Verified) is the regime where gram探索
  has the most headroom.
- **16 / 29 instances saturate `tool_call_budget = 30`** (6 Lite + 10
  Verified) — over half the runs exhaust the agent's call budget before
  converging.

### What's NOT yet measured

- **Live B0 vs B4 numbers**: the frozen γ pipeline (Phase 2.1, see below) is
  implemented and unit-tested but not yet executed against the live OpenAI
  Embeddings + Cohere Rerank APIs because the API keys are not yet
  configured locally. Mock-pipeline smoke runs reproduce the plumbing
  (step_reduction ≈ 0.91, paired bootstrap CI returned correctly) but
  yield zero recall because mock embeddings hash inputs without semantic
  understanding.
- **Patch success rate**: requires SWE-bench harness execution (Phase
  3+ scope per design doc §13).

## Phase 2.1 — frozen γ pipeline (B4) — live measurement complete (2026-05-22)

The frozen γ retrieval pipeline that produces the **B4 baseline** in the
design-doc evaluation matrix is fully implemented behind
`--gram-runner frozen-gamma`:

```
problem statement
   → embed (Gemini gemini-embedding-001 @ 768d, Matryoshka)
   → cosine top-200 (in-memory EmbeddingStore over chunked repo)
   → Cohere rerank top-30 (rerank-english-v3.0)
   → token-budgeted blob (8192 tokens default, line-based 50/10 chunks,
     deduplicated, Markdown-rendered with `## file:line-line` headers)
   → AgentRunner returns 1 tool call whose accessed_files = blob's
     distinct paths
```

The embedder migrated from OpenAI `text-embedding-3-small` to Gemini
`gemini-embedding-001` after the OpenAI account remained flagged as
`billing_not_active` despite restored billing (likely a >1y-old unpaid
invoice that the 12-month billing-history view does not surface).
`OpenAiEmbedder` is retained in the source tree for parity.

### Modules

| Module | Purpose |
| --- | --- |
| `scorers::Embedder` / `GeminiEmbedder` / `MockEmbedder` | text → vector, live + offline impls (OpenAiEmbedder retained, unwired) |
| `scorers::Reranker` / `CohereReranker` / `MockReranker` | (query, docs) → ranked indices |
| `retrieval::chunks::{Chunk, Chunker}` | line-based chunking with code-extension + vendor-skip filters |
| `retrieval::embedding_store::EmbeddingStore` | brute-force cosine top-N, insert-time L2 normalization |
| `retrieval::packer::{ContextBlob, Packer, ChunkCitation}` | greedy token-budgeted pack (chars / 4 token heuristic) |
| `retrieval::pipeline::FrozenGammaPipeline` | wires chunker + embedder + reranker + packer |
| `eval::agent::frozen_gamma::FrozenGammaRunner` | exposes pipeline as `AgentRunner` for the eval harness |

### Mock smoke verification (offline, no API keys)

```
default summary: recall@5 mean=0.333 median=0.000 std=0.471 (n=3)
gram summary:    recall@5 mean=0.000 std=0.000, step_reduction mean=0.908,
                 coverage mean=0.000
paired CI (95%, 1000 resamples):
  step_reduction (gram − default): [0.875, 0.933]
  recall@5       (gram − default): [-1.000, 0.000]
```

The recall is 0 with mock embeddings because they hash inputs without
semantic understanding. The number that matters here is **step_reduction
mean ≈ 0.91** — the structural compression gain from collapsing the
multi-step deterministic exploration into a single tool call. This is
upper-bounded by `(default_steps − 1) / default_steps` and so depends
only on the default trace's step count, not on either runner's recall.

### Running the live B0 vs B4 measurement

Requires two API keys. Gemini AI Studio must be on a paid tier (Tier 1
or higher) — the free tier's 5 RPM ceiling causes the embedder to spend
all of its retry budget on backoff. Estimated total cost for n=30 is
**\$3 – \$6**. The `GeminiEmbedder` self-throttles at 1.5 s/call and
retries on HTTP 429 with exponential backoff (2 → 60 s, 5 attempts).

```bash
export GEMINI_API_KEY='...'
export COHERE_API_KEY='...'

cd rust

# B0 (deterministic) vs B4 (frozen γ) on Lite n=15
cargo run --features eval --bin engramoid-eval --release -- \
    --data /path/to/baseline_n30_lite.jsonl \
    --out  docs/b0_vs_b4_lite.json \
    --gram-runner frozen-gamma

# B0 vs B4 on Verified multi-file n=14
cargo run --features eval --bin engramoid-eval --release -- \
    --data /path/to/baseline_n30_verified.jsonl \
    --out  docs/b0_vs_b4_verified.json \
    --gram-runner frozen-gamma
```

The CLI prints the two summaries plus a paired bootstrap CI on the three
primary metrics. Reports include per-instance records suitable for
ablations and downstream analysis.

### Live measurement results (2026-05-22, n=30 = 15 Lite + 15 Verified)

Embedder: Gemini `gemini-embedding-001` @ 768d (Matryoshka). Reranker:
Cohere `rerank-english-v3.0`. Both datasets sampled with seed 42; see
`scripts/build_n30_fixtures.py`. Reports persisted as
`docs/b0_vs_b4_lite.json` / `docs/b0_vs_b4_verified.json`.

| Metric | dataset | B0 (default) | B4 (γ) | Δ (B4 − B0) 95 % CI |
| --- | --- | ---: | ---: | --- |
| recall@5 mean | Lite | 0.333 | **0.600** | [-0.067, 0.600] |
| recall@5 median | Lite | 0.000 | **1.000** | — |
| step_reduction | Lite | — | 0.956 | **[0.947, 0.964]** |
| coverage | Lite | — | 0.240 | — |
| recall@5 mean | Verified | 0.236 | **0.291** | [-0.111, 0.256] |
| recall@5 median | Verified | 0.000 | **0.333** | — |
| step_reduction | Verified | — | 0.958 | **[0.945, 0.966]** |
| coverage | Verified | — | 0.107 | — |
| golden_files mean | Verified | 2.7 | 2.7 | — |

Bootstrap configuration: 1 000 resamples, 95 % paired CI on per-instance
B4 − B0 differences (`bootstrap::paired_primary_ci`).

**Interpretation**

- **step_reduction is robustly significant** on both datasets — B4
  collapses the deterministic baseline's ~30-step walk into a single
  retrieval call, reducing tool-call count by ~96 % with a tight CI.
- **recall@5 improves on the mean for both datasets** (Lite +0.267,
  Verified +0.055) but the 95 % paired CI crosses zero at n=15. The
  improvement is not statistically significant at this sample size,
  though the direction is consistent.
- **multi-file (Verified) is harder for B4**: the single-call retrieval
  bundles one blob whose distinct paths must cover ~2.7 golden files on
  average, vs. 1.0 in Lite. The smaller mean lift (+0.055 vs. +0.267)
  reflects that constraint.
- **coverage is low** (Lite 0.24, Verified 0.11): B4 retrieves a
  materially different set of files than the deterministic baseline,
  which is expected — the two pipelines explore the repo differently.

B4 delivers a clear structural win (step reduction) and a directional
recall improvement that would need n > 15 per dataset to confirm
statistically. That matches the Phase 2.1 success criterion well enough
to unblock Phase 2.2 (Bayesian-Hebbian online graph weight learning).

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
