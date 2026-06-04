//! Iterative multi-round retrieval (§6.7.2).
//!
//! Wraps the frozen-γ pipeline in a search → feedback → re-search loop:
//! 1. Round 1 — run the hybrid pipeline normally
//! 2. Feedback — extract import targets and high-signal keywords from results
//! 3. Round 2 — re-run with expanded signals; union candidates, re-rank, re-pack
//!
//! This addresses the structural single-round cap on multi-file (Verified)
//! recall by following cross-file references discovered in the first pass.

use crate::graph::builder::AstGraphBuilder;
use crate::retrieval::candidates::{CandidateSource, HybridCandidateSource};
use crate::retrieval::chunks::{Chunk, Chunker};
use crate::retrieval::embedding_store::EmbeddingStore;
use crate::retrieval::keyword;
use crate::retrieval::packer::{ContextBlob, Packer};
use crate::retrieval::query_expand::QueryExpander;
use crate::scorers::{Embedder, Reranker};
use std::collections::HashSet;
use std::path::Path;

/// Configuration for iterative multi-round retrieval.
pub struct IterativeRetriever {
    /// Maximum number of retrieval rounds (default 2).
    pub max_rounds: usize,
    /// Number of extra keywords to extract from round-1 chunk content.
    pub feedback_keywords: usize,
    /// Whether to follow Python import statements found in retrieved files.
    pub feedback_imports: bool,
    /// Embedding vector dimension.
    pub embedding_dim: usize,
    /// Batch size for embedding calls.
    pub embed_batch_size: usize,
    /// Envelope for the inner pipeline — shared across rounds.
    pub candidate_top_n: usize,
    pub rerank_top_k: usize,
}

impl Default for IterativeRetriever {
    fn default() -> Self {
        Self {
            max_rounds: 2,
            feedback_keywords: 5,
            feedback_imports: true,
            embedding_dim: 768,
            embed_batch_size: 64,
            candidate_top_n: 200,
            rerank_top_k: 30,
        }
    }
}

impl IterativeRetriever {
    /// Run iterative multi-round retrieval.
    ///
    /// Round 1 is the standard pipeline. If `max_rounds > 1`, feedback signals
    /// are extracted from round-1 results and a second round is run with
    /// expanded candidate sources. The final blob is re-ranked from the
    /// union of all rounds.
    #[allow(clippy::too_many_lines)]
    pub fn retrieve(
        &self,
        embedder: &dyn Embedder,
        reranker: &dyn Reranker,
        task: &str,
        repo_root: &Path,
    ) -> Result<ContextBlob, super::pipeline::PipelineError> {
        let chunker = Chunker::default();
        let packer = Packer::default();

        // ── Round 1 ────────────────────────────────────────────────────
        let chunks = chunker.chunk_repo(repo_root);
        if chunks.is_empty() {
            return Err(super::pipeline::PipelineError::EmptyRepo);
        }

        let mut store = EmbeddingStore::new(self.embedding_dim);
        for batch in chunks.chunks(self.embed_batch_size) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let vecs = embedder.embed_batch(&texts)?;
            store.insert_batch(vecs);
        }
        let task_vec = embedder.embed_batch(&[task.to_string()])?;
        let task_vec = task_vec.into_iter().next().unwrap_or_default();

        // Build AST graph for PPR support
        let ast_builder = AstGraphBuilder::default();
        let mut ast_engine = ast_builder.build(repo_root);
        let has_ast_graph = ast_engine.node_count() > 0;
        if has_ast_graph {
            let ast_chunks = ast_builder.extract_chunks(&ast_engine, repo_root);
            if !ast_chunks.is_empty() {
                let texts: Vec<String> = ast_chunks.iter().map(|(_, c)| c.text.clone()).collect();
                if let Ok(vecs) = embedder.embed_batch(&texts) {
                    for ((node_id, _), mut vec) in ast_chunks.iter().zip(vecs) {
                        let n = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                        for x in &mut vec {
                            *x /= n;
                        }
                        if let Some(node) = ast_engine.get_node_mut(node_id) {
                            node.embedding = Some(vec);
                        }
                    }
                }
            }
        }

        let candidate_source = HybridCandidateSource {
            embedding_top_n: self.candidate_top_n,
            include_ppr: has_ast_graph,
            ..Default::default()
        };
        let graph_opt = if has_ast_graph { Some(&ast_engine) } else { None };
        let round1_candidates =
            candidate_source.generate(&store, &task_vec, &chunks, task, repo_root, graph_opt);

        if round1_candidates.is_empty() || self.max_rounds <= 1 {
            return pack_candidates(
                &round1_candidates,
                &chunks,
                task,
                reranker,
                &packer,
                self.rerank_top_k,
            );
        }

        // ── Feedback extraction ────────────────────────────────────────
        let round1_files: HashSet<String> = round1_candidates
            .iter()
            .map(|(i, _)| chunks[*i].path.clone())
            .collect();

        // A. Import targets: parse Python imports from retrieved files
        let mut import_targets: HashSet<String> = HashSet::new();
        if self.feedback_imports {
            import_targets = extract_import_targets(repo_root, &round1_files);
        }

        // B. High-signal keywords from retrieved chunk text
        let mut feedback_kw: Vec<String> = Vec::new();
        if self.feedback_keywords > 0 {
            let round1_text: String = round1_candidates
                .iter()
                .take(10)
                .map(|(i, _)| chunks[*i].text.as_str())
                .collect::<Vec<&str>>()
                .join("\n");
            feedback_kw = keyword::extract_keywords(&round1_text, self.feedback_keywords, 3);
            let expander = QueryExpander::default();
            feedback_kw = expander.expand(&feedback_kw);
        }

        // ── Round 2: expand with feedback signals ──────────────────────
        let mut round2_candidates: Vec<(usize, CandidateSource)> = Vec::new();

        // Import-following: add chunks for imported files
        for import_path in &import_targets {
            for (idx, chunk) in chunks.iter().enumerate() {
                if chunk.path == *import_path || chunk.path.ends_with(import_path) {
                    round2_candidates.push((idx, CandidateSource::Keyword));
                }
            }
        }

        // Feedback-keyword grep
        for kw in &feedback_kw {
            let code_files = keyword::collect_code_files(repo_root, 256 * 1024);
            let hits = keyword::grep_files(&code_files, kw, candidate_source.max_files_per_keyword);
            for hit_path in hits {
                let rel_result = hit_path.strip_prefix(repo_root);
                let Ok(rel) = rel_result else { continue };
                let rel_str = rel.to_string_lossy();
                for (idx, chunk) in chunks.iter().enumerate() {
                    if chunk.path == rel_str {
                        round2_candidates.push((idx, CandidateSource::Keyword));
                    }
                }
            }
        }

        // Union rounds, preserving round-1 order first
        let mut seen: HashSet<usize> = HashSet::new();
        let mut union: Vec<(usize, CandidateSource)> = Vec::new();
        for (idx, src) in round1_candidates.iter().chain(round2_candidates.iter()) {
            if seen.insert(*idx) {
                union.push((*idx, *src));
            }
        }

        pack_candidates(&union, &chunks, task, reranker, &packer, self.rerank_top_k)
    }
}

/// Extract Python import targets from retrieved files.
///
/// Parses `import X` and `from X import Y` statements, returning the
/// top-level module/package paths that may correspond to other files in
/// the repo. Uses simple line-based parsing to avoid a regex dependency.
fn extract_import_targets(repo_root: &Path, file_paths: &HashSet<String>) -> HashSet<String> {
    let mut targets: HashSet<String> = HashSet::new();

    for path in file_paths {
        let Ok(source) = std::fs::read_to_string(repo_root.join(path)) else {
            continue;
        };
        for line in source.lines() {
            let trimmed = line.trim();
            let module = if let Some(rest) = trimmed.strip_prefix("from ") {
                // "from foo.bar import baz" → module = "foo.bar"
                rest.split(&[' ', '\t']).next()
            } else if let Some(rest) = trimmed.strip_prefix("import ") {
                // "import foo.bar" → module = "foo.bar"
                rest.split(&[' ', '\t', ',']).next()
            } else {
                None
            };

            if let Some(m) = module {
                let m = m.trim_end_matches(',');
                if m.is_empty() || m == "import" {
                    continue;
                }
                let top = m.split('.').next().unwrap_or(m);
                let file_path = top.replace('.', "/");
                targets.insert(format!("{file_path}.py"));
                targets.insert(format!("{file_path}/__init__.py"));
            }
        }
    }

    targets
}

/// Pack ranked candidates into a ContextBlob.
fn pack_candidates(
    candidates: &[(usize, CandidateSource)],
    chunks: &[Chunk],
    task: &str,
    reranker: &dyn Reranker,
    packer: &Packer,
    top_k: usize,
) -> Result<ContextBlob, super::pipeline::PipelineError> {
    if candidates.is_empty() {
        return Ok(ContextBlob {
            chunks: Vec::new(),
            total_tokens: 0,
        });
    }

    let cand_chunks: Vec<&Chunk> = candidates.iter().map(|(i, _)| &chunks[*i]).collect();
    let cand_texts: Vec<String> = cand_chunks.iter().map(|c| c.text.clone()).collect();

    let reranked = reranker.rerank(task, &cand_texts, top_k)?;
    let ranked: Vec<(Chunk, f32)> = reranked
        .into_iter()
        .map(|(idx, score)| (cand_chunks[idx].clone(), score))
        .collect();

    Ok(packer.pack(&ranked))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorers::cohere_rerank::MockReranker;
    use crate::scorers::openai_embed::MockEmbedder;

    fn multi_file_repo() -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();

        // Main validator that imports from util
        std::fs::write(
            root.join("src/validators.py"),
            "from util import helper\n\nclass Validator:\n    def check(self, name):\n        return helper.is_valid(name)\n",
        )
        .unwrap();

        // Utility module imported by validators
        std::fs::write(
            root.join("src/util.py"),
            "def helper():\n    pass\n\ndef is_valid(name):\n    return len(name) > 3\n",
        )
        .unwrap();

        // Unrelated file
        std::fs::write(
            root.join("src/other.py"),
            "def unrelated():\n    pass\n",
        )
        .unwrap();

        td
    }

    #[test]
    fn iterative_retrieval_includes_imported_files() {
        let repo = multi_file_repo();
        let embedder = MockEmbedder::new(64);
        let reranker = MockReranker;
        let retriever = IterativeRetriever {
            max_rounds: 2,
            feedback_keywords: 0,
            feedback_imports: true,
            embedding_dim: 64,
            embed_batch_size: 4,
            candidate_top_n: 10,
            rerank_top_k: 5,
        };

        let blob = retriever
            .retrieve(&embedder, &reranker, "Validator check", repo.path())
            .unwrap();
        let paths: Vec<&str> = blob.chunks.iter().map(|c| c.path.as_str()).collect();
        assert!(
            paths.iter().any(|p| p.contains("validators.py")),
            "should include validators.py, got: {paths:?}"
        );
        // Iterative round 2 should pull in imported util.py
        assert!(
            paths.iter().any(|p| p.contains("util.py")),
            "should include imported util.py via feedback, got: {paths:?}"
        );
    }

    #[test]
    fn single_round_still_works() {
        let repo = multi_file_repo();
        let embedder = MockEmbedder::new(64);
        let reranker = MockReranker;
        let retriever = IterativeRetriever {
            max_rounds: 1,
            feedback_keywords: 0,
            feedback_imports: false,
            embedding_dim: 64,
            embed_batch_size: 4,
            candidate_top_n: 10,
            rerank_top_k: 5,
        };

        let blob = retriever
            .retrieve(&embedder, &reranker, "Validator regex", repo.path())
            .unwrap();
        assert!(!blob.chunks.is_empty(), "single round should return results");
    }

    #[test]
    fn extract_import_targets_finds_modules() {
        let repo = multi_file_repo();
        let mut paths = HashSet::new();
        paths.insert("src/validators.py".to_string());
        let targets = extract_import_targets(repo.path(), &paths);
        assert!(
            targets.iter().any(|t| t.contains("util")),
            "should find util import target, got: {targets:?}"
        );
    }
}
