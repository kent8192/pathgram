use crate::graph::engine::GraphEngine;
use crate::graph::models::{MetaValue, NodeKind};
use crate::graph::traversal::personalized_pagerank;
use crate::retrieval::chunks::Chunk;
use crate::retrieval::embedding_store::EmbeddingStore;
use crate::retrieval::error_extract::ErrorSignalExtractor;
use crate::retrieval::hook_cache::HookExecutionCache;
use crate::retrieval::keyword;
use crate::retrieval::query_expand::QueryExpander;
use std::collections::HashSet;
use std::path::Path;

/// Which signal produced this candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandidateSource {
    Embedding,
    Keyword,
    ErrorMessage,
    Ppr,
    HookExecution,
}

/// Produces a union candidate set from embedding cosine, keyword-grep,
/// error-message extraction, and PPR sources. Deduplicates by chunk index
/// so each chunk appears once.
pub struct HybridCandidateSource {
    pub embedding_top_n: usize,
    pub max_keywords: usize,
    pub max_files_per_keyword: usize,
    pub include_keywords: bool,
    pub include_error_messages: bool,
    pub include_hook_execution: bool,
    pub include_ppr: bool,
    pub ppr_top_seeds: usize,
    pub ppr_top_n: usize,
    pub ppr_damping: f64,
    pub ppr_iterations: usize,
    /// Enable query expansion for keyword-grep (domain synonyms + variants).
    pub query_expansion: bool,
}

impl Default for HybridCandidateSource {
    fn default() -> Self {
        Self {
            embedding_top_n: 200,
            max_keywords: 8,
            max_files_per_keyword: 5,
            include_keywords: true,
            include_error_messages: true,
            include_hook_execution: false,
            include_ppr: false,
            ppr_top_seeds: 3,
            ppr_top_n: 100,
            ppr_damping: 0.85,
            ppr_iterations: 20,
            query_expansion: true,
        }
    }
}

impl HybridCandidateSource {
    /// Generate a deduplicated candidate set. Returns `(chunk_index, source)` pairs.
    /// Embedding candidates always come first, then keyword, error-message, and PPR.
    /// Pass `graph` to enable PPR candidate generation (AST graph with embedded nodes).
    #[must_use]
    pub fn generate(
        &self,
        store: &EmbeddingStore,
        task_vec: &[f32],
        chunks: &[Chunk],
        task: &str,
        repo_root: &Path,
        graph: Option<&GraphEngine>,
    ) -> Vec<(usize, CandidateSource)> {
        let mut out: Vec<(usize, CandidateSource)> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();

        // 1. Embedding cosine top-N (skip zero-score results: orthogonal
        //    chunks are irrelevant and would pollute the dedup set)
        for (idx, score) in store.top_n(task_vec, self.embedding_top_n) {
            if score > 0.0 && seen.insert(idx) {
                out.push((idx, CandidateSource::Embedding));
            }
        }

        // 2. Keyword-grep candidates
        if self.include_keywords {
            let mut keywords = keyword::extract_keywords(task, self.max_keywords, 3);
            if self.query_expansion {
                let expander = QueryExpander::default();
                keywords = expander.expand(&keywords);
            }
            let code_files = keyword::collect_code_files(repo_root, 256 * 1024);
            for kw in &keywords {
                let hits = keyword::grep_files(&code_files, kw, self.max_files_per_keyword);
                for hit_path in hits {
                    let rel_result = hit_path.strip_prefix(repo_root);
                    let Ok(rel) = rel_result else {
                        continue;
                    };
                    let rel_str = rel.to_string_lossy();
                    for (idx, chunk) in chunks.iter().enumerate() {
                        if chunk.path == rel_str && seen.insert(idx) {
                            out.push((idx, CandidateSource::Keyword));
                        }
                    }
                }
            }
        }

        // 3. Error-message file references (traceback paths, error types)
        if self.include_error_messages {
            let extractor = ErrorSignalExtractor::default();
            let error_paths = extractor.extract_file_paths(task);
            for err_path in &error_paths {
                // Try matching against chunk paths.
                // err_path may be absolute (e.g. /app/src/validators.py) — try
                // stripping repo_root first, then fall back to basename matching.
                let candidates_for_path = chunks.iter().enumerate().filter(|(_, chunk)| {
                    if chunk.path == *err_path {
                        return true;
                    }
                    // Basename match as fallback
                    let chunk_base =
                        std::path::Path::new(&chunk.path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("");
                    let err_base =
                        std::path::Path::new(err_path.as_str())
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("");
                    chunk_base == err_base
                });
                for (idx, _) in candidates_for_path {
                    if seen.insert(idx) {
                        out.push((idx, CandidateSource::ErrorMessage));
                    }
                }
            }
        }

        // 4. Hook execution: run error-referenced files, extract additional
        //    paths from runtime output (tracebacks, import errors, etc.)
        if self.include_hook_execution {
            let mut hook_cache = HookExecutionCache::default();
            hook_cache.execute_error_files(repo_root, task);
            let output_paths = hook_cache.extract_output_file_paths();
            for out_path in &output_paths {
                for (idx, chunk) in chunks.iter().enumerate() {
                    if chunk.path == *out_path
                        || std::path::Path::new(&chunk.path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            == std::path::Path::new(out_path.as_str())
                                .file_name()
                                .and_then(|n| n.to_str())
                    {
                        if seen.insert(idx) {
                            out.push((idx, CandidateSource::HookExecution));
                        }
                    }
                }
            }
        }

        // 5. PPR candidates from AST graph
        if self.include_ppr {
            if let Some(engine) = graph {
                let seeds = find_ppr_seeds(engine, task_vec, self.ppr_top_seeds);
                if !seeds.is_empty() {
                    let ppr_scores =
                        personalized_pagerank(engine, &seeds, self.ppr_damping, self.ppr_iterations);
                    for (node_id, _score) in ppr_scores.into_iter().take(self.ppr_top_n) {
                        if let Some(node) = engine.get_node(&node_id) {
                            if let Some(MetaValue::Text(file_path)) = node.get_meta("file") {
                                for (idx, chunk) in chunks.iter().enumerate() {
                                    if chunk.path == file_path.as_str() && seen.insert(idx) {
                                        out.push((idx, CandidateSource::Ppr));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        out
    }
}

/// Find top seed nodes for PPR by computing cosine similarity between
/// `task_vec` and each Function/Class node's stored embedding.
fn find_ppr_seeds(engine: &GraphEngine, task_vec: &[f32], top_n: usize) -> Vec<crate::graph::models::NodeId> {
    let norm = task_vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    let q: Vec<f32> = task_vec.iter().map(|x| x / norm).collect();

    let mut scored: Vec<(crate::graph::models::NodeId, f32)> = engine
        .all_nodes()
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Function | NodeKind::Class))
        .filter_map(|n| {
            n.embedding.as_ref().map(|emb| {
                let dot: f32 = emb.iter().zip(&q).map(|(a, b)| a * b).sum();
                (n.id, dot)
            })
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_n);
    scored.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::chunks::Chunker;

    fn fake_repo() -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("src/validators.py"),
            "class ASCIIUsernameValidator:\n    regex = r'^[\\w.@+-]+$'\n",
        )
        .unwrap();
        std::fs::write(root.join("src/util.py"), "def util():\n    pass\n").unwrap();
        std::fs::write(root.join("README.md"), "# project\n").unwrap();
        td
    }

    #[test]
    fn hybrid_source_includes_keyword_hits() {
        let repo = fake_repo();
        let chunker = Chunker::default();
        let chunks = chunker.chunk_repo(repo.path());
        // Build embedding store where the task vector is orthogonal to
        // all chunk vectors, so embedding top-N returns nothing. This
        // isolates keyword-grep as the sole candidate source.
        let mut store = EmbeddingStore::new(4);
        for _ in &chunks {
            store.insert_batch(vec![vec![1.0, 0.0, 0.0, 0.0]]);
        }
        // Task vector is orthogonal → cosine ≈ 0 for all chunks
        let task_vec = vec![0.0, 1.0, 0.0, 0.0_f32];
        let source = HybridCandidateSource::default();
        let candidates = source.generate(&store, &task_vec, &chunks, "ASCIIUsernameValidator", repo.path(), None);
        // Embedding top-N should be empty (zero scores), keyword should find validators.py
        assert!(!candidates.is_empty(), "keyword candidates should be present");
        let has_validators_keyword = candidates.iter().any(|(idx, src)| {
            *src == CandidateSource::Keyword && chunks[*idx].path.contains("validators.py")
        });
        assert!(has_validators_keyword);
        // Verify no embedding candidates were returned
        let has_embedding = candidates.iter().any(|(_, src)| *src == CandidateSource::Embedding);
        assert!(!has_embedding, "embedding should return empty for orthogonal vectors");
    }

    #[test]
    fn hybrid_source_includes_embedding_results() {
        let repo = fake_repo();
        let chunker = Chunker::default();
        let chunks = chunker.chunk_repo(repo.path());
        let mut store = EmbeddingStore::new(4);
        for _ in &chunks {
            store.insert_batch(vec![vec![0.1; 4]]);
        }
        let task_vec = vec![0.1_f32; 4];
        let source = HybridCandidateSource::default();
        let candidates = source.generate(&store, &task_vec, &chunks, "nonexistent_keyword_xyz", repo.path(), None);
        let embedding_count = candidates.iter().filter(|(_, s)| *s == CandidateSource::Embedding).count();
        assert!(embedding_count > 0);
    }

    #[test]
    fn keyword_only_mode_no_embedding_overflow() {
        let repo = fake_repo();
        let chunker = Chunker::default();
        let chunks = chunker.chunk_repo(repo.path());
        let mut store = EmbeddingStore::new(4);
        for _ in &chunks {
            store.insert_batch(vec![vec![0.1; 4]]);
        }
        let task_vec = vec![0.1_f32; 4];
        let source = HybridCandidateSource {
            embedding_top_n: 1,
            ..Default::default()
        };
        let candidates = source.generate(&store, &task_vec, &chunks, "ASCIIUsernameValidator util", repo.path(), None);
        // Should have both embedding and keyword sources
        let sources: HashSet<_> = candidates.iter().map(|(_, s)| *s).collect();
        assert!(sources.contains(&CandidateSource::Embedding));
        assert!(sources.contains(&CandidateSource::Keyword));
    }

    #[test]
    fn error_message_source_adds_traceback_files() {
        let repo = fake_repo();
        let chunker = Chunker::default();
        let chunks = chunker.chunk_repo(repo.path());
        let mut store = EmbeddingStore::new(4);
        for _ in &chunks {
            store.insert_batch(vec![vec![1.0, 0.0, 0.0, 0.0]]);
        }
        let task_vec = vec![0.0, 1.0, 0.0, 0.0_f32];
        // Task contains a traceback referencing validators.py but no
        // identifier-like keywords that would trigger keyword-grep.
        let task = r#"Traceback:
  File "src/validators.py", line 42, in validate
ValueError: something"#;
        let source = HybridCandidateSource {
            include_keywords: false,
            ..Default::default()
        };
        let candidates = source.generate(&store, &task_vec, &chunks, task, repo.path(), None);
        let has_err_source = candidates.iter().any(|(idx, src)| {
            *src == CandidateSource::ErrorMessage && chunks[*idx].path.contains("validators.py")
        });
        assert!(has_err_source, "error traceback should produce candidates");
    }
}
