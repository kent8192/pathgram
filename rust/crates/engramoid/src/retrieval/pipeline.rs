use crate::graph::builder::AstGraphBuilder;
use crate::retrieval::candidates::{CandidateSource, HybridCandidateSource};
use crate::retrieval::chunks::{Chunk, Chunker};
use crate::retrieval::embedding_store::EmbeddingStore;
use crate::retrieval::keyword;
use crate::retrieval::packer::{ContextBlob, Packer};
use crate::scorers::{Embedder, Reranker, ScorerError};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("scorer error: {0}")]
    Scorer(#[from] ScorerError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("empty repo: no code chunks extracted")]
    EmptyRepo,
}

pub struct FrozenGammaPipeline<'a> {
    pub embedder: &'a dyn Embedder,
    pub reranker: &'a dyn Reranker,
    pub chunker: Chunker,
    pub packer: Packer,
    pub candidate_top_n: usize,
    pub rerank_top_k: usize,
    pub embedding_dim: usize,
    pub embed_batch_size: usize,
    /// If the fraction of B0 (keyword-grep) files covered by B4 is below this,
    /// supplement the blob with B0-only chunks. Default 0.3.
    pub coverage_fallback_threshold: f64,
}

impl<'a> FrozenGammaPipeline<'a> {
    pub fn new(embedder: &'a dyn Embedder, reranker: &'a dyn Reranker) -> Self {
        Self {
            embedder,
            reranker,
            chunker: Chunker::default(),
            packer: Packer::default(),
            candidate_top_n: 200,
            rerank_top_k: 30,
            embedding_dim: 768,
            embed_batch_size: 64,
            coverage_fallback_threshold: 0.3,
        }
    }

    /// 3-stage retrieval against a sandboxed repo.
    #[allow(clippy::too_many_lines)]
    pub fn retrieve(&self, task: &str, repo_root: &Path) -> Result<ContextBlob, PipelineError> {
        // 1. Chunk the repo
        let chunks = self.chunker.chunk_repo(repo_root);
        if chunks.is_empty() {
            return Err(PipelineError::EmptyRepo);
        }

        // 2. Embed all chunks (batched) + the task
        let mut store = EmbeddingStore::new(self.embedding_dim);
        for batch in chunks.chunks(self.embed_batch_size) {
            let texts: Vec<String> = batch.iter().map(|c| c.text.clone()).collect();
            let vecs = self.embedder.embed_batch(&texts)?;
            store.insert_batch(vecs);
        }
        let task_vec = self.embedder.embed_batch(&[task.to_string()])?;
        let task_vec = task_vec.into_iter().next().unwrap_or_default();

        // 3. Build AST graph and embed Function/Class nodes for PPR
        let ast_builder = AstGraphBuilder::default();
        let mut ast_engine = ast_builder.build(repo_root);
        let has_ast_graph = ast_engine.node_count() > 0;
        if has_ast_graph {
            let ast_chunks = ast_builder.extract_chunks(&ast_engine, repo_root);
            if !ast_chunks.is_empty() {
                let texts: Vec<String> = ast_chunks.iter().map(|(_, c)| c.text.clone()).collect();
                if let Ok(vecs) = self.embedder.embed_batch(&texts) {
                    for ((node_id, _), mut vec) in ast_chunks.iter().zip(vecs) {
                        // L2-normalize so cosine = dot product
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

        // 4. Stage 1: hybrid candidate generation (embedding + keyword + PPR union)
        let candidate_source = HybridCandidateSource {
            embedding_top_n: self.candidate_top_n,
            include_ppr: has_ast_graph,
            ..Default::default()
        };
        let graph_opt = if has_ast_graph { Some(&ast_engine) } else { None };
        let candidates =
            candidate_source.generate(&store, &task_vec, &chunks, task, repo_root, graph_opt);
        if candidates.is_empty() {
            return Ok(ContextBlob {
                chunks: Vec::new(),
                total_tokens: 0,
            });
        }

        // Materialize candidate chunk texts for the reranker
        let cand_chunks: Vec<&Chunk> = candidates.iter().map(|(i, _)| &chunks[*i]).collect();
        let cand_texts: Vec<String> = cand_chunks.iter().map(|c| c.text.clone()).collect();

        // 5. Stage 2: rerank
        let reranked = self
            .reranker
            .rerank(task, &cand_texts, self.rerank_top_k)?;
        let ranked: Vec<(Chunk, f32)> = reranked
            .into_iter()
            .map(|(idx, score)| (cand_chunks[idx].clone(), score))
            .collect();

        // 6. Stage 3: pack into a token-budgeted blob
        let mut blob = self.packer.pack(&ranked);

        // 7. Coverage-aware fallback: if the blob misses a large fraction of
        //    the files that keyword-grep would have found, supplement.
        if self.coverage_fallback_threshold < 1.0 {
            let b0_paths = keyword::keyword_grep_hit_paths(
                task,
                repo_root,
                HybridCandidateSource::default().max_keywords,
                3,
                HybridCandidateSource::default().max_files_per_keyword,
                256 * 1024,
            );
            if !b0_paths.is_empty() {
                let b4_distinct = blob.distinct_paths();
                let b4_paths: std::collections::HashSet<&str> =
                    b4_distinct.iter().map(String::as_str).collect();
                let b0_set: std::collections::HashSet<&str> =
                    b0_paths.iter().map(String::as_str).collect();
                let overlap = b0_set.intersection(&b4_paths).count();
                #[allow(clippy::cast_precision_loss)]
                let coverage = overlap as f64 / b0_set.len() as f64;

                if coverage < self.coverage_fallback_threshold {
                    let b0_only: std::collections::HashSet<&str> = b0_set
                        .difference(&b4_paths)
                        .copied()
                        .collect();
                    // Augment candidates with B0-only-file chunks
                    let mut augmented: Vec<(usize, CandidateSource)> = candidates.clone();
                    let mut augmented_indices: Vec<usize> = Vec::new();
                    for (idx, chunk) in chunks.iter().enumerate() {
                        if b0_only.contains(chunk.path.as_str()) {
                            augmented.push((idx, CandidateSource::Keyword));
                            augmented_indices.push(idx);
                        }
                    }
                    if !augmented_indices.is_empty() {
                        let aug_chunks: Vec<&Chunk> =
                            augmented.iter().map(|(i, _)| &chunks[*i]).collect();
                        let aug_texts: Vec<String> =
                            aug_chunks.iter().map(|c| c.text.clone()).collect();
                        if let Ok(reranked) =
                            self.reranker.rerank(task, &aug_texts, self.rerank_top_k)
                        {
                            let ranked: Vec<(Chunk, f32)> = reranked
                                .into_iter()
                                .map(|(idx, score)| (aug_chunks[idx].clone(), score))
                                .collect();
                            blob = self.packer.pack(&ranked);
                        }
                    }
                }
            }
        }

        Ok(blob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorers::cohere_rerank::MockReranker;
    use crate::scorers::openai_embed::MockEmbedder;

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
        std::fs::write(root.join("src/other.py"), "def other():\n    pass\n").unwrap();
        td
    }

    #[test]
    fn pipeline_returns_a_blob_with_mocks() {
        let repo = fake_repo();
        let embedder = MockEmbedder::new(64);
        let reranker = MockReranker;
        let pipe = FrozenGammaPipeline {
            embedder: &embedder,
            reranker: &reranker,
            chunker: Chunker::default(),
            packer: Packer::default(),
            candidate_top_n: 10,
            rerank_top_k: 5,
            embedding_dim: 64,
            embed_batch_size: 4,
            coverage_fallback_threshold: 0.3,
        };
        let blob = pipe.retrieve("ASCIIUsernameValidator regex", repo.path()).unwrap();
        assert!(!blob.chunks.is_empty());
        let paths = blob.distinct_paths();
        assert!(paths.iter().all(|p| p.ends_with(".py")));
    }
}
