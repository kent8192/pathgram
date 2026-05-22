use crate::retrieval::chunks::{Chunk, Chunker};
use crate::retrieval::embedding_store::EmbeddingStore;
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
        }
    }

    /// 3-stage retrieval against a sandboxed repo.
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

        // 3. Stage 1: cosine top-N candidates
        let candidates = store.top_n(&task_vec, self.candidate_top_n);
        if candidates.is_empty() {
            return Ok(ContextBlob {
                chunks: Vec::new(),
                total_tokens: 0,
            });
        }

        // Materialize candidate chunk texts for the reranker
        let cand_chunks: Vec<&Chunk> = candidates.iter().map(|(i, _)| &chunks[*i]).collect();
        let cand_texts: Vec<String> = cand_chunks.iter().map(|c| c.text.clone()).collect();

        // 4. Stage 2: rerank
        let reranked = self
            .reranker
            .rerank(task, &cand_texts, self.rerank_top_k)?;
        let ranked: Vec<(Chunk, f32)> = reranked
            .into_iter()
            .map(|(idx, score)| (cand_chunks[idx].clone(), score))
            .collect();

        // 5. Stage 3: pack into a token-budgeted blob
        Ok(self.packer.pack(&ranked))
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
        };
        let blob = pipe.retrieve("ASCIIUsernameValidator regex", repo.path()).unwrap();
        assert!(!blob.chunks.is_empty());
        let paths = blob.distinct_paths();
        assert!(paths.iter().all(|p| p.ends_with(".py")));
    }
}
