//! Frozen semantic scorers used by the Phase 2.1 frozen γ retrieval
//! pipeline. Currently:
//!
//! - `gemini_embed::GeminiEmbedder` — gemini-embedding-001 @ 768 dims (Matryoshka)
//! - `cohere_rerank::CohereReranker` — rerank-english-v3.0
//!
//! `openai_embed::OpenAiEmbedder` is retained for parity / future paired
//! runs but is no longer wired into the CLI.
//!
//! Each implementation exposes a small trait so unit tests and offline
//! runs can plug in deterministic mocks.

pub mod cohere_rerank;
pub mod gemini_embed;
pub mod openai_embed;

#[derive(Debug, thiserror::Error)]
pub enum ScorerError {
    #[error("missing API key in env: {0}")]
    MissingApiKey(&'static str),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("scorer rejected the request: status={status} body={body}")]
    Status { status: u16, body: String },
    #[error("unexpected response shape: {0}")]
    Decode(String),
}

pub trait Embedder {
    /// Embed a batch of input strings, returning one vector per input in
    /// the same order.
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ScorerError>;
}

pub trait Reranker {
    /// Re-rank `documents` against `query`. Returns `(index_into_documents,
    /// relevance_score)` pairs sorted descending by score, length up to
    /// `top_k`.
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, ScorerError>;
}
