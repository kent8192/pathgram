use super::{Reranker, ScorerError};
use serde::{Deserialize, Serialize};

const RERANK_URL: &str = "https://api.cohere.com/v2/rerank";

pub struct CohereReranker {
    api_key: String,
    model: String,
    client: reqwest::blocking::Client,
}

impl CohereReranker {
    /// Construct from env var `COHERE_API_KEY`.
    pub fn from_env() -> Result<Self, ScorerError> {
        let api_key = std::env::var("COHERE_API_KEY")
            .map_err(|_| ScorerError::MissingApiKey("COHERE_API_KEY"))?;
        Ok(Self {
            api_key,
            model: "rerank-english-v3.0".to_string(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("client builds"),
        })
    }

    #[must_use]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

#[derive(Serialize)]
struct Req<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    top_n: usize,
}

#[derive(Deserialize)]
struct Resp {
    results: Vec<Result_>,
}

#[derive(Deserialize)]
#[allow(non_camel_case_types)]
struct Result_ {
    index: usize,
    relevance_score: f32,
}

impl Reranker for CohereReranker {
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, ScorerError> {
        if documents.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let body = Req {
            model: &self.model,
            query,
            documents,
            top_n: top_k.min(documents.len()),
        };
        let resp = self
            .client
            .post(RERANK_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(ScorerError::Status {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: Resp = resp
            .json()
            .map_err(|e| ScorerError::Decode(e.to_string()))?;
        Ok(parsed
            .results
            .into_iter()
            .map(|r| (r.index, r.relevance_score))
            .collect())
    }
}

/// Mock reranker that simply scores by reverse cosine of mock embeddings.
/// Useful for tests when no Cohere key is available. Acts as a stable
/// no-op: returns documents in original order with diminishing scores.
pub struct MockReranker;

impl Reranker for MockReranker {
    fn rerank(
        &self,
        _query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, ScorerError> {
        let n = documents.len().min(top_k);
        Ok((0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let score = 1.0 - (i as f32) / (documents.len().max(1) as f32);
                (i, score)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_reranker_returns_topk() {
        let docs = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let out = MockReranker.rerank("x", &docs, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 0);
        assert!(out[0].1 > out[1].1);
    }

    #[test]
    fn mock_reranker_empty_docs() {
        let out = MockReranker.rerank("x", &[], 5).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    #[ignore]
    fn live_cohere_rerank_smoke() {
        let r = CohereReranker::from_env().expect("COHERE_API_KEY set");
        let docs = vec![
            "the quick brown fox".into(),
            "rust programming language".into(),
            "machine learning models".into(),
        ];
        let out = r.rerank("rust code", &docs, 2).expect("rerank");
        assert!(!out.is_empty());
        assert!(out.len() <= 2);
    }
}
