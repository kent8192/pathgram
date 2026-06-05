use super::{Embedder, ScorerError};
use serde::{Deserialize, Serialize};

const EMBED_URL: &str = "https://api.openai.com/v1/embeddings";
const MAX_EMBED_INPUT_CHARS: usize = 12_000;
const MAX_RETRIES: usize = 5;

pub struct OpenAiEmbedder {
    api_key: String,
    model: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
}

impl OpenAiEmbedder {
    /// Construct from env var `OPENAI_API_KEY`.
    pub fn from_env() -> Result<Self, ScorerError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ScorerError::MissingApiKey("OPENAI_API_KEY"))?;
        Ok(Self {
            api_key,
            model: "text-embedding-3-small".to_string(),
            dimensions: 1024,
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

    #[must_use]
    pub fn with_dimensions(mut self, dim: usize) -> Self {
        self.dimensions = dim;
        self
    }
}

#[derive(Serialize)]
struct Req<'a> {
    input: &'a [String],
    model: &'a str,
    dimensions: usize,
    encoding_format: &'static str,
}

#[derive(Deserialize)]
struct Resp {
    data: Vec<DataItem>,
}

#[derive(Deserialize)]
struct DataItem {
    embedding: Vec<f32>,
    index: usize,
}

impl Embedder for OpenAiEmbedder {
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ScorerError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let normalized_inputs: Vec<String> =
            inputs.iter().map(|input| truncate_input(input)).collect();
        let body = Req {
            input: &normalized_inputs,
            model: &self.model,
            dimensions: self.dimensions,
            encoding_format: "float",
        };
        let mut last_status_body = None;
        for attempt in 0..=MAX_RETRIES {
            let resp = self
                .client
                .post(EMBED_URL)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()?;
            let status = resp.status();
            if status.is_success() {
                let parsed: Resp = resp
                    .json()
                    .map_err(|e| ScorerError::Decode(e.to_string()))?;
                // Sort by index in case API doesn't preserve order
                let mut data = parsed.data;
                data.sort_by_key(|d| d.index);
                return Ok(data.into_iter().map(|d| d.embedding).collect());
            }

            let body = resp.text().unwrap_or_default();
            let should_retry = status.as_u16() == 429 || status.is_server_error();
            if !should_retry || attempt == MAX_RETRIES {
                return Err(ScorerError::Status {
                    status: status.as_u16(),
                    body,
                });
            }
            last_status_body = Some((status.as_u16(), body));
            let delay_secs = 2_u64.pow(attempt.min(5) as u32);
            std::thread::sleep(std::time::Duration::from_secs(delay_secs));
        }

        if let Some((status, body)) = last_status_body {
            return Err(ScorerError::Status { status, body });
        }
        Err(ScorerError::Decode(
            "embedding request exhausted without a response".to_string(),
        ))
    }
}

fn truncate_input(input: &str) -> String {
    if input.len() <= MAX_EMBED_INPUT_CHARS {
        return input.to_string();
    }

    let mut end = 0;
    for (idx, _) in input.char_indices() {
        if idx > MAX_EMBED_INPUT_CHARS {
            break;
        }
        end = idx;
    }
    input[..end].to_string()
}

/// Deterministic mock embedder for tests / offline mode. Hashes the input
/// string into a `dim`-dimensional unit vector. Same input → same vector,
/// roughly preserves rough similarity for shared substrings.
pub struct MockEmbedder {
    pub dim: usize,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for MockEmbedder {
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ScorerError> {
        Ok(inputs.iter().map(|s| mock_embed(s, self.dim)).collect())
    }
}

fn mock_embed(input: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut v = vec![0.0f32; dim];
    // Each token contributes a hashed projection to a small set of dims.
    for tok in input.split_whitespace() {
        let mut h = DefaultHasher::new();
        tok.hash(&mut h);
        let seed = h.finish();
        for i in 0..4 {
            let idx = ((seed.wrapping_add(i as u64 * 2654435761)) as usize) % dim;
            let sign = if (seed >> i) & 1 == 0 { 1.0f32 } else { -1.0 };
            v[idx] += sign;
        }
    }
    // L2-normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= norm;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_embedder_returns_unit_vectors() {
        let m = MockEmbedder::new(64);
        let v = m.embed_batch(&["hello world".into()]).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].len(), 64);
        let norm: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm = {norm}");
    }

    #[test]
    fn mock_embedder_is_deterministic() {
        let m = MockEmbedder::new(32);
        let a = m.embed_batch(&["abc def".into()]).unwrap();
        let b = m.embed_batch(&["abc def".into()]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mock_embedder_distinguishes_inputs() {
        let m = MockEmbedder::new(32);
        let res = m.embed_batch(&["alpha".into(), "beta".into()]).unwrap();
        assert_ne!(res[0], res[1]);
    }

    #[test]
    fn truncate_input_caps_long_text_on_char_boundary() {
        let input = format!("{}é", "a".repeat(MAX_EMBED_INPUT_CHARS + 16));
        let truncated = truncate_input(&input);
        assert!(truncated.len() <= MAX_EMBED_INPUT_CHARS);
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    #[ignore]
    fn live_openai_embed_smoke() {
        // run with: cargo test --features eval -- --ignored live_openai
        let e = OpenAiEmbedder::from_env().expect("OPENAI_API_KEY set");
        let v = e
            .embed_batch(&["hello".into(), "world".into()])
            .expect("embed");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].len(), 1024);
    }
}
