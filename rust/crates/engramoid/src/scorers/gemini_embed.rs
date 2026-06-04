use super::{Embedder, ScorerError};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const EMBED_URL_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const MIN_INTERVAL: Duration = Duration::from_millis(1500);
const MAX_RETRIES: u32 = 5;

pub struct GeminiEmbedder {
    api_key: String,
    model: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
    last_call: Mutex<Option<Instant>>,
}

impl GeminiEmbedder {
    /// Construct from env var `GEMINI_API_KEY`. Defaults to
    /// `gemini-embedding-001` at 768 dims (Matryoshka truncation from the
    /// native 3072d). text-embedding-004 was deprecated by Google in 2026.
    pub fn from_env() -> Result<Self, ScorerError> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .map_err(|_| ScorerError::MissingApiKey("GEMINI_API_KEY"))?;
        Ok(Self {
            api_key,
            model: "gemini-embedding-001".to_string(),
            dimensions: 768,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("client builds"),
            last_call: Mutex::new(None),
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
struct BatchReq {
    requests: Vec<EmbedReq>,
}

#[derive(Serialize)]
struct EmbedReq {
    model: String,
    content: Content,
    #[serde(rename = "taskType")]
    task_type: &'static str,
    #[serde(rename = "outputDimensionality", skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<usize>,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Deserialize)]
struct BatchResp {
    embeddings: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    values: Vec<f32>,
}

impl GeminiEmbedder {
    /// Sleep just long enough to keep at least `MIN_INTERVAL` between
    /// consecutive batch calls. Cheap throttle to stay well under
    /// gemini-embedding-001's RPM/TPM ceilings on Tier 1.
    fn throttle(&self) {
        let mut guard = self.last_call.lock().expect("mutex poisoned");
        if let Some(prev) = *guard {
            let elapsed = prev.elapsed();
            if elapsed < MIN_INTERVAL {
                std::thread::sleep(MIN_INTERVAL - elapsed);
            }
        }
        *guard = Some(Instant::now());
    }
}

impl Embedder for GeminiEmbedder {
    fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, ScorerError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        // The `Embedder` trait can't distinguish document vs query context,
        // so we always tag as RETRIEVAL_DOCUMENT. The asymmetric-retrieval
        // gap with RETRIEVAL_QUERY is ~1 MTEB point — small enough for a
        // paired B0 vs B4 comparison.
        let model_str = format!("models/{}", self.model);
        let body = BatchReq {
            requests: inputs
                .iter()
                .map(|s| EmbedReq {
                    model: model_str.clone(),
                    content: Content {
                        parts: vec![Part { text: s.clone() }],
                    },
                    task_type: "RETRIEVAL_DOCUMENT",
                    // gemini-embedding-001 is Matryoshka-trained; native
                    // output is 3072d, so we always send the desired dim
                    // to get a stable 768d back.
                    output_dimensionality: Some(self.dimensions),
                })
                .collect(),
        };
        let url = format!(
            "{}/{}:batchEmbedContents?key={}",
            EMBED_URL_BASE, self.model, self.api_key
        );

        // Retry on 429 with exponential backoff. gemini-embedding-001's
        // per-minute quota (RPM + TPM) can briefly trip even on Tier 1
        // during dense bursts; backoff makes the embedder robust without
        // requiring the caller to know about rate limits.
        let mut backoff = Duration::from_secs(2);
        for attempt in 0..=MAX_RETRIES {
            self.throttle();
            let resp = self.client.post(&url).json(&body).send()?;
            let status = resp.status();
            if status.is_success() {
                let parsed: BatchResp = resp
                    .json()
                    .map_err(|e| ScorerError::Decode(e.to_string()))?;
                return Ok(parsed.embeddings.into_iter().map(|e| e.values).collect());
            }
            let body_text = resp.text().unwrap_or_default();
            if status.as_u16() == 429 && attempt < MAX_RETRIES {
                eprintln!(
                    "gemini embed 429, backoff {:?} (attempt {}/{})",
                    backoff,
                    attempt + 1,
                    MAX_RETRIES
                );
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_secs(60));
                continue;
            }
            return Err(ScorerError::Status {
                status: status.as_u16(),
                body: body_text,
            });
        }
        unreachable!("loop returns or breaks before exhausting iterator");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn live_gemini_embed_smoke() {
        // run with: cargo test --features eval -- --ignored live_gemini
        let e = GeminiEmbedder::from_env().expect("GEMINI_API_KEY set");
        let v = e
            .embed_batch(&["hello".into(), "world".into()])
            .expect("embed");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].len(), 768);
    }
}
