use crate::retrieval::chunks::Chunk;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkCitation {
    pub path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlob {
    pub chunks: Vec<ChunkCitation>,
    pub total_tokens: usize,
}

impl ContextBlob {
    /// Distinct file paths cited by this blob, in citation order.
    pub fn distinct_paths(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for c in &self.chunks {
            if seen.insert(c.path.clone()) {
                out.push(c.path.clone());
            }
        }
        out
    }

    /// Render as a single Markdown document with `## file:line-line` headers.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        for c in &self.chunks {
            s.push_str(&format!("## {}:{}-{}\n", c.path, c.line_start, c.line_end));
            s.push_str("```\n");
            s.push_str(&c.text);
            if !c.text.ends_with('\n') {
                s.push('\n');
            }
            s.push_str("```\n\n");
        }
        s
    }
}

/// Approximate tokens via the well-known `chars / 4` heuristic.
fn approx_tokens(s: &str) -> usize {
    s.chars().count() / 4 + 1
}

pub struct Packer {
    pub token_budget: usize,
}

impl Default for Packer {
    fn default() -> Self {
        Self { token_budget: 8192 }
    }
}

impl Packer {
    /// Greedy fill: ranked chunks added until the next would overflow the
    /// budget. Overlapping ranges in the same file are deduplicated.
    pub fn pack(&self, ranked: &[(Chunk, f32)]) -> ContextBlob {
        let mut chunks: Vec<ChunkCitation> = Vec::new();
        let mut total = 0usize;
        for (chunk, score) in ranked {
            let header_tokens = approx_tokens(&format!(
                "## {}:{}-{}\n```\n```\n\n",
                chunk.path, chunk.line_start, chunk.line_end
            ));
            let text_tokens = approx_tokens(&chunk.text);
            let cost = header_tokens + text_tokens;
            if total + cost > self.token_budget {
                break;
            }
            // Skip if this exact range is already in the blob
            let dup = chunks.iter().any(|c| {
                c.path == chunk.path
                    && c.line_start == chunk.line_start
                    && c.line_end == chunk.line_end
            });
            if dup {
                continue;
            }
            chunks.push(ChunkCitation {
                path: chunk.path.clone(),
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                text: chunk.text.clone(),
                score: *score,
            });
            total += cost;
        }
        ContextBlob {
            chunks,
            total_tokens: total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(path: &str, ls: usize, le: usize, n_lines: usize) -> Chunk {
        Chunk {
            path: path.to_string(),
            line_start: ls,
            line_end: le,
            text: (0..n_lines).map(|i| format!("line {i}\n")).collect(),
        }
    }

    #[test]
    fn pack_respects_budget() {
        // Each chunk ~50 lines × 7 chars ≈ 90 tokens. Budget 250 → ≤2 chunks.
        let p = Packer { token_budget: 250 };
        let ranked = vec![
            (ch("a.py", 1, 50, 50), 0.9),
            (ch("b.py", 1, 50, 50), 0.8),
            (ch("c.py", 1, 50, 50), 0.7),
        ];
        let blob = p.pack(&ranked);
        assert!(blob.chunks.len() <= 2);
        assert!(blob.total_tokens <= 250);
    }

    #[test]
    fn pack_deduplicates_same_range() {
        let p = Packer::default();
        let ranked = vec![
            (ch("a.py", 1, 50, 50), 0.9),
            (ch("a.py", 1, 50, 50), 0.8), // dup
        ];
        let blob = p.pack(&ranked);
        assert_eq!(blob.chunks.len(), 1);
    }

    #[test]
    fn distinct_paths_preserves_order() {
        let p = Packer::default();
        let ranked = vec![
            (ch("a.py", 1, 50, 50), 0.9),
            (ch("b.py", 1, 50, 50), 0.8),
            (ch("a.py", 51, 100, 50), 0.7), // same file, different range
        ];
        let blob = p.pack(&ranked);
        let paths = blob.distinct_paths();
        assert_eq!(paths, vec!["a.py", "b.py"]);
    }

    #[test]
    fn markdown_render_has_headers() {
        let p = Packer::default();
        let ranked = vec![(ch("x.py", 5, 10, 6), 0.5)];
        let blob = p.pack(&ranked);
        let md = blob.to_markdown();
        assert!(md.contains("## x.py:5-10"));
        assert!(md.contains("```"));
    }
}
