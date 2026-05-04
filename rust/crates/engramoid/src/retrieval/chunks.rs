use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// A single retrieval candidate: a contiguous slice of one file.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Path relative to the repo root.
    pub path: String,
    /// 1-indexed inclusive line range.
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
}

pub struct Chunker {
    pub lines_per_chunk: usize,
    pub overlap: usize,
    pub max_file_size_bytes: u64,
    pub max_chunks: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self {
            lines_per_chunk: 50,
            overlap: 10,
            max_file_size_bytes: 256 * 1024,
            max_chunks: 10_000,
        }
    }
}

impl Chunker {
    pub fn chunk_repo(&self, root: &Path) -> Vec<Chunk> {
        let mut out = Vec::new();
        let walker = WalkDir::new(root)
            .max_depth(10)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                let Some(name) = e.file_name().to_str() else { return false };
                !is_skip_component(name)
            });
        for entry in walker.flatten() {
            if out.len() >= self.max_chunks {
                break;
            }
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if !is_code_extension(path) {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if meta.len() > self.max_file_size_bytes {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(path) else {
                continue;
            };
            let rel = path
                .strip_prefix(root)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned();
            out.extend(self.chunk_file(&rel, &content));
        }
        out.truncate(self.max_chunks);
        out
    }

    fn chunk_file(&self, rel_path: &str, content: &str) -> Vec<Chunk> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Vec::new();
        }
        let step = self.lines_per_chunk.saturating_sub(self.overlap).max(1);
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < lines.len() {
            let end = (start + self.lines_per_chunk).min(lines.len());
            let text = lines[start..end].join("\n");
            out.push(Chunk {
                path: rel_path.to_string(),
                line_start: start + 1,
                line_end: end,
                text,
            });
            if end == lines.len() {
                break;
            }
            start += step;
        }
        out
    }
}

fn is_code_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "py" | "pyx"
            | "rs"
            | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "mjs"
            | "go"
            | "java"
            | "kt"
            | "scala"
            | "rb"
            | "c"
            | "cc"
            | "cpp"
            | "cxx"
            | "h"
            | "hpp"
            | "cs"
            | "swift"
            | "m"
            | "mm"
    )
}

fn is_skip_component(c: &str) -> bool {
    matches!(
        c,
        ".git"
            | "node_modules"
            | "vendor"
            | "third_party"
            | "build"
            | "dist"
            | "target"
            | "__pycache__"
            | ".pytest_cache"
            | ".mypy_cache"
            | ".tox"
            | "site-packages"
            | "venv"
            | ".venv"
            | "env"
            | ".env"
    )
}

#[allow(dead_code)]
fn _force_use(_p: PathBuf) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_repo() -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        // 120-line file → 3 chunks of 50 lines with 10 overlap (step=40)
        let mut content = String::new();
        for i in 1..=120 {
            content.push_str(&format!("line_{i}\n"));
        }
        std::fs::write(root.join("src/big.py"), content).unwrap();
        // small file → 1 chunk
        std::fs::write(root.join("src/small.py"), "a\nb\nc\n").unwrap();
        // skipped: not a code extension
        std::fs::write(root.join("README.md"), "# hi\n").unwrap();
        td
    }

    #[test]
    fn chunker_emits_overlapping_chunks() {
        let repo = fake_repo();
        let chunks = Chunker::default().chunk_repo(repo.path());
        let big_chunks: Vec<_> = chunks.iter().filter(|c| c.path.ends_with("big.py")).collect();
        assert!(
            big_chunks.len() >= 3,
            "expected ≥3 chunks for 120-line file, got {}",
            big_chunks.len()
        );
        // First chunk
        assert_eq!(big_chunks[0].line_start, 1);
        assert!(big_chunks[0].line_end >= 50);
        // Overlapping second chunk
        assert!(big_chunks[1].line_start <= big_chunks[0].line_end);
    }

    #[test]
    fn chunker_skips_non_code_files() {
        let repo = fake_repo();
        let chunks = Chunker::default().chunk_repo(repo.path());
        assert!(chunks.iter().all(|c| !c.path.ends_with("README.md")));
    }

    #[test]
    fn chunker_handles_small_files() {
        let repo = fake_repo();
        let chunks = Chunker::default().chunk_repo(repo.path());
        let small: Vec<_> = chunks.iter().filter(|c| c.path.ends_with("small.py")).collect();
        assert_eq!(small.len(), 1);
        assert_eq!(small[0].line_start, 1);
        assert_eq!(small[0].line_end, 3);
    }
}
