use std::cmp::Ordering;

/// In-memory embedding store with brute-force cosine top-N. Inputs are
/// L2-normalized at insert time so the cosine score is just a dot product.
pub struct EmbeddingStore {
    /// Pre-normalized vectors.
    vectors: Vec<Vec<f32>>,
    pub dim: usize,
}

impl EmbeddingStore {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Insert pre-built vectors; each must be `dim`-dimensional. They are
    /// L2-normalized in place.
    pub fn insert_batch(&mut self, vecs: Vec<Vec<f32>>) {
        for mut v in vecs {
            assert_eq!(
                v.len(),
                self.dim,
                "embedding store dim mismatch: expected {}, got {}",
                self.dim,
                v.len()
            );
            l2_normalize(&mut v);
            self.vectors.push(v);
        }
    }

    /// Return up to `top_n` `(index, cosine_score)` pairs sorted descending.
    pub fn top_n(&self, query: &[f32], top_n: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dim || self.vectors.is_empty() {
            return Vec::new();
        }
        let mut q = query.to_vec();
        l2_normalize(&mut q);
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(v, &q)))
            .collect();
        // Partial sort: top_n by score desc
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(top_n);
        scored
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v {
        *x /= n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_n_returns_most_similar_first() {
        let mut store = EmbeddingStore::new(3);
        store.insert_batch(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ]);
        let r = store.top_n(&[1.0, 0.0, 0.0], 3);
        // Most similar is itself (idx 0), then idx 2 (1,1,0) at 0.707, then idx 1 at 0
        assert_eq!(r[0].0, 0);
        assert_eq!(r[1].0, 2);
        assert_eq!(r[2].0, 1);
        assert!(r[0].1 > 0.99);
    }

    #[test]
    fn top_n_truncates() {
        let mut store = EmbeddingStore::new(2);
        store.insert_batch(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]]);
        let r = store.top_n(&[1.0, 0.0], 2);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn empty_store_returns_empty() {
        let store = EmbeddingStore::new(4);
        let r = store.top_n(&[0.0; 4], 5);
        assert!(r.is_empty());
    }

    #[test]
    fn dim_mismatch_query_returns_empty() {
        let mut store = EmbeddingStore::new(3);
        store.insert_batch(vec![vec![1.0, 0.0, 0.0]]);
        let r = store.top_n(&[1.0, 0.0], 5);
        assert!(r.is_empty());
    }
}
