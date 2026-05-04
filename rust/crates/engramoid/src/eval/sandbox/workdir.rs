use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// A scratch directory that auto-cleans on drop unless `keep()` was called.
pub struct Workdir {
    inner: Option<TempDir>,
    persisted_path: Option<PathBuf>,
}

impl Workdir {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            inner: Some(tempfile::tempdir()?),
            persisted_path: None,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        if let Some(p) = &self.persisted_path {
            p
        } else {
            self.inner.as_ref().expect("inner tempdir present").path()
        }
    }

    /// Take ownership of the underlying directory, preventing cleanup.
    pub fn keep(mut self) -> PathBuf {
        let td = self.inner.take().expect("already kept");
        let p = td.into_path();
        self.persisted_path = Some(p.clone());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workdir_creates_existing_directory() {
        let wd = Workdir::new().unwrap();
        assert!(wd.path().exists());
        assert!(wd.path().is_dir());
    }

    #[test]
    fn workdir_cleans_up_on_drop() {
        let path;
        {
            let wd = Workdir::new().unwrap();
            path = wd.path().to_path_buf();
            assert!(path.exists());
        }
        assert!(!path.exists(), "tempdir should be removed on drop");
    }
}
