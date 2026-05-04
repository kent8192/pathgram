use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum CheckoutError {
    #[error("git error: {0}")]
    Git(#[from] git2::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("commit not found: {0}")]
    CommitNotFound(String),
}

/// Clone `repo_url` into `dest`, then check out `commit_sha`.
/// Returns the absolute path of the checked-out worktree.
pub fn checkout_repo(
    repo_url: &str,
    commit_sha: &str,
    dest: &Path,
) -> Result<PathBuf, CheckoutError> {
    let repo = git2::Repository::clone(repo_url, dest)?;
    let oid = git2::Oid::from_str(commit_sha)
        .map_err(|_| CheckoutError::CommitNotFound(commit_sha.into()))?;
    let commit = repo
        .find_commit(oid)
        .map_err(|_| CheckoutError::CommitNotFound(commit_sha.into()))?;
    repo.set_head_detached(commit.id())?;
    let mut co = git2::build::CheckoutBuilder::new();
    co.force();
    repo.checkout_head(Some(&mut co))?;
    Ok(dest.to_path_buf())
}

/// Per-repo bare-clone cache to amortize cloning across multiple instances.
pub struct RepoCache {
    cache_dir: PathBuf,
}

impl RepoCache {
    #[must_use]
    pub fn new(cache_dir: &Path) -> Self {
        std::fs::create_dir_all(cache_dir).ok();
        Self {
            cache_dir: cache_dir.to_path_buf(),
        }
    }

    fn cache_path_for(&self, repo_url: &str) -> PathBuf {
        // "https://github.com/django/django" -> "django__django.git"
        let parts: Vec<&str> = repo_url
            .trim_end_matches('/')
            .trim_end_matches(".git")
            .rsplit('/')
            .take(2)
            .collect();
        let suffix = parts.into_iter().rev().collect::<Vec<_>>().join("__");
        self.cache_dir.join(format!("{suffix}.git"))
    }

    pub fn checkout(
        &self,
        repo_url: &str,
        commit_sha: &str,
        dest: &Path,
    ) -> Result<PathBuf, CheckoutError> {
        let cache_path = self.cache_path_for(repo_url);
        if !cache_path.exists() {
            git2::build::RepoBuilder::new()
                .bare(true)
                .clone(repo_url, &cache_path)?;
        }
        // Local clone from the bare cache is fast (file copies, no network).
        git2::build::RepoBuilder::new()
            .clone(&format!("file://{}", cache_path.display()), dest)?;
        let repo = git2::Repository::open(dest)?;
        let oid = git2::Oid::from_str(commit_sha)
            .map_err(|_| CheckoutError::CommitNotFound(commit_sha.into()))?;
        let commit = repo
            .find_commit(oid)
            .map_err(|_| CheckoutError::CommitNotFound(commit_sha.into()))?;
        repo.set_head_detached(commit.id())?;
        let mut co = git2::build::CheckoutBuilder::new();
        co.force();
        repo.checkout_head(Some(&mut co))?;
        Ok(dest.to_path_buf())
    }
}
