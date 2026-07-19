//! Weight downloading via hf-hub with offline caching.
//!
//! Handles:
//! - First-run download from Hugging Face
//! - Offline mode after initial download
//! - Clear error messages for 401/403 (gated model + HF_TOKEN)

use std::path::{Path, PathBuf};

/// Model repository identifier
pub const MODEL_REPO: &str = "google/embeddinggemma-300m";

/// Tokenizer config file
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Model weights file (safetensors)
const WEIGHTS_FILE: &str = "model.safetensors";

/// Config file
const CONFIG_FILE: &str = "config.json";

/// All files that need to be downloaded
const REQUIRED_FILES: &[&str] = &[
    TOKENIZER_FILE,
    WEIGHTS_FILE,
    CONFIG_FILE,
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
];

/// Error type for download operations
#[derive(Debug)]
pub enum DownloadError {
    /// Network error during download
    Network(String),
    /// Authentication error (401/403) - likely missing HF_TOKEN or unaccepted license
    Auth(String),
    /// Model not found (404)
    NotFound(String),
    /// Failed to parse config/tokenizer JSON
    ParseError(String),
    /// IO error during file operations
    IoError(std::io::Error),
    /// Model download incomplete or corrupted
    Incomplete(String),
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadError::Network(msg) => write!(f, "Network error: {msg}"),
            DownloadError::Auth(msg) => {
                write!(
                    f,
                    "Authentication error: {msg}\n\
                     EmbeddingGemma requires a Hugging Face account with accepted license.\n\
                     1. Visit https://huggingface.co/{MODEL_REPO}\n\
                     2. Log in and acknowledge the license\n\
                     3. Set HF_TOKEN environment variable or configure hf-hub\n\
                     Error: {msg}"
                )
            }
            DownloadError::NotFound(msg) => write!(f, "Model not found: {msg}"),
            DownloadError::ParseError(msg) => write!(f, "Config parse error: {msg}"),
            DownloadError::IoError(err) => write!(f, "IO error: {err}"),
            DownloadError::Incomplete(msg) => write!(f, "Download incomplete: {msg}"),
        }
    }
}

impl std::error::Error for DownloadError {}

impl From<std::io::Error> for DownloadError {
    fn from(err: std::io::Error) -> Self {
        DownloadError::IoError(err)
    }
}

/// Resolve the hf-hub repo directory name from a repo ID (e.g. "google/model" -> "models--google--model").
fn repo_dir_name(repo_id: &str) -> String {
    format!("{}--{}", "models", repo_id.replace('/', "--"))
}

/// Resolve the snapshot directory inside an hf-hub cache given a cache root and repo ID.
///
/// hf-hub stores files under:
///   <span><code>cache</code></span>/<span><code>repo-dir</code></span>/snapshots/<span><code>commit-hash</code></span>/
/// The commit hash is read from <span><code>cache</code></span>/<span><code>repo-dir</code></span>/refs/main.
fn resolve_snapshot_dir(cache_dir: &Path, repo_id: &str) -> Option<PathBuf> {
    let repo_dir = cache_dir.join(repo_dir_name(repo_id));
    let refs_main = repo_dir.join("refs").join("main");
    let commit_hash = std::fs::read_to_string(&refs_main).ok()?;
    let commit_hash = commit_hash.trim();
    Some(repo_dir.join("snapshots").join(commit_hash))
}

/// Create symlinks from flat layout (<cache_dir>/file) to hf-hub snapshot directory.
///
/// hf-hub stores downloaded files in a nested structure:
///   <span><code>cache</code></span>/models--author--repo/snapshots/<span><code>hash</code></span>/file
/// But load_model and is_model_ready expect a flat layout:
///   <span><code>cache</code></span>/file
/// This bridges the gap by creating symlinks.
fn link_flat_layout(cache_dir: &Path, snapshot_dir: &Path) -> Result<(), DownloadError> {
    for file in REQUIRED_FILES {
        let target = snapshot_dir.join(file);
        let link = cache_dir.join(file);

        // Ensure parent directory exists (e.g. for "1_Pooling/config.json")
        if let Some(parent) = link.parent() {
            std::fs::create_dir_all(parent).map_err(DownloadError::IoError)?;
        }

        // Remove existing file/link and create fresh symlink
        if link.exists() || link.is_symlink() {
            std::fs::remove_file(&link).map_err(DownloadError::IoError)?;
        }

        // Use absolute symlink for reliability across platforms
        let target_path = target.to_string_lossy().to_string();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target_path, &link).map_err(DownloadError::IoError)?;
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(&target_path, &link).map_err(DownloadError::IoError)?;

        tracing::debug!("Linked {file} <- {target_path}");
    }
    Ok(())
}

/// Download model weights and configs to the specified cache directory.
///
/// Uses hf-hub to download required files. On success, returns the path to the
/// downloaded model directory. Subsequent calls with the same cache_dir will
/// be fast (files already present in hf-hub cache).
pub async fn download_model(cache_dir: &Path) -> Result<PathBuf, DownloadError> {
    // Create cache directory if it doesn't exist
    std::fs::create_dir_all(cache_dir).map_err(DownloadError::IoError)?;

    tracing::info!(
        "Downloading EmbeddingGemma-300M model to {}",
        cache_dir.display()
    );

    // Use hf-hub with our custom cache directory
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| DownloadError::Network(format!("Failed to create HF client: {e}")))?;
    let api_repo = api.model(MODEL_REPO.to_string());

    let mut downloaded_files = Vec::new();

    for file_pattern in REQUIRED_FILES {
        match api_repo.download(file_pattern) {
            Ok(local_path) => {
                tracing::debug!("Downloaded: {file_pattern}");
                downloaded_files.push(local_path);
            }
            Err(hf_hub::api::sync::ApiError::RequestError(ureq_err)) => {
                // ureq encodes HTTP status codes in Error::Status variant
                let msg = format!("{ureq_err}");
                if msg.contains("401") || msg.contains("403") {
                    return Err(DownloadError::Auth(format!(
                        "Authentication error: You may need to accept the EmbeddingGemma license and provide a valid HF_TOKEN. {msg}"
                    )));
                }
                if msg.contains("404") {
                    return Err(DownloadError::NotFound(format!(
                        "Model or file not found: {msg}"
                    )));
                }
                return Err(DownloadError::Network(format!(
                    "Failed to download {file_pattern}: {msg}"
                )));
            }
            Err(e) => {
                return Err(DownloadError::Network(format!(
                    "Failed to download {file_pattern}: {e}"
                )));
            }
        }
    }

    // Resolve hf-hub's snapshot directory and create flat-layout symlinks
    let snapshot_dir = resolve_snapshot_dir(cache_dir, MODEL_REPO).ok_or_else(|| {
        DownloadError::Incomplete(
            "Could not resolve hf-hub snapshot directory. Check refs/main.".to_string(),
        )
    })?;

    link_flat_layout(cache_dir, &snapshot_dir)?;

    // Verify all required files are accessible via flat layout
    for file in REQUIRED_FILES {
        let expected = cache_dir.join(file);
        if !expected.exists() {
            return Err(DownloadError::Incomplete(format!(
                "Missing required file: {}",
                file
            )));
        }
    }

    tracing::info!(
        "Successfully downloaded {} files to {}",
        downloaded_files.len(),
        cache_dir.display()
    );

    Ok(cache_dir.to_path_buf())
}

/// Check if the model is already downloaded and valid.
pub fn is_model_ready(cache_dir: &Path) -> bool {
    REQUIRED_FILES
        .iter()
        .all(|file| cache_dir.join(file).exists())
}

/// Get the default model cache directory (~/.cache/notectl/search/models/)
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("notectl")
        .join("search")
        .join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cache_dir() {
        let cache = default_cache_dir();
        assert!(cache.to_string_lossy().contains("notectl/search/models"));
    }

    #[test]
    fn test_error_display_auth() {
        let err = DownloadError::Auth("invalid token".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("HF_TOKEN"));
        assert!(msg.contains(MODEL_REPO));
    }

    #[test]
    fn test_required_files_not_empty() {
        assert!(!REQUIRED_FILES.is_empty());
        assert!(REQUIRED_FILES.len() >= 8);
    }

    #[test]
    fn test_is_model_ready_missing_dir() {
        // Returns false when cache directory doesn't exist.
        let non_existent = PathBuf::from(format!(
            "/tmp/notectl-test-model-cache-xyz-{}",
            std::process::id()
        ));
        assert!(
            !non_existent.exists(),
            "Test precondition: path must not exist"
        );
        assert!(!is_model_ready(&non_existent));
    }

    #[test]
    fn test_is_model_ready_partial_files() {
        // Returns false when some required files are missing.
        let dir =
            std::env::temp_dir().join(format!("notectl-partial-model-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Create only one of the required files.
        std::fs::write(dir.join(TOKENIZER_FILE), "{}").unwrap();

        assert!(!is_model_ready(&dir));

        // Cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }
}
