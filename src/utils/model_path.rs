//! Resolve the DeepFilterNet3 model directory without user downloads.
//!
//! Search order:
//! 1. `NOISE_CANCEL_MODEL_DIR` environment variable
//! 2. `./models` (current working directory)
//! 3. `CARGO_MANIFEST_DIR/models` (crate source tree)
//! 4. Directory next to the running executable: `<exe>/models`

use std::env;
use std::path::{Path, PathBuf};

/// Returns true if `dir` looks like a usable DFN3 model directory.
pub fn is_model_dir(dir: &Path) -> bool {
    let has_split = dir.join("enc.onnx").is_file()
        && dir.join("erb_dec.onnx").is_file()
        && dir.join("df_dec.onnx").is_file();
    let has_combined = dir.join("deepfilter.onnx").is_file() || dir.join("dfn3.onnx").is_file();
    has_split || has_combined
}

/// Find a model directory, or `None` if none is available.
pub fn find_model_dir() -> Option<PathBuf> {
    if let Ok(p) = env::var("NOISE_CANCEL_MODEL_DIR") {
        let p = PathBuf::from(p);
        if is_model_dir(&p) {
            return Some(p);
        }
    }

    let cwd = PathBuf::from("models");
    if is_model_dir(&cwd) {
        return Some(cwd);
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    if is_model_dir(&manifest) {
        return Some(manifest);
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            let beside = parent.join("models");
            if is_model_dir(&beside) {
                return Some(beside);
            }
        }
    }

    None
}

/// Default model dir for APIs that take a path.
pub fn default_model_dir() -> PathBuf {
    find_model_dir().unwrap_or_else(|| PathBuf::from("models"))
}
