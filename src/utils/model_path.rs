//! Helpers to locate DeepFilterNet3 model directories.

use std::path::{Path, PathBuf};

/// Default relative model directory.
pub fn default_model_dir() -> PathBuf {
    PathBuf::from("models")
}

/// True if `dir` looks like a DeepFilterNet3 model directory.
pub fn is_model_dir(dir: &Path) -> bool {
    let has_split = dir.join("enc.onnx").is_file()
        && dir.join("erb_dec.onnx").is_file()
        && dir.join("df_dec.onnx").is_file();
    let has_combined = dir.join("deepfilter.onnx").is_file() || dir.join("dfn3.onnx").is_file();
    has_split || has_combined
}

/// Search common locations for model files.
pub fn find_model_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NOISE_CANCEL_MODEL_DIR") {
        let pb = PathBuf::from(p);
        if is_model_dir(&pb) {
            return Some(pb);
        }
    }

    let candidates = [
        PathBuf::from("models"),
        PathBuf::from("./models"),
        PathBuf::from("../models"),
        PathBuf::from("../../models"),
    ];

    for c in candidates {
        if is_model_dir(&c) {
            return Some(c);
        }
    }

    // Also try next to the executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let next_to_exe = parent.join("models");
            if is_model_dir(&next_to_exe) {
                return Some(next_to_exe);
            }
        }
    }

    None
}
