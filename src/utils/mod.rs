//! Shared utilities: buffer pool, model path resolution.

mod buffer_pool;
mod model_path;

pub use buffer_pool::{AudioBufferPool, PooledBuffer};
pub use model_path::{default_model_dir, find_model_dir, is_model_dir};
