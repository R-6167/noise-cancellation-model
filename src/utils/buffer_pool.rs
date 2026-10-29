//! Lock-free-friendly audio buffer pool for real-time paths.
//!
//! Pre-allocates a small set of `Vec<f32>` buffers and reuses them
//! instead of allocating on every process() call. Designed for a
//! single audio thread (or external mutex if shared).

use std::collections::VecDeque;

/// A buffer checked out from the pool.
///
/// When dropped, the inner `Vec` is automatically returned to the pool
/// (if the pool still exists and the buffer capacity is acceptable).
pub struct PooledBuffer {
    data: Vec<f32>,
    pool: *mut AudioBufferPool,
}

// Safety: PooledBuffer is only used on the audio thread that owns the pool.
unsafe impl Send for PooledBuffer {}

impl PooledBuffer {
    /// Mutable view of the samples.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Immutable view.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Current length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Ensure capacity and set length, zero-filling new elements.
    pub fn resize(&mut self, new_len: usize) {
        if self.data.len() >= new_len {
            self.data.truncate(new_len);
        } else {
            self.data.resize(new_len, 0.0);
        }
    }

    /// Clear length to 0 without freeing capacity.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Extend from a slice (may allocate only if capacity is insufficient).
    pub fn extend_from_slice(&mut self, src: &[f32]) {
        self.data.extend_from_slice(src);
    }

    /// Copy from a slice, resizing as needed.
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        self.resize(src.len());
        self.data.copy_from_slice(src);
    }

    /// Consume and return the inner Vec (does NOT return it to the pool).
    pub fn into_vec(mut self) -> Vec<f32> {
        let data = std::mem::take(&mut self.data);
        self.pool = std::ptr::null_mut();
        data
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if !self.pool.is_null() {
            unsafe {
                (*self.pool).give_back(std::mem::take(&mut self.data));
            }
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Simple audio buffer pool.
pub struct AudioBufferPool {
    free: VecDeque<Vec<f32>>,
    max_buffers: usize,
    default_capacity: usize,
}

impl AudioBufferPool {
    /// Create a pool that keeps up to `max_buffers` buffers,
    /// each pre-sized to `default_capacity` samples.
    pub fn new(max_buffers: usize, default_capacity: usize) -> Self {
        let mut free = VecDeque::with_capacity(max_buffers);
        for _ in 0..max_buffers.min(4) {
            free.push_back(Vec::with_capacity(default_capacity));
        }
        Self {
            free,
            max_buffers: max_buffers.max(1),
            default_capacity,
        }
    }

    /// Default pool sized for streaming 10 ms hops with some headroom.
    pub fn for_streaming() -> Self {
        Self::new(8, 2048)
    }

    /// Acquire a buffer with at least `min_capacity` capacity.
    /// Length is set to 0.
    pub fn acquire(&mut self, min_capacity: usize) -> PooledBuffer {
        let mut data = self.free.pop_front().unwrap_or_else(|| {
            Vec::with_capacity(min_capacity.max(self.default_capacity))
        });

        if data.capacity() < min_capacity {
            data.reserve(min_capacity - data.capacity());
        }
        data.clear();

        PooledBuffer {
            data,
            pool: self as *mut AudioBufferPool,
        }
    }

    fn give_back(&mut self, mut buf: Vec<f32>) {
        if self.free.len() >= self.max_buffers {
            return;
        }
        buf.clear();
        if buf.capacity() > self.default_capacity * 8 {
            buf.shrink_to(self.default_capacity);
        }
        self.free.push_back(buf);
    }

    /// Number of buffers currently sitting in the free list.
    pub fn available(&self) -> usize {
        self.free.len()
    }
}

unsafe impl Send for AudioBufferPool {}
