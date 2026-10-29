//! Safe audio buffer pool for reusable processing buffers.
//!
//! The pool uses a mutex-protected free list so checked-out buffers can
//! coexist without borrowing the pool mutably. A checked-out buffer holds a
//! shared borrow of its pool, preventing the pool from being dropped first.
//! The mutex is not lock-free; sharing a pool across latency-sensitive threads
//! should be considered carefully.

use std::collections::VecDeque;
use std::sync::Mutex;

pub struct PooledBuffer<'a> {
    data: Vec<f32>,
    pool: &'a AudioBufferPool,
}

impl PooledBuffer<'_> {
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] { &mut self.data }
    #[inline]
    pub fn as_slice(&self) -> &[f32] { &self.data }
    #[inline]
    pub fn len(&self) -> usize { self.data.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    pub fn resize(&mut self, new_len: usize) {
        if self.data.len() >= new_len { self.data.truncate(new_len); }
        else { self.data.resize(new_len, 0.0); }
    }
    pub fn clear(&mut self) { self.data.clear(); }
    pub fn extend_from_slice(&mut self, src: &[f32]) { self.data.extend_from_slice(src); }
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        self.resize(src.len());
        self.data.copy_from_slice(src);
    }

    /// Consume the buffer without returning its vector to the pool.
    pub fn into_vec(self) -> Vec<f32> {
        let mut this = std::mem::ManuallyDrop::new(self);
        std::mem::take(&mut this.data)
    }
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) { self.pool.give_back(std::mem::take(&mut self.data)); }
}
impl std::ops::Deref for PooledBuffer<'_> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target { &self.data }
}
impl std::ops::DerefMut for PooledBuffer<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
}

pub struct AudioBufferPool {
    free: Mutex<VecDeque<Vec<f32>>>,
    max_buffers: usize,
    default_capacity: usize,
}

impl AudioBufferPool {
    pub fn new(max_buffers: usize, default_capacity: usize) -> Self {
        let mut free = VecDeque::with_capacity(max_buffers);
        for _ in 0..max_buffers.min(4) {
            free.push_back(Vec::with_capacity(default_capacity));
        }
        Self {
            free: Mutex::new(free),
            max_buffers: max_buffers.max(1),
            default_capacity,
        }
    }

    pub fn for_streaming() -> Self { Self::new(8, 2048) }

    pub fn acquire(&self, min_capacity: usize) -> PooledBuffer<'_> {
        let mut data = self.free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or_else(|| Vec::with_capacity(min_capacity.max(self.default_capacity)));

        if data.capacity() < min_capacity {
            data.reserve_exact(min_capacity);
        }
        data.clear();
        PooledBuffer { data, pool: self }
    }

    fn give_back(&self, mut buf: Vec<f32>) {
        let mut free = self.free.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if free.len() >= self.max_buffers { return; }
        buf.clear();
        if buf.capacity() > self.default_capacity.saturating_mul(8) {
            buf.shrink_to(self.default_capacity);
        }
        free.push_back(buf);
    }

    pub fn available(&self) -> usize {
        self.free.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::AudioBufferPool;

    #[test]
    fn returns_buffer_to_pool_on_drop() {
        let pool = AudioBufferPool::new(2, 16);
        let initial = pool.available();
        {
            let mut buffer = pool.acquire(8);
            buffer.extend_from_slice(&[1.0, 2.0, 3.0]);
            assert_eq!(buffer.as_slice(), &[1.0, 2.0, 3.0]);
        }
        assert_eq!(pool.available(), initial + 1);
        let reused = pool.acquire(8);
        assert!(reused.is_empty());
    }

    #[test]
    fn supports_multiple_checked_out_buffers() {
        let pool = AudioBufferPool::new(4, 16);
        let first = pool.acquire(8);
        let second = pool.acquire(32);
        assert!(first.is_empty());
        assert!(second.is_empty());
    }

    #[test]
    fn grows_to_requested_capacity() {
        let pool = AudioBufferPool::new(1, 4);
        let buffer = pool.acquire(128);
        assert!(buffer.data.capacity() >= 128);
    }

    #[test]
    fn into_vec_does_not_return_buffer_to_pool() {
        let pool = AudioBufferPool::new(1, 16);
        let initial = pool.available();
        let buffer = pool.acquire(8);
        let _owned = buffer.into_vec();
        assert_eq!(pool.available(), initial);
    }
}
