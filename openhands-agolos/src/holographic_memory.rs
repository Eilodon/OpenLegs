//! Holographic Associative Memory (HAM)
//!
//! FFT-based distributed memory storage for content-addressable retrieval.
//! Information is stored as interference patterns across the entire memory trace,
//! enabling O(dim log dim) retrieval time.
//!
//! # Key Features
//! - Content-addressable recall by similarity
//! - Graceful degradation through superposition
//! - Temporal decay for natural forgetting
//! - Capacity-based eviction (soft LRU)
//!
//! # Feature Flag
//! This module requires the `memory-holographic` feature.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Complex32 type alias for convenience
pub type Complex32 = Complex<f32>;

/// Holographic Associative Memory
///
/// # Mathematical Model
/// - Write (Entangle): M_new = M_old + IFFT(FFT(key) ⊙ FFT(value))
/// - Read (Recall): result = IFFT(FFT(M) ⊙ conj(FFT(key)))
///
/// # Invariants
/// - `memory_trace.len() == dim` at all times
/// - Retrieval is O(dim log dim)
/// - Energy is bounded through decay
pub struct HolographicMemory {
    /// The holographic memory trace
    memory_trace: Vec<Complex32>,
    /// Dimension of the memory space
    dim: usize,
    /// Forward FFT planner
    fft: Arc<dyn Fft<f32>>,
    /// Inverse FFT planner
    ifft: Arc<dyn Fft<f32>>,
    /// Normalization factor for IFFT (1/dim)
    norm_factor: f32,
    /// Number of items stored
    item_count: usize,
    /// Maximum trace magnitude before decay
    max_magnitude: f32,
    /// Timestamp of last entanglement
    last_entangle_ts_us: Option<i64>,
    /// Maximum items before decay eviction
    max_items: usize,
    /// Decay factor for capacity eviction
    capacity_decay_factor: f32,
    /// Scratch buffers for zero-allocation FFT
    scratch_a: Vec<Complex32>,
    scratch_b: Vec<Complex32>,
    scratch_c: Vec<Complex32>,
}

impl std::fmt::Debug for HolographicMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HolographicMemory")
            .field("dim", &self.dim)
            .field("item_count", &self.item_count)
            .field("energy", &self.energy())
            .finish_non_exhaustive()
    }
}

impl HolographicMemory {
    /// Create a new holographic memory with given dimension.
    ///
    /// # Arguments
    /// * `dim` - Dimension of memory space. Recommended: 256-1024.
    ///
    /// # Panics
    /// Panics if dim is 0.
    pub fn new(dim: usize) -> Self {
        Self::with_capacity(dim, 10_000)
    }

    /// Create with dimension and capacity limit.
    ///
    /// # Arguments
    /// * `dim` - Dimension of memory space
    /// * `max_items` - Maximum items before decay eviction
    pub fn with_capacity(dim: usize, max_items: usize) -> Self {
        assert!(dim > 0, "Memory dimension must be positive");

        let mut planner = FftPlanner::new();
        Self {
            memory_trace: vec![Complex32::new(0.0, 0.0); dim],
            dim,
            fft: planner.plan_fft_forward(dim),
            ifft: planner.plan_fft_inverse(dim),
            norm_factor: 1.0 / (dim as f32),
            item_count: 0,
            max_magnitude: 100.0,
            last_entangle_ts_us: None,
            max_items,
            capacity_decay_factor: 0.9,
            scratch_a: vec![Complex32::new(0.0, 0.0); dim],
            scratch_b: vec![Complex32::new(0.0, 0.0); dim],
            scratch_c: vec![Complex32::new(0.0, 0.0); dim],
        }
    }

    /// Default configuration for OpenHands agent memory
    pub fn default_for_agent() -> Self {
        Self::with_capacity(512, 10_000)
    }

    /// Entangle (write) a key-value pair into memory.
    ///
    /// # Mathematical Operation
    /// M_new = M_old + IFFT(FFT(key) ⊙ FFT(value))
    ///
    /// # Panics
    /// Panics if key.len() != dim or value.len() != dim
    pub fn entangle(&mut self, key: &[Complex32], value: &[Complex32]) {
        debug_assert_eq!(key.len(), self.dim, "Key dimension mismatch");
        debug_assert_eq!(value.len(), self.dim, "Value dimension mismatch");

        // Energy cap to prevent overflow
        let current_energy = self.energy();
        let critical_threshold =
            self.max_magnitude * self.max_magnitude * (self.dim as f32) * 0.8;

        if current_energy > critical_threshold {
            let decay_factor = (critical_threshold / current_energy).sqrt().min(0.9);
            log::warn!(
                "HolographicMemory: Energy cap triggered ({}), decay={:.3}",
                current_energy, decay_factor
            );
            self.decay(decay_factor);
        }

        // Capacity eviction
        if self.item_count >= self.max_items {
            log::debug!("HolographicMemory: Capacity {} reached, applying decay", self.max_items);
            self.decay(self.capacity_decay_factor);
        }

        // FFT operations using scratch buffers
        self.scratch_a.copy_from_slice(key);
        self.fft.process(&mut self.scratch_a);

        self.scratch_b.copy_from_slice(value);
        self.fft.process(&mut self.scratch_b);

        // Hadamard product
        for (a, b) in self.scratch_a.iter_mut().zip(self.scratch_b.iter()) {
            *a = *a * b;
        }

        // IFFT
        self.scratch_c.copy_from_slice(&self.scratch_a);
        self.ifft.process(&mut self.scratch_c);

        // Superpose onto memory trace
        let norm = self.norm_factor;
        for (m, c) in self.memory_trace.iter_mut().zip(self.scratch_c.iter()) {
            *m = *m + *c * norm;
        }

        self.item_count += 1;

        // Safety check for NaN/Inf
        let max_mag = self.memory_trace.iter()
            .map(|c| c.norm())
            .fold(0.0f32, f32::max);

        if !max_mag.is_finite() {
            log::error!("HolographicMemory: NaN/Inf detected! Resetting.");
            self.clear();
            return;
        }

        if max_mag > self.max_magnitude {
            self.decay(0.9);
        }
    }

    /// Entangle from raw f32 slices
    pub fn entangle_real(&mut self, key: &[f32], value: &[f32]) {
        let key_c: Vec<Complex32> = key.iter().map(|&r| Complex32::new(r, 0.0)).collect();
        let val_c: Vec<Complex32> = value.iter().map(|&r| Complex32::new(r, 0.0)).collect();
        self.entangle(&key_c, &val_c);
    }

    /// Recall values associated with a key.
    ///
    /// # Mathematical Operation
    /// result = IFFT(FFT(M) ⊙ conj(FFT(key)))
    pub fn recall(&self, key: &[Complex32]) -> Vec<Complex32> {
        debug_assert_eq!(key.len(), self.dim, "Key dimension mismatch");

        // FFT of memory and key
        let mut m_fft = self.fft_process(&self.memory_trace);
        let k_fft = self.fft_process(key);

        // Correlation via conjugate multiplication
        for (m, k) in m_fft.iter_mut().zip(k_fft.iter()) {
            *m = *m * k.conj();
        }

        // IFFT to get recalled value
        self.ifft_process(&m_fft)
    }

    /// Recall from raw f32 slice
    pub fn recall_real(&self, key: &[f32]) -> Vec<f32> {
        let key_c: Vec<Complex32> = key.iter().map(|&r| Complex32::new(r, 0.0)).collect();
        self.recall(&key_c).iter().map(|c| c.re).collect()
    }

    /// Find most similar stored key and return its value
    ///
    /// # Arguments
    /// * `query` - Query key to search for
    /// * `threshold` - Minimum similarity (0.0-1.0) to return result
    ///
    /// # Returns
    /// Option of (similarity_score, recalled_value)
    pub fn find_similar(&self, query: &[f32], threshold: f32) -> Option<(f32, Vec<f32>)> {
        if self.item_count == 0 {
            return None;
        }

        let recalled = self.recall_real(query);

        // Compute similarity as normalized correlation
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let recalled_norm: f32 = recalled.iter().map(|x| x * x).sum::<f32>().sqrt();

        if query_norm < 1e-6 || recalled_norm < 1e-6 {
            return None;
        }

        let similarity = recalled.iter()
            .zip(query.iter())
            .map(|(r, q)| r * q)
            .sum::<f32>() / (query_norm * recalled_norm);

        if similarity >= threshold {
            Some((similarity, recalled))
        } else {
            None
        }
    }

    /// Decay the memory trace
    pub fn decay(&mut self, factor: f32) {
        for m in self.memory_trace.iter_mut() {
            *m = *m * factor;
        }
    }

    /// Temporal decay based on elapsed time
    pub fn decay_temporal(&mut self, now_us: i64, half_life_us: i64) {
        if let Some(last_ts) = self.last_entangle_ts_us {
            let age_us = now_us.saturating_sub(last_ts);
            if age_us > 0 && half_life_us > 0 {
                let exponent = -0.693147 * (age_us as f64 / half_life_us as f64);
                let factor = exponent.exp().clamp(0.01, 1.0) as f32;
                if factor < 0.999 {
                    self.decay(factor);
                }
            }
        }
    }

    /// Reset memory to empty state
    pub fn clear(&mut self) {
        self.memory_trace.fill(Complex32::new(0.0, 0.0));
        self.item_count = 0;
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get item count
    pub fn item_count(&self) -> usize {
        self.item_count
    }

    /// Get capacity status
    pub fn capacity_status(&self) -> (usize, usize) {
        (self.item_count, self.max_items)
    }

    /// Get total energy of memory trace
    pub fn energy(&self) -> f32 {
        self.memory_trace.iter().map(|c| c.norm_sqr()).sum()
    }

    // FFT helpers
    fn fft_process(&self, input: &[Complex32]) -> Vec<Complex32> {
        let mut buffer = input.to_vec();
        self.fft.process(&mut buffer);
        buffer
    }

    fn ifft_process(&self, input: &[Complex32]) -> Vec<Complex32> {
        let mut buffer = input.to_vec();
        self.ifft.process(&mut buffer);
        for c in buffer.iter_mut() {
            *c = *c * self.norm_factor;
        }
        buffer
    }
}

impl Default for HolographicMemory {
    fn default() -> Self {
        Self::default_for_agent()
    }
}

/// Encode a context into a key vector
pub fn encode_context_key(features: &[f32], dim: usize) -> Vec<Complex32> {
    let mut key = vec![Complex32::new(0.0, 0.0); dim];

    for (i, &f) in features.iter().enumerate() {
        let freq_idx = (i * dim / features.len().max(1)) % dim;
        let phase = f * std::f32::consts::PI;
        key[freq_idx] = Complex32::from_polar(1.0, phase);
    }

    key
}

/// Encode a state into a value vector
pub fn encode_state_value(state: &[f32], dim: usize) -> Vec<Complex32> {
    let mut val = vec![Complex32::new(0.0, 0.0); dim];

    for (i, &s) in state.iter().enumerate() {
        val[i % dim] = Complex32::new(s, 0.0);
    }

    // Fill remaining dimensions
    let n_features = state.len();
    if n_features > 0 && n_features < dim {
        for i in n_features..dim {
            val[i] = val[i % n_features];
        }
    }

    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_memory() {
        let mem = HolographicMemory::new(256);
        assert_eq!(mem.dim(), 256);
        assert_eq!(mem.item_count(), 0);
        assert!(mem.energy() < 1e-10);
    }

    #[test]
    fn test_entangle_and_recall() {
        let dim = 64;
        let mut mem = HolographicMemory::new(dim);

        let key: Vec<Complex32> = (0..dim)
            .map(|i| Complex32::new((i as f32).sin(), 0.0))
            .collect();
        let value: Vec<Complex32> = (0..dim)
            .map(|i| Complex32::new((i as f32 * 0.5).cos(), 0.0))
            .collect();

        mem.entangle(&key, &value);
        assert_eq!(mem.item_count(), 1);

        let recalled = mem.recall(&key);

        // Should have positive correlation
        let correlation: f32 = recalled.iter()
            .zip(value.iter())
            .map(|(r, v)| (r * v.conj()).re)
            .sum();

        assert!(correlation > 0.1, "Recall should show positive correlation");
    }

    #[test]
    fn test_multiple_entangle() {
        let dim = 128;
        let mut mem = HolographicMemory::new(dim);

        for i in 0..100 {
            let key: Vec<Complex32> = (0..dim)
                .map(|j| Complex32::new(((i + j) as f32 * 0.1).sin(), 0.0))
                .collect();
            let value: Vec<Complex32> = (0..dim)
                .map(|j| Complex32::new(((i * 2 + j) as f32 * 0.05).cos(), 0.0))
                .collect();
            mem.entangle(&key, &value);
        }

        assert_eq!(mem.item_count(), 100);
        assert!(mem.energy().is_finite());
    }

    #[test]
    fn test_decay() {
        let dim = 32;
        let mut mem = HolographicMemory::new(dim);

        let key = vec![Complex32::new(1.0, 0.0); dim];
        mem.entangle(&key, &key);

        let energy_before = mem.energy();
        mem.decay(0.5);
        let energy_after = mem.energy();

        assert!(energy_after < energy_before * 0.5);
    }

    #[test]
    fn test_clear() {
        let mut mem = HolographicMemory::new(64);
        let key = vec![Complex32::new(1.0, 0.0); 64];
        mem.entangle(&key, &key);

        mem.clear();
        assert_eq!(mem.item_count(), 0);
        assert!(mem.energy() < 1e-10);
    }

    #[test]
    fn test_real_methods() {
        let dim = 64;
        let mut mem = HolographicMemory::new(dim);

        let key: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let value: Vec<f32> = (0..dim).map(|i| (i as f32 * 2.0).cos()).collect();

        mem.entangle_real(&key, &value);
        let recalled = mem.recall_real(&key);

        assert_eq!(recalled.len(), dim);
    }

    #[test]
    fn test_find_similar() {
        let dim = 64;
        let mut mem = HolographicMemory::new(dim);

        let key: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let value: Vec<f32> = (0..dim).map(|i| (i as f32 * 2.0).cos()).collect();

        mem.entangle_real(&key, &value);

        // Query with same key should find it
        let result = mem.find_similar(&key, 0.0);
        assert!(result.is_some(), "Same key should find stored value");

        // Query with different key - just verify it doesn't crash
        // Holographic memory may or may not find correlation depending on patterns
        let diff_key: Vec<f32> = (0..dim).map(|i| ((i + 10) as f32).cos()).collect();
        let _result2 = mem.find_similar(&diff_key, 0.0);
        // No assertion - this is testing the API doesn't crash
    }
}
