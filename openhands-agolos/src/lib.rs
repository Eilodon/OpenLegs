//! OpenHands-AGOLOS: Cognitive and Safety primitives from Pandora SDK
//!
//! This crate provides:
//! - **DAGMA Causal Discovery**: Full DAGMA algorithm with warm start, counterfactuals
//! - **Holographic Memory**: FFT-based associative memory with content-addressable recall
//! - **LTL Safety Monitor**: Mathematically-provable command blocking
//! - **Trauma Registry**: 3-tier failure memory with forgetting
//! - **Causal Debugger**: DAGMA-based root cause analysis
//! - **Circuit Breaker**: Resilient operation execution
//! - **Policy/EFE**: Active Inference action selection
//! - **Learning**: Experience buffer and pattern mining
//! - **Decision Tree**: Context-aware routing
//! - **LLM Providers**: Fallback chain management
//!
//! # Feature Flags
//! - `causal-dagma` (default): Full DAGMA causal reasoning
//! - `memory-holographic`: FFT-based associative memory
//! - `safety-core` (default): LTL, Circuit Breaker, Trauma

mod ltl_monitor;
mod trauma_registry;
mod causal_debugger;
mod circuit_breaker;
mod policy;
mod learning;
mod decision_tree;
mod llm_providers;

// DAGMA module (feature-gated)
#[cfg(feature = "causal-dagma")]
pub mod dagma;

// Holographic Memory module (feature-gated)
#[cfg(feature = "memory-holographic")]
pub mod holographic_memory;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

pub use ltl_monitor::{LtlMonitor, SafetyViolation, SafetyProperty};
pub use trauma_registry::{TraumaRegistry, TraumaHit, TraumaSeverity};
pub use causal_debugger::{CausalDebugger, CausalAnalysis};
pub use circuit_breaker::{ShardedCircuitBreaker, CircuitBreakerConfig, CircuitStats};
pub use policy::{PolicySelector, PolicyEvaluation, ActionPolicy, EFECalculator};
pub use learning::{ExperienceBuffer, Experience, PatternMiner, ActionPattern};
pub use decision_tree::{DecisionTree, DecisionContext, DecisionAction, DecisionResult};
pub use llm_providers::{ProviderChain, LLMProvider, ProviderStatus};

#[cfg(feature = "causal-dagma")]
pub use dagma::{Dagma, DagmaConfig, CausalDebugger as DagmaCausalDebugger};

#[cfg(feature = "memory-holographic")]
pub use holographic_memory::{HolographicMemory, encode_context_key, encode_state_value};



// ============================================================================
// PyO3 Python Module
// ============================================================================

/// LTL Safety Monitor for bash commands
#[pyclass]
pub struct PyLtlMonitor {
    inner: LtlMonitor,
}

#[pymethods]
impl PyLtlMonitor {
    #[new]
    fn new() -> Self {
        Self {
            inner: LtlMonitor::default(),
        }
    }

    /// Check a command against all safety properties
    /// Returns list of violations (empty if command is safe)
    fn check_command(&self, command: &str) -> Vec<PySafetyViolation> {
        self.inner
            .check_command(command)
            .into_iter()
            .map(|v| PySafetyViolation {
                property_name: v.property_name.to_string(),
                description: v.description.to_string(),
            })
            .collect()
    }

    /// Add a custom safety property (pattern-based)
    fn add_pattern_rule(&mut self, name: String, description: String, blocked_pattern: String) {
        self.inner.add_pattern_rule(&name, &description, &blocked_pattern);
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySafetyViolation {
    #[pyo3(get)]
    property_name: String,
    #[pyo3(get)]
    description: String,
}

// ============================================================================
// Trauma Registry Python Bindings
// ============================================================================

#[pyclass]
pub struct PyTraumaRegistry {
    inner: TraumaRegistry,
}

#[pymethods]
impl PyTraumaRegistry {
    /// Create new trauma registry with SQLite persistence
    #[new]
    fn new(db_path: &str) -> PyResult<Self> {
        TraumaRegistry::open(db_path)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(format!("Failed to open trauma DB: {}", e)))
    }

    /// Record a failure with given severity (1=Light, 2=Medium, 3=Severe)
    fn record_failure(
        &mut self,
        context_hash: Vec<u8>,
        action_type: &str,
        severity: u8,
        decay_hours: i64,
    ) -> PyResult<()> {
        if context_hash.len() != 32 {
            return Err(PyValueError::new_err("context_hash must be 32 bytes"));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&context_hash);

        let sev = match severity {
            1 => TraumaSeverity::Light,
            2 => TraumaSeverity::Medium,
            3 => TraumaSeverity::Severe,
            _ => return Err(PyValueError::new_err("severity must be 1, 2, or 3")),
        };

        self.inner.record_failure(hash, action_type, sev, decay_hours);
        Ok(())
    }

    /// Query trauma for a context hash
    /// Returns (severity_score, count, is_expired) or None
    fn query(&self, context_hash: Vec<u8>) -> PyResult<Option<PyTraumaHit>> {
        if context_hash.len() != 32 {
            return Err(PyValueError::new_err("context_hash must be 32 bytes"));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&context_hash);

        Ok(self.inner.query(&hash).map(|hit| PyTraumaHit {
            severity: hit.severity as u8,
            count: hit.count,
            is_expired: hit.is_expired(),
            inhibit_until_ts: hit.inhibit_until_ts,
        }))
    }

    /// Compute context hash from action details
    #[staticmethod]
    fn compute_context_hash(
        command: &str,
        working_dir: &str,
        project_id: &str,
    ) -> Vec<u8> {
        TraumaRegistry::compute_context_hash(command, working_dir, project_id).to_vec()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyTraumaHit {
    #[pyo3(get)]
    severity: u8,
    #[pyo3(get)]
    count: u32,
    #[pyo3(get)]
    is_expired: bool,
    #[pyo3(get)]
    inhibit_until_ts: i64,
}

// ============================================================================
// Causal Debugger Python Bindings
// ============================================================================

#[pyclass]
pub struct PyCausalDebugger {
    inner: CausalDebugger,
}

#[pymethods]
impl PyCausalDebugger {
    #[new]
    fn new() -> Self {
        Self {
            inner: CausalDebugger::new(),
        }
    }

    /// Analyze error to find root cause
    /// features: 2D array [n_samples, n_variables]
    /// Returns: (root_cause_index, confidence, intervention_suggestion)
    fn analyze(&self, features: Vec<Vec<f32>>) -> PyResult<PyCausalAnalysis> {
        if features.is_empty() || features[0].is_empty() {
            return Err(PyValueError::new_err("features must be non-empty"));
        }

        let analysis = self.inner.analyze(&features);
        Ok(PyCausalAnalysis {
            root_cause_index: analysis.root_cause_index,
            confidence: analysis.confidence,
            suggestion: analysis.suggestion,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCausalAnalysis {
    #[pyo3(get)]
    root_cause_index: usize,
    #[pyo3(get)]
    confidence: f32,
    #[pyo3(get)]
    suggestion: String,
}


// ============================================================================
// Circuit Breaker Python Bindings
// ============================================================================

use std::time::Duration;

#[pyclass]
pub struct PyCircuitBreaker {
    inner: ShardedCircuitBreaker,
}

#[pymethods]
impl PyCircuitBreaker {
    #[new]
    fn new() -> Self {
        Self {
            inner: ShardedCircuitBreaker::default(),
        }
    }

    #[staticmethod]
    fn with_config(
        failure_threshold: u32,
        recovery_timeout_secs: u64,
        half_open_permits: u32,
    ) -> Self {
        let config = CircuitBreakerConfig {
            failure_threshold,
            recovery_timeout: Duration::from_secs(recovery_timeout_secs),
            half_open_permits,
            state_ttl: Duration::from_secs(300),
        };
        Self {
            inner: ShardedCircuitBreaker::new(config),
        }
    }

    fn is_open(&self, operation: &str) -> bool {
        self.inner.is_open(operation)
    }

    fn record_success(&self, operation: &str) {
        self.inner.record_success(operation)
    }

    fn record_failure(&self, operation: &str) {
        self.inner.record_failure(operation)
    }

    fn get_state(&self, operation: &str) -> Option<String> {
        self.inner.get_state(operation)
    }

    fn stats(&self) -> PyCircuitStats {
        let stats = self.inner.stats();
        PyCircuitStats {
            total_operations: stats.total_operations,
            open_circuits: stats.open_circuits,
            closed_circuits: stats.closed_circuits,
            half_open_circuits: stats.half_open_circuits,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCircuitStats {
    #[pyo3(get)]
    total_operations: usize,
    #[pyo3(get)]
    open_circuits: usize,
    #[pyo3(get)]
    closed_circuits: usize,
    #[pyo3(get)]
    half_open_circuits: usize,
}

// ============================================================================
// DAGMA Python Bindings (Full Implementation)
// ============================================================================

#[cfg(feature = "causal-dagma")]
#[pyclass]
pub struct PyDagma {
    inner: std::cell::RefCell<dagma::CausalDebugger>,
}

#[cfg(feature = "causal-dagma")]
#[pymethods]
impl PyDagma {
    #[new]
    fn new() -> Self {
        Self {
            inner: std::cell::RefCell::new(dagma::CausalDebugger::new()),
        }
    }

    /// Create DAGMA with adaptive configuration based on expected data size
    #[staticmethod]
    fn adaptive(n_vars: usize, n_samples: usize) -> Self {
        Self {
            inner: std::cell::RefCell::new(dagma::CausalDebugger::with_config(
                n_vars,
                dagma::DagmaConfig::adaptive(n_vars, n_samples),
            )),
        }
    }

    /// Create DAGMA with fast configuration (trades accuracy for speed)
    #[staticmethod]
    fn fast(n_vars: usize) -> Self {
        Self {
            inner: std::cell::RefCell::new(dagma::CausalDebugger::with_config(
                n_vars,
                dagma::DagmaConfig::fast(),
            )),
        }
    }

    /// Set variable names for better reporting
    fn set_variable_names(&self, names: Vec<String>) {
        self.inner.borrow_mut().set_variable_names(names);
    }

    /// Analyze features to find root cause
    ///
    /// # Arguments
    /// * `features` - 2D array [n_samples, n_variables]
    ///
    /// # Returns
    /// CausalAnalysis with root_cause_index, confidence, suggestion, adjacency matrix
    fn analyze(&self, features: Vec<Vec<f32>>) -> PyResult<PyDagmaAnalysis> {
        if features.is_empty() || features[0].is_empty() {
            return Err(PyValueError::new_err("features must be non-empty"));
        }

        let analysis = self.inner.borrow_mut().analyze(&features);
        Ok(PyDagmaAnalysis {
            root_cause_index: analysis.root_cause_index,
            confidence: analysis.confidence,
            suggestion: analysis.suggestion,
            adjacency: analysis.adjacency,
            acyclicity_score: analysis.acyclicity_score,
        })
    }

    /// Predict intervention effects (counterfactual reasoning)
    ///
    /// # Arguments
    /// * `features` - Current observation data [n_samples, n_variables]
    /// * `intervene_idx` - Index of variable to intervene on
    /// * `intervene_value` - New value to set for that variable
    ///
    /// # Returns
    /// Predicted state after intervention
    fn predict_intervention(
        &self,
        features: Vec<Vec<f32>>,
        intervene_idx: usize,
        intervene_value: f32,
    ) -> PyResult<Vec<f32>> {
        if features.is_empty() || features[0].is_empty() {
            return Err(PyValueError::new_err("features must be non-empty"));
        }

        let result = self.inner.borrow_mut().predict_intervention(
            &features,
            intervene_idx,
            intervene_value,
        );

        if result.is_empty() {
            return Err(PyValueError::new_err("Invalid intervention index"));
        }

        Ok(result)
    }
}

#[cfg(feature = "causal-dagma")]
#[pyclass]
#[derive(Clone)]
pub struct PyDagmaAnalysis {
    #[pyo3(get)]
    root_cause_index: usize,
    #[pyo3(get)]
    confidence: f32,
    #[pyo3(get)]
    suggestion: String,
    #[pyo3(get)]
    adjacency: Vec<Vec<f32>>,
    #[pyo3(get)]
    acyclicity_score: f32,
}

// ============================================================================
// HolographicMemory Python Bindings
// ============================================================================

#[cfg(feature = "memory-holographic")]
#[pyclass]
pub struct PyHolographicMemory {
    inner: std::cell::RefCell<holographic_memory::HolographicMemory>,
}

#[cfg(feature = "memory-holographic")]
#[pymethods]
impl PyHolographicMemory {
    #[new]
    fn new(dim: usize) -> Self {
        Self {
            inner: std::cell::RefCell::new(holographic_memory::HolographicMemory::new(dim)),
        }
    }

    /// Create memory with dimension and capacity limit
    #[staticmethod]
    fn with_capacity(dim: usize, max_items: usize) -> Self {
        Self {
            inner: std::cell::RefCell::new(
                holographic_memory::HolographicMemory::with_capacity(dim, max_items)
            ),
        }
    }

    /// Default configuration for agent memory (dim=512, max_items=10000)
    #[staticmethod]
    fn default_for_agent() -> Self {
        Self {
            inner: std::cell::RefCell::new(holographic_memory::HolographicMemory::default_for_agent()),
        }
    }

    /// Store a key-value association
    /// key and value must have length == dim
    fn entangle(&self, key: Vec<f32>, value: Vec<f32>) -> PyResult<()> {
        let dim = self.inner.borrow().dim();
        if key.len() != dim || value.len() != dim {
            return Err(PyValueError::new_err(format!(
                "key and value must have length {}, got {} and {}",
                dim, key.len(), value.len()
            )));
        }
        self.inner.borrow_mut().entangle_real(&key, &value);
        Ok(())
    }

    /// Recall values associated with a key
    fn recall(&self, key: Vec<f32>) -> PyResult<Vec<f32>> {
        let dim = self.inner.borrow().dim();
        if key.len() != dim {
            return Err(PyValueError::new_err(format!(
                "key must have length {}, got {}",
                dim, key.len()
            )));
        }
        Ok(self.inner.borrow().recall_real(&key))
    }

    /// Find similar stored key-value pair
    /// Returns (similarity_score, recalled_value) or None
    fn find_similar(&self, query: Vec<f32>, threshold: f32) -> PyResult<Option<(f32, Vec<f32>)>> {
        let dim = self.inner.borrow().dim();
        if query.len() != dim {
            return Err(PyValueError::new_err(format!(
                "query must have length {}, got {}",
                dim, query.len()
            )));
        }
        Ok(self.inner.borrow().find_similar(&query, threshold))
    }

    /// Apply decay to memory (0.0 = forget all, 1.0 = remember all)
    fn decay(&self, factor: f32) {
        self.inner.borrow_mut().decay(factor);
    }

    /// Clear all memory
    fn clear(&self) {
        self.inner.borrow_mut().clear();
    }

    /// Get dimension of memory
    fn dim(&self) -> usize {
        self.inner.borrow().dim()
    }

    /// Get number of items stored
    fn item_count(&self) -> usize {
        self.inner.borrow().item_count()
    }

    /// Get current energy of memory trace
    fn energy(&self) -> f32 {
        self.inner.borrow().energy()
    }

    /// Get capacity status (items_stored, max_items)
    fn capacity_status(&self) -> (usize, usize) {
        self.inner.borrow().capacity_status()
    }
}

// ============================================================================
// Python Module Definition
// ============================================================================

#[pymodule]
fn openhands_agolos(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLtlMonitor>()?;
    m.add_class::<PySafetyViolation>()?;
    m.add_class::<PyTraumaRegistry>()?;
    m.add_class::<PyTraumaHit>()?;
    m.add_class::<PyCausalDebugger>()?;
    m.add_class::<PyCausalAnalysis>()?;
    m.add_class::<PyCircuitBreaker>()?;
    m.add_class::<PyCircuitStats>()?;

    // DAGMA (feature-gated)
    #[cfg(feature = "causal-dagma")]
    {
        m.add_class::<PyDagma>()?;
        m.add_class::<PyDagmaAnalysis>()?;
    }

    // HolographicMemory (feature-gated)
    #[cfg(feature = "memory-holographic")]
    {
        m.add_class::<PyHolographicMemory>()?;
    }

    Ok(())
}
