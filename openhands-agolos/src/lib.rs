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
// Policy/EFE Calculator Python Bindings
// ============================================================================

#[pyclass]
pub struct PyEFECalculator {
    inner: EFECalculator,
}

#[pymethods]
impl PyEFECalculator {
    #[new]
    fn new() -> Self {
        Self {
            inner: EFECalculator::default(),
        }
    }

    #[staticmethod]
    fn with_params(temperature: f32, epistemic_weight: f32) -> Self {
        Self {
            inner: EFECalculator::new(temperature, epistemic_weight),
        }
    }

    /// Compute Expected Free Energy
    fn compute_efe(&self, pragmatic: f32, epistemic: f32) -> f32 {
        self.inner.compute_efe(pragmatic, epistemic)
    }

    /// Evaluate action and return policy recommendation
    fn evaluate_action(
        &self,
        action_type: &str,
        goal_alignment: f32,
        uncertainty: f32,
        past_success_rate: f32,
    ) -> PyPolicyEvaluation {
        let eval = self.inner.evaluate_action(action_type, goal_alignment, uncertainty, past_success_rate);
        PyPolicyEvaluation {
            policy: format!("{:?}", eval.policy),
            efe_value: eval.efe,
            pragmatic: eval.pragmatic_value,
            epistemic: eval.epistemic_value,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPolicyEvaluation {
    #[pyo3(get)]
    policy: String,
    #[pyo3(get)]
    efe_value: f32,
    #[pyo3(get)]
    pragmatic: f32,
    #[pyo3(get)]
    epistemic: f32,
}

#[pyclass]
pub struct PyPolicySelector {
    inner: std::cell::RefCell<PolicySelector>,
}

#[pymethods]
impl PyPolicySelector {
    #[new]
    fn new() -> Self {
        Self {
            inner: std::cell::RefCell::new(PolicySelector::new()),
        }
    }

    /// Select best policy for an action given goal alignment and uncertainty
    fn select(&self, action_type: &str, goal_alignment: f32, uncertainty: f32) -> PyPolicyEvaluation {
        let eval = self.inner.borrow().select(action_type, goal_alignment, uncertainty);
        PyPolicyEvaluation {
            policy: format!("{:?}", eval.policy),
            efe_value: eval.efe,
            pragmatic: eval.pragmatic_value,
            epistemic: eval.epistemic_value,
        }
    }

    /// Record action outcome for learning
    fn record_outcome(&self, action_type: &str, success: bool) {
        self.inner.borrow_mut().record_outcome(action_type, success);
    }

    /// Get success rate for action type
    fn get_success_rate(&self, action_type: &str) -> f32 {
        self.inner.borrow().get_success_rate(action_type)
    }
}

// ============================================================================
// Learning Module Python Bindings
// ============================================================================

#[pyclass]
pub struct PyExperienceBuffer {
    inner: std::cell::RefCell<ExperienceBuffer>,
}

#[pymethods]
impl PyExperienceBuffer {
    #[new]
    fn new(max_size: usize) -> Self {
        Self {
            inner: std::cell::RefCell::new(ExperienceBuffer::new(max_size)),
        }
    }

    /// Push a new experience
    fn push(&self, action_type: &str, action_signature: &str, success: bool, reward: f32) {
        let context_hash = [0u8; 32]; // Simplified - could compute from action
        let exp = Experience::new(action_type, action_signature, context_hash, success, reward);
        self.inner.borrow_mut().push(exp);
    }

    /// Get success rate for action type
    fn success_rate(&self, action_type: &str) -> Option<f32> {
        self.inner.borrow().success_rate(action_type)
    }

    /// Get buffer length
    fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    /// Check if buffer is empty
    fn is_empty(&self) -> bool {
        self.inner.borrow().is_empty()
    }
}

#[pyclass]
pub struct PyPatternMiner {
    inner: std::cell::RefCell<PatternMiner>,
}

#[pymethods]
impl PyPatternMiner {
    #[new]
    fn new() -> Self {
        Self {
            inner: std::cell::RefCell::new(PatternMiner::default()),
        }
    }

    #[staticmethod]
    fn with_params(min_support: f32, max_pattern_length: usize) -> Self {
        Self {
            inner: std::cell::RefCell::new(PatternMiner::new(min_support, max_pattern_length)),
        }
    }

    /// Add an action to the sequence buffer
    fn add_action(&self, action_type: &str, success: bool, reward: f32) {
        self.inner.borrow_mut().add_action(action_type, success, reward);
    }

    /// Suggest next action based on current sequence
    fn suggest_next(&self, recent_actions: Vec<String>) -> Option<String> {
        self.inner.borrow().suggest_next(&recent_actions).map(|s| s.to_string())
    }

    /// Get number of patterns discovered
    fn pattern_count(&self) -> usize {
        self.inner.borrow().pattern_count()
    }
}

// ============================================================================
// Decision Tree Python Bindings
// ============================================================================

#[pyclass]
pub struct PyDecisionTree {
    inner: DecisionTree,
}

#[pymethods]
impl PyDecisionTree {
    #[new]
    fn new() -> Self {
        Self {
            inner: DecisionTree::default(),
        }
    }

    /// Create default decision tree for OpenHands agent
    #[staticmethod]
    fn default_for_agent() -> Self {
        Self {
            inner: DecisionTree::default_for_agent(),
        }
    }

    /// Make decision based on context
    fn decide(&self, budget: f32, context: f32, confidence: f32, failures: u32, is_rate_limited: bool) -> PyDecisionResult {
        let ctx = DecisionContext {
            budget_remaining: budget,
            context_usage: context,
            confidence,
            failure_count: failures,
            is_rate_limited,
            ..Default::default()
        };
        let result = self.inner.decide(&ctx);
        PyDecisionResult {
            action: format!("{:?}", result.action),
            path: result.path_string(),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDecisionResult {
    #[pyo3(get)]
    action: String,
    #[pyo3(get)]
    path: String,
}

// ============================================================================
// LLM Provider Chain Python Bindings
// ============================================================================

#[pyclass]
pub struct PyProviderChain {
    inner: std::cell::RefCell<ProviderChain>,
}

#[pymethods]
impl PyProviderChain {
    #[new]
    fn new() -> Self {
        Self {
            inner: std::cell::RefCell::new(ProviderChain::new()),
        }
    }

    /// Create default chain with common providers
    #[staticmethod]
    fn default_chain() -> Self {
        Self {
            inner: std::cell::RefCell::new(ProviderChain::default_chain()),
        }
    }

    /// Add a provider to the chain
    fn add_provider(&self, name: &str, priority: u32, cost_per_token: f32, max_context: usize) {
        let provider = LLMProvider::new(name, priority, cost_per_token, max_context);
        self.inner.borrow_mut().add_provider(provider);
    }

    /// Get next available provider name
    fn next_available(&self) -> Option<String> {
        self.inner.borrow().next_available().map(|p| p.name.clone())
    }

    /// Record success for a provider
    fn record_success(&self, provider_name: &str, latency_ms: u64) {
        self.inner.borrow_mut().record_success(provider_name, latency_ms);
    }

    /// Record failure for a provider
    fn record_failure(&self, provider_name: &str, error: &str, latency_ms: u64) {
        self.inner.borrow_mut().record_failure(provider_name, error, latency_ms);
    }

    /// Mark provider as rate limited
    fn mark_rate_limited(&self, provider_name: &str, duration_secs: u64) {
        self.inner.borrow_mut().mark_rate_limited(provider_name, Duration::from_secs(duration_secs));
    }

    /// Get stats for a provider (error_rate, avg_latency_ms)
    fn get_stats(&self, provider_name: &str) -> Option<(f32, f32)> {
        self.inner.borrow().get_stats(provider_name)
    }

    /// Reset provider status for recovery
    fn reset_provider(&self, provider_name: &str) {
        self.inner.borrow_mut().reset_provider(provider_name);
    }

    /// Get all available provider names
    fn available_providers(&self) -> Vec<String> {
        self.inner.borrow().available_providers().iter().map(|p| p.name.clone()).collect()
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

    // Policy/EFE Calculator
    m.add_class::<PyEFECalculator>()?;
    m.add_class::<PyPolicySelector>()?;
    m.add_class::<PyPolicyEvaluation>()?;

    // Learning Module
    m.add_class::<PyExperienceBuffer>()?;
    m.add_class::<PyPatternMiner>()?;

    // Decision Tree
    m.add_class::<PyDecisionTree>()?;
    m.add_class::<PyDecisionResult>()?;

    // LLM Provider Chain
    m.add_class::<PyProviderChain>()?;

    Ok(())
}
