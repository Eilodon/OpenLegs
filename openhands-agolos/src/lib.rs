//! OpenHands-AGOLOS: Safety and Intelligence primitives from Pandora SDK
//!
//! This crate provides:
//! - **LTL Safety Monitor**: Mathematically-provable command blocking
//! - **Trauma Registry**: 3-tier failure memory with forgetting
//! - **Causal Debugger**: DAGMA-based root cause analysis
//! - **Circuit Breaker**: Resilient operation execution (NEW)
//! - **Policy/EFE**: Active Inference action selection (NEW)
//! - **Learning**: Experience buffer and pattern mining (NEW)
//! - **Decision Tree**: Context-aware routing (NEW)
//! - **LLM Providers**: Fallback chain management (NEW)

mod ltl_monitor;
mod trauma_registry;
mod causal_debugger;
mod circuit_breaker;
mod policy;
mod learning;
mod decision_tree;
mod llm_providers;

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
    Ok(())
}
