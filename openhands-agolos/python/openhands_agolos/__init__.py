"""OpenHands-AGOLOS: Cognitive and Safety primitives for OpenHands

This package provides:
- LTL Safety Monitor for bash commands
- 3-Tier Trauma Registry with forgetting
- DAGMA-based Causal Debugger
- Circuit Breaker for resilient operations
- Holographic Memory for associative recall
- Policy/EFE Calculator for Active Inference
- Experience Buffer and Pattern Miner for learning
- Decision Tree for context-aware routing
- LLM Provider Chain for fallback management
"""

# Re-export from Rust extension
from .openhands_agolos import (
    # Safety primitives
    PyLtlMonitor as LtlMonitor,
    PySafetyViolation as SafetyViolation,
    PyTraumaRegistry as TraumaRegistry,
    PyTraumaHit as TraumaHit,
    PyCircuitBreaker as CircuitBreaker,
    PyCircuitStats as CircuitStats,
    # Causal analysis
    PyCausalDebugger as CausalDebugger,
    PyCausalAnalysis as CausalAnalysis,
    PyDagma as Dagma,
    PyDagmaAnalysis as DagmaAnalysis,
    # Memory
    PyHolographicMemory as HolographicMemory,
    # Policy/EFE
    PyEFECalculator as EFECalculator,
    PyPolicySelector as PolicySelector,
    PyPolicyEvaluation as PolicyEvaluation,
    # Learning
    PyExperienceBuffer as ExperienceBuffer,
    PyPatternMiner as PatternMiner,
    # Decision Tree
    PyDecisionTree as DecisionTree,
    PyDecisionResult as DecisionResult,
    # LLM Providers
    PyProviderChain as ProviderChain,
)

__all__ = [
    # Safety
    "LtlMonitor",
    "SafetyViolation",
    "TraumaRegistry",
    "TraumaHit",
    "CircuitBreaker",
    "CircuitStats",
    # Causal
    "CausalDebugger",
    "CausalAnalysis",
    "Dagma",
    "DagmaAnalysis",
    # Memory
    "HolographicMemory",
    # Policy
    "EFECalculator",
    "PolicySelector",
    "PolicyEvaluation",
    # Learning
    "ExperienceBuffer",
    "PatternMiner",
    # Decision
    "DecisionTree",
    "DecisionResult",
    # Providers
    "ProviderChain",
]
