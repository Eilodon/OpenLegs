//! Sharded Circuit Breaker for resilient operation execution.
//!
//! Ported from Pandora's `circuit_breaker.rs`
//! Uses 16 shards to reduce lock contention in concurrent scenarios.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Number of shards for the circuit breaker state.
const SHARD_COUNT: usize = 16;
const SHARD_MASK: usize = SHARD_COUNT - 1;

/// Circuit breaker state
#[derive(Debug, Clone)]
pub enum CircuitState {
    /// Circuit is closed (healthy). Requests flow through normally.
    Closed {
        failures: u32,
        last_updated: Instant,
    },
    /// Circuit is open (unhealthy). Requests are rejected.
    Open {
        opened_at: Instant,
    },
    /// Circuit is testing. Limited requests allowed.
    HalfOpen {
        trial_permits: u32,
        last_updated: Instant,
    },
}

impl CircuitState {
    fn new_closed() -> Self {
        CircuitState::Closed {
            failures: 0,
            last_updated: Instant::now(),
        }
    }
}

/// Configuration for circuit breaker behavior.
#[derive(Clone, Debug)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Duration to wait before attempting recovery
    pub recovery_timeout: Duration,
    /// Number of trial requests in half-open state
    pub half_open_permits: u32,
    /// TTL for expired states (for cleanup)
    pub state_ttl: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            half_open_permits: 3,
            state_ttl: Duration::from_secs(300),
        }
    }
}

/// Sharded circuit breaker manager
pub struct ShardedCircuitBreaker {
    shards: Vec<Mutex<HashMap<String, CircuitState>>>,
    config: CircuitBreakerConfig,
}

impl ShardedCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let shards = (0..SHARD_COUNT)
            .map(|_| Mutex::new(HashMap::new()))
            .collect();
        Self { shards, config }
    }

    /// Calculate which shard an operation belongs to
    fn shard_index(&self, operation: &str) -> usize {
        let mut hash: usize = 0;
        for byte in operation.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
        }
        hash & SHARD_MASK
    }

    /// Check if circuit is open (requests should be rejected)
    pub fn is_open(&self, operation: &str) -> bool {
        let idx = self.shard_index(operation);
        let shard = self.shards[idx].lock().unwrap();

        match shard.get(operation) {
            Some(CircuitState::Open { opened_at }) => {
                // Check if recovery timeout has passed
                if opened_at.elapsed() >= self.config.recovery_timeout {
                    false // Allow trial (will transition to HalfOpen on next record)
                } else {
                    true // Still open
                }
            }
            Some(CircuitState::HalfOpen { trial_permits, .. }) => {
                *trial_permits == 0 // Open if no permits left
            }
            _ => false, // Closed or not tracked
        }
    }

    /// Record successful execution
    pub fn record_success(&self, operation: &str) {
        let idx = self.shard_index(operation);
        let mut shard = self.shards[idx].lock().unwrap();

        // Reset to closed on success
        shard.insert(operation.to_string(), CircuitState::new_closed());
    }

    /// Record failed execution
    pub fn record_failure(&self, operation: &str) {
        let idx = self.shard_index(operation);
        let mut shard = self.shards[idx].lock().unwrap();

        let current = shard.get(operation).cloned();
        let new_state = match current {
            None | Some(CircuitState::Closed { failures: 0, .. }) => {
                CircuitState::Closed {
                    failures: 1,
                    last_updated: Instant::now(),
                }
            }
            Some(CircuitState::Closed { failures, .. }) => {
                if failures + 1 >= self.config.failure_threshold {
                    CircuitState::Open {
                        opened_at: Instant::now(),
                    }
                } else {
                    CircuitState::Closed {
                        failures: failures + 1,
                        last_updated: Instant::now(),
                    }
                }
            }
            Some(CircuitState::Open { opened_at }) => {
                if opened_at.elapsed() >= self.config.recovery_timeout {
                    // Transition to half-open
                    CircuitState::HalfOpen {
                        trial_permits: self.config.half_open_permits.saturating_sub(1),
                        last_updated: Instant::now(),
                    }
                } else {
                    CircuitState::Open { opened_at }
                }
            }
            Some(CircuitState::HalfOpen { trial_permits, .. }) => {
                if trial_permits <= 1 {
                    // All trials failed, re-open
                    CircuitState::Open {
                        opened_at: Instant::now(),
                    }
                } else {
                    CircuitState::HalfOpen {
                        trial_permits: trial_permits - 1,
                        last_updated: Instant::now(),
                    }
                }
            }
        };

        shard.insert(operation.to_string(), new_state);
    }

    /// Get circuit state for an operation
    pub fn get_state(&self, operation: &str) -> Option<String> {
        let idx = self.shard_index(operation);
        let shard = self.shards[idx].lock().unwrap();

        shard.get(operation).map(|s| match s {
            CircuitState::Closed { failures, .. } => format!("closed(failures={})", failures),
            CircuitState::Open { .. } => "open".to_string(),
            CircuitState::HalfOpen { trial_permits, .. } => {
                format!("half_open(permits={})", trial_permits)
            }
        })
    }

    /// Get statistics
    pub fn stats(&self) -> CircuitStats {
        let mut total_ops = 0;
        let mut open_count = 0;
        let mut closed_count = 0;
        let mut half_open_count = 0;

        for shard in &self.shards {
            let guard = shard.lock().unwrap();
            for state in guard.values() {
                total_ops += 1;
                match state {
                    CircuitState::Closed { .. } => closed_count += 1,
                    CircuitState::Open { .. } => open_count += 1,
                    CircuitState::HalfOpen { .. } => half_open_count += 1,
                }
            }
        }

        CircuitStats {
            total_operations: total_ops,
            open_circuits: open_count,
            closed_circuits: closed_count,
            half_open_circuits: half_open_count,
        }
    }
}

impl Default for ShardedCircuitBreaker {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

/// Statistics about circuit breaker state
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub total_operations: usize,
    pub open_circuits: usize,
    pub closed_circuits: usize,
    pub half_open_circuits: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_opens_after_threshold() {
        let cb = ShardedCircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        });

        let op = "test_api";

        // First 2 failures - circuit stays closed
        cb.record_failure(op);
        cb.record_failure(op);
        assert!(!cb.is_open(op));

        // Third failure - circuit opens
        cb.record_failure(op);
        assert!(cb.is_open(op));
    }

    #[test]
    fn test_success_resets_circuit() {
        let cb = ShardedCircuitBreaker::default();
        let op = "test_api";

        cb.record_failure(op);
        cb.record_failure(op);
        cb.record_success(op);

        // Should be reset
        assert!(!cb.is_open(op));
        assert_eq!(cb.get_state(op), Some("closed(failures=0)".to_string()));
    }

    #[test]
    fn test_shard_distribution() {
        let cb = ShardedCircuitBreaker::default();

        // Different operations should go to different shards
        let idx1 = cb.shard_index("operation_a");
        let idx2 = cb.shard_index("operation_b");
        let idx3 = cb.shard_index("operation_c");

        // At least some should differ (probabilistic but very likely)
        assert!(idx1 != idx2 || idx2 != idx3 || idx1 != idx3);
    }
}
