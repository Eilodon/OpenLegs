//! LLM Provider Fallback Chain
//!
//! Ported from Pandora's provider patterns.
//! Provides automatic fallback between LLM providers.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Provider status
#[derive(Debug, Clone)]
pub enum ProviderStatus {
    Available,
    RateLimited { until: Instant },
    Unavailable { reason: String },
    Degraded { error_rate: f32 },
}

impl ProviderStatus {
    pub fn is_usable(&self) -> bool {
        match self {
            ProviderStatus::Available => true,
            ProviderStatus::Degraded { error_rate } => *error_rate < 0.5,
            ProviderStatus::RateLimited { until } => Instant::now() >= *until,
            ProviderStatus::Unavailable { .. } => false,
        }
    }
}

/// LLM Provider definition
#[derive(Debug, Clone)]
pub struct LLMProvider {
    pub name: String,
    pub priority: u32,  // Lower = higher priority
    pub cost_per_token: f32,
    pub max_context: usize,
    pub supports_streaming: bool,
}

impl LLMProvider {
    pub fn new(name: &str, priority: u32, cost: f32, context: usize) -> Self {
        Self {
            name: name.to_string(),
            priority,
            cost_per_token: cost,
            max_context: context,
            supports_streaming: true,
        }
    }
}

/// Provider chain with automatic fallback
pub struct ProviderChain {
    providers: Vec<LLMProvider>,
    status: HashMap<String, ProviderStatus>,
    stats: HashMap<String, ProviderStats>,
}

#[derive(Debug, Clone, Default)]
struct ProviderStats {
    success_count: u32,
    failure_count: u32,
    total_latency_ms: u64,
    last_error: Option<String>,
}

impl ProviderStats {
    fn error_rate(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.failure_count as f32 / total as f32
        }
    }

    fn avg_latency_ms(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.total_latency_ms as f32 / total as f32
        }
    }
}

impl Default for ProviderChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderChain {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            status: HashMap::new(),
            stats: HashMap::new(),
        }
    }

    /// Create default chain for OpenHands
    pub fn default_chain() -> Self {
        let mut chain = Self::new();

        // Add common providers in priority order
        chain.add_provider(LLMProvider::new("anthropic", 1, 0.015, 200_000));
        chain.add_provider(LLMProvider::new("openai", 2, 0.010, 128_000));
        chain.add_provider(LLMProvider::new("google", 3, 0.001, 1_000_000));
        chain.add_provider(LLMProvider::new("local", 10, 0.0, 32_000));

        chain
    }

    /// Add a provider to the chain
    pub fn add_provider(&mut self, provider: LLMProvider) {
        self.status.insert(provider.name.clone(), ProviderStatus::Available);
        self.stats.insert(provider.name.clone(), ProviderStats::default());
        self.providers.push(provider);
        self.providers.sort_by_key(|p| p.priority);
    }

    /// Get next available provider
    pub fn next_available(&self) -> Option<&LLMProvider> {
        for provider in &self.providers {
            if let Some(status) = self.status.get(&provider.name) {
                if status.is_usable() {
                    return Some(provider);
                }
            }
        }
        None
    }

    /// Get provider by name
    pub fn get(&self, name: &str) -> Option<&LLMProvider> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// Record success
    pub fn record_success(&mut self, provider_name: &str, latency_ms: u64) {
        if let Some(stats) = self.stats.get_mut(provider_name) {
            stats.success_count += 1;
            stats.total_latency_ms += latency_ms;
        }

        self.status.insert(provider_name.to_string(), ProviderStatus::Available);
    }

    /// Record failure
    pub fn record_failure(&mut self, provider_name: &str, error: &str, latency_ms: u64) {
        if let Some(stats) = self.stats.get_mut(provider_name) {
            stats.failure_count += 1;
            stats.total_latency_ms += latency_ms;
            stats.last_error = Some(error.to_string());

            let error_rate = stats.error_rate();

            // Update status based on error rate
            let new_status = if error_rate > 0.8 {
                ProviderStatus::Unavailable { reason: error.to_string() }
            } else if error_rate > 0.3 {
                ProviderStatus::Degraded { error_rate }
            } else {
                ProviderStatus::Available
            };

            self.status.insert(provider_name.to_string(), new_status);
        }
    }

    /// Mark provider as rate limited
    pub fn mark_rate_limited(&mut self, provider_name: &str, duration: Duration) {
        self.status.insert(
            provider_name.to_string(),
            ProviderStatus::RateLimited {
                until: Instant::now() + duration,
            },
        );
    }

    /// Get all available providers in priority order
    pub fn available_providers(&self) -> Vec<&LLMProvider> {
        self.providers
            .iter()
            .filter(|p| {
                self.status.get(&p.name).map(|s| s.is_usable()).unwrap_or(false)
            })
            .collect()
    }

    /// Get provider stats
    pub fn get_stats(&self, provider_name: &str) -> Option<(f32, f32)> {
        self.stats.get(provider_name).map(|s| (s.error_rate(), s.avg_latency_ms()))
    }

    /// Reset provider status (for recovery)
    pub fn reset_provider(&mut self, provider_name: &str) {
        self.status.insert(provider_name.to_string(), ProviderStatus::Available);
        if let Some(stats) = self.stats.get_mut(provider_name) {
            stats.failure_count = 0;
            stats.last_error = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_chain() {
        let chain = ProviderChain::default_chain();

        let first = chain.next_available().unwrap();
        assert_eq!(first.name, "anthropic"); // Highest priority
    }

    #[test]
    fn test_fallback_on_failure() {
        let mut chain = ProviderChain::default_chain();

        // Fail anthropic multiple times
        for _ in 0..10 {
            chain.record_failure("anthropic", "API Error", 100);
        }

        // Should fallback to openai
        let next = chain.next_available().unwrap();
        assert_eq!(next.name, "openai");
    }

    #[test]
    fn test_rate_limit() {
        let mut chain = ProviderChain::default_chain();

        chain.mark_rate_limited("anthropic", Duration::from_secs(60));

        let next = chain.next_available().unwrap();
        assert_eq!(next.name, "openai"); // Skips rate-limited
    }

    #[test]
    fn test_recovery() {
        let mut chain = ProviderChain::default_chain();

        // Fail provider
        for _ in 0..10 {
            chain.record_failure("anthropic", "Error", 100);
        }

        // Reset
        chain.reset_provider("anthropic");

        // Should be available again
        let next = chain.next_available().unwrap();
        assert_eq!(next.name, "anthropic");
    }
}
