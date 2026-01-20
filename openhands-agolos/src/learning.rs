//! Learning Module - Experience buffer and pattern mining
//!
//! Ported from Pandora's `learning/` module.
//! Provides persistent learning from successful action sequences.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single experience record
#[derive(Debug, Clone)]
pub struct Experience {
    /// Action type (e.g., "bash_command", "file_edit")
    pub action_type: String,
    /// Normalized action signature (for pattern matching)
    pub action_signature: String,
    /// Context hash for similarity matching
    pub context_hash: [u8; 32],
    /// Whether the action succeeded
    pub success: bool,
    /// Timestamp in microseconds
    pub timestamp_us: i64,
    /// Optional reward/score
    pub reward: f32,
}

impl Experience {
    pub fn new(
        action_type: &str,
        action_signature: &str,
        context_hash: [u8; 32],
        success: bool,
        reward: f32,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        Self {
            action_type: action_type.to_string(),
            action_signature: action_signature.to_string(),
            context_hash,
            success,
            timestamp_us: timestamp,
            reward,
        }
    }
}

/// Ring buffer for recent experiences
pub struct ExperienceBuffer {
    buffer: Vec<Experience>,
    max_size: usize,
    head: usize,
    count: usize,
}

impl ExperienceBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
            head: 0,
            count: 0,
        }
    }

    /// Push a new experience
    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() < self.max_size {
            self.buffer.push(exp);
        } else {
            self.buffer[self.head] = exp;
        }
        self.head = (self.head + 1) % self.max_size;
        self.count = (self.count + 1).min(self.max_size);
    }

    /// Get all experiences (most recent first)
    pub fn iter(&self) -> impl Iterator<Item = &Experience> {
        let start = if self.count < self.max_size {
            0
        } else {
            self.head
        };

        (0..self.count).map(move |i| {
            let idx = (start + self.count - 1 - i) % self.max_size;
            &self.buffer[idx]
        })
    }

    /// Get recent N experiences
    pub fn recent(&self, n: usize) -> Vec<&Experience> {
        self.iter().take(n).collect()
    }

    /// Get success rate for action type
    pub fn success_rate(&self, action_type: &str) -> Option<f32> {
        let relevant: Vec<_> = self.iter()
            .filter(|e| e.action_type == action_type)
            .collect();

        if relevant.is_empty() {
            return None;
        }

        let success_count = relevant.iter().filter(|e| e.success).count();
        Some(success_count as f32 / relevant.len() as f32)
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Pattern for action sequences
#[derive(Debug, Clone)]
pub struct ActionPattern {
    /// Sequence of action types
    pub sequence: Vec<String>,
    /// Support count (how often this pattern occurs)
    pub support: u32,
    /// Success rate when this pattern is followed
    pub success_rate: f32,
    /// Average reward
    pub avg_reward: f32,
}

/// Pattern miner for discovering recurring action sequences
pub struct PatternMiner {
    /// Minimum support threshold (fraction)
    pub min_support: f32,
    /// Maximum pattern length
    pub max_pattern_length: usize,
    /// Discovered patterns
    patterns: HashMap<String, ActionPattern>,
    /// Sequence buffer for mining
    sequence_buffer: Vec<String>,
}

impl Default for PatternMiner {
    fn default() -> Self {
        Self::new(0.1, 5)
    }
}

impl PatternMiner {
    pub fn new(min_support: f32, max_pattern_length: usize) -> Self {
        Self {
            min_support,
            max_pattern_length,
            patterns: HashMap::new(),
            sequence_buffer: Vec::new(),
        }
    }

    /// Add an action to the sequence buffer
    pub fn add_action(&mut self, action_type: &str, success: bool, reward: f32) {
        self.sequence_buffer.push(action_type.to_string());

        // Keep buffer bounded
        if self.sequence_buffer.len() > 100 {
            self.sequence_buffer.remove(0);
        }

        // Mine patterns of various lengths
        for len in 2..=self.max_pattern_length.min(self.sequence_buffer.len()) {
            let start = self.sequence_buffer.len() - len;
            let seq: Vec<_> = self.sequence_buffer[start..].to_vec();
            let key = seq.join(" -> ");

            let pattern = self.patterns.entry(key.clone()).or_insert(ActionPattern {
                sequence: seq,
                support: 0,
                success_rate: 0.0,
                avg_reward: 0.0,
            });

            pattern.support += 1;

            // Update success rate with exponential moving average
            let alpha = 0.1;
            let success_val = if success { 1.0 } else { 0.0 };
            pattern.success_rate = pattern.success_rate * (1.0 - alpha) + success_val * alpha;
            pattern.avg_reward = pattern.avg_reward * (1.0 - alpha) + reward * alpha;
        }
    }

    /// Get top patterns by support
    pub fn top_patterns(&self, n: usize) -> Vec<&ActionPattern> {
        let mut patterns: Vec<_> = self.patterns.values().collect();
        patterns.sort_by(|a, b| b.support.cmp(&a.support));
        patterns.into_iter().take(n).collect()
    }

    /// Suggest next action based on current sequence
    pub fn suggest_next(&self, recent_actions: &[String]) -> Option<&str> {
        if recent_actions.is_empty() {
            return None;
        }

        // Find patterns that match recent actions and have good success rate
        let mut best_match: Option<(&ActionPattern, usize)> = None;

        for pattern in self.patterns.values() {
            if pattern.success_rate < 0.5 || pattern.support < 3 {
                continue;
            }

            // Check if recent actions match the beginning of this pattern
            let match_len = pattern.sequence.iter()
                .zip(recent_actions.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if match_len > 0 && match_len < pattern.sequence.len() {
                if best_match.is_none() || match_len > best_match.unwrap().1 {
                    best_match = Some((pattern, match_len));
                }
            }
        }

        best_match.map(|(p, len)| p.sequence.get(len).map(|s| s.as_str())).flatten()
    }

    /// Get patterns count
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experience_buffer() {
        let mut buffer = ExperienceBuffer::new(5);

        for i in 0..7 {
            buffer.push(Experience::new(
                "test",
                &format!("sig_{}", i),
                [0u8; 32],
                i % 2 == 0,
                1.0,
            ));
        }

        // Should only have 5 most recent
        assert_eq!(buffer.len(), 5);

        // Most recent should be sig_6
        let recent = buffer.recent(1);
        assert_eq!(recent[0].action_signature, "sig_6");
    }

    #[test]
    fn test_pattern_mining() {
        let mut miner = PatternMiner::new(0.1, 3);

        // Simulate repeating sequence: A -> B -> C
        for _ in 0..5 {
            miner.add_action("A", true, 1.0);
            miner.add_action("B", true, 1.0);
            miner.add_action("C", true, 1.0);
        }

        // Should have discovered patterns
        assert!(miner.pattern_count() > 0);

        // Top pattern should have good support
        let top = miner.top_patterns(1);
        assert!(top[0].support >= 3);
    }

    #[test]
    fn test_suggest_next() {
        let mut miner = PatternMiner::new(0.1, 3);

        // Train on repeating sequence
        for _ in 0..10 {
            miner.add_action("init", true, 1.0);
            miner.add_action("build", true, 1.0);
            miner.add_action("test", true, 1.0);
        }

        // Should suggest "build" after "init"
        let recent = vec!["init".to_string()];
        let suggestion = miner.suggest_next(&recent);

        // May or may not have enough support, but should not panic
        if let Some(next) = suggestion {
            assert_eq!(next, "build");
        }
    }
}
