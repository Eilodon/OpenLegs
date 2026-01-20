//! 3-Tier Trauma Registry - Persistent failure memory with forgetting
//!
//! Adapted from Pandora SDK's zenb-core/src/safety_swarm.rs
//!
//! Provides 3-tier trauma memory:
//! - **Light**: Project-scoped, 24h decay
//! - **Medium**: Cross-project, 7d decay
//! - **Severe**: Global, 30d+ / manual reset

use blake3::Hasher;
use chrono::Utc;
use rusqlite::{Connection, params};
use std::path::Path;

/// Trauma severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TraumaSeverity {
    /// Project-scoped, short-term (24h decay)
    Light = 1,
    /// Cross-project with decay (7d decay)
    Medium = 2,
    /// Global, long-persistent (30d+)
    Severe = 3,
}

impl TraumaSeverity {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            1 => Some(Self::Light),
            2 => Some(Self::Medium),
            3 => Some(Self::Severe),
            _ => None,
        }
    }

    /// Default decay hours for each severity
    pub fn default_decay_hours(&self) -> i64 {
        match self {
            TraumaSeverity::Light => 24,
            TraumaSeverity::Medium => 168,  // 7 days
            TraumaSeverity::Severe => 720,  // 30 days
        }
    }
}

/// Trauma record
#[derive(Debug, Clone)]
pub struct TraumaHit {
    pub severity: TraumaSeverity,
    pub count: u32,
    pub inhibit_until_ts: i64,
    pub last_ts: i64,
    pub action_type: String,
}

impl TraumaHit {
    /// Check if this trauma has expired (past inhibit time)
    pub fn is_expired(&self) -> bool {
        let now = Utc::now().timestamp();
        now > self.inhibit_until_ts
    }

    /// Fear score based on severity and recency
    pub fn fear_score(&self) -> f32 {
        if self.is_expired() {
            return 0.0;
        }

        match self.severity {
            TraumaSeverity::Light => 0.3,
            TraumaSeverity::Medium => 0.6,
            TraumaSeverity::Severe => 1.0,
        }
    }
}

/// 3-Tier Trauma Registry with SQLite persistence
pub struct TraumaRegistry {
    conn: Connection,
}

impl TraumaRegistry {
    /// Open or create a trauma registry at the given path
    pub fn open<P: AsRef<Path>>(db_path: P) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS trauma (
                context_hash BLOB PRIMARY KEY,
                severity INTEGER NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                inhibit_until_ts INTEGER NOT NULL,
                last_ts INTEGER NOT NULL,
                action_type TEXT NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trauma_inhibit ON trauma(inhibit_until_ts)",
            [],
        )?;

        Ok(Self { conn })
    }

    /// Open in-memory database (for testing)
    pub fn in_memory() -> Result<Self, rusqlite::Error> {
        Self::open(":memory:")
    }

    /// Record a failure with exponential backoff
    pub fn record_failure(
        &mut self,
        context_hash: [u8; 32],
        action_type: &str,
        severity: TraumaSeverity,
        decay_hours: i64,
    ) {
        let now_ts = Utc::now().timestamp();

        // Check if exists
        let existing: Option<(u32, i64)> = self
            .conn
            .query_row(
                "SELECT count, inhibit_until_ts FROM trauma WHERE context_hash = ?",
                [&context_hash[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();

        let (new_count, new_inhibit) = if let Some((count, _old_inhibit)) = existing {
            // Exponential backoff: 2^(count-1) * base_decay
            let new_count = count.saturating_add(1);
            let multiplier = 1i64 << (new_count.min(10) - 1);
            let new_inhibit = now_ts + (decay_hours * 3600 * multiplier);
            (new_count, new_inhibit)
        } else {
            // First failure
            (1, now_ts + (decay_hours * 3600))
        };

        self.conn
            .execute(
                "INSERT OR REPLACE INTO trauma
                 (context_hash, severity, count, inhibit_until_ts, last_ts, action_type)
                 VALUES (?, ?, ?, ?, ?, ?)",
                params![
                    &context_hash[..],
                    severity as u8,
                    new_count,
                    new_inhibit,
                    now_ts,
                    action_type
                ],
            )
            .ok();

        log::info!(
            "TRAUMA RECORDED: action={}, severity={:?}, count={}, inhibit_until=+{}h",
            action_type,
            severity,
            new_count,
            (new_inhibit - now_ts) / 3600
        );
    }

    /// Query trauma for a context hash
    pub fn query(&self, context_hash: &[u8]) -> Option<TraumaHit> {
        if context_hash.len() != 32 {
            return None;
        }

        self.conn
            .query_row(
                "SELECT severity, count, inhibit_until_ts, last_ts, action_type
                 FROM trauma WHERE context_hash = ?",
                [context_hash],
                |row| {
                    let sev_u8: u8 = row.get(0)?;
                    let severity = TraumaSeverity::from_u8(sev_u8)
                        .unwrap_or(TraumaSeverity::Light);
                    Ok(TraumaHit {
                        severity,
                        count: row.get(1)?,
                        inhibit_until_ts: row.get(2)?,
                        last_ts: row.get(3)?,
                        action_type: row.get(4)?,
                    })
                },
            )
            .ok()
    }

    /// Clear expired trauma records (garbage collection)
    pub fn gc(&mut self) {
        let now = Utc::now().timestamp();
        self.conn
            .execute(
                "DELETE FROM trauma WHERE inhibit_until_ts < ? AND severity < 3",
                [now],
            )
            .ok();
    }

    /// Clear all trauma (admin reset)
    pub fn clear_all(&mut self) {
        self.conn.execute("DELETE FROM trauma", []).ok();
    }

    /// Compute context hash from action details
    pub fn compute_context_hash(
        command: &str,
        working_dir: &str,
        project_id: &str,
    ) -> [u8; 32] {
        let mut hasher = Hasher::new();

        // Hash command structure (not exact content for generalization)
        let normalized_cmd = Self::normalize_command(command);
        hasher.update(normalized_cmd.as_bytes());
        hasher.update(b"||");
        hasher.update(working_dir.as_bytes());
        hasher.update(b"||");
        hasher.update(project_id.as_bytes());

        *hasher.finalize().as_bytes()
    }

    /// Normalize command for generalization
    /// Strips specific paths/args to detect similar commands
    fn normalize_command(cmd: &str) -> String {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            return String::new();
        }

        // Keep first word (command) and flags, normalize paths
        let mut normalized = Vec::new();
        normalized.push(parts[0].to_string());

        for part in parts.iter().skip(1) {
            if part.starts_with('-') {
                // Keep flags
                normalized.push(part.to_string());
            } else if part.starts_with('/') || part.starts_with('.') {
                // Normalize paths to placeholder
                normalized.push("<PATH>".to_string());
            } else {
                // Keep other args (like table names)
                normalized.push(part.to_string());
            }
        }

        normalized.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_query() {
        let mut registry = TraumaRegistry::in_memory().unwrap();
        let hash = TraumaRegistry::compute_context_hash("rm -rf /", "/home", "proj1");

        registry.record_failure(hash, "rm -rf", TraumaSeverity::Severe, 24);

        let hit = registry.query(&hash).unwrap();
        assert_eq!(hit.severity, TraumaSeverity::Severe);
        assert_eq!(hit.count, 1);
        assert!(!hit.is_expired());
    }

    #[test]
    fn test_exponential_backoff() {
        let mut registry = TraumaRegistry::in_memory().unwrap();
        let hash = TraumaRegistry::compute_context_hash("drop table users", "/", "proj1");

        // First failure
        registry.record_failure(hash, "drop table", TraumaSeverity::Medium, 24);
        let hit1 = registry.query(&hash).unwrap();
        assert_eq!(hit1.count, 1);

        // Second failure - count increases
        registry.record_failure(hash, "drop table", TraumaSeverity::Medium, 24);
        let hit2 = registry.query(&hash).unwrap();
        assert_eq!(hit2.count, 2);

        // Inhibit time should be longer (exponential)
        assert!(hit2.inhibit_until_ts > hit1.inhibit_until_ts);
    }

    #[test]
    fn test_command_normalization() {
        let norm1 = TraumaRegistry::normalize_command("rm -rf /home/user/project");
        let norm2 = TraumaRegistry::normalize_command("rm -rf /var/log/app");

        // Both should normalize to same pattern
        assert_eq!(norm1, norm2);
        assert_eq!(norm1, "rm -rf <PATH>");
    }

    #[test]
    fn test_fear_score() {
        let light_hit = TraumaHit {
            severity: TraumaSeverity::Light,
            count: 1,
            inhibit_until_ts: Utc::now().timestamp() + 3600,
            last_ts: Utc::now().timestamp(),
            action_type: "test".to_string(),
        };
        assert!(light_hit.fear_score() > 0.0);

        let severe_hit = TraumaHit {
            severity: TraumaSeverity::Severe,
            count: 1,
            inhibit_until_ts: Utc::now().timestamp() + 3600,
            last_ts: Utc::now().timestamp(),
            action_type: "test".to_string(),
        };
        assert!(severe_hit.fear_score() > light_hit.fear_score());
    }
}
