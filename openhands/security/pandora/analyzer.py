"""Pandora Security Analyzer - LTL-based safety verification for OpenHands

Uses AGOLOS safety primitives from Pandora SDK to provide mathematically-provable
action rejection with confirmation mode.
"""

import os
from typing import Optional

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, CmdRunAction
from openhands.events.action.action import ActionSecurityRisk, ActionConfirmationStatus
from openhands.security.analyzer import SecurityAnalyzer

# Try to import the Rust extension
try:
    import openhands_agolos
    AGOLOS_AVAILABLE = True
except ImportError:
    AGOLOS_AVAILABLE = False
    logger.warning(
        "openhands_agolos not available. Run 'cd openhands-agolos && maturin develop' to build."
    )


class PandoraSecurityAnalyzer(SecurityAnalyzer):
    """LTL-based security analyzer using AGOLOS safety primitives.

    Provides mathematically-provable command blocking:
    - rm -rf / (root deletion)
    - DROP TABLE/DATABASE (data destruction)
    - sudo/su (privilege escalation)
    - curl | bash (remote code execution)
    - /dev/sd* writes (disk destruction)
    - Fork bombs

    Flagged commands return AWAITING_CONFIRMATION instead of hard rejection.
    """

    def __init__(
        self,
        trauma_db_path: Optional[str] = None,
    ) -> None:
        """Initialize Pandora Security Analyzer.

        Args:
            trauma_db_path: Path to trauma SQLite database. Defaults to ~/.openhands/trauma.db
        """
        super().__init__()

        if not AGOLOS_AVAILABLE:
            raise ImportError(
                "openhands_agolos is not available. "
                "Build it with: cd openhands-agolos && maturin develop"
            )

        # Initialize LTL Monitor
        self.ltl_monitor = openhands_agolos.PyLtlMonitor()

        # Initialize Trauma Registry
        if trauma_db_path is None:
            trauma_db_path = os.path.expanduser("~/.openhands/trauma.db")

        # Ensure directory exists
        os.makedirs(os.path.dirname(trauma_db_path), exist_ok=True)

        self.trauma_registry = openhands_agolos.PyTraumaRegistry(trauma_db_path)
        self.project_id = "default"

        logger.info("PandoraSecurityAnalyzer initialized with LTL monitor and trauma registry")

    def set_project_id(self, project_id: str) -> None:
        """Set the current project ID for trauma scoping."""
        self.project_id = project_id

    async def security_risk(self, action: Action) -> ActionSecurityRisk:
        """Evaluate security risk of an action using LTL verification.

        Args:
            action: The action to evaluate

        Returns:
            ActionSecurityRisk level (LOW, MEDIUM, HIGH, UNKNOWN)
        """
        # Only check CmdRunAction for now
        if not isinstance(action, CmdRunAction):
            return ActionSecurityRisk.LOW

        command = action.command

        # 1. Check LTL violations
        violations = self.ltl_monitor.check_command(command)

        if violations:
            # Log violations
            violation_details = "; ".join(
                f"{v.property_name}: {v.description}" for v in violations
            )
            logger.warning(
                f"LTL Safety Violation detected: {violation_details}"
            )

            # Set confirmation required (not hard block)
            action.confirmation_state = ActionConfirmationStatus.AWAITING_CONFIRMATION

            return ActionSecurityRisk.HIGH

        # 2. Check trauma registry for past failures
        try:
            context_hash = openhands_agolos.PyTraumaRegistry.compute_context_hash(
                command,
                getattr(action, 'cwd', '/'),
                self.project_id,
            )

            trauma_hit = self.trauma_registry.query(context_hash)

            if trauma_hit and not trauma_hit.is_expired:
                logger.info(
                    f"Trauma detected for command (severity={trauma_hit.severity}, count={trauma_hit.count})"
                )

                # Map severity to risk
                if trauma_hit.severity >= 3:  # Severe
                    action.confirmation_state = ActionConfirmationStatus.AWAITING_CONFIRMATION
                    return ActionSecurityRisk.HIGH
                elif trauma_hit.severity >= 2:  # Medium
                    return ActionSecurityRisk.MEDIUM
                else:  # Light
                    return ActionSecurityRisk.LOW
        except Exception as e:
            logger.debug(f"Trauma query failed: {e}")

        return ActionSecurityRisk.LOW

    def record_failure(
        self,
        action: Action,
        outcome: str,
        severity: int,
        decay_hours: int = 24,
    ) -> None:
        """Record a failure for trauma tracking.

        Args:
            action: The action that failed
            outcome: Description of the failure
            severity: 1=Light, 2=Medium, 3=Severe
            decay_hours: Hours until trauma expires
        """
        if not isinstance(action, CmdRunAction):
            return

        try:
            context_hash = openhands_agolos.PyTraumaRegistry.compute_context_hash(
                action.command,
                getattr(action, 'cwd', '/'),
                self.project_id,
            )

            self.trauma_registry.record_failure(
                context_hash,
                action.action,
                severity,
                decay_hours,
            )

            logger.info(
                f"Recorded trauma: severity={severity}, outcome={outcome}"
            )
        except Exception as e:
            logger.error(f"Failed to record trauma: {e}")

    def add_custom_rule(self, name: str, description: str, blocked_pattern: str) -> None:
        """Add a custom LTL blocking rule.

        Args:
            name: Rule name
            description: Human-readable description
            blocked_pattern: Pattern to block (substring match)
        """
        self.ltl_monitor.add_pattern_rule(name, description, blocked_pattern)
        logger.info(f"Added custom LTL rule: {name}")

    async def close(self) -> None:
        """Cleanup resources."""
        pass  # SQLite connection is managed by Rust
