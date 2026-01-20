"""Safety Monitor for Runtime verification.

Provides LTL-based runtime verification of safety properties.
"""
import logging
from typing import Optional

try:
    from openhands_agolos import PyLtlMonitor
    LTL_MONITOR_AVAILABLE = True
except ImportError:
    LTL_MONITOR_AVAILABLE = False
    PyLtlMonitor = None  # type: ignore

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Runtime safety monitor using LTL verification.

    Monitors commands before execution to prevent dangerous operations.
    """

    def __init__(self):
        """Initialize safety monitor with default rules."""
        if not LTL_MONITOR_AVAILABLE:
            logger.warning(
                "openhands_agolos not available. Safety monitor disabled. "
                "Build with: cd openhands-agolos && maturin develop"
            )
            self._monitor = None
            return

        self._monitor = PyLtlMonitor()

        # Add default safety rules
        self._add_default_rules()
        logger.info("Safety monitor initialized with LTL verification")

    def _add_default_rules(self):
        """Add default safety properties."""
        if self._monitor is None:
            return

        # Block dangerous root deletion patterns
        self._monitor.add_pattern_rule(
            "no_root_deletion",
            "Prevent deletion of root or critical system directories",
            r"rm\s+(-[rf]+\s+)*/"
        )

        # Block recursive force removal of important dirs
        self._monitor.add_pattern_rule(
            "no_force_recursive_root",
            "Prevent rm -rf on root directory",
            r"rm\s+-[rf]+(f|r)\s+/"
        )

    def check_command(self, command: str) -> Optional[str]:
        """Check if command violates safety properties.

        Args:
            command: Shell command to check

        Returns:
            Error message if command is unsafe, None if safe
        """
        if self._monitor is None:
            return None

        violations = self._monitor.check_command(command)
        if violations:
            # Return first violation description
            return f"Safety violation: {violations[0].description}"

        return None

    def is_safe(self, command: str) -> bool:
        """Check if command is safe to execute.

        Args:
            command: Shell command to check

        Returns:
            True if safe, False if violates safety properties
        """
        return self.check_command(command) is None
