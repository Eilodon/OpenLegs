"""3-Tier Trauma Memory for OpenHands agents.

Provides persistent memory of agent failures with severity-based scoping and forgetting.
"""

import os
from enum import IntEnum
from typing import Optional

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, CmdRunAction

# Try to import the Rust extension
try:
    import openhands_agolos

    AGOLOS_AVAILABLE = True
except ImportError:
    AGOLOS_AVAILABLE = False


class TraumaSeverity(IntEnum):
    """Trauma severity levels with different scoping and forgetting."""

    LIGHT = 1  # Project-scoped, 24h decay
    MEDIUM = 2  # Cross-project, 7d decay
    SEVERE = 3  # Global, 30d+ decay


# Default decay hours for each severity
DECAY_HOURS = {
    TraumaSeverity.LIGHT: 24,
    TraumaSeverity.MEDIUM: 168,  # 7 days
    TraumaSeverity.SEVERE: 720,  # 30 days
}


class TraumaMemory:
    """3-Tier Trauma Memory for OpenHands agents.

    Manages two registries:
    - Project registry: Light severity traumas (project-scoped)
    - Global registry: Medium and Severe traumas (cross-project)

    Severity levels:
    | Tier   | Scope         | Decay   | Example                    |
    |--------|---------------|---------|----------------------------|
    | Light  | Project-only  | 24h     | Test failure, lint error   |
    | Medium | Cross-project | 7d      | Build break, API crash     |
    | Severe | Global        | 30d+    | rm -rf, DROP TABLE         |
    """

    def __init__(
        self,
        project_db_path: Optional[str] = None,
        global_db_path: Optional[str] = None,
        project_id: str = 'default',
    ):
        """Initialize 3-tier trauma memory.

        Args:
            project_db_path: Path to project-specific trauma database
            global_db_path: Path to global trauma database (cross-project)
            project_id: Current project identifier
        """
        if not AGOLOS_AVAILABLE:
            raise ImportError(
                'openhands_agolos is not available. '
                'Build it with: cd openhands-agolos && maturin develop'
            )

        self.project_id = project_id

        # Set default paths
        openhands_dir = os.path.expanduser('~/.openhands')

        if project_db_path is None:
            project_db_path = os.path.join(openhands_dir, 'trauma', f'{project_id}.db')

        if global_db_path is None:
            global_db_path = os.path.join(openhands_dir, 'trauma', 'global.db')

        # Ensure directories exist
        os.makedirs(os.path.dirname(project_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(global_db_path), exist_ok=True)

        self.project_registry = openhands_agolos.PyTraumaRegistry(project_db_path)
        self.global_registry = openhands_agolos.PyTraumaRegistry(global_db_path)

        logger.info(f"TraumaMemory initialized for project '{project_id}'")

    def record_failure(
        self,
        action: Action,
        outcome: str,
        severity: TraumaSeverity,
        decay_hours: Optional[int] = None,
    ) -> None:
        """Record a failure with appropriate tier.

        Args:
            action: The action that failed
            outcome: Description of the failure
            severity: Light, Medium, or Severe
            decay_hours: Override default decay for severity
        """
        if not isinstance(action, CmdRunAction):
            return

        if decay_hours is None:
            decay_hours = DECAY_HOURS[severity]

        context_hash = openhands_agolos.PyTraumaRegistry.compute_context_hash(
            action.command,
            getattr(action, 'cwd', '/'),
            self.project_id,
        )

        # Route to appropriate registry
        if severity == TraumaSeverity.LIGHT:
            self.project_registry.record_failure(
                context_hash,
                action.action,
                severity.value,
                decay_hours,
            )
        else:
            # Medium and Severe go to global registry
            self.global_registry.record_failure(
                context_hash,
                action.action,
                severity.value,
                decay_hours,
            )

        logger.info(
            f'Trauma recorded: severity={severity.name}, outcome={outcome}, decay={decay_hours}h'
        )

    def query_fear(
        self,
        action: Action,
    ) -> tuple[float, Optional[TraumaSeverity]]:
        """Query trauma across all tiers, return highest severity match.

        Args:
            action: The action to check

        Returns:
            Tuple of (fear_score, severity) or (0.0, None) if no trauma
        """
        if not isinstance(action, CmdRunAction):
            return (0.0, None)

        context_hash = openhands_agolos.PyTraumaRegistry.compute_context_hash(
            action.command,
            getattr(action, 'cwd', '/'),
            self.project_id,
        )

        # Check global registry first (higher priority)
        global_hit = self.global_registry.query(context_hash)
        if global_hit and not global_hit.is_expired:
            if global_hit.severity >= TraumaSeverity.SEVERE.value:
                return (1.0, TraumaSeverity.SEVERE)
            elif global_hit.severity >= TraumaSeverity.MEDIUM.value:
                return (0.6, TraumaSeverity.MEDIUM)

        # Check project registry
        project_hit = self.project_registry.query(context_hash)
        if project_hit and not project_hit.is_expired:
            return (0.3, TraumaSeverity.LIGHT)

        return (0.0, None)

    def get_fear_message(
        self, fear_score: float, severity: Optional[TraumaSeverity]
    ) -> str:
        """Generate a human-readable fear message.

        Args:
            fear_score: Fear score from query_fear
            severity: Severity level from query_fear

        Returns:
            Warning message for the agent
        """
        if severity is None:
            return ''

        messages = {
            TraumaSeverity.LIGHT: (
                '⚠️ Minor past issue detected. Proceed with caution.'
            ),
            TraumaSeverity.MEDIUM: (
                '⚠️ Previous failure recorded for similar action. '
                'Consider alternative approaches.'
            ),
            TraumaSeverity.SEVERE: (
                '🛑 SEVERE TRAUMA: This action pattern caused significant damage previously. '
                'Confirmation required before proceeding.'
            ),
        }

        return messages.get(severity, '')
