"""Policy Selector for Active Inference action selection.

Provides a Python wrapper around the Rust EFE (Expected Free Energy)
calculator for intelligent action selection based on goal alignment
and uncertainty.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import Rust bindings
try:
    import openhands_agolos as ag
    POLICY_AVAILABLE = True
except ImportError:
    POLICY_AVAILABLE = False
    ag = None


@dataclass
class PolicyDecision:
    """Result of policy selection."""
    policy: str  # Execute, RequestConfirmation, Skip, Fallback, Explore
    efe_value: float  # Expected Free Energy (lower = better)
    pragmatic_value: float  # Goal alignment component
    epistemic_value: float  # Information gain component

    @property
    def requires_confirmation(self) -> bool:
        return self.policy == "RequestConfirmation"

    @property
    def should_skip(self) -> bool:
        return self.policy == "Skip"

    @property
    def should_explore(self) -> bool:
        return self.policy == "Explore"


class PolicySelector:
    """Active Inference policy selector for agent actions.

    Uses Expected Free Energy to balance:
    - Pragmatic value: Goal alignment (exploitation)
    - Epistemic value: Information gain (exploration)

    Usage:
        selector = PolicySelector()
        decision = selector.select_action("cmd_run", goal_alignment=0.8, uncertainty=0.3)
        if decision.requires_confirmation:
            await request_user_confirmation()
    """

    def __init__(self, temperature: float = 1.0, epistemic_weight: float = 0.3):
        """Initialize policy selector.

        Args:
            temperature: Softmax temperature (higher = more exploration)
            epistemic_weight: Weight for exploration vs exploitation
        """
        self._selector = None
        self._efe_calc = None

        if POLICY_AVAILABLE and ag is not None:
            try:
                self._selector = ag.PolicySelector()
                self._efe_calc = ag.EFECalculator.with_params(temperature, epistemic_weight)
                logger.info("PolicySelector: Rust backend initialized")
            except Exception as e:
                logger.warning(f"PolicySelector: Failed to init Rust backend: {e}")
        else:
            logger.warning("PolicySelector: Rust backend not available, using fallback")

    def select_action(
        self,
        action_type: str,
        goal_alignment: float,
        uncertainty: float,
    ) -> PolicyDecision:
        """Select best policy for an action.

        Args:
            action_type: Type of action (e.g., "cmd_run", "file_edit")
            goal_alignment: How well action aligns with goal (0-1)
            uncertainty: How uncertain we are about outcome (0-1)

        Returns:
            PolicyDecision with recommended policy
        """
        if self._selector is not None:
            try:
                eval_result = self._selector.select(action_type, goal_alignment, uncertainty)
                return PolicyDecision(
                    policy=eval_result.policy.replace("ActionPolicy::", ""),
                    efe_value=eval_result.efe_value,
                    pragmatic_value=eval_result.pragmatic,
                    epistemic_value=eval_result.epistemic,
                )
            except Exception as e:
                logger.warning(f"PolicySelector: Rust selection failed: {e}")

        # Fallback Python implementation
        return self._fallback_select(action_type, goal_alignment, uncertainty)

    def _fallback_select(
        self,
        action_type: str,
        goal_alignment: float,
        uncertainty: float,
    ) -> PolicyDecision:
        """Fallback policy selection when Rust not available."""
        # Simple heuristic matching Rust implementation
        success_rate = self._action_success_rates.get(action_type, 0.5)

        pragmatic = goal_alignment * success_rate
        epistemic = uncertainty * 0.5
        efe = -(0.7 * pragmatic + 0.3 * epistemic)

        if goal_alignment < 0.2:
            policy = "Skip"
        elif uncertainty > 0.8:
            policy = "Explore"
        elif success_rate < 0.3:
            policy = "Fallback"
        elif goal_alignment > 0.8 and success_rate > 0.7:
            policy = "Execute"
        else:
            policy = "RequestConfirmation"

        return PolicyDecision(
            policy=policy,
            efe_value=efe,
            pragmatic_value=pragmatic,
            epistemic_value=epistemic,
        )

    _action_success_rates: dict = {}

    def record_outcome(self, action_type: str, success: bool) -> None:
        """Record action outcome for learning.

        Args:
            action_type: Type of action
            success: Whether action succeeded
        """
        if self._selector is not None:
            try:
                self._selector.record_outcome(action_type, success)
                return
            except Exception as e:
                logger.warning(f"PolicySelector: Failed to record outcome: {e}")

        # Fallback: update local success rates
        if action_type not in self._action_success_rates:
            self._action_success_rates[action_type] = 0.5

        rate = self._action_success_rates[action_type]
        # Exponential moving average
        alpha = 0.1
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * rate
        self._action_success_rates[action_type] = new_rate

    def get_success_rate(self, action_type: str) -> float:
        """Get current success rate for action type."""
        if self._selector is not None:
            try:
                return self._selector.get_success_rate(action_type)
            except:
                pass
        return self._action_success_rates.get(action_type, 0.5)


# Convenience function for quick policy checks
def should_execute(action_type: str, goal_alignment: float, uncertainty: float) -> bool:
    """Quick check if action should be executed without confirmation."""
    selector = PolicySelector()
    decision = selector.select_action(action_type, goal_alignment, uncertainty)
    return decision.policy == "Execute"
