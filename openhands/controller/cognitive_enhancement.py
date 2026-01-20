"""Cognitive Enhancement Mixin for OpenHands agents.

Provides DAGMA causal analysis and holographic memory capabilities
to enhance agent reasoning and recall.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from openhands.core.logger import openhands_logger as logger

if TYPE_CHECKING:
    from openhands.events.action import Action
    from openhands.events.observation import Observation

# Try to import cognitive modules
try:
    from openhands.memory.holographic_memory import HolographicMemory, HOLOGRAPHIC_AVAILABLE
except ImportError:
    HOLOGRAPHIC_AVAILABLE = False
    HolographicMemory = None

try:
    from openhands.memory.causal_analyzer import CausalAnalyzer, DAGMA_AVAILABLE
except ImportError:
    DAGMA_AVAILABLE = False
    CausalAnalyzer = None


class CognitiveEnhancementMixin:
    """Mixin to add cognitive capabilities to OpenHands agents.

    Provides:
    - **Holographic Memory**: Store and recall past solutions by similarity
    - **Causal Analysis**: Understand why commands fail and predict outcomes

    # Usage:
    ```python
    class EnhancedCodeActAgent(CognitiveEnhancementMixin, CodeActAgent):
        def __init__(self, config, llm_registry):
            CodeActAgent.__init__(self, config, llm_registry)
            CognitiveEnhancementMixin.__init__(self)
    ```

    Or use the `enhance_agent()` function to add capabilities dynamically.
    """

    def __init__(
        self,
        enable_memory: bool = True,
        enable_causal: bool = True,
        memory_dim: int = 256,
        memory_max_items: int = 10000,
    ):
        """Initialize cognitive enhancement.

        Args:
            enable_memory: Enable holographic memory
            enable_causal: Enable causal analysis
            memory_dim: Dimension of memory space
            memory_max_items: Maximum items in memory
        """
        self._cognitive_memory: Optional[HolographicMemory] = None
        self._cognitive_analyzer: Optional[CausalAnalyzer] = None
        self._action_history: List[Dict[str, Any]] = []
        self._last_action_time_ms: float = 0
        self._total_overhead_ms: float = 0
        self._step_count: int = 0

        # Initialize holographic memory
        if enable_memory and HOLOGRAPHIC_AVAILABLE:
            try:
                self._cognitive_memory = HolographicMemory(
                    dim=memory_dim, max_items=memory_max_items
                )
                logger.info("CognitiveEnhancement: HolographicMemory enabled")
            except Exception as e:
                logger.warning(f"CognitiveEnhancement: Failed to init memory: {e}")

        # Initialize causal analyzer
        if enable_causal and DAGMA_AVAILABLE:
            try:
                self._cognitive_analyzer = CausalAnalyzer(mode="fast")
                logger.info("CognitiveEnhancement: CausalAnalyzer enabled")
            except Exception as e:
                logger.warning(f"CognitiveEnhancement: Failed to init analyzer: {e}")

    def on_action_start(self, action: "Action", context: str = "") -> Optional[str]:
        """Called before an action is executed.

        Can provide suggestions based on past experiences.

        Args:
            action: The action about to be executed
            context: Additional context (e.g., current task)

        Returns:
            Optional suggestion from memory
        """
        start_time = time.perf_counter()
        suggestion = None

        if self._cognitive_memory and context:
            try:
                result = self._cognitive_memory.recall_similar(context, threshold=0.3)
                if result:
                    similarity, recalled_solution = result
                    if similarity > 0.5:  # Only suggest for strong matches
                        suggestion = f"Similar past scenario (similarity={similarity:.1%}): {recalled_solution}"
                        logger.debug(f"CognitiveEnhancement: Found similar experience")
            except Exception as e:
                logger.debug(f"CognitiveEnhancement: Memory recall error: {e}")

        self._last_action_time_ms = (time.perf_counter() - start_time) * 1000
        return suggestion

    def on_action_success(
        self,
        action: "Action",
        observation: "Observation",
        context: str = "",
        solution: str = "",
    ) -> None:
        """Called when an action succeeds.

        Stores the successful pattern in memory.

        Args:
            action: The action that succeeded
            observation: The result observation
            context: Problem context (e.g., error that was solved)
            solution: The solution that worked
        """
        start_time = time.perf_counter()

        # Store in holographic memory
        if self._cognitive_memory and context and solution:
            try:
                self._cognitive_memory.store_experience(context, solution)
                logger.debug("CognitiveEnhancement: Stored successful experience")
            except Exception as e:
                logger.debug(f"CognitiveEnhancement: Memory store error: {e}")

        # Record for causal analysis
        self._action_history.append({
            "action_type": type(action).__name__,
            "success": True,
            "timestamp": time.time(),
        })

        self._total_overhead_ms += (time.perf_counter() - start_time) * 1000
        self._step_count += 1

    def on_action_failure(
        self,
        action: "Action",
        observation: "Observation",
        error: str = "",
    ) -> Optional[str]:
        """Called when an action fails.

        Analyzes the failure pattern and suggests remediation.

        Args:
            action: The action that failed
            observation: The error observation
            error: Error message

        Returns:
            Optional suggestion for remediation
        """
        start_time = time.perf_counter()
        suggestion = None

        # Record for causal analysis
        self._action_history.append({
            "action_type": type(action).__name__,
            "success": False,
            "error": error[:100] if error else "",
            "timestamp": time.time(),
        })

        # Try to find similar past solutions
        if self._cognitive_memory and error:
            try:
                result = self._cognitive_memory.recall_similar(error, threshold=0.2)
                if result:
                    similarity, recalled_solution = result
                    suggestion = f"Past solution for similar error: {recalled_solution}"
            except Exception as e:
                logger.debug(f"CognitiveEnhancement: Memory recall error: {e}")

        self._total_overhead_ms += (time.perf_counter() - start_time) * 1000
        self._step_count += 1

        return suggestion

    def analyze_failure_pattern(
        self,
        variables: Dict[str, float],
        failure_var: str = "success",
    ) -> Optional[str]:
        """Analyze the causal pattern of recent failures.

        Args:
            variables: Current state variables (e.g., {"file_exists": 1.0, "permissions": 0.0})
            failure_var: Name of the failure variable

        Returns:
            Explanation of root cause
        """
        if not self._cognitive_analyzer:
            return None

        try:
            explanation = self._cognitive_analyzer.explain_failure(variables, failure_var)
            return explanation
        except Exception as e:
            logger.debug(f"CognitiveEnhancement: Causal analysis error: {e}")
            return None

    def predict_intervention(
        self,
        current_state: Dict[str, float],
        change_var: str,
        new_value: float,
    ) -> Optional[Dict[str, float]]:
        """Predict what would happen if we change a variable.

        Answers counterfactual questions like "What if file permissions were fixed?"

        Args:
            current_state: Current variable values
            change_var: Variable to change
            new_value: New value

        Returns:
            Predicted state after intervention
        """
        if not self._cognitive_analyzer:
            return None

        try:
            var_names = list(current_state.keys())
            observations = [list(current_state.values())]

            var_idx = var_names.index(change_var) if change_var in var_names else -1
            if var_idx < 0:
                return None

            predicted = self._cognitive_analyzer.predict_intervention(
                observations, var_idx, new_value
            )
            return dict(zip(var_names, predicted)) if predicted else None
        except Exception as e:
            logger.debug(f"CognitiveEnhancement: Intervention error: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for cognitive operations.

        Returns:
            Dict with overhead_ms, avg_overhead_per_step_ms, step_count
        """
        avg = self._total_overhead_ms / self._step_count if self._step_count > 0 else 0
        return {
            "total_overhead_ms": self._total_overhead_ms,
            "avg_overhead_per_step_ms": avg,
            "step_count": self._step_count,
            "memory_items": self._cognitive_memory.item_count() if self._cognitive_memory else 0,
            "memory_energy": self._cognitive_memory.energy() if self._cognitive_memory else 0,
        }

    def clear_memory(self) -> None:
        """Clear all stored memories."""
        if self._cognitive_memory:
            self._cognitive_memory.clear()
            logger.info("CognitiveEnhancement: Memory cleared")


def enhance_agent(agent, **kwargs) -> None:
    """Add cognitive capabilities to an existing agent instance.

    Args:
        agent: The agent instance to enhance
        **kwargs: Arguments passed to CognitiveEnhancementMixin.__init__
    """
    mixin = CognitiveEnhancementMixin(**kwargs)
    agent._cognitive_memory = mixin._cognitive_memory
    agent._cognitive_analyzer = mixin._cognitive_analyzer
    agent._action_history = mixin._action_history
    agent._last_action_time_ms = mixin._last_action_time_ms
    agent._total_overhead_ms = mixin._total_overhead_ms
    agent._step_count = mixin._step_count

    # Bind methods
    agent.on_action_start = lambda action, context="": mixin.on_action_start(action, context)
    agent.on_action_success = mixin.on_action_success
    agent.on_action_failure = mixin.on_action_failure
    agent.analyze_failure_pattern = mixin.analyze_failure_pattern
    agent.predict_intervention = mixin.predict_intervention
    agent.get_performance_stats = mixin.get_performance_stats
    agent.clear_memory = mixin.clear_memory

    logger.info(f"CognitiveEnhancement: Enhanced agent {type(agent).__name__}")


# Export status
COGNITIVE_AVAILABLE = HOLOGRAPHIC_AVAILABLE or DAGMA_AVAILABLE
