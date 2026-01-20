"""Causal Analyzer for OpenHands agents.

Provides DAGMA-based causal reasoning to understand:
- Why did a command fail?
- What is the root cause of an error?
- What would happen if I used different parameters?
"""

from typing import Dict, List, Optional, Tuple

from openhands.core.logger import openhands_logger as logger

# Try to import the Rust extension
try:
    import openhands_agolos
    AGOLOS_AVAILABLE = True
    # Check if DAGMA feature is enabled
    DAGMA_AVAILABLE = hasattr(openhands_agolos, 'PyDagma')
except ImportError:
    AGOLOS_AVAILABLE = False
    DAGMA_AVAILABLE = False


class CausalAnalyzer:
    """DAGMA-based causal analyzer for OpenHands agents.

    Uses DAG learning via M-matrices and log-det acyclicity constraint
    (20x faster than NOTEARS) to discover causal relationships.

    # Usage Example:
    ```python
    analyzer = CausalAnalyzer()
    analyzer.set_variable_names([
        "file_exists",
        "permissions_ok",
        "command_run",
        "test_passed"
    ])

    # Record observations
    observations = [
        [1.0, 1.0, 1.0, 1.0],  # All good
        [0.0, 1.0, 1.0, 0.0],  # File missing -> test failed
        [1.0, 0.0, 0.0, 0.0],  # Permission denied -> nothing worked
        ...
    ]

    result = analyzer.analyze(observations)
    print(f"Root cause: {result.root_cause_variable}")
    print(f"Suggestion: {result.suggestion}")
    ```
    """

    def __init__(self, mode: str = "adaptive"):
        """Initialize causal analyzer.

        Args:
            mode: Analysis mode - "adaptive" (recommended), "fast", or "default"
        """
        if not DAGMA_AVAILABLE:
            raise ImportError(
                "PyDagma is not available. "
                "Build with: cd openhands-agolos && maturin develop --features causal-dagma"
            )

        self.mode = mode
        self._dagma: Optional[openhands_agolos.PyDagma] = None
        self._variable_names: List[str] = []
        logger.debug(f"CausalAnalyzer initialized (mode={mode})")

    def set_variable_names(self, names: List[str]) -> None:
        """Set variable names for better reporting.

        Args:
            names: List of variable names (e.g., ["file_exists", "permissions_ok"])
        """
        self._variable_names = names

    def analyze(self, observations: List[List[float]]) -> "CausalAnalysisResult":
        """Analyze observations to find root cause.

        Args:
            observations: 2D array [n_samples, n_variables]
                Each row is an observation, each column is a variable.

        Returns:
            CausalAnalysisResult with root cause, confidence, and suggestion.
        """
        if not observations or not observations[0]:
            return CausalAnalysisResult(
                root_cause_index=0,
                root_cause_variable=None,
                confidence=0.0,
                suggestion="No data to analyze",
                adjacency_matrix=[],
                acyclicity_score=0.0,
            )

        n_vars = len(observations[0])
        n_samples = len(observations)

        # Create DAGMA instance
        if self.mode == "adaptive":
            self._dagma = openhands_agolos.PyDagma.adaptive(n_vars, n_samples)
        elif self.mode == "fast":
            self._dagma = openhands_agolos.PyDagma.fast(n_vars)
        else:
            self._dagma = openhands_agolos.PyDagma()

        # Set variable names if provided
        if self._variable_names:
            self._dagma.set_variable_names(self._variable_names)

        # Run analysis
        analysis = self._dagma.analyze(observations)

        # Get variable name if available
        root_cause_var = None
        if self._variable_names and analysis.root_cause_index < len(self._variable_names):
            root_cause_var = self._variable_names[analysis.root_cause_index]

        return CausalAnalysisResult(
            root_cause_index=analysis.root_cause_index,
            root_cause_variable=root_cause_var,
            confidence=analysis.confidence,
            suggestion=analysis.suggestion,
            adjacency_matrix=analysis.adjacency,
            acyclicity_score=analysis.acyclicity_score,
        )

    def predict_intervention(
        self,
        observations: List[List[float]],
        variable_index: int,
        new_value: float,
    ) -> List[float]:
        """Predict the effect of intervening on a variable.

        Answers counterfactual questions like:
        "What would happen if I changed X to Y?"

        Args:
            observations: Current observation data
            variable_index: Index of variable to change
            new_value: New value to set

        Returns:
            Predicted state after intervention
        """
        if self._dagma is None:
            # Need to run analyze first if not already done
            if not observations or not observations[0]:
                return []
            n_vars = len(observations[0])
            n_samples = len(observations)
            self._dagma = openhands_agolos.PyDagma.adaptive(n_vars, n_samples)

        return self._dagma.predict_intervention(observations, variable_index, new_value)

    def explain_failure(
        self,
        variables: Dict[str, float],
        failure_variable: str,
    ) -> str:
        """Explain why a failure occurred.

        Args:
            variables: Current state of all variables
            failure_variable: Name of the variable that failed

        Returns:
            Human-readable explanation of the failure
        """
        if not self._variable_names:
            self._variable_names = list(variables.keys())

        # Convert to observations format
        observations = [[variables[name] for name in self._variable_names]]

        # Analyze
        result = self.analyze(observations)

        if result.root_cause_variable == failure_variable:
            return f"The variable '{failure_variable}' is itself the root cause."
        elif result.root_cause_variable:
            return (
                f"'{failure_variable}' failed because '{result.root_cause_variable}' "
                f"is the root cause (confidence: {result.confidence:.1%}). "
                f"{result.suggestion}"
            )
        else:
            return "Unable to determine root cause with available data."


class CausalAnalysisResult:
    """Result of causal analysis."""

    def __init__(
        self,
        root_cause_index: int,
        root_cause_variable: Optional[str],
        confidence: float,
        suggestion: str,
        adjacency_matrix: List[List[float]],
        acyclicity_score: float,
    ):
        self.root_cause_index = root_cause_index
        self.root_cause_variable = root_cause_variable
        self.confidence = confidence
        self.suggestion = suggestion
        self.adjacency_matrix = adjacency_matrix
        self.acyclicity_score = acyclicity_score

    def __repr__(self) -> str:
        var = self.root_cause_variable or f"index:{self.root_cause_index}"
        return (
            f"CausalAnalysisResult(root_cause={var}, "
            f"confidence={self.confidence:.2f}, "
            f"acyclicity={self.acyclicity_score:.2e})"
        )


# Convenience functions
def analyze_command_failure(
    command: str,
    error: str,
    context: Dict[str, float],
) -> CausalAnalysisResult:
    """Analyze why a command failed.

    Args:
        command: The command that failed
        error: The error message
        context: Context variables (e.g., {"file_exists": 0.0, "permissions": 1.0})

    Returns:
        CausalAnalysisResult with root cause analysis
    """
    if not DAGMA_AVAILABLE:
        raise ImportError("DAGMA not available")

    analyzer = CausalAnalyzer(mode="fast")
    analyzer.set_variable_names(list(context.keys()))

    # Single observation
    observations = [list(context.values())]
    return analyzer.analyze(observations)
