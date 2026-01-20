"""Circuit Breaker wrapper for OpenHands Runtime.

Provides Python-friendly interface to the Rust-based circuit breaker.
"""
import logging
from typing import Optional

try:
    from openhands_agolos import PyCircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    PyCircuitBreaker = None  # type: ignore

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker for runtime action execution.

    Prevents cascading failures by tracking operation health and
    temporarily blocking operations that are failing repeatedly.

    Example:
        cb = CircuitBreaker()

        if not cb.is_open("cmd_run"):
            try:
                result = execute_command()
                cb.record_success("cmd_run")
            except Exception:
                cb.record_failure("cmd_run")
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_secs: int = 30,
        half_open_permits: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before circuit opens
            recovery_timeout_secs: Seconds to wait before attempting recovery
            half_open_permits: Number of trial requests in half-open state
        """
        if not CIRCUIT_BREAKER_AVAILABLE:
            logger.warning(
                "openhands_agolos not available. Circuit breaker disabled. "
                "Build with: cd openhands-agolos && maturin develop"
            )
            self._inner = None
            return

        self._inner = PyCircuitBreaker.with_config(
            failure_threshold=failure_threshold,
            recovery_timeout_secs=recovery_timeout_secs,
            half_open_permits=half_open_permits,
        )
        logger.info(
            f"Circuit breaker initialized: threshold={failure_threshold}, "
            f"timeout={recovery_timeout_secs}s"
        )

    def is_open(self, operation: str) -> bool:
        """Check if circuit is open for an operation.

        Args:
            operation: Operation name to check

        Returns:
            True if circuit is open (requests should be rejected)
        """
        if self._inner is None:
            return False
        return self._inner.is_open(operation)

    def record_success(self, operation: str) -> None:
        """Record successful operation execution.

        Args:
            operation: Operation name
        """
        if self._inner is not None:
            self._inner.record_success(operation)

    def record_failure(self, operation: str) -> None:
        """Record failed operation execution.

        Args:
            operation: Operation name
        """
        if self._inner is not None:
            self._inner.record_failure(operation)

    def get_state(self, operation: str) -> Optional[str]:
        """Get circuit state for an operation.

        Args:
            operation: Operation name

        Returns:
            State string (e.g., "closed(failures=2)", "open", "half_open(permits=1)")
            or None if not tracked
        """
        if self._inner is None:
            return None
        return self._inner.get_state(operation)

    def stats(self) -> dict:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with keys: total_operations, open_circuits,
            closed_circuits, half_open_circuits
        """
        if self._inner is None:
            return {
                "total_operations": 0,
                "open_circuits": 0,
                "closed_circuits": 0,
                "half_open_circuits": 0,
            }

        stats = self._inner.stats()
        return {
            "total_operations": stats.total_operations,
            "open_circuits": stats.open_circuits,
            "closed_circuits": stats.closed_circuits,
            "half_open_circuits": stats.half_open_circuits,
        }
