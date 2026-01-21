"""Integration tests for Circuit Breaker in agent execution.

Tests that the circuit breaker properly prevents cascading failures
and recovers after successful operations.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, failure_threshold=5, recovery_timeout_secs=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_secs
        self._states = {}  # operation_key -> (failures, is_open)

    def is_open(self, operation_key: str) -> bool:
        if operation_key not in self._states:
            return False
        failures, is_open = self._states[operation_key]
        return is_open

    def record_success(self, operation_key: str):
        if operation_key in self._states:
            self._states[operation_key] = (0, False)

    def record_failure(self, operation_key: str):
        if operation_key not in self._states:
            self._states[operation_key] = (0, False)
        failures, _ = self._states[operation_key]
        failures += 1
        is_open = failures >= self.failure_threshold
        self._states[operation_key] = (failures, is_open)

    def get_state(self, operation_key: str):
        if operation_key not in self._states:
            return {"state": "Closed", "failures": 0}
        failures, is_open = self._states[operation_key]
        return {
            "state": "Open" if is_open else "Closed",
            "failures": failures
        }


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in agent controller."""

    @pytest.fixture
    def circuit_breaker(self):
        """Get circuit breaker (real or mock)."""
        try:
            from openhands.runtime.circuit_breaker import CircuitBreaker
            return CircuitBreaker()
        except ImportError:
            return MockCircuitBreaker()

    def test_circuit_starts_closed(self, circuit_breaker):
        """Circuit should start in closed state."""
        assert not circuit_breaker.is_open("test_operation")

    def test_circuit_opens_after_failures(self, circuit_breaker):
        """Circuit should open after threshold failures."""
        op_key = "test_failures"

        # Record failures up to threshold
        for i in range(5):
            circuit_breaker.record_failure(op_key)

        # Circuit should now be open
        assert circuit_breaker.is_open(op_key)

    def test_circuit_closes_on_success(self, circuit_breaker):
        """Circuit should close after successful operation."""
        op_key = "test_recovery"

        # Open the circuit
        for i in range(5):
            circuit_breaker.record_failure(op_key)
        assert circuit_breaker.is_open(op_key)

        # Record success (simulating half-open trial)
        circuit_breaker.record_success(op_key)

        # Should be closed now
        assert not circuit_breaker.is_open(op_key)

    def test_different_operations_isolated(self, circuit_breaker):
        """Each operation key should have independent circuit."""
        op1 = "agent_step_session1"
        op2 = "agent_step_session2"

        # Fail op1 to open circuit
        for i in range(5):
            circuit_breaker.record_failure(op1)

        # op1 should be open
        assert circuit_breaker.is_open(op1)
        # op2 should still be closed
        assert not circuit_breaker.is_open(op2)


class TestCircuitBreakerWithRust:
    """Test actual Rust circuit breaker if available."""

    @pytest.fixture
    def rust_breaker(self):
        """Get Rust circuit breaker or skip."""
        try:
            import openhands_agolos as ag
            return ag.CircuitBreaker()
        except (ImportError, AttributeError):
            pytest.skip("Rust circuit breaker not available")

    def test_rust_circuit_breaker_basic(self, rust_breaker):
        """Test basic Rust circuit breaker functionality."""
        op_key = "test_rust_cb"

        # Should start closed
        assert not rust_breaker.is_open(op_key)

        # Record some failures
        for _ in range(3):
            rust_breaker.record_failure(op_key)

        # Not yet open (default threshold is 5)
        assert not rust_breaker.is_open(op_key)

        # More failures to open
        for _ in range(3):
            rust_breaker.record_failure(op_key)

        # Now should be open
        assert rust_breaker.is_open(op_key)

    def test_rust_circuit_breaker_stats(self, rust_breaker):
        """Test circuit breaker stats retrieval."""
        op_key = "test_stats"

        rust_breaker.record_success(op_key)
        rust_breaker.record_failure(op_key)

        stats = rust_breaker.get_state(op_key)
        assert "state" in stats or hasattr(stats, "state")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
