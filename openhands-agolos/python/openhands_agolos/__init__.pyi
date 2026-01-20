"""OpenHands-AGOLOS: Safety primitives for OpenHands

Type stubs for the Rust extension.
"""

from typing import List, Optional, Tuple

class PySafetyViolation:
    """Safety violation from LTL monitor"""
    property_name: str
    description: str

class PyLtlMonitor:
    """LTL Safety Monitor for bash commands"""

    def __init__(self) -> None: ...

    def check_command(self, command: str) -> List[PySafetyViolation]:
        """Check command against safety properties. Returns violations."""
        ...

    def add_pattern_rule(self, name: str, description: str, blocked_pattern: str) -> None:
        """Add a custom pattern-based blocking rule."""
        ...

class PyTraumaHit:
    """Trauma record from registry"""
    severity: int  # 1=Light, 2=Medium, 3=Severe
    count: int
    is_expired: bool
    inhibit_until_ts: int

class PyTraumaRegistry:
    """3-Tier Trauma Registry with SQLite persistence"""

    def __init__(self, db_path: str) -> None: ...

    def record_failure(
        self,
        context_hash: bytes,
        action_type: str,
        severity: int,
        decay_hours: int
    ) -> None:
        """Record a failure. severity: 1=Light, 2=Medium, 3=Severe"""
        ...

    def query(self, context_hash: bytes) -> Optional[PyTraumaHit]:
        """Query trauma for a context hash."""
        ...

    @staticmethod
    def compute_context_hash(command: str, working_dir: str, project_id: str) -> bytes:
        """Compute context hash from action details."""
        ...

class PyCausalAnalysis:
    """Causal analysis result"""
    root_cause_index: int
    confidence: float
    suggestion: str

class PyCausalDebugger:
    """DAGMA-based causal debugger"""

    def __init__(self) -> None: ...

    def analyze(self, features: List[List[float]]) -> PyCausalAnalysis:
        """Analyze features to find root cause."""
        ...
