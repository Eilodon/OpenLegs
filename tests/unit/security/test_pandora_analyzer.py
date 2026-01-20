"""Unit tests for Pandora Security Analyzer."""

import pytest
from unittest.mock import MagicMock, patch

from openhands.events.action import CmdRunAction
from openhands.events.action.action import ActionSecurityRisk, ActionConfirmationStatus


# Mock the openhands_agolos module for testing without Rust extension
class MockViolation:
    def __init__(self, property_name: str, description: str):
        self.property_name = property_name
        self.description = description


class MockLtlMonitor:
    def check_command(self, command: str):
        dangerous_patterns = [
            ("rm -rf /", "no_rm_rf_critical", "rm -rf must not target /"),
            ("DROP TABLE", "no_drop_statements", "DROP TABLE requires confirmation"),
            ("sudo ", "no_privilege_escalation", "sudo commands require approval"),
            ("curl", "no_remote_code_execution", "Remote scripts blocked"),
        ]

        violations = []
        for pattern, name, desc in dangerous_patterns:
            if pattern in command.upper() or pattern in command:
                # Special check for curl - only if piped to shell
                if pattern == "curl" and not any(x in command for x in ["| sh", "|sh", "| bash", "|bash"]):
                    continue
                violations.append(MockViolation(name, desc))

        return violations

    def add_pattern_rule(self, name: str, description: str, pattern: str):
        pass


class MockTraumaHit:
    def __init__(self, severity: int, count: int, is_expired: bool):
        self.severity = severity
        self.count = count
        self.is_expired = is_expired


class MockTraumaRegistry:
    def __init__(self, db_path: str):
        self.records = {}

    def query(self, context_hash):
        return self.records.get(tuple(context_hash))

    def record_failure(self, context_hash, action_type, severity, decay_hours):
        self.records[tuple(context_hash)] = MockTraumaHit(severity, 1, False)

    @staticmethod
    def compute_context_hash(command: str, cwd: str, project_id: str):
        import hashlib
        h = hashlib.blake2b(f"{command}||{cwd}||{project_id}".encode(), digest_size=32)
        return h.digest()


@pytest.fixture
def mock_agolos():
    """Mock the openhands_agolos module."""
    mock_module = MagicMock()
    mock_module.PyLtlMonitor = MockLtlMonitor
    mock_module.PyTraumaRegistry = MockTraumaRegistry

    with patch.dict('sys.modules', {'openhands_agolos': mock_module}):
        yield mock_module


@pytest.mark.asyncio
async def test_ltl_blocks_rm_rf(mock_agolos):
    """LTL monitor should flag rm -rf / as HIGH risk."""
    from openhands.security.pandora.analyzer import PandoraSecurityAnalyzer

    with patch('openhands.security.pandora.analyzer.openhands_agolos', mock_agolos):
        with patch('openhands.security.pandora.analyzer.AGOLOS_AVAILABLE', True):
            analyzer = PandoraSecurityAnalyzer.__new__(PandoraSecurityAnalyzer)
            analyzer.ltl_monitor = MockLtlMonitor()
            analyzer.trauma_registry = MockTraumaRegistry("/tmp/test.db")
            analyzer.project_id = "test"

            action = CmdRunAction(command="rm -rf /")
            risk = await analyzer.security_risk(action)

            assert risk == ActionSecurityRisk.HIGH
            assert action.confirmation_state == ActionConfirmationStatus.AWAITING_CONFIRMATION


@pytest.mark.asyncio
async def test_ltl_allows_safe_rm(mock_agolos):
    """LTL monitor should allow rm -rf on safe paths."""
    from openhands.security.pandora.analyzer import PandoraSecurityAnalyzer

    with patch('openhands.security.pandora.analyzer.openhands_agolos', mock_agolos):
        with patch('openhands.security.pandora.analyzer.AGOLOS_AVAILABLE', True):
            analyzer = PandoraSecurityAnalyzer.__new__(PandoraSecurityAnalyzer)
            analyzer.ltl_monitor = MockLtlMonitor()
            analyzer.trauma_registry = MockTraumaRegistry("/tmp/test.db")
            analyzer.project_id = "test"

            action = CmdRunAction(command="rm -rf ./build")
            risk = await analyzer.security_risk(action)

            assert risk == ActionSecurityRisk.LOW


@pytest.mark.asyncio
async def test_sudo_blocked(mock_agolos):
    """sudo commands should be flagged."""
    from openhands.security.pandora.analyzer import PandoraSecurityAnalyzer

    with patch('openhands.security.pandora.analyzer.openhands_agolos', mock_agolos):
        with patch('openhands.security.pandora.analyzer.AGOLOS_AVAILABLE', True):
            analyzer = PandoraSecurityAnalyzer.__new__(PandoraSecurityAnalyzer)
            analyzer.ltl_monitor = MockLtlMonitor()
            analyzer.trauma_registry = MockTraumaRegistry("/tmp/test.db")
            analyzer.project_id = "test"

            action = CmdRunAction(command="sudo apt install package")
            risk = await analyzer.security_risk(action)

            assert risk == ActionSecurityRisk.HIGH


@pytest.mark.asyncio
async def test_safe_curl_allowed(mock_agolos):
    """curl without pipe to shell should be allowed."""
    from openhands.security.pandora.analyzer import PandoraSecurityAnalyzer

    with patch('openhands.security.pandora.analyzer.openhands_agolos', mock_agolos):
        with patch('openhands.security.pandora.analyzer.AGOLOS_AVAILABLE', True):
            analyzer = PandoraSecurityAnalyzer.__new__(PandoraSecurityAnalyzer)
            analyzer.ltl_monitor = MockLtlMonitor()
            analyzer.trauma_registry = MockTraumaRegistry("/tmp/test.db")
            analyzer.project_id = "test"

            action = CmdRunAction(command="curl https://api.example.com/data")
            risk = await analyzer.security_risk(action)

            assert risk == ActionSecurityRisk.LOW
