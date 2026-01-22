"""Experience Learning Module for OpenHands agents.

Provides Python wrappers around Rust ExperienceBuffer and PatternMiner
for learning from past actions and discovering successful action patterns.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Rust bindings
try:
    import openhands_agolos as ag

    LEARNING_AVAILABLE = hasattr(ag, 'PyExperienceBuffer') and hasattr(
        ag, 'PyPatternMiner'
    )
except ImportError:
    LEARNING_AVAILABLE = False
    ag = None


class ExperienceBuffer:
    """Circular buffer for storing action experiences.

    Tracks action outcomes and provides success rate statistics
    for intelligent decision making.

    Usage:
        buffer = ExperienceBuffer(max_size=1000)
        buffer.push("CmdRunAction", "ls -la", success=True, reward=1.0)
        rate = buffer.get_success_rate("CmdRunAction")
    """

    def __init__(self, max_size: int = 1000):
        """Initialize experience buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self._buffer = None

        if LEARNING_AVAILABLE and ag is not None:
            try:
                self._buffer = ag.PyExperienceBuffer(max_size)
                logger.info(
                    f'ExperienceBuffer: Rust backend initialized (max_size={max_size})'
                )
            except Exception as e:
                logger.warning(f'ExperienceBuffer: Failed to init Rust backend: {e}')
        else:
            logger.warning(
                'ExperienceBuffer: Rust backend not available, using fallback'
            )
            self._fallback_buffer: list[dict] = []
            self._max_size = max_size

    def push(
        self,
        action_type: str,
        action_signature: str,
        success: bool,
        reward: float = 0.0,
    ) -> None:
        """Push a new experience to the buffer.

        Args:
            action_type: Type of action (e.g., "CmdRunAction")
            action_signature: Signature/identifier for the specific action
            success: Whether the action succeeded
            reward: Reward value for the action (optional)
        """
        if self._buffer is not None:
            self._buffer.push(action_type, action_signature, success, reward)
        else:
            # Fallback implementation
            if len(self._fallback_buffer) >= self._max_size:
                self._fallback_buffer.pop(0)
            self._fallback_buffer.append(
                {
                    'action_type': action_type,
                    'signature': action_signature,
                    'success': success,
                    'reward': reward,
                }
            )

    def get_success_rate(self, action_type: str) -> Optional[float]:
        """Get success rate for a specific action type.

        Args:
            action_type: Type of action to query

        Returns:
            Success rate (0.0-1.0) or None if no data
        """
        if self._buffer is not None:
            return self._buffer.success_rate(action_type)
        else:
            # Fallback implementation
            matching = [
                e for e in self._fallback_buffer if e['action_type'] == action_type
            ]
            if not matching:
                return None
            success_count = sum(1 for e in matching if e['success'])
            return success_count / len(matching)

    def __len__(self) -> int:
        if self._buffer is not None:
            return self._buffer.len()
        return len(self._fallback_buffer)

    def is_empty(self) -> bool:
        if self._buffer is not None:
            return self._buffer.is_empty()
        return len(self._fallback_buffer) == 0


class PatternMiner:
    """Pattern mining for discovering successful action sequences.

    Analyzes action history to find patterns that lead to success,
    and suggests next actions based on current sequence.

    Usage:
        miner = PatternMiner()
        miner.add_action("FileReadAction", success=True)
        miner.add_action("FileEditAction", success=True)
        suggestion = miner.suggest_next(["FileReadAction"])
    """

    def __init__(self, min_support: float = 0.1, max_pattern_length: int = 5):
        """Initialize pattern miner.

        Args:
            min_support: Minimum support threshold for pattern discovery
            max_pattern_length: Maximum length of patterns to discover
        """
        self._miner = None

        if LEARNING_AVAILABLE and ag is not None:
            try:
                self._miner = ag.PyPatternMiner.with_params(
                    min_support, max_pattern_length
                )
                logger.info('PatternMiner: Rust backend initialized')
            except Exception as e:
                logger.warning(f'PatternMiner: Failed to init Rust backend: {e}')
        else:
            logger.warning('PatternMiner: Rust backend not available, using fallback')
            self._fallback_actions: list[dict] = []
            self._max_pattern_length = max_pattern_length

    def add_action(self, action_type: str, success: bool, reward: float = 0.0) -> None:
        """Add an action to the sequence buffer for pattern mining.

        Args:
            action_type: Type of action
            success: Whether action succeeded
            reward: Reward value
        """
        if self._miner is not None:
            self._miner.add_action(action_type, success, reward)
        else:
            self._fallback_actions.append(
                {
                    'type': action_type,
                    'success': success,
                    'reward': reward,
                }
            )

    def suggest_next(self, recent_actions: list[str]) -> Optional[str]:
        """Suggest next action based on current sequence.

        Args:
            recent_actions: List of recent action types

        Returns:
            Suggested next action type, or None if no suggestion
        """
        if self._miner is not None:
            return self._miner.suggest_next(recent_actions)
        else:
            # Simple fallback: return most common successful action
            successful = [a['type'] for a in self._fallback_actions if a['success']]
            if not successful:
                return None
            from collections import Counter

            most_common = Counter(successful).most_common(1)
            return most_common[0][0] if most_common else None

    def pattern_count(self) -> int:
        """Get number of discovered patterns."""
        if self._miner is not None:
            return self._miner.pattern_count()
        return 0


# Module-level instances for singleton access
_experience_buffer: Optional[ExperienceBuffer] = None
_pattern_miner: Optional[PatternMiner] = None


def get_experience_buffer(max_size: int = 1000) -> ExperienceBuffer:
    """Get or create the global experience buffer."""
    global _experience_buffer
    if _experience_buffer is None:
        _experience_buffer = ExperienceBuffer(max_size)
    return _experience_buffer


def get_pattern_miner() -> PatternMiner:
    """Get or create the global pattern miner."""
    global _pattern_miner
    if _pattern_miner is None:
        _pattern_miner = PatternMiner()
    return _pattern_miner
