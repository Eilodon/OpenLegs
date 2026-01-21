"""Causal Feature Extractor for OpenHands agents.

Converts event history into a DAGMA-compatible feature matrix for
causal structure learning and root cause analysis.

Features extracted per event:
  - action_type: Encoded action type (CmdRun=1, FileEdit=2, etc.)
  - is_error: 1.0 if ErrorObservation, 0.0 otherwise
  - error_category: Encoded error type (permission=1, not_found=2, syntax=3, etc.)
  - action_success: 1.0 if action succeeded, 0.0 otherwise
  - time_delta: Normalized time since last event (0-1 range)
  - repetition_score: Similarity to recent actions (detect loops)
"""

from typing import Dict, List, Optional, Tuple
import hashlib
import time

from openhands.events.event import Event
from openhands.events.action import (
    Action,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    BrowseInteractiveAction,
    MessageAction,
    AgentFinishAction,
)
from openhands.events.observation import (
    Observation,
    ErrorObservation,
    CmdOutputObservation,
)

# Feature names for interpretability
FEATURE_NAMES = [
    "action_type",
    "is_error",
    "error_category",
    "action_success",
    "time_delta",
    "repetition_score",
]

# Action type encoding
ACTION_TYPE_MAP = {
    "CmdRunAction": 1.0,
    "FileEditAction": 2.0,
    "FileReadAction": 3.0,
    "IPythonRunCellAction": 4.0,
    "BrowseInteractiveAction": 5.0,
    "MessageAction": 6.0,
    "AgentFinishAction": 7.0,
    "AgentDelegateAction": 8.0,
    "ChangeAgentStateAction": 9.0,
    # Add more as needed
}

# Error category encoding
ERROR_CATEGORY_MAP = {
    "permission": 1.0,
    "not_found": 2.0,
    "file_not_found": 2.0,
    "syntax": 3.0,
    "timeout": 4.0,
    "connection": 5.0,
    "authentication": 6.0,
    "rate_limit": 7.0,
    "context_window": 8.0,
    "unknown": 0.0,
}


def _classify_error(error_content: str) -> float:
    """Classify error message into a category."""
    error_lower = error_content.lower()

    if "permission" in error_lower or "denied" in error_lower:
        return ERROR_CATEGORY_MAP["permission"]
    elif "not found" in error_lower or "no such file" in error_lower:
        return ERROR_CATEGORY_MAP["not_found"]
    elif "syntax" in error_lower or "invalid" in error_lower:
        return ERROR_CATEGORY_MAP["syntax"]
    elif "timeout" in error_lower or "timed out" in error_lower:
        return ERROR_CATEGORY_MAP["timeout"]
    elif "connection" in error_lower or "network" in error_lower:
        return ERROR_CATEGORY_MAP["connection"]
    elif "auth" in error_lower or "credential" in error_lower:
        return ERROR_CATEGORY_MAP["authentication"]
    elif "rate" in error_lower or "limit" in error_lower:
        return ERROR_CATEGORY_MAP["rate_limit"]
    elif "context" in error_lower or "token" in error_lower:
        return ERROR_CATEGORY_MAP["context_window"]
    else:
        return ERROR_CATEGORY_MAP["unknown"]


def _compute_action_hash(action: Action) -> str:
    """Compute a hash of the action for repetition detection."""
    if isinstance(action, CmdRunAction):
        return hashlib.md5(action.command.encode()).hexdigest()[:8]
    elif isinstance(action, FileEditAction):
        return hashlib.md5(f"{action.path}:{action.content[:100]}".encode()).hexdigest()[:8]
    elif isinstance(action, FileReadAction):
        return hashlib.md5(action.path.encode()).hexdigest()[:8]
    elif isinstance(action, IPythonRunCellAction):
        return hashlib.md5(action.code.encode()).hexdigest()[:8]
    else:
        return hashlib.md5(type(action).__name__.encode()).hexdigest()[:8]


def _compute_repetition_score(
    current_hash: str,
    recent_hashes: List[str],
    window: int = 10
) -> float:
    """Compute how often this action has been repeated recently."""
    if not recent_hashes:
        return 0.0

    recent = recent_hashes[-window:]
    count = sum(1 for h in recent if h == current_hash)
    return min(count / window, 1.0)  # Normalize to 0-1


def extract_features(
    events: List[Event],
    max_events: int = 100
) -> Tuple[List[List[float]], List[str]]:
    """Extract feature matrix from event history.

    Args:
        events: List of events from state.history
        max_events: Maximum number of events to process

    Returns:
        Tuple of (feature_matrix, feature_names)
        feature_matrix: List of feature vectors [n_events, n_features]
        feature_names: List of feature names
    """
    if not events:
        return [], FEATURE_NAMES

    # Limit to recent events
    events = events[-max_events:]

    features: List[List[float]] = []
    action_hashes: List[str] = []
    last_timestamp = None

    for event in events:
        feature_vector = [0.0] * len(FEATURE_NAMES)

        # Feature 0: action_type
        if isinstance(event, Action):
            action_type_name = type(event).__name__
            feature_vector[0] = ACTION_TYPE_MAP.get(action_type_name, 0.0)

            # Compute hash for repetition detection
            action_hash = _compute_action_hash(event)
            action_hashes.append(action_hash)

            # Feature 5: repetition_score
            feature_vector[5] = _compute_repetition_score(action_hash, action_hashes[:-1])

        # Feature 1: is_error
        if isinstance(event, ErrorObservation):
            feature_vector[1] = 1.0
            # Feature 2: error_category
            feature_vector[2] = _classify_error(event.content)
            # Feature 3: action_success = 0 for errors
            feature_vector[3] = 0.0
        elif isinstance(event, Observation):
            feature_vector[1] = 0.0
            feature_vector[2] = 0.0
            # Feature 3: action_success = 1 for non-error observations
            feature_vector[3] = 1.0

        # Feature 4: time_delta (normalized)
        if hasattr(event, 'timestamp') and event.timestamp:
            if last_timestamp:
                delta = event.timestamp - last_timestamp
                # Normalize: 0-60 seconds maps to 0-1
                feature_vector[4] = min(delta / 60.0, 1.0)
            last_timestamp = event.timestamp

        features.append(feature_vector)

    return features, FEATURE_NAMES


def extract_current_state(events: List[Event]) -> Dict[str, float]:
    """Extract current state as a dictionary for explain_failure.

    Args:
        events: List of events from state.history

    Returns:
        Dictionary mapping feature names to current values
    """
    features, names = extract_features(events)
    if not features:
        return {name: 0.0 for name in names}

    # Return the last feature vector as current state
    return dict(zip(names, features[-1]))


def get_feature_names() -> List[str]:
    """Get the list of feature names."""
    return FEATURE_NAMES.copy()
