from openhands.security.analyzer import SecurityAnalyzer
from openhands.security.grayswan.analyzer import GraySwanAnalyzer
from openhands.security.invariant.analyzer import InvariantAnalyzer
from openhands.security.llm.analyzer import LLMRiskAnalyzer

# Optional: Pandora analyzer (requires openhands_agolos Rust extension)
try:
    from openhands.security.pandora.analyzer import PandoraSecurityAnalyzer
    PANDORA_AVAILABLE = True
except ImportError:
    PANDORA_AVAILABLE = False
    PandoraSecurityAnalyzer = None  # type: ignore

SecurityAnalyzers: dict[str, type[SecurityAnalyzer]] = {
    'invariant': InvariantAnalyzer,
    'llm': LLMRiskAnalyzer,
    'grayswan': GraySwanAnalyzer,
}

# Add Pandora if available
if PANDORA_AVAILABLE and PandoraSecurityAnalyzer is not None:
    SecurityAnalyzers['pandora'] = PandoraSecurityAnalyzer
