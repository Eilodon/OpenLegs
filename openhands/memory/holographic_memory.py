"""Holographic Associative Memory for OpenHands agents.

Provides content-addressable memory using FFT-based holographic encoding.
Enables agents to recall past solutions by similarity rather than exact match.
"""

import hashlib
from typing import Optional

from openhands.core.logger import openhands_logger as logger

# Try to import the Rust extension
try:
    import openhands_agolos

    AGOLOS_AVAILABLE = True
    # Check if HolographicMemory feature is enabled
    HOLOGRAPHIC_AVAILABLE = hasattr(openhands_agolos, 'PyHolographicMemory')
except ImportError:
    AGOLOS_AVAILABLE = False
    HOLOGRAPHIC_AVAILABLE = False


class HolographicMemory:
    """Holographic Associative Memory for OpenHands agents.

    Stores experiences as interference patterns, enabling:
    - Content-addressable recall by similarity
    - Graceful degradation (partial matches still work)
    - Natural forgetting through decay

    # Usage Example:
    ```python
    memory = HolographicMemory(dim=256)

    # Store a context-solution pair
    context = "File not found error when running tests"
    solution = "Create missing __init__.py files"
    memory.store_experience(context, solution)

    # Later, recall by similar context
    query = "ModuleNotFoundError during pytest"
    result = memory.recall_similar(query, threshold=0.3)
    if result:
        similarity, solution = result
        print(f"Found solution (similarity={similarity:.2f}): {solution}")
    ```
    """

    def __init__(
        self,
        dim: int = 256,
        max_items: int = 10000,
    ):
        """Initialize holographic memory.

        Args:
            dim: Dimension of memory space (larger = more capacity, slower)
            max_items: Maximum items before automatic decay eviction
        """
        if not HOLOGRAPHIC_AVAILABLE:
            raise ImportError(
                'PyHolographicMemory is not available. '
                'Build with: cd openhands-agolos && maturin develop --features full'
            )

        self.dim = dim
        self.max_items = max_items
        self._memory = openhands_agolos.PyHolographicMemory.with_capacity(
            dim, max_items
        )
        self._text_cache: dict[str, str] = {}  # hash -> original text
        logger.debug(
            f'HolographicMemory initialized (dim={dim}, max_items={max_items})'
        )

    @classmethod
    def default_for_agent(cls) -> 'HolographicMemory':
        """Create default memory configuration for agents."""
        return cls(dim=512, max_items=10000)

    def _text_to_embedding(self, text: str) -> list[float]:
        """Convert text to embedding vector using simple hash-based method.

        For production, replace with actual embedding model (OpenAI, sentence-transformers).
        """
        # Simple hash-based embedding (deterministic, fast)
        # Each character's position contributes to the vector
        embedding = [0.0] * self.dim

        if not text:
            return embedding

        # Normalize text
        text = text.lower().strip()

        # Hash-based embedding with overlapping n-grams
        for i, char in enumerate(text):
            # Character contribution
            idx = (ord(char) * (i + 1)) % self.dim
            embedding[idx] += 1.0 / (i + 1)

            # 3-gram contribution
            if i + 3 <= len(text):
                trigram = text[i : i + 3]
                h = int(hashlib.md5(trigram.encode()).hexdigest()[:8], 16)
                embedding[h % self.dim] += 0.5

        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def store_experience(self, context: str, solution: str) -> None:
        """Store a context-solution pair in memory.

        Args:
            context: The problem context (e.g., error message, task description)
            solution: The solution that worked (e.g., command, explanation)
        """
        # Encode context as key
        key = self._text_to_embedding(context)

        # Encode solution as value (use same embedding for simplicity)
        value = self._text_to_embedding(solution)

        # Store in holographic memory
        self._memory.entangle(key, value)

        # Cache original text for retrieval
        key_hash = hashlib.md5(context.encode()).hexdigest()[:16]
        self._text_cache[key_hash] = solution

        logger.debug(f'Stored experience: {context[:50]}... -> {solution[:50]}...')

    def recall_similar(
        self,
        query: str,
        threshold: float = 0.3,
    ) -> Optional[tuple[float, str]]:
        """Recall a solution by similar context.

        Args:
            query: The context to search for
            threshold: Minimum similarity (0.0-1.0) to return a result

        Returns:
            Tuple of (similarity_score, solution) or None if no match
        """
        key = self._text_to_embedding(query)

        result = self._memory.find_similar(key, threshold)

        if result is None:
            return None

        similarity, recalled_embedding = result

        # Find closest cached text by comparing embeddings
        best_match = None
        best_sim = -1.0

        for key_hash, solution in self._text_cache.items():
            sol_embedding = self._text_to_embedding(solution)
            # Cosine similarity
            dot = sum(r * s for r, s in zip(recalled_embedding, sol_embedding))
            if dot > best_sim:
                best_sim = dot
                best_match = solution

        if best_match:
            return (similarity, best_match)

        return None

    def recall_vector(self, query: str) -> list[float]:
        """Recall raw embedding vector for a query.

        Lower-level API for advanced use cases.
        """
        key = self._text_to_embedding(query)
        return self._memory.recall(key)

    def decay(self, factor: float = 0.9) -> None:
        """Apply decay to memory (for aging/forgetting).

        Args:
            factor: Decay multiplier (0.0 = forget all, 1.0 = remember all)
        """
        self._memory.decay(factor)

    def clear(self) -> None:
        """Clear all memory."""
        self._memory.clear()
        self._text_cache.clear()
        logger.info('HolographicMemory cleared')

    def item_count(self) -> int:
        """Get number of items stored."""
        return self._memory.item_count()

    def energy(self) -> float:
        """Get current energy of memory trace."""
        return self._memory.energy()

    def capacity_status(self) -> tuple[int, int]:
        """Get capacity status (items_stored, max_items)."""
        return self._memory.capacity_status()


# Convenience function for quick usage
def create_agent_memory() -> HolographicMemory:
    """Create a HolographicMemory configured for agent use."""
    if not HOLOGRAPHIC_AVAILABLE:
        raise ImportError(
            "HolographicMemory requires openhands_agolos with 'full' feature. "
            'Build with: cd openhands-agolos && maturin develop --features full'
        )
    return HolographicMemory.default_for_agent()
