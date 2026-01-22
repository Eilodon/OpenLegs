#!/usr/bin/env python3
"""Performance benchmark for cognitive enhancement modules.

Verifies that cognitive operations meet the ≤30% overhead target.
"""

import time


def benchmark_holographic_memory() -> dict[str, float]:
    """Benchmark holographic memory operations."""
    try:
        from openhands.memory.holographic_memory import (
            HOLOGRAPHIC_AVAILABLE,
            HolographicMemory,
        )

        if not HOLOGRAPHIC_AVAILABLE:
            return {'error': 'HolographicMemory not available'}
    except ImportError:
        return {'error': 'Import failed'}

    results = {}
    dim = 256
    n_items = 1000

    # Create memory
    start = time.perf_counter()
    memory = HolographicMemory(dim=dim, max_items=10000)
    results['create_ms'] = (time.perf_counter() - start) * 1000

    # Benchmark entangle (store)
    contexts = [
        f'Error context {i}: file not found in directory {i}' for i in range(n_items)
    ]
    solutions = [
        f'Solution {i}: create file using touch command' for i in range(n_items)
    ]

    start = time.perf_counter()
    for ctx, sol in zip(contexts, solutions):
        memory.store_experience(ctx, sol)
    entangle_time = (time.perf_counter() - start) * 1000
    results['entangle_total_ms'] = entangle_time
    results['entangle_per_item_ms'] = entangle_time / n_items

    # Benchmark recall
    n_recalls = 100
    queries = [f'ModuleNotFoundError in test {i}' for i in range(n_recalls)]

    start = time.perf_counter()
    for query in queries:
        memory.recall_similar(query, threshold=0.3)
    recall_time = (time.perf_counter() - start) * 1000
    results['recall_total_ms'] = recall_time
    results['recall_per_query_ms'] = recall_time / n_recalls

    results['items_stored'] = memory.item_count()
    results['energy'] = memory.energy()

    return results


def benchmark_causal_analyzer() -> dict[str, float]:
    """Benchmark causal analyzer operations."""
    try:
        from openhands.memory.causal_analyzer import DAGMA_AVAILABLE, CausalAnalyzer

        if not DAGMA_AVAILABLE:
            return {'error': 'CausalAnalyzer not available'}
    except ImportError:
        return {'error': 'Import failed'}

    results = {}
    n_vars = 5
    n_samples = 100

    # Create analyzer
    start = time.perf_counter()
    analyzer = CausalAnalyzer(mode='fast')
    results['create_ms'] = (time.perf_counter() - start) * 1000

    # Create observations
    import random

    observations = [[random.random() for _ in range(n_vars)] for _ in range(n_samples)]
    analyzer.set_variable_names(['var1', 'var2', 'var3', 'var4', 'var5'])

    # Benchmark analyze
    start = time.perf_counter()
    result = analyzer.analyze(observations)
    results['analyze_ms'] = (time.perf_counter() - start) * 1000
    results['confidence'] = result.confidence

    # Benchmark intervention prediction
    start = time.perf_counter()
    for _ in range(10):
        analyzer.predict_intervention(observations, 0, 1.0)
    intervention_time = (time.perf_counter() - start) * 1000
    results['intervention_total_ms'] = intervention_time
    results['intervention_per_call_ms'] = intervention_time / 10

    return results


def benchmark_cognitive_mixin() -> dict[str, float]:
    """Benchmark the cognitive enhancement mixin."""
    try:
        from openhands.controller.cognitive_enhancement import (
            COGNITIVE_AVAILABLE,
            CognitiveEnhancementMixin,
        )

        if not COGNITIVE_AVAILABLE:
            return {'error': 'CognitiveEnhancement not available'}
    except ImportError:
        return {'error': 'Import failed'}

    results = {}

    # Create mixin
    start = time.perf_counter()
    mixin = CognitiveEnhancementMixin(
        enable_memory=True,
        enable_causal=True,
        memory_dim=256,
    )
    results['create_ms'] = (time.perf_counter() - start) * 1000

    # Simulate action cycle
    n_cycles = 100

    class MockAction:
        pass

    class MockObservation:
        pass

    action = MockAction()
    observation = MockObservation()

    total_pre_ms = 0
    total_post_ms = 0

    for i in range(n_cycles):
        context = f'Error {i}: command failed'
        solution = f'Solution {i}: fix command'

        # Pre-action hook
        start = time.perf_counter()
        mixin.on_action_start(action, context)
        total_pre_ms += (time.perf_counter() - start) * 1000

        # Post-action hook (success)
        start = time.perf_counter()
        mixin.on_action_success(action, observation, context, solution)
        total_post_ms += (time.perf_counter() - start) * 1000

    results['pre_action_total_ms'] = total_pre_ms
    results['pre_action_per_call_ms'] = total_pre_ms / n_cycles
    results['post_action_total_ms'] = total_post_ms
    results['post_action_per_call_ms'] = total_post_ms / n_cycles

    stats = mixin.get_performance_stats()
    results['mixin_overhead_ms'] = stats['total_overhead_ms']
    results['mixin_avg_per_step_ms'] = stats['avg_overhead_per_step_ms']

    return results


def run_benchmarks() -> None:
    """Run all benchmarks and report results."""
    print('=' * 60)
    print('OpenHands Cognitive Enhancement Benchmarks')
    print('=' * 60)
    print()

    # Holographic Memory
    print('Holographic Memory Benchmark')
    print('-' * 40)
    holo_results = benchmark_holographic_memory()
    if 'error' in holo_results:
        print(f'  SKIPPED: {holo_results["error"]}')
    else:
        print(f'  Create:           {holo_results["create_ms"]:.2f} ms')
        print(f'  Entangle (1000x): {holo_results["entangle_total_ms"]:.2f} ms')
        print(f'  Entangle/item:    {holo_results["entangle_per_item_ms"]:.3f} ms')
        print(f'  Recall (100x):    {holo_results["recall_total_ms"]:.2f} ms')
        print(f'  Recall/query:     {holo_results["recall_per_query_ms"]:.3f} ms')
        print(f'  Items stored:     {holo_results["items_stored"]}')
    print()

    # Causal Analyzer
    print('Causal Analyzer Benchmark')
    print('-' * 40)
    causal_results = benchmark_causal_analyzer()
    if 'error' in causal_results:
        print(f'  SKIPPED: {causal_results["error"]}')
    else:
        print(f'  Create:           {causal_results["create_ms"]:.2f} ms')
        print(f'  Analyze (5 vars): {causal_results["analyze_ms"]:.2f} ms')
        print(
            f'  Intervention/call: {causal_results["intervention_per_call_ms"]:.3f} ms'
        )
    print()

    # Cognitive Mixin
    print('Cognitive Enhancement Mixin Benchmark')
    print('-' * 40)
    mixin_results = benchmark_cognitive_mixin()
    if 'error' in mixin_results:
        print(f'  SKIPPED: {mixin_results["error"]}')
    else:
        print(f'  Create:           {mixin_results["create_ms"]:.2f} ms')
        print(f'  Pre-action/call:  {mixin_results["pre_action_per_call_ms"]:.3f} ms')
        print(f'  Post-action/call: {mixin_results["post_action_per_call_ms"]:.3f} ms')
        print(f'  Total overhead:   {mixin_results["mixin_overhead_ms"]:.2f} ms')
        print(f'  Avg/step:         {mixin_results["mixin_avg_per_step_ms"]:.3f} ms')
    print()

    # Summary
    print('=' * 60)
    print('Performance Summary')
    print('=' * 60)

    baseline_step_ms = 100  # Typical LLM call is 100-2000ms
    overhead_target_pct = 30

    if 'error' not in mixin_results:
        total_overhead = (
            mixin_results['pre_action_per_call_ms']
            + mixin_results['post_action_per_call_ms']
        )
        overhead_pct = (total_overhead / baseline_step_ms) * 100
        status = '✅ PASS' if overhead_pct <= overhead_target_pct else '❌ FAIL'

        print(f'  Baseline step time: {baseline_step_ms} ms (LLM call)')
        print(f'  Cognitive overhead: {total_overhead:.2f} ms/step')
        print(f'  Overhead percentage: {overhead_pct:.1f}%')
        print(f'  Target: ≤{overhead_target_pct}%')
        print(f'  Status: {status}')
    else:
        print('  Unable to measure - modules not available')

    print()


if __name__ == '__main__':
    run_benchmarks()
