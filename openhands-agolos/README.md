# OpenHands-AGOLOS

AGOLOS safety primitives for OpenHands, providing mathematically-grounded safety, causal debugging, and trauma-based memory.

## Features

### LTL Safety Monitor
Mathematically-provable command blocking based on Linear Temporal Logic properties:
- Blocks `rm -rf /`, `DROP TABLE`, `sudo`, fork bombs, etc.
- Returns confirmation request instead of hard block
- Extensible with custom pattern rules

### 3-Tier Trauma Registry
Persistent memory of agent failures with forgetting:
| Tier | Scope | Decay | Example |
|------|-------|-------|---------|
| Light | Project | 24h | Test failures |
| Medium | Global | 7d | Build breaks |
| Severe | Global | 30d+ | Data loss |

### Causal Debugger
DAGMA-based root cause analysis for error logs.

## Installation

```bash
# Development build
cd openhands-agolos
maturin develop

# Release build
maturin build --release
```

## Usage

```python
import openhands_agolos as agolos

# LTL Monitor
monitor = agolos.PyLtlMonitor()
violations = monitor.check_command("rm -rf /")
if violations:
    print(f"Command blocked: {violations[0].description}")

# Trauma Registry
registry = agolos.PyTraumaRegistry("./trauma.db")
context_hash = agolos.PyTraumaRegistry.compute_context_hash("rm -rf", "/home", "project1")
registry.record_failure(context_hash, "rm", 3, 720)  # Severe, 30 days

# Causal Debugger
debugger = agolos.PyCausalDebugger()
result = debugger.analyze(features)
print(f"Root cause: {result.root_cause_index}")
```
