# Algorithms

Algorithms own training logic.

They:
- validate algorithm-specific kwargs
- compute advantages or other training targets
- compute loss terms and training metrics

They do not:
- generate tasks
- score environment responses
- implement environment-specific reward logic

Config entry point:
- `algo.factory`
- `algo.kwargs`

Factory:
- `algo.factory` should point to a class, e.g. `src.algo.grpo:GRPO`.
- The class should subclass `Algorithm` and accept `config` plus `**kwargs`.

Example:
- `GRPO` validates grouped-rollout params and computes grouped normalized advantages.
