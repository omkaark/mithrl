# Environments

Environments own task generation and reward production.

They:
- generate prompts or examples
- evaluate responses
- emit rewards and metadata used during training

They do not:
- implement algorithm loss logic
- compute training objectives
- own optimizer-side behavior

Config entry point:
- `env.factory`
- `env.kwargs`

Factory:
- `env.factory` should point to a class, e.g. `environments.simple_math:SimpleMathEnvironmentFactory`.
- The class should subclass `EnvironmentFactory` and implement `create(rollout_idx)`.

Example:
- `simple_math` samples arithmetic tasks and emits `group_id` metadata for grouped training.
