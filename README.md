# mithrl

"Mithrl. Lighter than feather, harder than steel." - Gandalf, maybe.

Mithrl is an RL stack written to be decently performant, while being hackable and modular. The main loop is < 500 LoC. I want to make new ideas in RL easy to implement.

## What It Does

`mithrl` runs RL against a chat model served by `vllm`, then trains a local PEFT adapter against those rollouts.

Current examples:
- [`configs/simple_math.yaml`](/workspace/mithrl/configs/simple_math.yaml)
- [`configs/gsm8k.yaml`](/workspace/mithrl/configs/gsm8k.yaml)

Training entrypoint:
```bash
python -m src.rl.train --config configs/simple_math.yaml
```

## Requirements

- Python 3.11+
- CUDA-capable GPU
- `uv`
- vllm installed

## Install

Base project dependencies:

```bash
uv sync
```

Install FlashAttention support if you want training to use `attn_implementation="flash_attention_2"`:

```bash
uv sync --extra flash_attn --no-build-isolation
```

Install `vllm` separately. It is used as an external runtime and is not declared in [`pyproject.toml`](/workspace/mithrl/pyproject.toml):

```bash
uv pip install vllm
```

If you want to use the GSM8K environment, install `datasets` as well:

```bash
uv pip install datasets
```

## Start vLLM

The trainer expects a local server at `http://localhost:8000` unless `VLLM_BASE_URL` is overridden.

Start `vllm` with LoRA loading enabled (configure as per load):

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
VLLM_SERVER_DEV_MODE=1 \
vllm serve Qwen/Qwen2.5-3B \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --enable-sleep-mode \
  --enable-lora
```

Why these flags matter here:

- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` enables runtime adapter load/unload
- `VLLM_SERVER_DEV_MODE=1` enables the nonstandard server endpoints used by this repo
- `--enable-sleep-mode` is used so training can temporarily reclaim GPU memory
- `--enable-lora` is required because the trainer hot-reloads the adapter each step

## Run RL

Simple math example:
```bash
python -m src.rl.train --config configs/simple_math.yaml
```

GSM8K example:
```bash
python -m src.rl.train --config configs/gsm8k.yaml
```

High-level loop:
1. Verify `vllm` is reachable.
2. Initialize or reload the LoRA adapter from `train.adapter_path`.
3. Generate grouped rollouts through the configured environment.
4. Query `vllm` for token logprobs from the base model.
5. Sleep the `vllm` server, move the trainable adapter onto GPU, and run GRPO updates.
6. Save the adapter and repeat.

## Config

Main config sections are defined in [`src/utils/config.py`](/workspace/mithrl/src/utils/config.py):
- `train`: model, adapter path, optimizer settings, steps, optional FlashAttention
- `rollout`: number of rollouts, parallelism, timeout, assistant token masking
- `algo`: grouping, PPO-style clip epsilon, KL coefficient
- `env`: environment factory import path plus kwargs

`train.use_flash_attn` is auto-enabled when `flash_attn` is found in packages.

## Environments

An environment factory must return instances implementing the interfaces in [`src/train/env.py`](/workspace/mithrl/src/train/env.py).

Required environment behavior:
- expose a `system_prompt`
- expose the next user query via `next_query`
- implement `step(response)` to update reward, metadata, and done state

The trainer loads the factory dynamically from `env.factory`, for example:
```yaml
env:
  factory: environments.gsm8k:GSM8KEnvironmentFactory
```

I'd read through environment examples. It should be generalizable and modular enough.

## Notes

- The LoRA adapter directory is created automatically if it does not already exist.
- `rollout.n_rollouts` must be divisible by `algo.n_groups`.
- The default examples assume Qwen chat formatting and use assistant token spans for loss masking.

## Feedback

Please direct any feedback to omkaar [at] extensible [dot] dev or make issues on our repo.
