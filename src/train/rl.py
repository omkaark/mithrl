import argparse
import asyncio
import random
from collections import defaultdict

import numpy as np
import torch
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM

from ..utils import vllm
from ..utils.loaders import load_algorithm
from ..utils.config import MithrlConfig
from ..utils.torch_utils import move_opt_to_device, pad_2d
from .rollout import RolloutSample, run_rollouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config")
    args = parser.parse_args()
    config = MithrlConfig.from_yaml(args.config)
    algorithm = load_algorithm(config)

    wandb.init(
        mode="online" if config.train.use_wandb else "disabled",
        project=config.train.wandb_project_name,
        name=config.train.wandb_run_name,
        config=config.to_dict(),
    )

    # Determinism
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Check to make sure vLLM server is running
    vllm._ping()
    vllm._sleep()  # Put to sleep to allow for space while loading base model and adapter

    # Setup
    init_model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
    }
    if config.train.use_flash_attn:
        init_model_kwargs["attn_implementation"] = "flash_attention_2"

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=(  # TODO: needs to be part of config
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )

    # Init adapter if not found
    if not vllm.adapter_exists(config.train.adapter_path):
        print("No policy adapter found on disk. Initializing default LoRA adapter at path.")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.train.model_name, **init_model_kwargs
        )
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        adapter_model = get_peft_model(base_model, lora_config)
        adapter_model.save_pretrained(config.train.adapter_path)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.train.model_name, **init_model_kwargs
        )
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        adapter_model = PeftModel.from_pretrained(
            base_model,
            config.train.adapter_path,
            is_trainable=True,
        )
    adapter_model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        (p for p in adapter_model.parameters() if p.requires_grad),
        lr=config.train.lr,
        fused=True,
    )
    adapter_model.to("cpu")
    move_opt_to_device(optimizer, "cpu")
    torch.cuda.empty_cache()

    # Start training
    for step_idx in range(config.train.n_steps):
        # Load adapter to vLLM
        vllm._reload_with_lora("adapter", config.train.adapter_path)

        # Run rollouts
        result: tuple[list[RolloutSample], list[float]] = asyncio.run(
            asyncio.wait_for(
                run_rollouts(config=config),
                timeout=config.rollout.rollout_timeout,
            )
        )
        rollouts, rewards = result

        print(f"[DEBUG] Rollout #0 got reward {rewards[0]}", rollouts[0].messages) # TODO: add config --debug flag

        # Run base model and get ref logprobs
        token_ids: list[list[int]] = []
        ref_logprobs: list[list[float]] = []
        old_logprobs: list[list[float]] = []
        masks: list[list[int]] = []
        metadatas: list[dict] = []
        for rollout in rollouts:
            token_ids.append(rollout.token_ids)
            old_logprobs.append(rollout.logprobs)
            masks.append(rollout.mask)
            ref_logprobs.append(
                vllm._get_model_logps(
                    config.train.model_name, rollout.token_ids, return_token_ids=False
                )
            )
            metadatas.append(rollout.metadata)

        # vLLM goes to sleep
        vllm._sleep()

        # Bring model and optimizer to gpu
        adapter_model.to("cuda")
        move_opt_to_device(optimizer, "cuda")

        # Pad data and tensorize
        input_ids = pad_2d(token_ids, pad_value=0, dtype=torch.long, device="cuda")
        attention_mask = pad_2d(
            [[1] * len(seq) for seq in token_ids], pad_value=0, dtype=torch.long, device="cuda"
        )
        old_logprobs = pad_2d(old_logprobs, pad_value=0.0, dtype=torch.float32, device="cuda")
        ref_logprobs = pad_2d(ref_logprobs, pad_value=0.0, dtype=torch.float32, device="cuda")
        masks = pad_2d(masks, pad_value=0.0, dtype=torch.float32, device="cuda")
        rewards = torch.tensor(rewards, dtype=torch.float32, device="cuda")
        advantages = algorithm.compute_advantages(rewards=rewards, metadatas=metadatas)

        old_logprobs = old_logprobs[:, 1:]
        ref_logprobs = ref_logprobs[:, 1:]
        masks = masks[:, 1:]
        train_batch_size = input_ids.shape[0]
        microbatch_size = config.train.train_microbatch_size or train_batch_size

        # Train for M iters
        for iter in range(config.train.n_iters):
            wandb_log_stats: dict[str, float] = defaultdict(float)

            # Gradient accumulation
            for start_idx in range(0, train_batch_size, microbatch_size):
                end_idx = min(start_idx + microbatch_size, train_batch_size)
                microbatch_weight = (end_idx - start_idx) / train_batch_size

                mb_input_ids = input_ids[start_idx:end_idx]
                mb_attention_mask = attention_mask[start_idx:end_idx]
                mb_old_logprobs = old_logprobs[start_idx:end_idx]
                mb_ref_logprobs = ref_logprobs[start_idx:end_idx]
                mb_masks = masks[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = adapter_model(
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                        use_cache=False,
                    )
                    logits = outputs.logits

                logits = logits[:, :-1, :]
                labels = mb_input_ids[:, 1:]
                selected = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                log_probs = selected - logits.logsumexp(dim=-1)

                loss, microbatch_log = algorithm.loss(
                    current_logprobs=log_probs,
                    old_logprobs=mb_old_logprobs,
                    ref_logprobs=mb_ref_logprobs,
                    masks=mb_masks,
                    advantages=mb_advantages,
                )

                (loss * microbatch_weight).backward()

                for key, value in microbatch_log.items():
                    wandb_log_stats[key] += value * microbatch_weight

                del selected, outputs, loss, logits, labels, log_probs

            torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        wandb.log(
            {
                "reward_mean": rewards.mean().item(),
                "completion_tokens_mean": masks.sum(dim=1).mean().item(),
                **wandb_log_stats,
            },
            step=step_idx,
        )

        # Save adapter
        adapter_model.save_pretrained(config.train.adapter_path)

        # Offload model and optimizer to cpu
        adapter_model.to("cpu")
        move_opt_to_device(optimizer, "cpu")

        # Clear vars from VRAM
        del (
            input_ids,
            attention_mask,
            old_logprobs,
            ref_logprobs,
            masks,
            rewards,
            advantages,
        )
        torch.cuda.empty_cache()

    wandb.finish()


if __name__ == "__main__":
    main()
