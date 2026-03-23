import importlib
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer


def use_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


class TrainConfig(BaseModel):
    use_wandb: bool = Field(default=True)
    wandb_project_name: str = Field(default="untitled-project")
    wandb_run_name: str = Field(default="untitled-run")

    model_name: str = Field(...)
    adapter_path: str = Field(...)
    n_iters: int = Field(default=4)
    lr: float = Field(default=5e-7)
    n_steps: int = Field(default=100)
    train_microbatch_size: int | None = Field(default=None)
    use_flash_attn: bool = Field(default_factory=use_flash_attn)

    @model_validator(mode="after")
    def apply_derived_defaults(self) -> "TrainConfig":
        self.adapter_path = os.path.abspath(self.adapter_path)
        return self


class RolloutConfig(BaseModel):
    n_rollouts: int = Field(default=32)
    max_parallel_rollouts: int | None = Field(
        default=None, description="# of rollouts to run in parallel threads"
    )
    rollout_timeout: float = Field(
        default=600.0, description="Timeout after which rollout will error"
    )
    assistant_start_string: str = Field(default="<|im_start|>assistant\n")
    assistant_end_string: str = Field(default="<|im_end|>")
    mask_start_token_ids: list[int] = Field(default_factory=list)
    mask_end_token_ids: list[int] = Field(default_factory=list)


class AlgoConfig(BaseModel):
    factory: str = Field(default="src.algo.grpo:GRPO")
    kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_groups": 4,
            "clip_eps": 0.2,
            "kl_coef": 0.04,
            "group_adv_eps": 1e-6,
        }
    )


class EnvConfig(BaseModel):
    factory: str = Field(default="environments.gsm8k:GSM8KEnvironmentFactory")
    kwargs: dict[str, Any] = Field(default_factory=dict)


class MithrlConfig(BaseModel):
    train: TrainConfig = Field(default_factory=TrainConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    algo: AlgoConfig = Field(default_factory=AlgoConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)

    @model_validator(mode="after")
    def apply_derived_defaults(self) -> "MithrlConfig":
        tokenizer = AutoTokenizer.from_pretrained(self.train.model_name)
        self.rollout.mask_start_token_ids = tokenizer.encode(
            self.rollout.assistant_start_string,
            add_special_tokens=False,
        )
        self.rollout.mask_end_token_ids = tokenizer.encode(
            self.rollout.assistant_end_string,
            add_special_tokens=False,
        )
        return self

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "MithrlConfig":
        config_path = Path(path)
        data = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a YAML mapping at {config_path}, got {type(data).__name__}."
            )
        return cls.model_validate(data)

    @classmethod
    def from_yaml_or_dict(cls, value: str | os.PathLike[str] | dict[str, Any]) -> "MithrlConfig":
        if isinstance(value, dict):
            return cls.model_validate(value)
        return cls.from_yaml(value)

    def to_dict(self, **kwargs: Any) -> dict:
        return self.model_dump()
