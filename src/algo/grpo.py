import torch

from .base import Algorithm
from ..utils.config import MithrlConfig


class GRPO(Algorithm):
    def __init__(self, config: MithrlConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        metadatas: list[dict],
    ) -> torch.Tensor:
        try:
            group_ids = torch.tensor([md["group_id"] for md in metadatas], device=rewards.device)
        except KeyError:
            print("group_id not set in environment metadata.")
            raise

        advantages = torch.empty_like(rewards)
        for gid in group_ids.unique():
            m = group_ids == gid
            g = rewards[m]
            advantages[m] = (g - g.mean()) / (
                g.std(unbiased=False) + self.config.algo.kwargs["group_adv_eps"]
            )

        return advantages

    def loss(
        self,
        current_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        masks: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask_denom = masks.sum(dim=1).clamp_min(1.0)

        importance = torch.exp(current_logprobs - old_logprobs)
        clip_eps = self.config.algo.kwargs["clip_eps"]
        clipped = importance.clamp(1 - clip_eps, 1 + clip_eps)

        adv = advantages[:, None]
        policy_loss = -torch.min(importance * adv, clipped * adv)
        policy_loss = (policy_loss * masks).sum(dim=1) / mask_denom
        policy_loss = policy_loss.mean()

        log_ratio = current_logprobs - ref_logprobs
        kl = (torch.exp(log_ratio) - log_ratio - 1) * masks
        kl_loss = self.config.algo.kwargs["kl_coef"] * (kl.sum(dim=1) / mask_denom).mean()

        combined = policy_loss + kl_loss
        return combined, {
            "loss": combined.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
        }
