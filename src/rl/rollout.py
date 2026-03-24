import asyncio
from dataclasses import dataclass, field

from ..utils import vllm
from ..utils.client import LMClient
from ..utils.config import MithrlConfig
from ..utils.loaders import load_environment_factory
from ..utils.torch_utils import get_masks_from_tokens
from .env import Environment, EnvironmentFactory


@dataclass
class RolloutSample:
    messages: list[dict]
    seed: int | None = None
    group_id: int | None = None
    turn_count: int = 0
    token_ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    metadata: dict = field(
        default_factory=dict
    )  # Stuff like group_id, etc. that will be used by the algorithm


async def _run_single_rollout(
    rollout_idx: int,
    config: MithrlConfig,
    sem: asyncio.Semaphore,
    env_factory: EnvironmentFactory,
) -> tuple[int, RolloutSample, float]:
    async with sem:
        env: Environment = env_factory.create(rollout_idx)
        llm = LMClient(model="adapter", system_prompt=env.system_prompt)

        while not env.done:
            response = await llm.query(env.next_query)
            env.step(response)

        token_ids = vllm._tokenize_messages(
            config.train.model_name, llm.messages
        ) # make async
        logprobs = vllm._get_model_logps(
            config.train.model_name, token_ids, return_token_ids=False
        )
        mask = get_masks_from_tokens(token_ids, config.rollout)

        sample = RolloutSample(
            messages=llm.messages,
            turn_count=env.turn_count,
            token_ids=token_ids,
            mask=mask,
            logprobs=logprobs,
            metadata=env.metadata,
        )

        return rollout_idx, sample, env.reward


async def run_rollouts(config: MithrlConfig) -> tuple[list[RolloutSample], list[float]]:
    n_parallel_rollouts = (
        min(config.rollout.n_rollouts, config.rollout.max_parallel_rollouts)
        if config.rollout.max_parallel_rollouts
        else config.rollout.n_rollouts
    )
    sem = asyncio.Semaphore(n_parallel_rollouts)
    env_factory = load_environment_factory(config)

    tasks = [
        _run_single_rollout(
            rollout_idx=i,
            config=config,
            sem=sem,
            env_factory=env_factory,
        )
        for i in range(config.rollout.n_rollouts)
    ]

    results = await asyncio.gather(*tasks)
    results.sort(key=lambda item: item[0])

    rollouts = [sample for _, sample, _ in results]
    rewards = [reward for _, _, reward in results]

    reward_mean = sum(rewards) / len(rewards)
    print(f"Rollout summary: n={len(rollouts)}, reward_mean={reward_mean:.3f}")

    return rollouts, rewards
