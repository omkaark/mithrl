# vibecoded

from __future__ import annotations

import random
import re

from src.rl.env import Environment, EnvironmentFactory
from src.utils.config import MithrlConfig

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

SYSTEM_PROMPT = (
    "You are solving a basic arithmetic problem. "
    f"Write your reasoning inside {THINK_OPEN}{THINK_CLOSE}. "
    f"Then write only the final integer answer inside {ANSWER_OPEN}{ANSWER_CLOSE}."
)


class SimpleMathEnvironment(Environment):
    def __init__(
        self,
        group_id: int,
        seed: int,
        left: int,
        operator: str,
        right: int,
    ) -> None:
        super().__init__()
        self._response = ""
        self.reward = 0.0
        self.group_id = group_id
        self.seed = seed
        self.left = left
        self.operator = operator
        self.right = right
        self.answer = self._solve(self.left, self.operator, self.right)
        self._metadata.update(
            {
                "group_id": self.group_id,
                "seed": self.seed,
                "problem": f"{self.left} {self.operator} {self.right}",
                "expected_answer": self.answer,
            }
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def next_query(self) -> str:
        return (
            f"Solve: {self.left} {self.operator} {self.right}\n"
            f"Write your reasoning inside {THINK_OPEN}{THINK_CLOSE}. "
            f"Then write only the final integer answer inside {ANSWER_OPEN}{ANSWER_CLOSE}."
        )

    def step(self, response: str) -> None:
        self.turn_count += 1
        self._response = response
        self.done = True

        used_think_tokens = self._has_think_block(response)
        used_answer_tokens = self._has_answer_block(response)
        parsed_answer = self._extract_answer(response)
        is_correct = parsed_answer == self.answer

        reward = 0.0
        if used_think_tokens:
            reward += 0.1
        if used_answer_tokens:
            reward += 0.1
        if is_correct:
            reward += 1.0

        self.reward = reward
        self._metadata.update(
            {
                "used_think_tokens": used_think_tokens,
                "parsed_answer": parsed_answer,
                "is_correct": is_correct,
                "reward": reward,
                "response": response,
            }
        )

    @staticmethod
    def _solve(left: int, operator: str, right: int) -> int:
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        return left * right

    @staticmethod
    def _has_think_block(response: str) -> bool:
        match = re.search(
            rf"{re.escape(THINK_OPEN)}(.*?){re.escape(THINK_CLOSE)}",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match is not None and bool(match.group(1).strip())

    @staticmethod
    def _has_answer_block(response: str) -> bool:
        match = re.search(
            rf"{re.escape(ANSWER_OPEN)}(.*?){re.escape(ANSWER_CLOSE)}",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        return match is not None and bool(match.group(1).strip())

    @staticmethod
    def _extract_answer(response: str) -> int | None:
        answer_match = re.search(r"Answer:\s*(-?\d+)\b", response, flags=re.IGNORECASE)
        if answer_match:
            return int(answer_match.group(1))

        all_numbers = re.findall(r"-?\d+", response)
        if all_numbers:
            return int(all_numbers[-1])
        return None


class SimpleMathEnvironmentFactory(EnvironmentFactory):
    def __init__(self, config: MithrlConfig) -> None:
        if config.rollout.n_rollouts % config.algo.kwargs["n_groups"] != 0:
            raise ValueError(
                "rollout.n_rollouts must be divisible by algo.kwargs.n_groups for grouped rollouts."
            )

        self._rollouts_per_group = config.rollout.n_rollouts // config.algo.kwargs["n_groups"]
        self._group_specs: dict[int, tuple[int, int, str, int]] = {}
        seed_rng = random.SystemRandom(42)

        for group_id in range(config.algo.kwargs["n_groups"]):
            seed = seed_rng.randrange(0, 2**31)
            left, operator, right = self._sample_problem(seed)
            self._group_specs[group_id] = (seed, left, operator, right)

    def create(self, rollout_idx: int) -> SimpleMathEnvironment:
        group_id = rollout_idx // self._rollouts_per_group
        seed, left, operator, right = self._group_specs[group_id]
        return SimpleMathEnvironment(
            group_id=group_id,
            seed=seed,
            left=left,
            operator=operator,
            right=right,
        )

    @staticmethod
    def _sample_problem(seed: int) -> tuple[int, str, int]:
        rng = random.Random(seed)
        operator = rng.choice(["+", "-", "*"])
        if operator == "*":
            left = rng.randint(0, 14)
            right = rng.randint(0, 14)
        else:
            left = rng.randint(0, 99)
            right = rng.randint(0, 99)
        return left, operator, right
