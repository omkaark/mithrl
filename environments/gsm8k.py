# vibecoded

from __future__ import annotations

import random
import re
import threading
from decimal import Decimal, InvalidOperation

from datasets import load_dataset
from src.train.env import Environment, EnvironmentFactory
from src.utils.config import MithrlConfig

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
THINK_TAGS = (THINK_OPEN, THINK_CLOSE)
ANSWER_TAGS = (ANSWER_OPEN, ANSWER_CLOSE)
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
TRAIN_SPLIT = "train"
ANSWER_REWARD = 1.0
REASONING_FORMAT_REWARD = 0.1
ANSWER_FORMAT_REWARD = 0.1
PARSEABLE_ANSWER_REWARD = 0.05
SHUFFLE_BUFFER_SIZE = 1024


class _GSM8KStreamState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rng = random.Random(42)  # Important to set this for fully deterministic runs
        self._iterator = self._build_iterator()

    def next_examples(self, count: int) -> list[dict]:
        examples: list[dict] = []
        with self._lock:
            while len(examples) < count:
                try:
                    examples.append(next(self._iterator))
                except StopIteration:
                    self._iterator = self._build_iterator()
        return examples

    def next_seed(self) -> int:
        with self._lock:
            return self._rng.randrange(0, 2**31)

    def _build_iterator(self):
        dataset = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split=TRAIN_SPLIT,
            streaming=True,
        )
        shuffle_seed = self._rng.randrange(0, 2**31)
        shuffled = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=shuffle_seed)
        return iter(shuffled)


_STREAM_STATE = _GSM8KStreamState()


class GSM8KEnvironment(Environment):
    def __init__(
        self,
        *,
        config: MithrlConfig,
        group_id: int,
        seed: int,
        question: str,
        reference_answer: str,
    ) -> None:
        super().__init__()
        self.group_id = group_id
        self.seed = seed
        self.question = question
        self.reference_answer = reference_answer
        self._metadata.update(
            {
                "env_name": "gsm8k",
                "group_id": self.group_id,
                "seed": self.seed,
                "question": self.question,
                "expected_answer": self.reference_answer,
            }
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are solving a grade-school math word problem. "
            f"Write your reasoning inside {THINK_OPEN}{THINK_CLOSE}. "
            f"Then write only the final answer, a number with no punctuation or words, inside {ANSWER_OPEN}{ANSWER_CLOSE}." # noqa
        )

    @property
    def next_query(self) -> str:
        return f"Solve this math word problem:\n{self.question}\n\n"

    def step(self, response: str) -> None:
        self.turn_count += 1
        self.done = True

        used_think_block = self._has_tagged_block(response, THINK_TAGS)
        used_answer_block = self._has_tagged_block(response, ANSWER_TAGS)
        parsed_answer = self._extract_model_answer(response, ANSWER_TAGS)
        normalized_answer = self.normalize_answer(parsed_answer)

        self.reward = (
            REASONING_FORMAT_REWARD * used_think_block
            + ANSWER_FORMAT_REWARD * used_answer_block
            + PARSEABLE_ANSWER_REWARD * (normalized_answer is not None)
            + ANSWER_REWARD * (normalized_answer == self.reference_answer)
        )

    @staticmethod
    def _find_tagged_block(response: str, tags: tuple[str, str]) -> re.Match[str] | None:
        open_tag, close_tag = tags
        return re.search(
            rf"{re.escape(open_tag)}(.*?){re.escape(close_tag)}",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )

    @classmethod
    def _has_tagged_block(cls, response: str, tags: tuple[str, str]) -> bool:
        match = cls._find_tagged_block(response, tags)
        return match is not None and bool(match.group(1).strip())

    @classmethod
    def _extract_model_answer(cls, response: str, tags: tuple[str, str]) -> str | None:
        match = cls._find_tagged_block(response, tags)
        return None if match is None else match.group(1).strip()

    @classmethod
    def normalize_answer(cls, answer: str | None) -> str | None:
        if answer is None:
            return None

        normalized = answer.strip()
        if not normalized:
            return None

        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.rstrip(".").replace(",", "").replace("$", "").strip()

        if not normalized:
            return None

        match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized)
        if match is not None:
            try:
                decimal_value = Decimal(normalized).normalize()
                if decimal_value == decimal_value.to_integral():
                    return str(decimal_value.quantize(Decimal("1")))
                return format(decimal_value, "f").rstrip("0").rstrip(".")
            except InvalidOperation:
                return normalized

        return normalized

    @classmethod
    def parse_reference_answer(cls, answer: str) -> str:
        if "####" not in answer:
            raise ValueError("Expected GSM8K answer to contain '####' delimiter.")
        final_answer = answer.split("####", 1)[1].strip()
        normalized_answer = cls.normalize_answer(final_answer)
        if normalized_answer is None:
            raise ValueError("Failed to parse GSM8K reference answer.")
        return normalized_answer


class GSM8KEnvironmentFactory(EnvironmentFactory):
    def __init__(self, config: MithrlConfig) -> None:
        if config.rollout.n_rollouts % config.algo.kwargs["n_groups"] != 0:
            raise ValueError(
                "rollout.n_rollouts must be divisible by algo.kwargs.n_groups for grouped rollouts."
            )

        self._config = config
        n_groups = config.algo.kwargs["n_groups"]
        self._rollouts_per_group = config.rollout.n_rollouts // n_groups
        examples = _STREAM_STATE.next_examples(n_groups)
        self._group_specs: dict[int, tuple[int, str, str]] = {}

        for group_id, example in enumerate(examples):
            seed = _STREAM_STATE.next_seed()
            self._group_specs[group_id] = (
                seed,
                example["question"],
                GSM8KEnvironment.parse_reference_answer(example["answer"]),
            )

    def create(self, rollout_idx: int) -> GSM8KEnvironment:
        group_id = rollout_idx // self._rollouts_per_group
        seed, question, reference_answer = self._group_specs[group_id]
        return GSM8KEnvironment(
            config=self._config,
            group_id=group_id,
            seed=seed,
            question=question,
            reference_answer=reference_answer,
        )
