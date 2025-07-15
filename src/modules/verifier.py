from typing import List, Tuple
from abc import ABC, abstractmethod
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
import networkx as nx

from .constructor import AutoConstructor
from ..utils.io import read_txt
from ..utils.parser import parse_from_boxed
from .client import OpenaiClient


def _verify_one_helper(args):
    """Helper function to run _verify_one in multiprocessing."""
    _verify_one_func, i, sample, generation_kwargs = args
    return _verify_one_func(i, sample, **generation_kwargs)


class Verifier(ABC):
    def __init__(
        self,
        client: OpenaiClient,
        show_progress: bool = True,
    ):
        self.client = client
        self.prompt_template = self._get_prompt_template()
        self.show_progress = show_progress

    @abstractmethod
    def _get_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _verify_one(
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single sample (dictionary)."""
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(
        self,
        samples: List[dict],
        num_workers: int = 4,
        **generation_kwargs,
    ) -> List[str]:
        """Verify multiple samples using multiprocessing, or sequentially if num_workers == 1."""
        if not samples:
            return []

        # Prepare arguments for the helper function
        tasks = [
            (self._verify_one, i, sample, generation_kwargs)
            for i, sample in enumerate(samples)
        ]

        results = []
        with Pool(processes=num_workers) as pool:
            if self.show_progress:
                results = list(
                    tqdm(
                        pool.imap_unordered(_verify_one_helper, tasks),
                        total=len(tasks),
                    )
                )
            else:
                results = list(pool.imap_unordered(_verify_one_helper, tasks))

        # Sort results by ID to maintain original order
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results]


class SequentialVerifier(Verifier):

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/SEQUENTIAL_VERIFICATION.txt"
        )

    def _verify_one(
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single sample (dictionary)."""
        tagged_steps = "\n".join(
            [
                f"<step_{i}>\n{step}\n</step_{i}>"
                for i, step in enumerate(sample["steps"])
            ]
        )
        user_input = self.prompt_template.format(
            problem=sample["problem"], tagged_steps=tagged_steps
        )
        responses = self.client(
            batch_messages=[[{"role": "user", "content": user_input}]],
            **generation_kwargs,
        )
        return id, responses[0]


class StepwiseVerifier(Verifier):
    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/STEPWISE_VERIFICATION.txt"
        )

    def _verify_one(
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single sample (dictionary) with early stopping and majority voting."""
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        no_wrong_step = True
        for step_idx in range(len(sample["steps"])):
            if sample["label"] != -1 and step_idx > sample["label"]:
                for result in results:
                    result.append(r"Final answer: \boxed{None}")
                break
            tagged_steps = "\n".join(
                [
                    f"<step_{i}>\n{step}\n</step_{i}>"
                    for i, step in enumerate(sample["steps"][: step_idx + 1])
                ]
            )
            user_input = self.prompt_template.format(
                problem=sample["problem"], tagged_steps=tagged_steps
            )
            # Create chat completion request
            step_results = self.client(
                batch_messages=[[{"role": "user", "content": user_input}]],
                **generation_kwargs,
            )[0]
            for result, step_result in zip(results, step_results):
                result.append(step_result)
            # Majority voting
            parsed = [parse_from_boxed(step_result) for step_result in step_results]
            majority, _ = Counter(parsed).most_common(1)[0]
            if majority == "0":
                for result in results:
                    result.append(f"Final answer: \\boxed{{{step_idx}}}")
                no_wrong_step = False
                break
        if no_wrong_step:
            for result in results:
                result.append("Final answer: \\boxed{-1}")
        return (id, ["<|sep|>".join(result) for result in results])


class PerlVerifier(Verifier):

    def __init__(
        self,
        client: OpenaiClient,
        constructor_type: str = "targeted",
        show_progress: bool = True,
    ):
        super().__init__(client, show_progress)
        self.constructor = AutoConstructor.from_type(constructor_type, client=client)

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/PERL_VERIFICATION.txt"
        )

    def _verify_one(
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        graph = nx.DiGraph()
        graph.add_node(0, content=sample["steps"][0], resolved=True)
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        no_wrong_step = True
        for step_idx in range(len(sample["steps"])):
            if sample["label"] != -1 and step_idx > sample["label"]:
                no_wrong_step = False
                for result in results:
                    result.append(r"Final answer: \boxed{None}")
                break
            graph.add_node(step_idx, content=sample["steps"][step_idx], resolved=False)
            self.constructor._track_one_step(
                sample, step_idx, graph, max_window_size=step_idx, **generation_kwargs
            )
            tracked_premises = "\n".join(
                f"<step_{i}>\n{graph.nodes[i]['content']}\n</step_{i}>"
                for i in graph.predecessors(step_idx)
            )
            target_step = (
                f"<step_{step_idx}>\n{sample['steps'][step_idx]}\n</step_{step_idx}>"
            )
            user_input = self.prompt_template.format(
                problem=sample["problem"],
                tracked_premises=tracked_premises,
                target_step=target_step,
            )
            # Create chat completion request
            step_results = self.client(
                batch_messages=[[{"role": "user", "content": user_input}]],
                **generation_kwargs,
            )[0]
            for result, step_result in zip(results, step_results):
                result.append(step_result)
            # Majority voting
            parsed = [parse_from_boxed(step_result) for step_result in step_results]
            majority, _ = Counter(parsed).most_common(1)[0]
            if majority == "0":
                for result in results:
                    result.append(f"Final answer: \\boxed{{{step_idx}}}")
                no_wrong_step = False
                break
        if no_wrong_step:
            for result in results:
                result.append("Final answer: \\boxed{-1}")
        return (id, ["<|sep|>".join(result) for result in results])


class LogicFlowVerifier(Verifier):
    pass


class AutoVerifier:
    TYPE_MAP = {
        "sequential": SequentialVerifier,
        "stepwise": StepwiseVerifier,
        "perl": PerlVerifier,
        "logicflow": LogicFlowVerifier,
    }

    @classmethod
    def from_type(cls, verifier_type: str, **kwargs):
        verifier_type = verifier_type.lower()
        if verifier_type not in cls.TYPE_MAP:
            raise ValueError(f"Unknown verifier type: {verifier_type}")
        return cls.TYPE_MAP[verifier_type](**kwargs)
