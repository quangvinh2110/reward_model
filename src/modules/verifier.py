from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
import networkx as nx
import time
import json
from networkx.readwrite import node_link_data, node_link_graph

from .constructor import AutoConstructor
from ..utils.io import read_txt
from ..utils.data import parse_from_boxed
from .client import Client


def _verify_one_helper(args):
    """Helper function to run _verify_one in multiprocessing."""
    _verify_one_func, i, sample, construction_kwargs, generation_kwargs = args
    return _verify_one_func(i, sample, construction_kwargs, generation_kwargs)


class Verifier(ABC):
    def __init__(
        self,
        client: Client,
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
        self, id: int, sample: dict, construction_kwargs: dict, generation_kwargs: dict
    ) -> Tuple[int, str]:
        """Verify a single sample (dictionary)."""
        raise NotImplementedError("Subclasses must implement this method")

    # Remove the __call__ method from Verifier. Only keep _verify_one and related logic.


class SequentialVerifier(Verifier):

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/home/admin/NLP/reward_model/reward_model/resources/prompt_templates/SEQUENTIAL_VERIFICATION.txt"
        )

    def _verify_one(
        self,
        id: int,
        sample: dict,
        construction_kwargs: dict = {},
        generation_kwargs: dict = {},
    ) -> dict:
        start_time = time.time()
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
            generation_kwargs=generation_kwargs,
        )
        return {
            "id": id,
            "generated_critique": responses[0],
            "graph": None,
            "time": time.time() - start_time,
        }


class StepwiseVerifier(Verifier):
    def _get_prompt_template(self) -> str:
        return read_txt(
            "/home/admin/NLP/reward_model/reward_model/resources/prompt_templates/STEPWISE_VERIFICATION.txt"
        )

    def _verify_one(
        self,
        id: int,
        sample: dict,
        construction_kwargs: dict = {},
        generation_kwargs: dict = {},
    ) -> dict:
        start_time = time.time()
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        for step_idx in range(len(sample["steps"])):
            tagged_steps = "\n".join(
                [
                    f"<step_{i}>\n{step}\n</step_{i}>"
                    for i, step in enumerate(sample["steps"][: step_idx + 1])
                ]
            )
            user_input = self.prompt_template.format(
                problem=sample["problem"], tagged_steps=tagged_steps, idx=step_idx
            )
            step_results = self.client(
                batch_messages=[[{"role": "user", "content": user_input}]],
                generation_kwargs=generation_kwargs,
            )[0]
            for result, step_result in zip(results, step_results):
                result.append(step_result)
            # Majority voting
            parsed = [parse_from_boxed(step_result) for step_result in step_results]
            majority, _ = Counter(parsed).most_common(1)[0]
            if majority == "0":
                for result in results:
                    result.append(f"Final answer: \\boxed{{{step_idx}}}")
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": None,
                    "time": time.time() - start_time,
                }
            if step_idx == sample["label"]:
                for result in results:
                    result.append(r"Final answer: \\boxed{None}")
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": None,
                    "time": time.time() - start_time,
                }
        for result in results:
            result.append("Final answer: \\boxed{-1}")
        return {
            "id": id,
            "generated_critique": ["<|sep|>".join(result) for result in results],
            "graph": None,
            "time": time.time() - start_time,
        }


class ParcVerifier(Verifier):

    def __init__(
        self,
        client: Client,
        show_progress: bool = True,
    ):
        super().__init__(client, show_progress)
        self.constructor = AutoConstructor.from_type("targeted", client=client)

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/home/admin/NLP/reward_model/reward_model/resources/prompt_templates/PARC_VERIFICATION.txt"
        )

    def _verify_one(
        self,
        id: int,
        sample: dict,
        construction_kwargs: dict = {},
        generation_kwargs: dict = {},
    ) -> dict:
        start_time = time.time()
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        solution_graph = nx.DiGraph()
        for step_idx in range(len(sample["steps"])):
            solution_graph.add_node(
                step_idx, content=sample["steps"][step_idx], resolved=False
            )
        solution_graph.nodes[0]["resolved"] = True
        for step_idx in range(len(sample["steps"])):
            self.constructor(
                problem=sample["problem"],
                solution_graph=solution_graph,
                target_idx=step_idx,
                construction_kwargs=construction_kwargs,
                generation_kwargs=generation_kwargs,
            )
            tracked_premises = "\n".join(
                f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                for i in solution_graph.predecessors(step_idx)
            )
            target_step = (
                f"<step_{step_idx}>\n{sample['steps'][step_idx]}\n</step_{step_idx}>"
            )
            user_input = self.prompt_template.format(
                problem=sample["problem"],
                tracked_premises=tracked_premises,
                target_step=target_step,
            )
            step_results = self.client(
                batch_messages=[[{"role": "user", "content": user_input}]],
                generation_kwargs=generation_kwargs,
            )[0]
            for result, step_result in zip(results, step_results):
                result.append(step_result)
            # Majority voting
            parsed = [parse_from_boxed(step_result) for step_result in step_results]
            majority, _ = Counter(parsed).most_common(1)[0]
            if majority == "0":
                for result in results:
                    result.append(f"Final answer: \\boxed{{{step_idx}}}")
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
                    "time": time.time() - start_time,
                }
            if step_idx == sample["label"]:
                for result in results:
                    result.append(r"Final answer: \\boxed{None}")
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
                    "time": time.time() - start_time,
                }
        for result in results:
            result.append("Final answer: \\boxed{-1}")
        return {
            "id": id,
            "generated_critique": ["<|sep|>".join(result) for result in results],
            "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
            "time": time.time() - start_time,
        }


class LogicFlowVerifier(Verifier):
    def __init__(
        self,
        client: Client,
        constructor_type: str = "targeted",
        prefix: str = "predecessors",
        suffix: str = "successors",
        prefix_size: int = -1,
        suffix_size: int = 10,
        show_progress: bool = True,
    ):
        self.constructor = AutoConstructor.from_type(constructor_type, client=client)
        self.prefix = prefix
        self.suffix = suffix
        self.prefix_size = prefix_size
        self.suffix_size = suffix_size
        super().__init__(client, show_progress)

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/home/admin/NLP/reward_model/reward_model/resources/prompt_templates/LOGICFLOW_VERIFICATION.txt"
        )

    def _verify_one_step(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        label: Optional[int] = None,
        generation_kwargs: dict = {},
    ) -> Tuple[int, str]:
        results = [[] for _ in range(generation_kwargs.get("n", 1))]

        for step_idx in sorted(solution_graph.nodes):
            if self.prefix == "predecessors":
                prefix_indices = sorted(
                    list(solution_graph.predecessors(step_idx)), reverse=True
                )
            elif self.prefix == "sequential":
                prefix_indices = sorted(
                    [i for i in (solution_graph.nodes) if i < step_idx], reverse=True
                )
            else:
                raise ValueError(f"Unknown prefix: {self.prefix}")
            if self.prefix_size > 0:
                prefix_indices = prefix_indices[: self.prefix_size]
            if self.suffix == "successors":
                suffix_indices = sorted(list(solution_graph.successors(step_idx)))
            elif self.suffix == "null":
                suffix_indices = []
            elif self.suffix == "sequential":
                suffix_indices = sorted(
                    [i for i in (solution_graph.nodes) if i > step_idx],
                )
            else:
                raise ValueError(f"Unknown suffix: {self.suffix}")
            if self.suffix_size > 0:
                suffix_indices = suffix_indices[: self.suffix_size]

            tagged_steps = []
            for i in sorted(solution_graph.nodes):
                if i in prefix_indices or i == step_idx or i in suffix_indices:
                    tagged_steps.append(
                        f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                    )
                elif len(tagged_steps) == 0 or tagged_steps[-1] != "...":
                    tagged_steps.append("...")
            tagged_steps = "\n".join(tagged_steps)

            user_input = self.prompt_template.format(
                problem=problem,
                tagged_steps=tagged_steps,
                idx=step_idx,
            )
            step_results = self.client(
                batch_messages=[[{"role": "user", "content": user_input}]],
                generation_kwargs=generation_kwargs,
            )[0]
            for result, step_result in zip(results, step_results):
                result.append(step_result)
            parsed = [parse_from_boxed(step_result) for step_result in step_results]
            majority, _ = Counter(parsed).most_common(1)[0]
            if majority == "0":
                for result in results:
                    result.append(f"Final answer: \\boxed{{{step_idx}}}")
                return ["<|sep|>".join(result) for result in results]
            if label is not None and step_idx == label:
                for result in results:
                    result.append(r"Final answer: \boxed{None}")
                return ["<|sep|>".join(result) for result in results]
            solution_graph.nodes[step_idx]["state"] = True

        for result in results:
            result.append("Final answer: \\boxed{-1}")
        return ["<|sep|>".join(result) for result in results]

    def _verify_one(
        self,
        id: int,
        sample: dict,
        construction_kwargs: dict = {},
        generation_kwargs: dict = {},
    ) -> dict:
        start_time = time.time()
        if "graph" in sample:
            solution_graph = node_link_graph(json.loads(sample["graph"]), edges="edges")
        else:
            solution_graph = nx.DiGraph()
            for step_idx in range(len(sample["steps"])):
                solution_graph.add_node(
                    step_idx,
                    content=sample["steps"][step_idx],
                    resolved=False,
                    state=None,
                )
            solution_graph.nodes[0]["resolved"] = True
            self.constructor(
                problem=sample["problem"],
                solution_graph=solution_graph,
                construction_kwargs=construction_kwargs,
                generation_kwargs=generation_kwargs,
            )
        generated_critique = self._verify_one_step(
            problem=sample["problem"],
            solution_graph=solution_graph,
            label=sample["label"],
            generation_kwargs=generation_kwargs,
        )
        return {
            "id": id,
            "generated_critique": generated_critique,
            "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
            "time": time.time() - start_time,
        }


class AutoVerifier:
    TYPE_MAP = {
        "sequential": SequentialVerifier,
        "stepwise": StepwiseVerifier,
        "parc": ParcVerifier,
        "logicflow": LogicFlowVerifier,
    }

    @classmethod
    def from_type(cls, verifier_type: str, **kwargs):
        verifier_type = verifier_type.lower()
        if verifier_type not in cls.TYPE_MAP:
            raise ValueError(f"Unknown verifier type: {verifier_type}")
        return cls.TYPE_MAP[verifier_type](**kwargs)
