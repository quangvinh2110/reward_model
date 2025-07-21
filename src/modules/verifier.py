from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
import networkx as nx
import time
import json
from networkx.readwrite import node_link_data

from .constructor import AutoConstructor
from ..utils.io import read_txt
from ..utils.data import parse_from_boxed
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
    ) -> List[dict]:
        """Verify multiple samples using multiprocessing, or sequentially if num_workers == 1. Returns a list of dicts."""
        if not samples:
            return []
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
        results.sort(key=lambda x: x["id"])
        return results


class SequentialVerifier(Verifier):

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/SEQUENTIAL_VERIFICATION.txt"
        )

    def _verify_one(self, id: int, sample: dict, **generation_kwargs) -> dict:
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
            **generation_kwargs,
        )
        elapsed = time.time() - start_time
        return {
            "id": id,
            "generated_critique": responses[0],
            "graph": None,
            "time": elapsed,
        }


class StepwiseVerifier(Verifier):
    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/STEPWISE_VERIFICATION.txt"
        )

    def _verify_one(self, id: int, sample: dict, **generation_kwargs) -> dict:
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
                elapsed = time.time() - start_time
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": None,
                    "time": elapsed,
                }
            if step_idx == sample["label"]:
                for result in results:
                    result.append(r"Final answer: \\boxed{None}")
                elapsed = time.time() - start_time
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": None,
                    "time": elapsed,
                }
        for result in results:
            result.append("Final answer: \\boxed{-1}")
        elapsed = time.time() - start_time
        return {
            "id": id,
            "generated_critique": ["<|sep|>".join(result) for result in results],
            "graph": None,
            "time": elapsed,
        }


class ParcVerifier(Verifier):

    def __init__(
        self,
        client: OpenaiClient,
        show_progress: bool = True,
    ):
        super().__init__(client, show_progress)
        self.constructor = AutoConstructor.from_type("targeted", client=client)

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/PERL_VERIFICATION.txt"
        )

    def _verify_one(self, id: int, sample: dict, **generation_kwargs) -> dict:
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
                max_window_size=step_idx,
                **generation_kwargs,
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
                elapsed = time.time() - start_time
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
                    "time": elapsed,
                }
            if step_idx == sample["label"]:
                for result in results:
                    result.append(r"Final answer: \\boxed{None}")
                elapsed = time.time() - start_time
                return {
                    "id": id,
                    "generated_critique": [
                        "<|sep|>".join(result) for result in results
                    ],
                    "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
                    "time": elapsed,
                }
        for result in results:
            result.append("Final answer: \\boxed{-1}")
        elapsed = time.time() - start_time
        return {
            "id": id,
            "generated_critique": ["<|sep|>".join(result) for result in results],
            "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
            "time": elapsed,
        }


class LogicFlowVerifier(Verifier):
    def __init__(
        self,
        client: OpenaiClient,
        constructor_type: str = "targeted",
        level: str = "step",
        show_progress: bool = True,
    ):
        super().__init__(client, show_progress)
        self.constructor = AutoConstructor.from_type(constructor_type, client=client)
        self.level = level

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/STEPWISE_VERIFICATION.txt"
        )

    def _identify_root_nodes(self, graph: nx.DiGraph) -> List[int]:
        """Identify *root* nodes in the solution graph.

        A node *i* is a root node if **any** of its immediate successors is not
        *i + 1* (meaning the result of step *i* is reused later in the
        solution) or if *i* is the final step, or if *i* has two or more
        predecessors (a synthesis node).
        """
        if graph.number_of_nodes() == 0:
            return []

        root_nodes = set()
        for node in graph.nodes():
            succs = list(graph.successors(node))
            if len(succs) == 0 or any(succ != node + 1 for succ in succs):
                root_nodes.add(node)
            # elif graph.in_degree(node) >= 2 or graph.in_degree(node) == 0:
            #     root_nodes.add(node)

        return sorted(list(root_nodes))

    def _build_subgraph(
        self, graph: nx.DiGraph, root: int, root_lst: List[int]
    ) -> nx.DiGraph:
        """Construct the sub-graph rooted at *root*."""
        visited = set([root])
        stack = [root]
        while stack:
            current = stack.pop()
            for pred in graph.predecessors(current):
                if pred in visited:
                    continue
                visited.add(pred)
                # Stop traversing beyond other root nodes but still keep
                # them inside the current sub-graph (to provide context).
                if pred in root_lst and pred != root:
                    continue
                stack.append(pred)

        return graph.subgraph(visited).copy()

    def _verify_one_step(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        label: Optional[int] = None,
        **generation_kwargs,
    ) -> Tuple[int, str]:
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        root_lst = self._identify_root_nodes(solution_graph)

        for root in root_lst:
            subgraph = self._build_subgraph(solution_graph, root, root_lst)
            target_step_indices = sorted(
                [i for i in subgraph.nodes if subgraph.nodes[i]["state"] is None]
            )
            tagged_steps = []
            for i in sorted(solution_graph.nodes):
                if i in list(subgraph.nodes) or i == max(solution_graph.nodes):
                    tagged_steps.append(
                        f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                    )
                elif len(tagged_steps) == 0 or tagged_steps[-1] != "...":
                    tagged_steps.append("...")
            tagged_steps = "\n".join(tagged_steps)
            for step_idx in target_step_indices:
                user_input = self.prompt_template.format(
                    problem=problem,
                    tagged_steps=tagged_steps,
                    idx=step_idx,
                )
                step_results = self.client(
                    batch_messages=[[{"role": "user", "content": user_input}]],
                    **generation_kwargs,
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

    def _verify_one_subgraph(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        label: Optional[int] = None,
        **generation_kwargs,
    ) -> Tuple[int, str]:
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        root_lst = self._identify_root_nodes(solution_graph)

        for root in root_lst:
            subgraph = self._build_subgraph(solution_graph, root, root_lst)
            target_step_indices = sorted(
                [i for i in subgraph.nodes if subgraph.nodes[i]["state"] is None]
            )
            tagged_steps = []
            for i in sorted(solution_graph.nodes):
                if i in list(subgraph.nodes) or i == max(solution_graph.nodes):
                    tagged_steps.append(
                        f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                    )
                elif len(tagged_steps) == 0 or tagged_steps[-1] != "...":
                    tagged_steps.append("...")
            tagged_steps = "\n".join(tagged_steps)
            user_input = self.prompt_template.format(
                problem=problem,
                tagged_steps=tagged_steps,
                indices=", ".join(map(str, target_step_indices)),
            )
            subgraph_results = self.client(
                batch_messages=[[{"role": "user", "content": user_input}]],
                **generation_kwargs,
            )[0]
            for result, subgraph_result in zip(results, subgraph_results):
                result.append(subgraph_result)
            parsed = [
                parse_from_boxed(subgraph_result)
                for subgraph_result in subgraph_results
            ]
            majority, _ = Counter(parsed).most_common(1)[0]
            if majority != "-1":
                for result in results:
                    result.append(f"Final answer: \\boxed{{{majority}}}")
                return ["<|sep|>".join(result) for result in results]
            if label is not None and label in target_step_indices:
                for result in results:
                    result.append(r"Final answer: \boxed{None}")
                return ["<|sep|>".join(result) for result in results]
            for step_idx in target_step_indices:
                solution_graph.nodes[step_idx]["state"] = True

        for result in results:
            result.append("Final answer: \\boxed{-1}")
        return ["<|sep|>".join(result) for result in results]

    def _verify_one(self, id: int, sample: dict, **generation_kwargs) -> dict:
        start_time = time.time()
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
            **generation_kwargs,
        )
        if self.level == "step":
            generated_critique = self._verify_one_step(
                problem=sample["problem"],
                solution_graph=solution_graph,
                label=sample["label"],
                **generation_kwargs,
            )
        elif self.level == "subgraph":
            generated_critique = self._verify_one_subgraph(
                problem=sample["problem"],
                solution_graph=solution_graph,
                label=sample["label"],
                **generation_kwargs,
            )
        else:
            raise ValueError(f"Unknown level: {self.level}")
        elapsed = time.time() - start_time
        return {
            "id": id,
            "generated_critique": generated_critique,
            "graph": json.dumps(node_link_data(solution_graph, edges="edges")),
            "time": elapsed,
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
