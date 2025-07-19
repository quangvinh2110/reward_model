from typing import List, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
import networkx as nx

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
        for step_idx in range(len(sample["steps"])):
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
                return (id, ["<|sep|>".join(result) for result in results])
            if step_idx == sample["label"]:
                for result in results:
                    result.append(r"Final answer: \boxed{None}")
                return (id, ["<|sep|>".join(result) for result in results])
            for result in results:
                result.append("Final answer: \\boxed{-1}")
        return (id, ["<|sep|>".join(result) for result in results])


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

    def _verify_one(
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
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
                return (id, ["<|sep|>".join(result) for result in results])
            if step_idx == sample["label"]:
                for result in results:
                    result.append(r"Final answer: \boxed{None}")
                return (id, ["<|sep|>".join(result) for result in results])
        for result in results:
            result.append("Final answer: \\boxed{-1}")
        return (id, ["<|sep|>".join(result) for result in results])


class LogicFlowVerifier(Verifier):
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

        last_idx = max(graph.nodes())
        root_nodes = set()
        for node in graph.nodes():
            succs = list(graph.successors(node))
            if any(succ != node + 1 for succ in succs):
                root_nodes.add(node)
            # elif graph.in_degree(node) >= 2 or graph.in_degree(node) == 0:
            #     root_nodes.add(node)

        root_nodes.add(last_idx)
        # root_nodes.add(0)
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

    def _verify_one(
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
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
        root_lst = self._identify_root_nodes(solution_graph)

        for root in root_lst:
            subgraph = self._build_subgraph(solution_graph, root, root_lst)
            target_step_indices = sorted(
                [i for i in subgraph.nodes if subgraph.nodes[i]["state"] is None]
            )
            # tagged_steps = "\n".join(
            #     f"<step_{i}>\n{subgraph.nodes[i]['content']}\n</step_{i}>"
            #     for i in sorted(subgraph.nodes)
            # )
            for step_idx in target_step_indices:
                tagged_steps = []
                tagged_step_indices = [subgraph.predecessors(step_idx)] + [step_idx]
                for i in sorted(subgraph.nodes):
                    if i == tagged_step_indices[-1] + 1:
                        tagged_step_indices.apppend(i)
                if max(solution_graph.nodes) not in tagged_step_indices:
                    tagged_step_indices.append(max(solution_graph.nodes))
                for i in sorted(solution_graph.nodes):
                    if i in tagged_step_indices:
                        tagged_steps.append(
                            f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                        )
                    elif len(tagged_steps) == 0 or tagged_steps[-1] != "...":
                        tagged_steps.append("...")
                tagged_steps = "\n".join(tagged_steps)
                user_input = self.prompt_template.format(
                    problem=sample["problem"],
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
                    return (id, ["<|sep|>".join(result) for result in results])
                if step_idx == sample["label"]:
                    for result in results:
                        result.append(r"Final answer: \boxed{None}")
                    return (id, ["<|sep|>".join(result) for result in results])
        for result in results:
            result.append("Final answer: \\boxed{-1}")
        return (id, ["<|sep|>".join(result) for result in results])


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
