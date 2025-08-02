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

    def __call__(
        self,
        samples: List[dict],
        num_workers: int = 4,
        construction_kwargs: dict = {},
        generation_kwargs: dict = {},
    ) -> List[dict]:
        """Verify multiple samples using multiprocessing, or sequentially if num_workers == 1. Returns a list of dicts."""
        if not samples:
            return []
        tasks = [
            (self._verify_one, i, sample, construction_kwargs, generation_kwargs)
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
        return read_txt("/your/path/to/prompt_templates/SEQUENTIAL_VERIFICATION.txt")

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
        return read_txt("/your/path/to/prompt_templates/LOGICFLOW_VERIFICATION.txt")

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
        self,
        graph: nx.DiGraph,
        target: int,
        root_lst: List[int],
        type: str = "backward",
    ) -> List[int]:
        """Construct the sub-graph rooted at *root* or traversed from target as described."""
        visited = set([target])
        if type == "backward":
            stack = [target]
            while stack:
                current = stack.pop()
                for pred in graph.predecessors(current):
                    if pred in visited:
                        continue
                    visited.add(pred)
                    # Stop traversing beyond other root nodes but still keep them inside the current sub-graph (to provide context).
                    if pred not in root_lst:
                        stack.append(pred)
            # All visited nodes are in the subgraph
        elif type == "forward":
            current = target
            while True:
                next_node = current + 1
                if next_node in graph.successors(current):
                    visited.add(next_node)
                    if next_node in root_lst:
                        break
                    current = next_node
                else:
                    # Stop if next_node is not a successor or is a root
                    break
        return list(visited)

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
            elif self.prefix == "subgraph":
                root_nodes = self._identify_root_nodes(solution_graph)
                prefix_indices = sorted(
                    self._build_subgraph(
                        solution_graph, step_idx, root_nodes, "backward"
                    ),
                    reverse=True,
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
            elif self.suffix == "subgraph":
                root_nodes = self._identify_root_nodes(solution_graph)
                suffix_indices = sorted(
                    self._build_subgraph(
                        solution_graph, step_idx, root_nodes, "forward"
                    ),
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
