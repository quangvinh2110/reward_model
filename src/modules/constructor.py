import math
import networkx as nx
from typing import Optional

from ..utils.io import read_txt
from ..utils.data import parse_from_json, to_int
from .client import OpenaiClient


def group_index_generator(
    n: int,
    max_window_size: int,
    overlap_size: int,
):
    if max_window_size >= n:
        yield 0, n
        return
    if max_window_size < overlap_size:
        raise ValueError("max_window_size must be greater than overlap_size")

    n_groups = max(1, math.ceil((n - overlap_size) / (max_window_size - overlap_size)))
    group_size = (n + (n_groups - 1) * overlap_size) // n_groups
    remainder = (n + (n_groups - 1) * overlap_size) % n_groups
    end = start = n
    while start > 0:
        if remainder > 0:
            start = max(0, end - group_size - 1)
            yield start, end
            end = start + overlap_size
            remainder -= 1
        else:
            start = max(0, end - group_size)
            yield start, end
            end = start + overlap_size


class TargetedConstructor:

    def __init__(
        self,
        client: OpenaiClient,
    ):
        self.client = client
        self.prompt_template = read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/TARGETED_TRACKING.txt"
        )

    def _track_one_step(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        step_idx: int,
        max_window_size: int,
        **generation_kwargs,
    ):
        if step_idx == 0:
            return
        if solution_graph.nodes[step_idx]["resolved"]:
            return
        for start_idx, end_idx in group_index_generator(step_idx, max_window_size, 0):
            tagged_steps = "\n".join(
                f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                for i in range(start_idx, end_idx)
            )
            target_step = f"<step_{step_idx}>\n{solution_graph.nodes[step_idx]['content']}\n</step_{step_idx}>"
            tracked_premises = "\n".join(
                f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                for i in solution_graph.predecessors(step_idx)
            )
            user_input = self.prompt_template.format(
                problem=problem,
                tagged_steps=tagged_steps,
                target_step=target_step,
                tracked_premises=tracked_premises,
            )
            response = self.client(
                batch_messages=[[{"role": "user", "content": user_input}]],
                **generation_kwargs,
            )[0][0]
            output = parse_from_json(response)
            if "premises" not in output or not output["premises"]:
                continue
            for prem_idx in output["premises"]:
                prem_idx = to_int(prem_idx)
                if prem_idx in range(start_idx, step_idx):
                    solution_graph.add_edge(prem_idx, step_idx)
            if "resolved" in output and output["resolved"]:
                solution_graph.nodes[step_idx]["resolved"] = True
                break

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        target_idx: Optional[int] = None,
        max_window_size: Optional[int] = None,
        **generation_kwargs,
    ) -> nx.DiGraph:
        max_window_size = (
            max_window_size if max_window_size else len(solution_graph.nodes)
        )
        if target_idx:
            self._track_one_step(
                problem,
                solution_graph,
                target_idx,
                max_window_size=max_window_size,
                **generation_kwargs,
            )
            return solution_graph
        for step_idx in range(len(solution_graph.nodes)):
            self._track_one_step(
                problem,
                solution_graph,
                step_idx,
                max_window_size=max_window_size,
                **generation_kwargs,
            )
        return solution_graph


class GroupedConstructor:

    def __init__(
        self,
        client: OpenaiClient,
    ):
        self.client = client
        self.prompt_template = read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/GROUPED_TRACKING.txt"
        )

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        **generation_kwargs,
    ) -> nx.DiGraph:
        start_idx = start_idx if start_idx else 0
        end_idx = end_idx if end_idx else len(solution_graph.nodes)
        tagged_steps = "\n".join(
            f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
            for i in range(start_idx, end_idx)
        )
        user_input = self.prompt_template.format(
            problem=problem,
            tagged_steps=tagged_steps,
        )
        response = self.client(
            batch_messages=[[{"role": "user", "content": user_input}]],
            **generation_kwargs,
        )[0][0]
        output = parse_from_json(response)
        for step_idx in range(start_idx, end_idx):
            if str(step_idx) not in output:
                continue
            if (
                "premises" not in output[str(step_idx)]
                or not output[str(step_idx)]["premises"]
            ):
                continue
            for prem_idx in output[str(step_idx)]["premises"]:
                prem_idx = to_int(prem_idx)
                if prem_idx in range(start_idx, step_idx):
                    solution_graph.add_edge(prem_idx, step_idx)
            if "resolved" in output[str(step_idx)]:
                solution_graph.nodes[step_idx]["resolved"] = True
        return solution_graph


class HybridConstructor:
    def __init__(
        self,
        client: OpenaiClient,
    ):
        self.client = client
        self.grouped_constructor = GroupedConstructor(client)
        self.targeted_constructor = TargetedConstructor(client)

    def _construct_local(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        start: int,
        end: int,
        **generation_kwargs,
    ):
        pass

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        **generation_kwargs,
    ) -> nx.DiGraph:
        pass


class AutoConstructor:
    TYPE_MAP = {
        "targeted": TargetedConstructor,
        "grouped": GroupedConstructor,
        "hybrid": HybridConstructor,
    }

    @classmethod
    def from_type(cls, constructor_type: str, **kwargs):
        constructor_type = constructor_type.lower()
        if constructor_type not in cls.TYPE_MAP:
            raise ValueError(f"Unknown constructor type: {constructor_type}")
        return cls.TYPE_MAP[constructor_type](**kwargs)
