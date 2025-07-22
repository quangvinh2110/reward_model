import math
import networkx as nx
from typing import Optional, List

from ..utils.io import read_txt
from ..utils.data import parse_from_json, to_int
from .client import Client


def group_index_generator(
    idx_list: List[int],
    max_window_size: int,
    overlap_size: int,
    reverse: bool = True,
):
    if max_window_size >= len(idx_list):
        yield idx_list
        return
    if max_window_size < overlap_size:
        raise ValueError("max_window_size must be greater than overlap_size")

    n_groups = max(
        1, math.ceil((len(idx_list) - overlap_size) / (max_window_size - overlap_size))
    )
    group_size = (len(idx_list) + (n_groups - 1) * overlap_size) // n_groups
    remainder = (len(idx_list) + (n_groups - 1) * overlap_size) % n_groups
    end = start = len(idx_list)
    idx_list = idx_list[::-1] if not reverse else idx_list
    while start > 0:
        if remainder > 0:
            start = max(0, end - group_size - 1)
            yield idx_list[start:end]
            end = start + overlap_size
            remainder -= 1
        else:
            start = max(0, end - group_size)
            yield idx_list[start:end]
            end = start + overlap_size


class NullConstructor:
    def __init__(self, client: Client):
        self.client = client

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        construction_kwargs: dict = {},
    ) -> nx.DiGraph:
        return solution_graph


class TargetedConstructor:

    def __init__(
        self,
        client: Client,
    ):
        self.client = client
        self.prompt_template = read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/TARGETED_TRACKING_V2.txt"
        )

    def _track_one_step(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        step_idx: int,
        candidate_idx_list: Optional[List[int]] = None,
        max_window_size: int = 5,
        generation_kwargs: Optional[dict] = None,
    ):
        if step_idx == 0:
            return
        if solution_graph.nodes[step_idx]["resolved"]:
            return
        if candidate_idx_list:
            candidate_idx_list = sorted(
                i
                for i in candidate_idx_list
                if i < step_idx and i in solution_graph.nodes
            )
        else:
            candidate_idx_list = sorted(i for i in solution_graph.nodes if i < step_idx)
        if not candidate_idx_list:
            return
        for group_idx_list in group_index_generator(
            candidate_idx_list, max_window_size, 0, reverse=True
        ):
            tagged_steps = "\n".join(
                f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
                for i in group_idx_list
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
                generation_kwargs=generation_kwargs,
            )[0][0]
            output = parse_from_json(response)
            if "premises" not in output or not output["premises"]:
                continue
            for prem_idx in output["premises"]:
                prem_idx = to_int(prem_idx)
                if prem_idx in group_idx_list:
                    solution_graph.add_edge(prem_idx, step_idx)
            if "resolved" in output and output["resolved"]:
                solution_graph.nodes[step_idx]["resolved"] = True
                break

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        target_idx: Optional[int] = None,
        construction_kwargs: dict = {},
        generation_kwargs: dict = {},
    ) -> nx.DiGraph:
        max_window_size = construction_kwargs.get(
            "max_window_size", len(solution_graph.nodes)
        )
        generation_kwargs["n"] = 1
        if target_idx:
            self._track_one_step(
                problem=problem,
                solution_graph=solution_graph,
                step_idx=target_idx,
                max_window_size=max_window_size,
                generation_kwargs=generation_kwargs,
            )
            return solution_graph
        for step_idx in range(len(solution_graph.nodes)):
            self._track_one_step(
                problem=problem,
                solution_graph=solution_graph,
                step_idx=step_idx,
                max_window_size=max_window_size,
                generation_kwargs=generation_kwargs,
            )
        return solution_graph


class GroupedConstructor:

    def __init__(
        self,
        client: Client,
    ):
        self.client = client
        self.prompt_template = read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/GROUPED_TRACKING.txt"
        )

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        group_idx_list: Optional[List[int]] = None,
        generation_kwargs: dict = {},
    ) -> nx.DiGraph:
        if group_idx_list:
            group_idx_list = sorted(
                i for i in group_idx_list if i in solution_graph.nodes
            )
        else:
            group_idx_list = sorted(i for i in solution_graph.nodes)
        tagged_steps = "\n".join(
            f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
            for i in group_idx_list
        )
        user_input = self.prompt_template.format(
            problem=problem,
            tagged_steps=tagged_steps,
        )
        response = self.client(
            batch_messages=[[{"role": "user", "content": user_input}]],
            generation_kwargs=generation_kwargs,
        )[0][0]
        output = parse_from_json(response)
        for step_idx in group_idx_list:
            if str(step_idx) not in output:
                continue
            if (
                "premises" not in output[str(step_idx)]
                or not output[str(step_idx)]["premises"]
            ):
                continue
            for prem_idx in output[str(step_idx)]["premises"]:
                prem_idx = to_int(prem_idx)
                if prem_idx in group_idx_list and prem_idx < step_idx:
                    solution_graph.add_edge(prem_idx, step_idx)
            if (
                "resolved" in output[str(step_idx)]
                and output[str(step_idx)]["resolved"]
            ):
                solution_graph.nodes[step_idx]["resolved"] = True
        return solution_graph


class HybridConstructor:
    def __init__(
        self,
        client: Client,
    ):
        self.client = client
        self.grouped_constructor = GroupedConstructor(client)
        self.targeted_constructor = TargetedConstructor(client)

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        construction_kwargs: dict = {},
        generation_kwargs: dict = {},
    ) -> nx.DiGraph:
        max_window_size = construction_kwargs.get("max_window_size", 5)
        overlap_size = construction_kwargs.get("overlap_size", 1)
        for group_idx_list in group_index_generator(
            sorted(list(solution_graph.nodes)), max_window_size, overlap_size, False
        ):
            self.grouped_constructor(
                problem=problem,
                solution_graph=solution_graph,
                group_idx_list=group_idx_list,
                generation_kwargs=generation_kwargs,
            )
            remaining_idx_list = [
                i for i in solution_graph.nodes if i not in group_idx_list
            ]
            for step_idx in group_idx_list:
                if solution_graph.nodes[step_idx]["resolved"]:
                    continue
                self.targeted_constructor._track_one_step(
                    problem=problem,
                    solution_graph=solution_graph,
                    step_idx=step_idx,
                    candidate_idx_list=remaining_idx_list,
                    max_window_size=max_window_size * 2,
                    generation_kwargs=generation_kwargs,
                )
                solution_graph.nodes[step_idx]["resolved"] = True
        return solution_graph


class DacConstructor:
    def __init__(
        self,
        client: Client,
    ):
        self.client = client
        self.targeted_constructor = TargetedConstructor(client)
        self.prompt_template = read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/DAC_TRACKING.txt"
        )

    def __call__(
        self,
        problem: str,
        solution_graph: nx.DiGraph,
        generation_kwargs: dict = {},
    ) -> nx.DiGraph:
        tagged_steps = "\n".join(
            f"<step_{i}>\n{solution_graph.nodes[i]['content']}\n</step_{i}>"
            for i in sorted(solution_graph.nodes)
        )
        user_input = self.prompt_template.format(
            problem=problem, tagged_steps=tagged_steps
        )
        response = self.client(
            batch_messages=[[{"role": "user", "content": user_input}]],
            generation_kwargs=generation_kwargs,
        )[0][0]
        try:
            root_indices = parse_from_json(response)["root_steps"]
            root_indices = sorted(to_int(i) for i in root_indices)
            if max(solution_graph.nodes) not in root_indices:
                root_indices.append(max(solution_graph.nodes))
            groups = []
            for i in range(len(root_indices) - 1):
                group = [
                    idx
                    for idx in solution_graph.nodes
                    if idx >= root_indices[i] and idx <= root_indices[i + 1]
                ]
                for root_idx in root_indices[:i]:
                    group.append(root_idx)
                group = sorted(group)
                groups.append(group)
        except:
            groups = [sorted(solution_graph.nodes)]
        for group_idx_list in groups:
            for i in range(len(group_idx_list)):
                if solution_graph.nodes[group_idx_list[i]]["resolved"]:
                    continue
                self.targeted_constructor._track_one_step(
                    problem=problem,
                    solution_graph=solution_graph,
                    step_idx=group_idx_list[i],
                    candidate_idx_list=group_idx_list[:i],
                    max_window_size=len(group_idx_list),
                    generation_kwargs=generation_kwargs,
                )
                solution_graph.nodes[group_idx_list[i]]["resolved"] = True
        return solution_graph


class AutoConstructor:
    TYPE_MAP = {
        "targeted": TargetedConstructor,
        "grouped": GroupedConstructor,
        "hybrid": HybridConstructor,
        "dac": DacConstructor,
        "null": NullConstructor,
    }

    @classmethod
    def from_type(cls, constructor_type: str, **kwargs):
        constructor_type = constructor_type.lower()
        if constructor_type not in cls.TYPE_MAP:
            raise ValueError(f"Unknown constructor type: {constructor_type}")
        return cls.TYPE_MAP[constructor_type](**kwargs)
