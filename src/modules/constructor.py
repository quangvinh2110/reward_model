import math
import networkx as nx
from abc import ABC, abstractmethod
from openai import OpenAI

from ..utils.io import read_txt
from ..utils.parser import parse_from_json


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


class AbstractConstructor(ABC):
    def __init__(
        self,
        model: str,
        client: OpenAI,
    ):
        self.model = model
        self.client = client
        self.prompt_template = self._get_prompt_template()

    @abstractmethod
    def _get_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __call__(self, sample: dict, **generation_kwargs) -> nx.DiGraph:
        raise NotImplementedError("Subclasses must implement this method")


class TargetedConstructor(AbstractConstructor):

    def _get_prompt_template(self) -> str:
        return read_txt(r"E:\AAAI-26\resources\prompt_templates\TARGETED_TRACKING.txt")

    def _track_one_step(
        self,
        sample: dict,
        step_idx: int,
        graph: nx.DiGraph,
        max_window_size: int = 5,
        **generation_kwargs,
    ):
        if graph.nodes[step_idx]["resolved"]:
            return
        for start_idx, end_idx in group_index_generator(step_idx, max_window_size, 0):
            tagged_steps = "\n".join(
                f"<step_{i}>\n{sample['steps'][i]}\n</step_{i}>"
                for i in range(start_idx, end_idx)
            )
            target_step = (
                f"<step_{step_idx}>\n{sample['steps'][step_idx]}\n</step_{step_idx}>"
            )
            tracked_premises = "\n".join(
                f"<step_{i}>\n{graph.nodes[i]['content']}\n</step_{i}>"
                for i in graph.predecessors(step_idx)
            )
            user_input = self.prompt_template.format(
                problem=sample["problem"],
                tagged_steps=tagged_steps,
                target_step=target_step,
                tracked_premises=tracked_premises,
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_input}],
                **generation_kwargs,
            )
            output = response.choices[0].message.content
            output = parse_from_json(output)
            if "premises" not in output or not output["premises"]:
                continue
            for prem_idx in output["premises"]:
                graph.add_edge(prem_idx, step_idx)
            if "resolved" in output and output["resolved"]:
                graph.nodes[step_idx]["resolved"] = True
                break

    def __call__(
        self, sample: dict, max_window_size: int = 5, **generation_kwargs
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node(0, content=sample["steps"][0], resolved=True)
        for step_idx in range(1, len(sample["steps"])):
            graph.add_node(step_idx, content=sample["steps"][step_idx], resolved=False)
            self._track_one_step(
                sample, step_idx, graph, max_window_size, **generation_kwargs
            )
        return graph


class GroupedConstructor(AbstractConstructor):
    def _get_prompt_template(self) -> str:
        return read_txt(r"E:\AAAI-26\resources\prompt_templates\GROUPED_TRACKING.txt")

    def __call__(self, sample: dict, **generation_kwargs) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_node(0, content=sample["steps"][0], resolved=True)
        tagged_steps = "\n".join(
            f"<step_{i}>\n{step}\n</step_{i}>" for i, step in enumerate(sample["steps"])
        )
        user_input = self.prompt_template.format(
            problem=sample["problem"],
            tagged_steps=tagged_steps,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_input}],
            **generation_kwargs,
        )
        output = response.choices[0].message.content
        output = parse_from_json(output)
        for step_idx in range(1, len(sample["steps"])):
            graph.add_node(step_idx, content=sample["steps"][step_idx], resolved=False)
            if str(step_idx) not in output:
                continue
            if (
                "premises" not in output[str(step_idx)]
                or not output[str(step_idx)]["premises"]
            ):
                continue
            for prem_idx in output[str(step_idx)]["premises"]:
                if prem_idx < step_idx:
                    graph.add_edge(prem_idx, step_idx)
            if "resolved" in output[str(step_idx)]:
                graph.nodes[step_idx]["resolved"] = True
        return graph


class HybridConstructor:
    def __init__(
        self,
        model: str,
        client: OpenAI,
    ):
        self.client = client
        self.model = model
        pass

    def _construct_subgraph(
        self, sample: dict, start: int, end: int, graph: nx.DiGraph, **generation_kwargs
    ):
        tagged_steps = "\n".join(
            f"<step_{i}>\n{sample['steps'][i]}\n</step_{i}>" for i in range(start, end)
        )
        user_input = self.grouped_prompt_template.format(
            problem=sample["problem"],
            tagged_steps=tagged_steps,
        )
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            **generation_kwargs,
        )
        output = response.choices[0].message.content
        output = parse_from_json(output)
        for step_idx in range(start, end):
            if step_idx not in output:
                continue
            if "premises" not in output[step_idx] or not output[step_idx]["premises"]:
                continue
            for prem_idx in output[step_idx]["premises"]:
                if prem_idx < step_idx:
                    graph.add_edge(prem_idx, step_idx)
            if "resolved" in output[step_idx] and output[step_idx]["resolved"]:
                graph.nodes[step_idx]["resolved"] = True

    def __call__(self, sample: dict, **generation_kwargs) -> nx.DiGraph:
        max_window_size = generation_kwargs.pop("max_window_size", len(sample["steps"]))
        overlap_size = generation_kwargs.pop("overlap_size", 1)
        graph = nx.DiGraph()
        for step_idx in range(len(sample["steps"])):
            graph.add_node(step_idx, content=sample["steps"][step_idx], resolved=False)
        graph.nodes[0]["resolved"] = True
        for start_idx, end_idx in group_index_generator(
            len(sample["steps"]), max_window_size, overlap_size
        ):
            self._construct_subgraph(
                sample, start_idx, end_idx, graph, **generation_kwargs
            )
        for step_idx in range(1, len(sample["steps"])):
            if graph.nodes[step_idx]["resolved"]:
                continue
            self._construct_subgraph(
                sample, step_idx, step_idx + 1, graph, **generation_kwargs
            )
        return graph


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
