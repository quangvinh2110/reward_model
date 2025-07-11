import json
import math
import networkx as nx
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from transformers import AutoTokenizer

from ..utils.io import read_txt
from ..utils.parser import parse_from_json
from .client import AutoLlmClient


def group_index_generator(
    n: int,
    max_group_size: int,
    overlap_size: int,
):
    if max_group_size >= n:
        yield 0, n
        return
    if max_group_size < overlap_size:
        raise ValueError("max_group_size must be greater than overlap_size")

    n_groups = max(1, math.ceil((n - overlap_size) / (max_group_size - overlap_size)))
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
        model_name_or_path: str,
        endpoint: str,
        client_type: str = "openai",
        served_model_name: Optional[str] = None,
        show_progress: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.prompt_template = self._get_prompt_template()
        self.show_progress = show_progress
        self.client = AutoLlmClient.from_type(
            client_type=client_type,
            endpoint=endpoint,
            served_model_name=served_model_name,
        )

    @abstractmethod
    def _get_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def __call__(self, sample: dict, **generation_kwargs) -> List[Node]:
        raise NotImplementedError("Subclasses must implement this method")


class TargetedConstructor(AbstractConstructor):

    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/DEPENDENCY_TRACKING_ONE.txt"
        )

    def __call__(self, sample: dict, **generation_kwargs) -> nx.DiGraph:
        group_size = generation_kwargs.pop("group_size", len(sample["steps"]))
        enable_thinking = generation_kwargs.pop("enable_thinking", False)
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking:
            chat_template_kwargs["enable_thinking"] = True
        graph = nx.DiGraph()
        graph.add_node(0, content=sample["steps"][0], resolved=True)
        for step_idx in range(1, len(sample["steps"])):
            graph.add_node(step_idx, content=sample["steps"][step_idx], resolved=False)
            for start, end in group_index_generator(step_idx, group_size, 0):
                tagged_steps = "\n".join(
                    [
                        f"<step_{i}>\n{sample['steps'][i]}\n</step_{i}>"
                        for i in range(start, end)
                    ]
                )
                tracked_premises = "\n".join(
                    [
                        f"<step_{i}>\n{graph.nodes[i]['content']}\n</step_{i}>"
                        for i in graph.predecessors(step_idx)
                    ]
                )
                user_input = self.prompt_template.format(
                    problem=sample["problem"],
                    tagged_steps=tagged_steps,
                    target_step=sample["steps"][step_idx],
                    tracked_premises=tracked_premises,
                )
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_input}], **chat_template_kwargs
                )
                output = self.client([prompt], **generation_kwargs)[0][0]
                output = parse_from_json(output)
                if "premises" not in output or not output["premises"]:
                    continue
                for prem_idx in output["premises"]:
                    graph.add_edge(prem_idx, step_idx)
                if "resolved" in output and output["resolved"]:
                    graph.nodes[step_idx]["resolved"] = True
                    break
        return graph


class DyadicConstructor(AbstractConstructor):
    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/DEPENDENCY_TRACKING_ALL.txt"
        )

    def __call__(self, sample: dict, **generation_kwargs) -> nx.DiGraph:
        pass


class GroupedConstructor(AbstractConstructor):
    def _get_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/DEPENDENCY_TRACKING_ALL.txt"
        )

    def __call__(self, sample: dict, **generation_kwargs) -> nx.DiGraph:
        pass


class Constructor:
    def __init__(
        self,
        model_name_or_path: str,
        endpoint: str,
        provider: str = "openai",
        served_model_name: Optional[str] = None,
        group_tracking_prompt_template_path: Optional[str] = None,
        targeted_tracking_prompt_template_path: Optional[str] = None,
        group_size: int = 4,
        overlap_size: int = 1,
        show_progress: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.group_size = group_size
        self.overlap_size = overlap_size
        self.show_progress = show_progress
        # Load two prompt templates
        if group_tracking_prompt_template_path:
            self.group_tracking_prompt_template = read_txt(
                group_tracking_prompt_template_path
            )
        else:
            self.group_tracking_prompt_template = read_txt(
                "/raid/vinh/reward_model/resources/prompt_templates/DEPENDENCY_TRACKING_ALL.txt"
            )
        if targeted_tracking_prompt_template_path:
            self.targeted_tracking_prompt_template = read_txt(
                targeted_tracking_prompt_template_path
            )
        else:
            self.targeted_tracking_prompt_template = read_txt(
                "/raid/vinh/reward_model/resources/prompt_templates/DEPENDENCY_TRACKING_ONE.txt"
            )
        if provider == "openai":
            from .client import OpenAIClient

            self.client = OpenAIClient(
                endpoint=endpoint, served_model_name=served_model_name
            )
        elif provider == "huggingface":
            from .client import HuggingFaceClient

            self.client = HuggingFaceClient(
                endpoint=endpoint, served_model_name=served_model_name
            )
        else:
            raise ValueError(f"Invalid provider: {provider}")

    def _format_steps(self, steps: List[str], offset: int = 0) -> str:
        return "\n".join([f"[{i+offset}] {step}" for i, step in enumerate(steps)])

    def _distribute_steps(self, total_steps: int) -> List[List[int]]:
        step_indices = list(range(total_steps))
        # Initial grouping
        groups = []
        start = 0
        while start < total_steps:
            end = min(start + self.group_size, total_steps)
            groups.append(step_indices[start:end])
            start = end - self.overlap_size
        # Check if redistribution is needed
        if len(groups) > 1:
            last_group_size = len(groups[-1])
            if self.group_size - last_group_size >= 2:
                # Redistribute elements from previous groups
                groups = self._redistribute_steps(groups)
        return groups

    def _redistribute_steps(self, groups: List[List[int]]) -> List[List[int]]:
        # Work backwards from the last group
        current_group_idx = len(groups) - 1
        while current_group_idx > 0:
            current_group = groups[current_group_idx]
            prev_group = groups[current_group_idx - 1]
            # If current group is too small, take one element from previous group
            if len(current_group) < self.group_size - 1:  # Allow some flexibility
                # Remove one element from the end of previous group (maintaining overlap)
                if len(prev_group) > self.overlap_size:
                    moved_element = prev_group.pop()
                    # Insert at the beginning of current group
                    current_group.insert(0, moved_element)
                    # Update the groups list
                    groups[current_group_idx] = current_group
                    groups[current_group_idx - 1] = prev_group
                else:
                    # Can't take more from previous group, try earlier groups
                    break
            else:
                # Current group is now balanced enough
                break

            current_group_idx -= 1

        return groups

    def _build_groups(self, nodes: List[Node]) -> List[List[Node]]:
        # Get the step distribution
        step_groups = self._distribute_steps(len(nodes))

        # Convert to the expected format with start, end, and nodes
        groups = []
        for step_group in step_groups:
            if step_group:  # Ensure group is not empty
                start = step_group[0]
                end = step_group[-1] + 1
                group_nodes = [nodes[i] for i in step_group]
                groups.append({"start": start, "end": end, "nodes": group_nodes})

        return groups

    def _build_group_prompts(self, nodes: List[Node]) -> List[str]:
        pass

    def _build_targeted_prompts(self, nodes: List[Node]) -> List[str]:
        pass

    def _parse_json_from_output(self, output: str) -> Optional[Dict]:
        try:
            json_strs = [s for s in output.split("```json") if "}" in s]
            if not json_strs:
                json_strs = [s for s in output.split("````") if "}" in s]
            if not json_strs:
                json_strs = [s for s in output.split("```") if "}" in s]
            if not json_strs:
                return None
            json_part = json_strs[-1]
            json_part = json_part.split("```", 1)[0]
            json_part = (
                json_part.replace("\n", " ").replace(", }", "}").replace(",]", "]")
            )
            json_part = json_part[: json_part.rfind("}") + 1]
            json_part = json_part.replace("{ ", "{").replace(" }", "}")
            try:
                return json.loads(json_part)
            except Exception:
                return eval(json_part)
        except Exception:
            return None

    def build_dag(
        self, problem: str, steps: List[str], max_iters: int = 3, **generation_kwargs
    ) -> Dict[int, List[int]]:
        n = len(steps)
        nodes = [Node(i, step) for i, step in enumerate(steps)]
        for _ in range(max_iters):
            groups = self._build_groups(nodes)
            prompts = []
            group_indices = []
            for group in groups:
                tagged_steps = self._format_steps(
                    [node.content for node in group["nodes"]], offset=group["start"]
                )
                prompt = self.group_tracking_prompt_template.format(
                    problem=problem, tagged_steps=tagged_steps
                )
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                group_indices.append((group["start"], group["end"]))
            outputs = self.client(prompts, **generation_kwargs)
            for (start, end), output_list in zip(group_indices, outputs):
                output = (
                    output_list[0] if isinstance(output_list, list) else output_list
                )
                dep_json = self._parse_json_from_output(output)
                if not dep_json:
                    continue
                for i in range(start, end):
                    if str(i) in dep_json:
                        nodes[i].dependencies = sorted(
                            set(nodes[i].dependencies) | set(dep_json[str(i)])
                        )
                        if nodes[i].dependencies or i == 0:
                            nodes[i].resolved = True
            if all(node.resolved for node in nodes):
                break
            unresolved_nodes = [node for node in nodes if not node.resolved]
            if not unresolved_nodes:
                break
            # For each unresolved node, expand group context and use dependency tracking prompt
            for node in unresolved_nodes:
                # Expand context: include up to subgraph_size previous steps
                start = max(0, node.index - self.group_size + 1)
                end = node.index + 1
                context_nodes = nodes[start:end]
                tagged_steps = self._format_steps(
                    [n.content for n in context_nodes], offset=start
                )
                prompt = self.targeted_tracking_prompt_template.format(
                    problem=problem, tagged_steps=tagged_steps
                )
                dep_output = self.client(
                    [
                        self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    ],
                    **generation_kwargs,
                )[0]
                dep_json = self._parse_json_from_output(
                    dep_output[0] if isinstance(dep_output, list) else dep_output
                )
                if dep_json and str(node.index) in dep_json:
                    node.dependencies = sorted(
                        set(node.dependencies) | set(dep_json[str(node.index)])
                    )
                    if node.dependencies or node.index == 0:
                        node.resolved = True
            self.group_size = min(self.group_size + 1, n)
        # Return as adjacency list
        return {node.index: node.dependencies for node in nodes}
