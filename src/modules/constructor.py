import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from ..utils.io import read_txt
from transformers import AutoTokenizer


@dataclass
class Node:
    index: int
    content: str
    dependencies: List["Node"] = field(default_factory=list)
    resolved: bool = False


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
