from typing import List, Tuple
import re
from ..utils import read_txt
from .base_model import BaseGenerativeModel


class MonolithicGenerativeTracker(BaseGenerativeModel):
    pass


class PolylithicGenerativeTracker(BaseGenerativeModel):
    """A generative tracker that predicts dependencies between solution steps.

    This model takes a problem, a current step, and a list of previous steps, and predicts
    the dependencies of the current step.
    """

    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        return read_txt("/absolute/path/to/DEPENDENCY_TRACKER_ONE.txt")

    def _format_prompt(self, problem: str, solution: List[str]) -> str:
        """Format the prompt for tracking dependencies between solution steps.

        Args:
            problem (str): The math problem
            solution (List[str]): List of solution steps

        Returns:
            List[str]: List of formatted prompts, one per solution step
        """
        prompts = []
        for i in range(len(solution)):
            tagged_previous_steps = "\n".join(
                [f"[{j}] {step}" for j, step in enumerate(solution[:i])]
            )
            tagged_current_step = f"[{i}] {solution[i]}"
            user_prompt = self.prompt_template.format(
                problem=problem,
                tagged_previous_steps=tagged_previous_steps,
                tagged_current_step=tagged_current_step,
            )
            prompts.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return prompts

    def __call__(
        self,
        problem_solution_pairs: List[Tuple[str, List[str]]],
        **generation_kwargs,
    ) -> List[int]:
        """Predict dependencies for the current step.

        Args:
            problem (str): The math problem
            current_step (str): The current step to analyze
            previous_steps (List[str]): List of previous steps to consider
            **generation_kwargs: Additional generation parameters

        Returns:
            List[int]: List of indices (0-based) of steps that the current step depends on
        """
        # Format prompt
        prompts = [
            self._format_prompt(problem, solution)
            for problem, solution in problem_solution_pairs
        ]

        # Flatten prompts while keeping track of which problem they belong to
        idx = [
            i for i, prompt_list in enumerate(prompts) for _ in range(len(prompt_list))
        ]
        flat_prompts = [prompt for prompt_list in prompts for prompt in prompt_list]

        # Generate dependencies
        outputs = self.generate(flat_prompts, **generation_kwargs)

        # Reformat outputs back into per-problem lists
        reformatted_outputs = []
        current_problem_outputs = []

        for i, output in enumerate(outputs):
            if i > 0 and idx[i] == idx[i - 1]:
                # Merge outputs for the same problem
                current_problem_outputs = [
                    prev + "<|sep|>" + curr
        return outputs
