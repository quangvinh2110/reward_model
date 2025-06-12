from typing import List
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

    def _parse_dependencies(self, generated_text: str) -> List[int]:
        """Parse the generated text to extract step dependencies.

        Expected format: "Step t depends on steps: 1, 3, 5"
        Returns list of step indices (0-based)
        """
        # Extract numbers after "steps:"
        pattern = r"steps:\s*([\d,\s]+)"
        match = re.search(pattern, generated_text)
        if not match:
            return []

        # Convert to list of integers (0-based)
        try:
            return [int(x.strip()) - 1 for x in match.group(1).split(",")]
        except ValueError:
            return []

    def __call__(
        self,
        problem: str,
        current_step: str,
        previous_steps: List[str],
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
        prompt = self._format_prompt(problem, current_step, previous_steps)

        # Generate and parse dependencies
        generated = self.generate([prompt], **generation_kwargs)[0]
        return self._parse_dependencies(generated)
