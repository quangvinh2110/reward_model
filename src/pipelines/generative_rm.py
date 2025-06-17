from typing import Tuple, List
from ..utils import read_txt, extract_from_boxed
from ..modules.base_model import BaseGenerativeModel


class MonolithicGenerativeRM(BaseGenerativeModel):
    """A generative reward model that verifies full solutions all at once.

    This model takes a problem and its complete solution, then generates a single
    critique for the entire solution. It inherits from BaseGenerativeModel and
    supports all its backend options.
    """

    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/CRITIQUE_ALL.txt"
        )

    def _format_prompt(self, problem: str, solution: List[str]) -> str:
        """Format the prompt for a single problem-solution pair.

        Args:
            problem (str): The problem text
            solution (List[str]): List of solution steps

        Returns:
            str: Formatted prompt ready for generation
        """
        tagged_response = "\n".join(
            [
                f"<paragraph_{i}>\n{step}</paragraph_{i}>\n\n"
                for i, step in enumerate(solution)
            ]
        )
        user_prompt = self.prompt_template.format(
            problem=problem, tagged_response=tagged_response
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(
        self, problem_solution_pairs: List[Tuple[str, List[str]]], **generation_kwargs
    ) -> List[str]:
        """Generate critiques for the given problem-solution pairs.

        Args:
            problem_solution_pairs (List[Tuple[str, List[str]]]): List of (problem, solution) pairs
            **generation_kwargs: Additional generation parameters

        Returns:
            List[str]: Generated critiques, one per problem-solution pair
        """
        prompts = [
            self._format_prompt(problem, solution)
            for problem, solution in problem_solution_pairs
        ]
        return self.generate(prompts, **generation_kwargs)


class PolylithicGenerativeRM(BaseGenerativeModel):
    """A generative reward model that verifies solutions step by step.

    This model takes a problem and its solution, then generates critiques for each
    step of the solution. It inherits from BaseGenerativeModel and supports all
    its backend options.
    """

    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/CRITIQUE_ONE.txt"
        )

    def _format_prompt(self, problem: str, solution: List[str]) -> List[str]:
        """Format prompts for each step of the solution.

        Args:
            problem (str): The problem text
            solution (List[str]): List of solution steps

        Returns:
            List[str]: List of formatted prompts, one per solution step
        """
        prompts = []
        for i in range(len(solution)):
            tagged_response = "\n".join(
                [
                    f"<paragraph_{j}>\n{step}</paragraph_{j}>\n\n"
                    for j, step in enumerate(solution[: i + 1])
                ]
            )
            user_prompt = self.prompt_template.format(
                problem=problem, tagged_response=tagged_response
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
        self, problem_solution_pairs: List[Tuple[str, List[str]]], **generation_kwargs
    ) -> List[str]:
        """Generate step-by-step critiques for the given problem-solution pairs.

        Args:
            problem_solution_pairs (List[Tuple[str, List[str]]]): List of (problem, solution) pairs
            **generation_kwargs: Additional generation parameters

        Returns:
            List[str]: Generated critique(s)
        """
        # Format prompts for each step of each solution
        prompts = [
            self._format_prompt(problem, solution)
            for problem, solution in problem_solution_pairs
        ]

        # Flatten prompts while keeping track of which problem they belong to
        idx = [
            i for i, prompt_list in enumerate(prompts) for _ in range(len(prompt_list))
        ]
        flat_prompts = [prompt for prompt_list in prompts for prompt in prompt_list]

        # Generate critiques
        outputs = self.generate(flat_prompts, **generation_kwargs)

        # Reformat outputs back into per-problem lists
        reformatted_outputs = []
        current_problem_outputs = []

        for i, output in enumerate(outputs):
            if i > 0 and idx[i] == idx[i - 1]:
                # Merge outputs for the same problem
                current_problem_outputs = [
                    prev + "<|sep|>" + curr
                    for prev, curr in zip(current_problem_outputs, output)
                ]
            else:
                # Start new problem outputs
                if current_problem_outputs:
                    reformatted_outputs.append(current_problem_outputs)
                current_problem_outputs = output.copy()

        # Add the last problem's outputs
        if current_problem_outputs:
            reformatted_outputs.append(current_problem_outputs)

        # Extract answer and add final answer
        final_outputs = []
        for problem_outputs in reformatted_outputs:
            new_problem_outputs = []
            for output in problem_outputs:
                steps = output.split("<|sep|>")
                # Find first incorrect step
                incorrect_step = next(
                    (
                        i
                        for i, step in enumerate(steps)
                        if extract_from_boxed(step) == "0"
                    ),
                    -1,
                )
                # Add final answer
                new_output = (
                    output + f"<|sep|> Final answer: \\boxed{{{incorrect_step}}}"
                )
                new_problem_outputs.append(new_output)
            final_outputs.append(new_problem_outputs)

        return final_outputs
