from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from ..utils.io import read_txt
from ..utils.parser import parse_from_boxed
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter


class VerifierAPI(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        endpoint: str,
        provider: str,
        served_model_name: Optional[str] = None,
        prompt_template: Optional[str] = None,
        show_progress: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = self._get_default_prompt_template()

        self.show_progress = show_progress

        if provider == "openai":
            from .client import OpenAIClient

            self.client = OpenAIClient(
                endpoint=endpoint,
                served_model_name=served_model_name,
            )
        elif provider == "huggingface":
            from .client import HuggingFaceClient

            self.client = HuggingFaceClient(
                endpoint=endpoint,
                served_model_name=served_model_name,
            )
        else:
            raise ValueError(f"Invalid provider: {provider}")

    @abstractmethod
    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _verify_one(
        self, id: int, problem: str, solution: List[str], **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single problem-solution pair."""
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(
        self,
        problem_solution_pairs: List[Tuple[str, List[str]]],
        num_workers: int = 4,
        **generation_kwargs,
    ) -> List[str]:
        """Verify multiple problem-solution pairs using multiprocessing."""
        if not problem_solution_pairs:
            return []

        # Create tasks with IDs for tracking
        tasks = [
            lambda: self._verify_one(i, problem, solution, **generation_kwargs)
            for i, (problem, solution) in enumerate(problem_solution_pairs)
        ]

        results = []
        with Pool(processes=num_workers) as pool:
            if self.show_progress:
                results = list(
                    tqdm(
                        pool.imap_unordered(lambda task: task(), tasks),
                        total=len(tasks),
                    )
                )
            else:
                results = list(pool.imap_unordered(lambda task: task(), tasks))

        # Sort results by ID to maintain original order
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results]


class AggregativeVerifierAPI(VerifierAPI):

    def _get_default_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/CRITIQUE_ALL.txt"
        )

    def _verify_one(
        self, id: int, problem: str, solution: List[str], **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single problem-solution pair."""
        tagged_solution = "\n".join(
            [f"<step_{i}>\n{step}\n</step_{i}>\n\n" for i, step in enumerate(solution)]
        )
        user_input = self.prompt_template.format(
            problem=problem, tagged_solution=tagged_solution
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return (id, self.client([prompt], **generation_kwargs)[0])


class IterativeVerifierAPI(VerifierAPI):
    def _get_default_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/CRITIQUE_ONE.txt"
        )

    def _verify_one(
        self, id: int, problem: str, solution: List[str], **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single problem-solution pair with early stopping and majority voting."""
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        for step_idx in range(len(solution)):
            tagged_solution = "\n".join(
                [
                    f"<step_{i}>\n{step}\n</step_{i}>\n\n"
                    for i, step in enumerate(solution[: step_idx + 1])
                ]
            )
            user_input = self.prompt_template.format(
                problem=problem, tagged_solution=tagged_solution
            )
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_input}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # Generate n responses the boxed answer for this step
            step_results = self.client([prompt], **generation_kwargs)[0]
            for result, step_result in zip(results, step_results):
                result.append(step_result)
            # Majority voting
            parsed = [parse_from_boxed(step_result) for step_result in step_results]
            majority, _ = Counter(parsed).most_common(1)[0]
            if majority == "0":
                for result in results:
                    result.append(f"Final answer: \\boxed{{{step_idx}}}")
                break
        # Return a string summarizing the results for each step
        return (id, ["<|sep|>".join(result) for result in results])
