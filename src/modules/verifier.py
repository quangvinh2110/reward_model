from typing import Optional, List, Tuple
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from ..utils.io import read_txt
from ..utils.parser import parse_from_boxed
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter


def _verify_one_helper(args):
    """Helper function to run _verify_one in multiprocessing."""
    _verify_one_func, i, sample, generation_kwargs = args
    return _verify_one_func(i, sample, **generation_kwargs)


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
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single sample (dictionary)."""
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(
        self,
        samples: List[dict],
        num_workers: int = 4,
        **generation_kwargs,
    ) -> List[str]:
        """Verify multiple samples using multiprocessing."""
        if not samples:
            return []

        # Prepare arguments for the helper function
        tasks = [
            (self._verify_one, i, sample, generation_kwargs)
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

        # Sort results by ID to maintain original order
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results]


class AggregativeVerifierAPI(VerifierAPI):

    def _get_default_prompt_template(self) -> str:
        return read_txt(
            "/raid/vinh/reward_model/resources/prompt_templates/CRITIQUE_ALL.txt"
        )

    def _verify_one(
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single sample (dictionary)."""
        tagged_solution = "\n".join(
            [
                f"<step_{i}>\n{step}\n</step_{i}>\n\n"
                for i, step in enumerate(sample["steps"])
            ]
        )
        user_input = self.prompt_template.format(
            problem=sample["problem"], tagged_solution=tagged_solution
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
        self, id: int, sample: dict, **generation_kwargs
    ) -> Tuple[int, str]:
        """Verify a single sample (dictionary) with early stopping and majority voting."""
        results = [[] for _ in range(generation_kwargs.get("n", 1))]
        no_wrong_step = True
        for step_idx in range(len(sample["steps"])):
            if sample["label"] != -1 and step_idx > sample["label"]:
                for result in results:
                    result.append(r"Final answer: \boxed{None}")
                break
            tagged_solution = "\n".join(
                [
                    f"<step_{i}>\n{step}\n</step_{i}>\n\n"
                    for i, step in enumerate(sample["steps"][: step_idx + 1])
                ]
            )
            user_input = self.prompt_template.format(
                problem=sample["problem"], tagged_solution=tagged_solution
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
                no_wrong_step = False
                break
        if no_wrong_step:
            for result in results:
                result.append("Final answer: \\boxed{-1}")
        # Return a string summarizing the results for each step
        return (id, ["<|sep|>".join(result) for result in results])
