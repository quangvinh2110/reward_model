import re
import torch
from typing import Optional, Tuple, List, Iterable
from transformers import AutoTokenizer
from ..utils.utils import read_txt, batch_iter
from tqdm import tqdm


def extract_answer(solution_text: str):
    boxed_pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


class MonolithicGenerativeRM:
    """A generative reward model that verify full solution all at once.

    This class supports multiple model types:
    - transformers: Uses HuggingFace transformers directly
    - vllm: Uses vLLM for inference
    - vllm_api: Uses vLLM API client
    - tgi_api: Uses TGI API client

    Args:
        model_type (str): Type of model to use ('transformers', 'vllm', 'vllm_api', or 'tgi_api')
        model_path (str): Path to the model or model name
        endpoint (Optional[str]): API endpoint URL (required for API types)
        prompt_template (Optional[str]): Path to prompt template file or template string
        progress_bar (bool): Whether to show progress bar during generation
    """

    def __init__(
        self,
        backend: str,
        model_name_or_path: str,
        endpoint: Optional[str] = None,
        served_model_name: Optional[str] = None,
        prompt_template: Optional[str] = None,
        progress_bar: bool = True,
    ):
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        # Load prompt template
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = read_txt(
                "/raid/vinh/reward_model/resources/prompt_templates/CRITIQUE_ALL.txt"
            )

        # Initialize model based on type
        if backend == "transformers":
            from transformers import AutoConfig, AutoModelForCausalLM

            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            self.model.eval()

        elif backend == "vllm":
            from vllm import LLM

            self.model = LLM(
                model=model_name_or_path,
                trust_remote_code=True,
                max_num_batched_tokens=16384,
                max_context_len_to_capture=4096,
                max_model_len=4096,
                max_num_seqs=32,
                dtype=torch.bfloat16,
            )

        elif backend in ["vllm_api", "tgi_api"]:
            if endpoint is None:
                raise ValueError(f"endpoint is required for {backend}")

            if backend == "vllm_api":
                from ..modules.client import VllmClient

                self.model = VllmClient(
                    endpoint=endpoint, served_model_name=served_model_name
                )
            else:  # tgi_api
                from ..modules.client import TgiClient

                self.model = TgiClient(
                    endpoint=endpoint, served_model_name=served_model_name
                )

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.progress_bar = progress_bar

    def _format_prompt(self, problem: str, solution: List[str]) -> str:
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

    def _get_batch_iterator(
        self, items: List[str], batch_size: int = 4, desc: str = "Processing"
    ) -> Iterable:
        """Get a batch iterator with optional progress bar.

        Args:
            items (List[str]): List of items to iterate over
            batch_size (int): Size of each batch
            desc (str): Description for progress bar

        Returns:
            Iterable: Batch iterator with optional progress bar
        """
        total_batches = (len(items) + batch_size - 1) // batch_size
        iterator = batch_iter(items, batch_size=batch_size)
        if self.progress_bar:
            iterator = tqdm(iterator, total=total_batches, desc=desc)
        return iterator

    def _generate_transformers(
        self, prompts: List[str], **generation_kwargs
    ) -> List[str]:
        from transformers import GenerationConfig

        # Set up default generation config
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            repetition_penalty=1.0,
            eos_token_id=[self.tokenizer.eos_token_id],
            pad_token_id=[self.tokenizer.pad_token_id],
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )

        # Update config with any provided kwargs
        for key, value in generation_kwargs.items():
            setattr(generation_config, key, value)

        all_outputs = []
        for batch_prompts in self._get_batch_iterator(
            prompts, desc="Generating with transformers"
        ):
            _inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True)
            input_ids = _inputs["input_ids"].to(self.model.device)
            attention_mask = _inputs["attention_mask"].to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )

            gen_tokens = generated["sequences"].cpu()
            batch_outputs = self.tokenizer.batch_decode(gen_tokens)
            batch_outputs = [
                output.replace(self.tokenizer.pad_token, "") for output in batch_outputs
            ]
            all_outputs.extend(batch_outputs)

        return all_outputs

    def _generate_vllm(self, prompts: List[str], **generation_kwargs) -> List[str]:
        from vllm import SamplingParams

        # Set up default generation config
        generation_config = SamplingParams(
            n=1,
            best_of=1,
            use_beam_search=False,
            max_tokens=1024,
            repetition_penalty=1.0,
            temperature=0,
            top_p=1,
            top_k=-1,
            stop=self.tokenizer.eos_token,
        )

        # Update config with any provided kwargs
        for key, value in generation_kwargs.items():
            setattr(generation_config, key, value)

        all_outputs = []
        for batch_prompts in self._get_batch_iterator(
            prompts, desc="Generating with vLLM"
        ):
            outputs = self.model.generate(
                batch_prompts,
                generation_config,
            )
            all_outputs.extend(
                [output.prompt + output.outputs[0].text for output in outputs]
            )
        return all_outputs

    def _generate_api(self, prompts: List[str], **generation_kwargs) -> List[str]:
        all_outputs = []
        for batch_prompts in self._get_batch_iterator(
            prompts, desc="Generating with API"
        ):
            batch_outputs = self.model(batch_prompts, **generation_kwargs)
            all_outputs.extend(batch_outputs)
        return all_outputs

    def __call__(
        self, problem_solution_pairs: List[Tuple[str, List[str]]], **generation_kwargs
    ) -> List[str]:
        """Generate a critique for the given problem and solution.

        Args:
            problem_solution_pairs (List[Tuple[str, List[str]]]): List of problem-solution pairs
            **generation_kwargs: Additional generation parameters

        Returns:
            List[str]: Generated critique(s)
        """
        # Format prompt
        prompts = [
            self._format_prompt(problem, solution)
            for problem, solution in problem_solution_pairs
        ]

        # Generate based on model type
        if self.backend == "transformers":
            return self._generate_transformers(prompts, **generation_kwargs)

        elif self.backend == "vllm":
            return self._generate_vllm(prompts, **generation_kwargs)

        else:  # API types
            return self._generate_api(prompts, **generation_kwargs)


class PolylithicGenerativeRM(MonolithicGenerativeRM):
    """A generative reward model that verify solution step by step.

    This class supports multiple model types:
    - transformers: Uses HuggingFace transformers directly
    - vllm: Uses vLLM for inference
    - vllm_api: Uses vLLM API client
    - tgi_api: Uses TGI API client

    Args:
        model_type (str): Type of model to use ('transformers', 'vllm', 'vllm_api', or 'tgi_api')
        model_path (str): Path to the model or model name
        endpoint (Optional[str]): API endpoint URL (required for API types)
        prompt_template (Optional[str]): Path to prompt template file or template string
        progress_bar (bool): Whether to show progress bar during generation
    """

    def __init__(
        self,
        backend: str,
        model_name_or_path: str,
        endpoint: Optional[str] = None,
        served_model_name: Optional[str] = None,
        prompt_template: Optional[str] = None,
        progress_bar: bool = True,
    ):
        super().__init__(
            backend,
            model_name_or_path,
            endpoint,
            served_model_name,
            prompt_template,
            progress_bar,
        )
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = read_txt(
                "/raid/vinh/reward_model/resources/prompt_templates/CRITIQUE_ONE.txt"
            )

    def _format_prompt(self, problem: str, solution: List[str]) -> str:
        prompts = []
        for i in range(len(solution)):
            tagged_response = "\n".join(
                [
                    f"<paragraph_{i}>\n{step}</paragraph_{i}>\n\n"
                    for i, step in enumerate(solution[:i])
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
        """Generate a critique for the given problem and solution.

        Args:
            problem_solution_pairs (List[Tuple[str, List[str]]]): List of problem-solution pairs
            **generation_kwargs: Additional generation parameters

        Returns:
            List[str]: Generated critique(s)
        """
        # Format prompt
        prompts = [
            self._format_prompt(problem, solution)
            for problem, solution in problem_solution_pairs
        ]
        idx = [
            i for i, prompt_list in enumerate(prompts) for _ in range(len(prompt_list))
        ]
        prompts = [prompt for prompt_list in prompts for prompt in prompt_list]

        # Generate based on model type
        if self.backend == "transformers":
            outputs = self._generate_transformers(prompts, **generation_kwargs)

        elif self.backend == "vllm":
            outputs = self._generate_vllm(prompts, **generation_kwargs)

        else:  # API types
            outputs = self._generate_api(prompts, **generation_kwargs)

        # Reformat outputs
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
                    (i for i, step in enumerate(steps) if extract_answer(step) == "0"),
                    -1,
                )
                # Add final answer
                new_output = output + f"\n\nFinal answer: \\boxed{{{incorrect_step}}}"
                new_problem_outputs.append(new_output)
            final_outputs.append(new_problem_outputs)

        return final_outputs
