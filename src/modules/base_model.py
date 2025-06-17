from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
import torch
from transformers import AutoTokenizer
from ..utils import batch_iter
from tqdm import tqdm


class BaseGenerativeModel(ABC):
    """Base class for generative models that support multiple backends.

    This class supports multiple model types:
    - transformers: Uses HuggingFace transformers directly
    - vllm: Uses vLLM for inference
    - vllm_api: Uses vLLM API client
    - tgi_api: Uses TGI API client

    Args:
        backend (str): Type of model to use ('transformers', 'vllm', 'vllm_api', or 'tgi_api')
        model_name_or_path (str): Path to the model or model name
        endpoint (Optional[str]): API endpoint URL (required for API types)
        served_model_name (Optional[str]): Name of the served model (for API types)
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
            self.prompt_template = self._get_default_prompt_template()

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
                from .client import VllmClient

                self.model = VllmClient(
                    endpoint=endpoint, served_model_name=served_model_name
                )
            else:  # tgi_api
                from .client import TgiClient

                self.model = TgiClient(
                    endpoint=endpoint, served_model_name=served_model_name
                )

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.progress_bar = progress_bar

    @abstractmethod
    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for the model."""
        pass

    @abstractmethod
    def _format_prompt(self, *args, **kwargs) -> str:
        """Format the prompt for generation."""
        pass

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
        """Generate using transformers backend."""
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
        """Generate using vLLM backend."""
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
        """Generate using API backend."""
        all_outputs = []
        for batch_prompts in self._get_batch_iterator(
            prompts, desc="Generating with API"
        ):
            batch_outputs = self.model(batch_prompts, **generation_kwargs)
            all_outputs.extend(batch_outputs)
        return all_outputs

    def generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate based on model type.

        Args:
            prompts (List[str]): List of prompts to generate from
            **generation_kwargs: Additional generation parameters

        Returns:
            List[List[str]]: Generated outputs, one list per prompt
        """
        if self.backend == "transformers":
            return self._generate_transformers(prompts, **generation_kwargs)
        elif self.backend == "vllm":
            return self._generate_vllm(prompts, **generation_kwargs)
        else:  # API types
            return self._generate_api(prompts, **generation_kwargs)
