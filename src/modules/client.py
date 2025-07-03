import os
import aiohttp
import requests
import asyncio
import traceback
from collections.abc import Iterable
from typing import List, Optional

from tqdm.asyncio import tqdm

from abc import ABC, abstractmethod

from ..utils.io import batch_iter


class LlmClient(ABC):
    """Base class for LLM API clients.

    This abstract base class provides common functionality for interacting with
    different LLM API endpoints. It handles request formatting, error handling,
    and batch processing of prompts.

    Args:
        endpoint (str): Base URL of the LLM API server
        served_model_name (Optional[str]): Name of the model being served
    """

    def __init__(
        self,
        endpoint: str,
        served_model_name: Optional[str] = None,
    ):
        self.endpoint = endpoint
        self.served_model_name = served_model_name

    @abstractmethod
    def _format_request_payload(self, prompt: str, **generation_kwargs) -> dict:
        """Format the request payload for the specific LLM API.

        Args:
            prompt (str): Input prompt text
            **generation_kwargs: Additional generation parameters

        Returns:
            dict: Formatted request payload
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def _agenerate_one(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        **generation_kwargs,
    ) -> List[str]:
        """Generate text for a single prompt.

        Args:
            session (aiohttp.ClientSession): Active aiohttp session
            prompt (str): Input prompt text
            **generation_kwargs: Additional generation parameters

        Returns:
            List[str]: List of generated text responses
        """
        data = self._format_request_payload(prompt, **generation_kwargs)
        api_key = os.getenv("API_KEY")
        if api_key:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        else:
            headers = {"Content-Type": "application/json"}
        try:
            async with session.post(
                self.endpoint, headers=headers, json=data, timeout=600000
            ) as resp:
                try:
                    resp = await resp.json()
                    return [answer["text"] for answer in resp["choices"]]
                except:
                    resp = await resp.text()
                    return [resp]
        except:
            return ["Failed: " + str(traceback.format_exc())]

    async def _agenerate(
        self, prompts: Iterable[str], progress_bar: bool = False, **generation_kwargs
    ) -> List[List[str]]:
        """Generate text for multiple prompts in parallel.

        Args:
            prompts (Iterable[str]): List of input prompts
            progress_bar (bool): Whether to show a progress bar
            **generation_kwargs: Additional generation parameters

        Returns:
            List[List[str]]: List of generated text responses for each prompt
        """
        async with asyncio.BoundedSemaphore(8):
            session_timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                tasks = []
                for prompt in prompts:
                    tasks.append(
                        asyncio.ensure_future(
                            self._agenerate_one(session, prompt, **generation_kwargs)
                        )
                    )
                if progress_bar:
                    answers = await tqdm.gather(*tasks)
                else:
                    answers = await asyncio.gather(*tasks)

        return answers

    def _generate_one(self, prompt: str, **generation_kwargs) -> List[str]:
        """Generate text for multiple prompts using synchronous interface.

        Args:
            prompt (str): Input prompt text
            **generation_kwargs: Additional generation parameters

        Returns:
            List[str]: List of generated text responses
        """
        api_key = os.getenv("API_KEY")
        if api_key:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        else:
            headers = {"Content-Type": "application/json"}
        data = self._format_request_payload(prompt, **generation_kwargs)
        resp = requests.request(
            "POST", self.endpoint, headers=headers, json=data, timeout=600000
        )
        try:
            resp = resp.json()
            return [answer["text"] for answer in resp["choices"]]
        except:
            return [resp.text]

    def __call__(
        self,
        prompts: Iterable[str],
        run_async: bool = False,
        **generation_kwargs,
    ) -> List[List[str]]:
        """Generate text for multiple prompts using synchronous interface.

        Args:
            prompts (Iterable[str]): List of input prompts
            run_async (bool): Whether to run the generation asynchronously
            **generation_kwargs: Additional generation parameters

        Returns:
            List[List[str]]: List of generated text responses for each prompt
        """
        results = []
        if run_async:
            for batch in batch_iter(prompts, batch_size=256):
                results.extend(
                    asyncio.run(self._agenerate(prompts=batch, **generation_kwargs))
                )
        else:
            for prompt in prompts:
                results.append(self._generate_one(prompt, **generation_kwargs))
        return results


class OpenAIClient(LlmClient):
    """Client for OpenAI API endpoints.

    Args:
        endpoint (str): Base URL of the vLLM API server
        served_model_name (Optional[str]): Name of the model being served
    """

    def __init__(
        self,
        endpoint: str,
        served_model_name: Optional[str] = None,
    ):
        if not served_model_name:
            raise ValueError("served_model_name is required")
        super().__init__(endpoint.strip("/") + "/v1/completions", served_model_name)

    def _format_request_payload(self, prompt: str, **generation_kwargs) -> dict:
        """Format the request payload for vLLM API.

        Args:
            prompt (str): Input prompt text
            **generation_kwargs: Additional generation parameters

        Returns:
            dict: Formatted request payload for vLLM API
        """
        return {
            "model": self.served_model_name,
            "prompt": prompt,
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "max_tokens": 1024,
            "repetition_penalty": 1.0,
            "temperature": 0,
            "top_p": 0.9,
            "top_k": -1,
            **generation_kwargs,
        }


class HuggingFaceClient(LlmClient):
    """Client for Text Generation Inference (TGI) API endpoints.

    This client is designed to work with Hugging Face's Text Generation
    Inference API endpoints, which provide optimized LLM inference.

    Args:
        endpoint (str): Base URL of the TGI API server
        served_model_name (Optional[str]): Name of the model being served
    """

    def __init__(
        self,
        endpoint: str,
        served_model_name: Optional[str] = None,
    ):
        super().__init__(endpoint, served_model_name)

    def _format_request_payload(self, prompt: str, **generation_kwargs) -> dict:
        """Format the request payload for TGI API.

        Args:
            prompt (str): Input prompt text
            **generation_kwargs: Additional generation parameters

        Returns:
            dict: Formatted request payload for TGI API
        """
        return {}


class TeiClient:
    def __init__(
        self,
        endpoint: str,
    ):
        """Initialize TEI client.

        Args:
            endpoint (str): Base URL of the TEI server (e.g. "http://localhost:8080")
        """
        self.endpoint = endpoint.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def request(self, url: str, data: dict) -> dict:
        """Make a request to the TEI server.

        Args:
            url (str): Endpoint URL
            data (dict): Request payload

        Returns:
            dict: Response from the server

        Raises:
            Exception: If request fails or response cannot be parsed
        """
        try:
            response = self.session.post(url, json=data, timeout=600000)
            response.raise_for_status()
            try:
                return response.json()
            except:
                return {"error": f"Failed to parse response: {response.text}"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(traceback.format_exc())}"}

    def embed(self, text: str | List[str]) -> List[List[float]]:
        """Get embeddings for input text(s).

        Args:
            text (str | List[str]): Input text or list of texts

        Returns:
            List[List[float]]: List of embeddings
        """
        if isinstance(text, str):
            text = [text]

        url = f"{self.endpoint}/embed"
        data = {"inputs": text}
        response = self.request(url, data)
        return response

    def rerank(
        self, query: str, texts: str | List[str], raw_scores: bool = False
    ) -> List[dict]:
        """Rerank texts based on query relevance.

        Args:
            query (str): Query text
            texts (str | List[str]): Text(s) to rerank
            raw_scores (bool): Whether to return raw scores

        Returns:
            List[dict]: Reranked results with scores
        """
        if isinstance(texts, str):
            texts = [texts]

        url = f"{self.endpoint}/rerank"
        data = {"query": query, "texts": texts, "raw_scores": raw_scores}
        response = self.request(url, data)
        return response

    def classify(self, text: str | List[str]) -> List[dict]:
        """Classify input text(s).

        Args:
            text (str | List[str]): Input text or list of texts

        Returns:
            List[dict]: Classification results
        """
        if isinstance(text, str):
            text = [text]

        url = f"{self.endpoint}/predict"
        data = {"inputs": text}
        response = self.request(url, data)
        return response


if __name__ == "__main__":
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
    ]

    # Test VLLM client
    print("\nTesting VLLM Client:")
    vllm_client = VllmClient(
        endpoint="http://localhost:8000", served_model_name="llama-2-7b"
    )
    vllm_results = vllm_client(test_prompts)
    for prompt, responses in zip(test_prompts, vllm_results):
        print(f"\nPrompt: {prompt}")
        print(f"Responses: {responses}")

    # Test TGI client
    # print("\nTesting TGI Client:")
    # tgi_client = TgiClient(
    #     endpoint="http://localhost:8080", served_model_name="tiiuae/falcon-7b"
    # )
    # tgi_results = tgi_client(test_prompts)
    # for prompt, responses in zip(test_prompts, tgi_results):
    #     print(f"\nPrompt: {prompt}")
    #     print(f"Responses: {responses}")

    # Test TEI client
    print("\nTesting TEI Client:")
    tei_client = TeiClient(endpoint="http://localhost:8080")

    # Test embedding
    print("\nTesting TEI Embedding:")
    embeddings = tei_client.embed(test_prompts)
    print(f"Embeddings shape: {len(embeddings)}")

    # Test reranking
    print("\nTesting TEI Reranking:")
    query = "What is the capital of France?"
    rerank_results = tei_client.rerank(query, test_prompts)
    print(f"Reranking results: {rerank_results}")

    # Test classification
    print("\nTesting TEI Classification:")
    classify_results = tei_client.classify(test_prompts)
    print(f"Classification results: {classify_results}")
