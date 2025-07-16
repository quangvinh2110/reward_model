import aiohttp
import requests
import asyncio
import traceback
from collections.abc import Iterable
from typing import List, Optional
from abc import ABC, abstractmethod

from tqdm.asyncio import tqdm

from ..utils.io import batch_iter


class Client(ABC):
    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        if api_key:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        else:
            self.headers = {"Content-Type": "application/json"}

    @abstractmethod
    def _format_request_payload(
        self, messages: List[dict], **generation_kwargs
    ) -> dict:
        raise NotImplementedError("Subclass must implement this method")

    async def _agenerate_one(
        self,
        session: aiohttp.ClientSession,
        messages: List[dict],
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
        data = self._format_request_payload(messages, **generation_kwargs)
        try:
            async with session.post(
                self.endpoint, headers=self.headers, json=data, timeout=600000
            ) as resp:
                try:
                    resp = await resp.json()
                    return [answer["message"]["content"] for answer in resp["choices"]]
                except:
                    resp = await resp.text()
                    return [resp]
        except:
            return ["Failed: " + str(traceback.format_exc())]

    async def _agenerate(
        self,
        batch_messages: Iterable[List[dict]],
        progress_bar: bool = False,
        **generation_kwargs,
    ) -> List[List[str]]:
        """Generate text for multiple prompts in parallel.

        Args:
            batch_messages (Iterable[List[dict]]): List of input messages
            progress_bar (bool): Whether to show a progress bar
            **generation_kwargs: Additional generation parameters

        Returns:
            List[List[str]]: List of generated text responses for each prompt
        """
        async with asyncio.BoundedSemaphore(8):
            session_timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                tasks = []
                for messages in batch_messages:
                    tasks.append(
                        asyncio.ensure_future(
                            self._agenerate_one(session, messages, **generation_kwargs)
                        )
                    )
                if progress_bar:
                    answers = await tqdm.gather(*tasks)
                else:
                    answers = await asyncio.gather(*tasks)

        return answers

    def _generate_one(self, messages: List[dict], **generation_kwargs) -> List[str]:
        """Generate text for multiple prompts using synchronous interface.

        Args:
            prompt (str): Input prompt text
            **generation_kwargs: Additional generation parameters

        Returns:
            List[str]: List of generated text responses
        """
        data = self._format_request_payload(messages, **generation_kwargs)
        resp = requests.request(
            "POST", self.endpoint, headers=self.headers, json=data, timeout=600000
        )
        try:
            resp = resp.json()
            return [answer["message"]["content"] for answer in resp["choices"]]
        except:
            return [resp.text]

    def __call__(
        self,
        batch_messages: Iterable[List[dict]],
        run_async: bool = False,
        **generation_kwargs,
    ) -> List[List[str]]:
        """Generate text for multiple prompts using synchronous interface.

        Args:
            batch_messages (Iterable[List[dict]]): List of input messages
            run_async (bool): Whether to run the generation asynchronously
            **generation_kwargs: Additional generation parameters

        Returns:
            List[List[str]]: List of generated text responses for each prompt
        """
        results = []
        if run_async:
            for mini_batch in batch_iter(batch_messages, batch_size=256):
                results.extend(
                    asyncio.run(
                        self._agenerate(messages=mini_batch, **generation_kwargs)
                    )
                )
        else:
            for messages in batch_messages:
                results.append(self._generate_one(messages, **generation_kwargs))
        return results


class OpenaiClient:
    """Base class for LLM API clients.

    This abstract base class provides common functionality for interacting with
    different LLM API endpoints. It handles request formatting, error handling,
    and batch processing of prompts.

    Args:
        endpoint (str): Base URL of the LLM API server
        model (Optional[str]): Name of the model being served
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
    ):
        full_endpoint = endpoint.rstrip("/") + "/v1/chat/completions"
        super().__init__(full_endpoint, model, api_key)

    def _format_request_payload(
        self, messages: List[dict], **generation_kwargs
    ) -> dict:
        """Format the request payload for OpenAI API.

        Args:
            messages (List[dict]): Input messages
            **generation_kwargs: Additional generation parameters

        Returns:
            dict: Formatted request payload for OpenAI API
        """
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            **generation_kwargs,
        }


class AzureOpenaiClient(Client):
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        api_version: str,
    ):
        full_endpoint = (
            endpoint.rstrip("/")
            + "/openai/deployments/"
            + model
            + "/chat/completions?api-version="
            + api_version
        )
        super().__init__(full_endpoint, model, api_key)

    def _format_request_payload(
        self, messages: List[dict], **generation_kwargs
    ) -> dict:
        return {
            "messages": messages,
            "max_tokens": 1024,
            **generation_kwargs,
        }
