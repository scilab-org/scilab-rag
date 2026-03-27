"""
LlamaIndex-compatible embedding wrapper for OpenRouter API.

Uses the OpenRouter embeddings endpoint with models like:
- nvidia/llama-nemotron-embed-vl-1b-v2:free (1024 dimensions, free tier)

Compatible with PropertyGraphIndex, VectorStoreIndex, and all LlamaIndex pipelines.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

_MAX_RETRIES = 4
_RETRY_BACKOFF_BASE = 2


class EmbeddingResponseError(Exception):
    """Raised when the API returns HTTP 200 but the response body is malformed or an error payload."""

    def __init__(self, message: str, raw_response: Dict[str, Any]):
        super().__init__(message)
        self.raw_response = raw_response


def _validate_embedding_response(result: Dict[str, Any], context: str) -> None:
    """
    Validate that an OpenRouter embedding response contains the expected 'data' field.

    OpenRouter occasionally returns HTTP 200 with an error payload such as:
        {"error": {"message": "...", "code": 429}}
    instead of the expected:
        {"data": [{"embedding": [...], "index": 0}]}

    Raises EmbeddingResponseError (which is retryable) rather than letting a bare
    KeyError bubble up as an unhandled crash.
    """
    if "data" not in result:
        error_detail = result.get("error", result)
        raise EmbeddingResponseError(
            f"OpenRouter embedding response missing 'data' field [{context}]: {error_detail}",
            raw_response=result,
        )


class OpenRouterEmbedding(BaseEmbedding):
    """
    OpenRouter embedding wrapper for LlamaIndex.

    Uses the OpenRouter API with an API key.
    Compatible with all LlamaIndex indexing and retrieval pipelines.

    """

    model_name: str = Field(
        default="nvidia/llama-nemotron-embed-vl-1b-v2:free",
        description="OpenRouter embedding model name",
    )
    api_key: str = Field(
        description="OpenRouter API key",
    )
    api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    batch_size: int = Field(
        default=20,
        description="Max texts per batch request",
    )
    requests_per_minute: int = Field(
        default=200,
        description="Rate limit: requests per minute",
    )
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds",
    )

    # Private attributes
    _last_request_time: float = PrivateAttr(default=0.0)
    _min_request_interval: float = PrivateAttr(default=0.0)
    _client: Optional[httpx.Client] = PrivateAttr(default=None)
    _rate_limit_lock: Any = PrivateAttr(default=None)  # threading.Lock, set in __init__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Rate limiting
        self._min_request_interval = 60.0 / self.requests_per_minute
        self._last_request_time = 0.0
        self._rate_limit_lock = threading.Lock()

        # HTTP client
        self._client = httpx.Client(timeout=self.timeout)

    def __del__(self):
        """Cleanup HTTP client."""
        if self._client:
            self._client.close()

    def _rate_limit(self):
        """Enforce rate limiting between requests (thread-safe)."""
        if self._rate_limit_lock is None:
            self._rate_limit_lock = threading.Lock()
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self._min_request_interval:
                sleep_time = self._min_request_interval - time_since_last
                time.sleep(sleep_time)

            self._last_request_time = time.time()

    @classmethod
    def class_name(cls) -> str:
        return "OpenRouterEmbedding"

    def _get_headers(self) -> dict:
        """Get request headers with auth."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "SciLab RAG",
        }

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenRouter API with retry on transient errors."""
        all_embeddings: List[List[float]] = []

        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            self._rate_limit()

            last_error = None
            for attempt in range(_MAX_RETRIES):
                try:
                    response = self._client.post(
                        f"{self.api_base}/embeddings",
                        headers=self._get_headers(),
                        json={
                            "model": self.model_name,
                            "input": batch,
                        },
                    )
                    response.raise_for_status()

                    result = response.json()

                    # Validate the response shape before accessing 'data'.
                    # OpenRouter may return HTTP 200 with an error payload on
                    _validate_embedding_response(result, context=f"batch[{i}:{i+len(batch)}]")

                    # OpenRouter returns embeddings in the same format as OpenAI
                    # {"data": [{"embedding": [...], "index": 0}, ...]}
                    batch_embeddings = [item["embedding"] for item in result["data"]]

                    # Sort by index to ensure correct order
                    sorted_embeddings = sorted(
                        zip(result["data"], batch_embeddings),
                        key=lambda x: x[0]["index"]
                    )
                    all_embeddings.extend([emb for _, emb in sorted_embeddings])
                    last_error = None
                    break  # success, move to next batch

                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500:
                        last_error = e
                        wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                        logger.warning(
                            "Embedding batch request returned %s (attempt %d/%d), retrying in %ds",
                            e.response.status_code, attempt + 1, _MAX_RETRIES, wait,
                        )
                        time.sleep(wait)
                        continue
                    logger.error(
                        "OpenRouter embedding request failed (non-retryable): %s - %s",
                        e.response.status_code, e.response.text,
                    )
                    raise
                except EmbeddingResponseError as e:
                    last_error = e
                    wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Malformed embedding response (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, _MAX_RETRIES, wait, e,
                    )
                    time.sleep(wait)
                    continue
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_error = e
                    wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Embedding batch request error: %s (attempt %d/%d), retrying in %ds",
                        e, attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                except Exception as e:
                    logger.error("OpenRouter embedding failed: %s", e)
                    raise

            if last_error is not None:
                logger.error(
                    "All %d retries exhausted for embedding batch request", _MAX_RETRIES
                )
                raise last_error

        return all_embeddings

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with retry on transient errors."""
        self._rate_limit()

        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)

        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.post(
                    f"{self.api_base}/embeddings",
                    headers=self._get_headers(),
                    json={
                        "model": self.model_name,
                        "input": text,
                    },
                )
                response.raise_for_status()

                result = response.json()
                _validate_embedding_response(result, context="single")
                return result["data"][0]["embedding"]

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = e
                    wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Embedding single request returned %s (attempt %d/%d), retrying in %ds",
                        e.response.status_code, attempt + 1, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "OpenRouter embedding request failed (non-retryable): %s - %s",
                    e.response.status_code, e.response.text,
                )
                raise
            except EmbeddingResponseError as e:
                last_error = e
                wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Malformed embedding response (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, _MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
                continue
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Embedding single request error: %s (attempt %d/%d), retrying in %ds",
                    e, attempt + 1, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            except Exception as e:
                logger.error("OpenRouter embedding failed: %s", e)
                raise

        logger.error(
            "All %d retries exhausted for embedding single request", _MAX_RETRIES
        )
        if not last_error:
            last_error = Exception("Unknown error in embedding request")
        raise last_error
    # --- Required by BaseEmbedding ---

    def _get_query_embedding(self, query: str) -> List[float]:
        """Embed a query string."""
        return self._embed_single(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Embed a single document text."""
        return self._embed_single(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        return self._embed_texts(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async query embedding — run in thread pool."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_query_embedding, query
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async single text embedding — run in thread pool."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_text_embedding, text
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async batch text embedding — run the batched sync path in a thread pool.

        LlamaIndex's default async implementation fans out to individual
        _aget_text_embedding calls, bypassing the batching in _embed_texts.
        This override restores batching on the async code path (used by
        PropertyGraphIndex during ingestion).
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_text_embeddings, texts
        )
