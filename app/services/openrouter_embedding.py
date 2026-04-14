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

_MAX_RETRIES = 6
_RETRY_BACKOFF_BASE = 2

# All transient network-level httpx exceptions that should trigger a retry.
# httpx.ReadError (broken pipe / [Errno 32]) is NOT a subclass of TimeoutException
# or ConnectError — it must be listed explicitly. WriteError and CloseError are
# included for the same reason (connection reset mid-stream).
_RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,    # "Broken pipe" — the bug that only surfaces on Cloud Run
    httpx.WriteError,
    httpx.CloseError,
)

# HTTP status codes that warrant a retry (server errors + rate limiting).
# 429 is intentionally included here; previously only >= 500 was retried.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


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

        # HTTP client — configured for Cloud Run's proxy environment
        self._client = self._create_client()

    def __del__(self):
        """Cleanup HTTP client."""
        if self._client:
            self._client.close()

    def _create_client(self) -> httpx.Client:
        """Create an httpx.Client with transport-level retries and Cloud-Run-friendly pool limits.

        Key settings:
        - transport retries=2: automatic low-level retry for TCP connection errors
          before our application-level retry loop even kicks in.
        - keepalive_expiry=30: Cloud Run's load balancer/NAT can silently close idle
          TCP connections. Setting a short keepalive expiry (30s) ensures we don't
          try to reuse connections that the proxy has already torn down, which would
          produce a "Broken pipe" (ReadError) on the next request.
        - max_keepalive_connections=5: limits the pool size to avoid holding too many
          connections open in a single-worker Cloud Run container.
        """
        transport = httpx.HTTPTransport(retries=2)
        return httpx.Client(
            timeout=self.timeout,
            transport=transport,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30,  # seconds
            ),
        )

    def _reset_client(self) -> None:
        """Close and recreate the HTTP client to discard all stale pooled connections.

        Called after a ReadError / WriteError / CloseError to ensure the next retry
        opens a fresh TCP connection rather than reusing a broken one from the pool.
        """
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
        self._client = self._create_client()
        logger.debug("httpx.Client reset after connection error")

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
            self._client = self._create_client()

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
                    # rate limits or model errors.
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
                    if e.response.status_code in _RETRYABLE_STATUS_CODES:
                        last_error = e
                        # Respect Retry-After header when present (e.g. 429 responses)
                        retry_after = e.response.headers.get("retry-after")
                        wait = (
                            float(retry_after)
                            if retry_after
                            else _RETRY_BACKOFF_BASE ** (attempt + 1)
                        )
                        logger.warning(
                            "Embedding batch request returned %s (attempt %d/%d), retrying in %.1fs",
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

                except _RETRYABLE_NETWORK_ERRORS as e:
                    last_error = e
                    # For broken-pipe style errors, reset the client so the next attempt
                    # opens a fresh TCP connection rather than reusing the broken one.
                    if isinstance(e, (httpx.ReadError, httpx.WriteError, httpx.CloseError)):
                        self._reset_client()
                    wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Embedding batch network error %s (attempt %d/%d), retrying in %ds: %s",
                        type(e).__name__, attempt + 1, _MAX_RETRIES, wait, e,
                    )
                    time.sleep(wait)
                    continue

                except Exception as e:
                    logger.error("OpenRouter embedding failed with unexpected error: %s", e)
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
            self._client = self._create_client()

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
                if e.response.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = e
                    retry_after = e.response.headers.get("retry-after")
                    wait = (
                        float(retry_after)
                        if retry_after
                        else _RETRY_BACKOFF_BASE ** (attempt + 1)
                    )
                    logger.warning(
                        "Embedding single request returned %s (attempt %d/%d), retrying in %.1fs",
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

            except _RETRYABLE_NETWORK_ERRORS as e:
                last_error = e
                if isinstance(e, (httpx.ReadError, httpx.WriteError, httpx.CloseError)):
                    self._reset_client()
                wait = _RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Embedding single network error %s (attempt %d/%d), retrying in %ds: %s",
                    type(e).__name__, attempt + 1, _MAX_RETRIES, wait, e,
                )
                time.sleep(wait)
                continue

            except Exception as e:
                logger.error("OpenRouter embedding failed with unexpected error: %s", e)
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
        This override restores batching on the async code path.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_text_embeddings, texts
        )
