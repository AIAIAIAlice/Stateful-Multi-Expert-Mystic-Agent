from collections.abc import Callable, Mapping
import http.client
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from pydantic import ValidationError

from yhj_agent.common.config import MimoConfig, get_mimo_config
from yhj_agent.common.errors import (
    LLMAuthError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest, LLMResponse


HttpTransport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, dict[str, Any]]]
logger = logging.getLogger(__name__)


def default_urllib_transport(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    """使用标准库发送 HTTP 请求，避免引入额外运行依赖。"""
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=dict(headers),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return response.status, json.loads(response_body)
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8")
        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError:
            payload = {"error": {"message": response_body}}
        return exc.code, payload
    except (TimeoutError, socket.timeout) as exc:
        raise LLMTimeoutError("MiMo request timed out.") from exc
    except urllib.error.URLError as exc:
        raise LLMProviderError(f"MiMo request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise LLMResponseValidationError("MiMo response is not valid JSON.") from exc


class MimoLLMClient:
    """MiMo OpenAI-compatible LLM client."""

    def __init__(
        self,
        config: MimoConfig | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config or get_mimo_config()
        self.transport = transport or default_urllib_transport

    def chat(self, request: LLMRequest) -> LLMResponse:
        """调用 MiMo `/chat/completions` 并返回统一响应。"""
        if request.stream:
            raise LLMProviderError("Streaming is not implemented in the current demo client.")

        url = f"{self.config.openai_base_url}/chat/completions"
        body = self._build_body(request)
        api_keys = self.config.require_api_keys()
        current_key_index = 0
        retry_attempt = 0

        last_error: Exception | None = None
        while True:
            try:
                headers = self._build_headers(api_keys[current_key_index])
                status_code, raw_response = self.transport(
                    url,
                    headers,
                    body,
                    self.config.timeout_seconds,
                )
                self._raise_for_status(status_code, raw_response)
                return self._parse_chat_response(raw_response)
            except (LLMRateLimitError, LLMTimeoutError, LLMProviderError, http.client.RemoteDisconnected) as exc:
                last_error = exc
                should_rotate = self._should_rotate_key(exc)
                logger.warning(
                    "MiMo chat error: type=%s key_index=%d/%d rotate=%s retry_attempt=%d model=%s message=%s",
                    type(exc).__name__,
                    current_key_index + 1,
                    len(api_keys),
                    should_rotate,
                    retry_attempt,
                    request.model,
                    str(exc)[:200],
                )
                if should_rotate:
                    if current_key_index + 1 >= len(api_keys):
                        logger.error(
                            "MiMo key rotation exhausted: last_error=%s total_keys=%d model=%s",
                            type(exc).__name__,
                            len(api_keys),
                            request.model,
                        )
                        raise
                    logger.warning(
                        "MiMo rotating key: from_index=%d to_index=%d total_keys=%d model=%s",
                        current_key_index + 1,
                        current_key_index + 2,
                        len(api_keys),
                        request.model,
                    )
                    current_key_index += 1
                    retry_attempt = 0
                    continue

                retry_attempt += 1
                if retry_attempt >= self.config.max_attempts:
                    logger.error(
                        "MiMo retry exhausted: error=%s attempts=%d key_index=%d/%d model=%s",
                        type(exc).__name__,
                        retry_attempt,
                        current_key_index + 1,
                        len(api_keys),
                        request.model,
                    )
                    raise
                logger.warning(
                    "MiMo retrying same key: next_attempt=%d key_index=%d/%d model=%s",
                    retry_attempt + 1,
                    current_key_index + 1,
                    len(api_keys),
                    request.model,
                )
                time.sleep(self._compute_retry_delay_seconds(retry_attempt))

        if last_error is not None:
            raise last_error
        raise LLMProviderError("MiMo request failed without a captured exception.")

    def chat_structured(self, request: LLMRequest, response_model: type) -> object:
        """调用 LLM 并解析为 Pydantic 模型。"""
        response = self.chat(request)

        try:
            return response_model.model_validate_json(response.content)
        except Exception:
            request.messages.append(
                LLMMessage(role="user", content="请严格按 schema 输出合法 JSON，不要包含 markdown 代码块。")
            )
            response = self.chat(request)
            try:
                return response_model.model_validate_json(response.content)
            except Exception as exc:
                raise LLMResponseValidationError(
                    f"{response_model.__name__} validation failed after JSON repair retry."
                ) from exc

    def _build_body(self, request: LLMRequest) -> bytes:
        payload = request.model_dump(exclude_none=True)
        payload.pop("metadata", None)
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _build_headers(self, api_key: str | None = None) -> dict[str, str]:
        api_key = api_key or self.config.require_api_key()
        headers = {"Content-Type": "application/json"}
        if self.config.auth_header_mode == "api-key":
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _compute_retry_delay_seconds(self, attempt: int) -> float:
        delay_ms = self.config.retry_base_delay_ms * (2 ** max(attempt - 1, 0))
        return min(delay_ms, self.config.retry_max_delay_ms) / 1000.0

    def _raise_for_status(self, status_code: int, raw_response: dict[str, Any]) -> None:
        if 200 <= status_code < 300:
            return

        message = self._extract_error_message(raw_response)
        if status_code in {401, 403}:
            raise LLMAuthError(message)
        if status_code == 429:
            raise LLMRateLimitError(message)
        if status_code >= 500:
            raise LLMProviderError(message)
        raise LLMProviderError(message)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        if not text:
            return text

        stripped = text.strip()
        if stripped.startswith("```"):
            nl_pos = stripped.find("\n")
            if nl_pos > 0:
                stripped = stripped[nl_pos + 1 :]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3]
            return stripped.strip()
        return stripped

    def _parse_chat_response(self, raw_response: dict[str, Any]) -> LLMResponse:
        try:
            choices = raw_response.get("choices") or []
            if not choices:
                raise KeyError("choices")

            first_choice = choices[0]
            message = first_choice.get("message") or {}
            usage = raw_response.get("usage") or {}
            content = message.get("content")

            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            if isinstance(content, str):
                content = self._strip_markdown_fences(content)

            return LLMResponse(
                model=raw_response.get("model") or "",
                content=content or "",
                reasoning_content=message.get("reasoning_content"),
                tool_calls=message.get("tool_calls") or [],
                finish_reason=first_choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                raw_response=raw_response,
            )
        except (KeyError, TypeError, ValidationError) as exc:
            raise LLMResponseValidationError("MiMo response does not match LLMResponse schema.") from exc

    def _extract_error_message(self, raw_response: dict[str, Any]) -> str:
        error = raw_response.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if isinstance(error, str):
            return error
        return "MiMo API request failed."

    @staticmethod
    def _should_rotate_key(exc: Exception) -> bool:
        if isinstance(exc, http.client.RemoteDisconnected):
            return True
        return MimoLLMClient._is_ssl_eof_error(exc)

    @staticmethod
    def _is_ssl_eof_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "unexpected_eof_while_reading" in message
            or "eof occurred in violation of protocol" in message
            or ("ssl" in message and "eof" in message)
        )
