from __future__ import annotations

import asyncio
import http.client
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from yhj_agent.agents.specialist_llm import SpecialistAgentLLM
from yhj_agent.common.config import MimoConfig, get_mimo_config
from yhj_agent.common.errors import (
    LLMAuthError,
    LLMProviderError,
    LLMResponseValidationError,
)
from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.evaluators.critic_evaluator_llm import CriticEvaluatorLLM
from yhj_agent.nodes.report_generator_llm import ReportGeneratorLLM
from yhj_agent.nodes.synthesis_node import SynthesisNode
from yhj_agent.routers.intent_router_llm import IntentRouterLLM
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PingSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool


def make_request() -> LLMRequest:
    return LLMRequest(
        model="test-model",
        messages=[LLMMessage(role="user", content="return json")],
        temperature=0,
    )


def make_client(transport) -> MimoLLMClient:
    config = MimoConfig(
        api_key="test-key",
        openai_base_url="https://example.com/v1",
        anthropic_base_url="https://example.com/anthropic",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )
    return MimoLLMClient(config=config, transport=transport)


def success_payload(content: str) -> tuple[int, dict]:
    return (
        200,
        {
            "model": "test-model",
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def test_chat_retries_recoverable_provider_errors():
    attempts = {"count": 0}

    def transport(url, headers, body, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise LLMProviderError("temporary eof")
        return success_payload('{"ok": true}')

    client = make_client(transport)

    response = client.chat(make_request())

    assert response.content == '{"ok": true}'
    assert attempts["count"] == 3


def test_get_mimo_config_prefers_api_keys_and_trims_to_first_three():
    config = get_mimo_config(
        env={
            "MIMO_API_KEYS": " key-1 , key-2 ,, key-3 , key-4 ",
            "MIMO_API_KEY": "legacy-key",
        },
        dotenv_path=PROJECT_ROOT / "tests" / "__missing__.env",
    )

    assert config.require_api_keys() == ["key-1", "key-2", "key-3"]
    assert config.require_api_key() == "key-1"


def test_get_mimo_config_falls_back_to_single_api_key():
    config = get_mimo_config(
        env={"MIMO_API_KEY": "legacy-key"},
        dotenv_path=PROJECT_ROOT / "tests" / "__missing__.env",
    )

    assert config.require_api_keys() == ["legacy-key"]
    assert config.require_api_key() == "legacy-key"


def test_chat_rotates_keys_on_ssl_eof_until_success():
    attempts = []
    config = MimoConfig(
        api_keys=["key-1", "key-2", "key-3"],
        openai_base_url="https://example.com/v1",
        anthropic_base_url="https://example.com/anthropic",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )

    def transport(url, headers, body, timeout):
        auth = headers["Authorization"]
        attempts.append(auth)
        if auth != "Bearer key-3":
            raise LLMProviderError(
                "MiMo request failed: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
            )
        return success_payload('{"ok": true}')

    client = MimoLLMClient(config=config, transport=transport)

    response = client.chat(make_request())

    assert response.content == '{"ok": true}'
    assert attempts == ["Bearer key-1", "Bearer key-2", "Bearer key-3"]


def test_chat_raises_after_all_keys_hit_ssl_eof_once():
    attempts = []
    config = MimoConfig(
        api_keys=["key-1", "key-2", "key-3"],
        openai_base_url="https://example.com/v1",
        anthropic_base_url="https://example.com/anthropic",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )

    def transport(url, headers, body, timeout):
        attempts.append(headers["Authorization"])
        raise LLMProviderError(
            "MiMo request failed: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
        )

    client = MimoLLMClient(config=config, transport=transport)

    with pytest.raises(LLMProviderError):
        client.chat(make_request())

    assert attempts == ["Bearer key-1", "Bearer key-2", "Bearer key-3"]


def test_chat_rotates_keys_on_remote_disconnected_until_success():
    attempts = []
    config = MimoConfig(
        api_keys=["key-1", "key-2", "key-3"],
        openai_base_url="https://example.com/v1",
        anthropic_base_url="https://example.com/anthropic",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )

    def transport(url, headers, body, timeout):
        del url, body, timeout
        auth = headers["Authorization"]
        attempts.append(auth)
        if auth != "Bearer key-3":
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return success_payload('{"ok": true}')

    client = MimoLLMClient(config=config, transport=transport)

    response = client.chat(make_request())

    assert response.content == '{"ok": true}'
    assert attempts == ["Bearer key-1", "Bearer key-2", "Bearer key-3"]


def test_chat_does_not_rotate_keys_for_non_ssl_provider_error():
    attempts = []
    config = MimoConfig(
        api_keys=["key-1", "key-2", "key-3"],
        openai_base_url="https://example.com/v1",
        anthropic_base_url="https://example.com/anthropic",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )

    def transport(url, headers, body, timeout):
        attempts.append(headers["Authorization"])
        if len(attempts) < 3:
            raise LLMProviderError("MiMo request failed: upstream 502")
        return success_payload('{"ok": true}')

    client = MimoLLMClient(config=config, transport=transport)

    response = client.chat(make_request())

    assert response.content == '{"ok": true}'
    assert attempts == ["Bearer key-1", "Bearer key-1", "Bearer key-1"]


def test_chat_does_not_retry_auth_errors():
    attempts = {"count": 0}

    def transport(url, headers, body, timeout):
        attempts["count"] += 1
        return 401, {"error": {"message": "invalid key"}}

    client = make_client(transport)

    with pytest.raises(LLMAuthError):
        client.chat(make_request())

    assert attempts["count"] == 1


def test_chat_structured_retries_schema_fix_once():
    responses = iter(
        [
            success_payload('{"wrong": true}'),
            success_payload('{"ok": true}'),
        ]
    )

    def transport(url, headers, body, timeout):
        return next(responses)

    client = make_client(transport)
    request = make_request()

    result = client.chat_structured(request, PingSchema)

    assert result.ok is True
    assert request.messages[-1].role == "user"
    assert "schema" in request.messages[-1].content.lower()


def test_chat_structured_repair_restarts_key_rotation_from_first_key():
    attempts = []
    config = MimoConfig(
        api_keys=["key-1", "key-2", "key-3"],
        openai_base_url="https://example.com/v1",
        anthropic_base_url="https://example.com/anthropic",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )

    def transport(url, headers, body, timeout):
        auth = headers["Authorization"]
        attempts.append(auth)
        if attempts == ["Bearer key-1"]:
            raise LLMProviderError(
                "MiMo request failed: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol"
            )
        if attempts == ["Bearer key-1", "Bearer key-2"]:
            return success_payload('{"wrong": true}')
        return success_payload('{"ok": true}')

    client = MimoLLMClient(config=config, transport=transport)

    result = client.chat_structured(make_request(), PingSchema)

    assert result.ok is True
    assert attempts == ["Bearer key-1", "Bearer key-2", "Bearer key-1"]


def test_chat_structured_raises_when_schema_retry_still_fails():
    def transport(url, headers, body, timeout):
        return success_payload('{"wrong": true}')

    client = make_client(transport)

    with pytest.raises(LLMResponseValidationError):
        client.chat_structured(make_request(), PingSchema)


class RaisingLLM:
    class _Config:
        light_model = "test-model"

    def __init__(self) -> None:
        self.config = self._Config()

    def chat(self, request):
        raise LLMProviderError("temporary eof")

    def chat_structured(self, request, response_model):
        raise LLMProviderError("temporary eof")


def test_router_raises_instead_of_returning_general_fallback():
    router = IntentRouterLLM(llm_client=RaisingLLM(), prompt_loader=PromptLoader())

    with pytest.raises(LLMProviderError):
        router.route(normalized_input={}, question="事业咨询")


def test_specialist_raises_instead_of_returning_default_claim():
    agent = SpecialistAgentLLM("practical_advisor", llm_client=RaisingLLM(), prompt_loader=PromptLoader())

    with pytest.raises(LLMProviderError):
        agent.run_sync({"question": "事业咨询", "pruned_context": {}, "symbolic_result": {}})


def test_critic_raises_instead_of_returning_default_scores():
    critic = CriticEvaluatorLLM(llm_client=RaisingLLM(), prompt_loader=PromptLoader())

    with pytest.raises(LLMProviderError):
        critic.evaluate(synthesis={"synthesis_text": "test"}, question="事业咨询", specialist_outputs={})


def test_report_generator_raises_instead_of_returning_default_report():
    generator = ReportGeneratorLLM(llm_client=RaisingLLM(), prompt_loader=PromptLoader())

    with pytest.raises(LLMProviderError):
        generator.generate(
            synthesis={"synthesis_text": "test"},
            specialist_outputs={},
            evaluation={"safety_score": 4.5, "overall_score": 4.2},
            raw_input="事业咨询",
            focus_question="事业咨询",
        )


def test_synthesis_raises_when_all_samples_fail(monkeypatch):
    node = SynthesisNode(llm_client=RaisingLLM(), prompt_loader=PromptLoader())

    async def always_fail(*args, **kwargs):
        raise LLMProviderError("temporary eof")

    monkeypatch.setattr(node, "_single_synthesis", always_fail)

    with pytest.raises(LLMProviderError):
        asyncio.run(node.synthesize(specialist_outputs={}, debate_output=None))
