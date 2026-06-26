from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api.main as api_main
from yhj_agent.common.config import MimoConfig
from yhj_agent.common.errors import LLMProviderError
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest


SSL_EOF_MESSAGE = (
    "MiMo request failed: [SSL: UNEXPECTED_EOF_WHILE_READING] "
    "EOF occurred in violation of protocol"
)
REMOTE_DISCONNECTED_MESSAGE = "Remote end closed connection without response"


def _post_turn(monkeypatch: Any, orchestrator: object) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(api_main, "ORCHESTRATOR", orchestrator)

    server = api_main.ThreadingHTTPServer((api_main.HOST, 0), api_main.DemoRequestHandler)
    try:
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        conn = http.client.HTTPConnection(api_main.HOST, port, timeout=10)
        conn.request(
            "POST",
            "/api/turns",
            body=json.dumps({"session_id": "demo-session", "message": "事业咨询"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        status = response.status
        conn.close()
        thread.join(timeout=5)
    finally:
        server.server_close()

    return status, payload


def _build_client(headers_seen: list[str], succeed_on_key: str | None) -> MimoLLMClient:
    config = MimoConfig(
        api_key="key-1",
        api_keys=["key-1", "key-2", "key-3"],
        openai_base_url="https://unit.test/v1",
        anthropic_base_url="https://unit.test/anthropic",
        default_model="test-model",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )

    def transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, dict[str, Any]]:
        del url, body, timeout
        headers_seen.append(headers["Authorization"])
        current_key = headers["Authorization"].removeprefix("Bearer ").strip()
        if current_key == succeed_on_key:
            return (
                200,
                {
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"content": "rotated success"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
        raise LLMProviderError(SSL_EOF_MESSAGE)

    return MimoLLMClient(config=config, transport=transport)


def _build_client_remote_disconnected(
    headers_seen: list[str],
    succeed_on_key: str | None,
) -> MimoLLMClient:
    config = MimoConfig(
        api_key="key-1",
        api_keys=["key-1", "key-2", "key-3"],
        openai_base_url="https://unit.test/v1",
        anthropic_base_url="https://unit.test/anthropic",
        default_model="test-model",
        light_model="test-model",
        timeout_seconds=1,
        auth_header_mode="bearer",
        max_attempts=3,
        retry_base_delay_ms=1,
        retry_max_delay_ms=2,
    )

    def transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, dict[str, Any]]:
        del url, body, timeout
        headers_seen.append(headers["Authorization"])
        current_key = headers["Authorization"].removeprefix("Bearer ").strip()
        if current_key == succeed_on_key:
            return (
                200,
                {
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"content": "rotated success"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
        raise http.client.RemoteDisconnected(REMOTE_DISCONNECTED_MESSAGE)

    return MimoLLMClient(config=config, transport=transport)


class _DirectRaiseOrchestrator:
    def run_turn(self, session_id: str, user_input: str, user_id: str = "") -> dict[str, Any]:
        del session_id, user_input, user_id
        raise RuntimeError("ReportGeneratorLLM: LLMProviderError")


class _SslRotationOrchestrator:
    def __init__(self, succeed_on_key: str | None) -> None:
        self.headers_seen: list[str] = []
        self.client = _build_client(self.headers_seen, succeed_on_key=succeed_on_key)

    def run_turn(self, session_id: str, user_input: str, user_id: str = "") -> dict[str, Any]:
        del session_id, user_id
        response = self.client.chat(
            LLMRequest(
                model="test-model",
                messages=[LLMMessage(role="user", content=user_input)],
                temperature=0,
            )
        )
        return {"final_report": response.content}


class _RemoteDisconnectedRotationOrchestrator:
    def __init__(self, succeed_on_key: str | None) -> None:
        self.headers_seen: list[str] = []
        self.client = _build_client_remote_disconnected(self.headers_seen, succeed_on_key=succeed_on_key)

    def run_turn(self, session_id: str, user_input: str, user_id: str = "") -> dict[str, Any]:
        del session_id, user_id
        response = self.client.chat(
            LLMRequest(
                model="test-model",
                messages=[LLMMessage(role="user", content=user_input)],
                temperature=0,
            )
        )
        return {"final_report": response.content}


def test_api_turns_returns_500_when_orchestrator_raises(monkeypatch: Any) -> None:
    status, payload = _post_turn(monkeypatch, _DirectRaiseOrchestrator())

    assert status == 500
    assert "RuntimeError" in payload["error"]
    assert "LLMProviderError" in payload["error"]


def test_api_turns_succeeds_after_ssl_eof_key_rotation(monkeypatch: Any) -> None:
    orchestrator = _SslRotationOrchestrator(succeed_on_key="key-3")

    status, payload = _post_turn(monkeypatch, orchestrator)

    assert status == 200
    assert payload["final_report"] == "rotated success"
    assert orchestrator.headers_seen == [
        "Bearer key-1",
        "Bearer key-2",
        "Bearer key-3",
    ]


def test_api_turns_returns_500_after_all_keys_hit_ssl_eof(monkeypatch: Any) -> None:
    orchestrator = _SslRotationOrchestrator(succeed_on_key=None)

    status, payload = _post_turn(monkeypatch, orchestrator)

    assert status == 500
    assert "LLMProviderError" in payload["error"]
    assert "EOF" in payload["error"]
    assert orchestrator.headers_seen == [
        "Bearer key-1",
        "Bearer key-2",
        "Bearer key-3",
    ]


def test_api_turns_succeeds_after_remote_disconnected_key_rotation(monkeypatch: Any) -> None:
    orchestrator = _RemoteDisconnectedRotationOrchestrator(succeed_on_key="key-3")

    status, payload = _post_turn(monkeypatch, orchestrator)

    assert status == 200
    assert payload["final_report"] == "rotated success"
    assert orchestrator.headers_seen == [
        "Bearer key-1",
        "Bearer key-2",
        "Bearer key-3",
    ]
