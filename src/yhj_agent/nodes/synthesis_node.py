"""Node8: SynthesisNode - combine specialist outputs with self-consistency."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any
from urllib import request as urllib_request

from yhj_agent.common.config import _read_env
from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.evaluation import SynthesisResult
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest


class SynthesisNode:
    """Combine specialist outputs and choose the most consistent synthesis."""

    def __init__(
        self,
        llm_client: MimoLLMClient | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.llm = llm_client or MimoLLMClient()
        self.prompts = prompt_loader or PromptLoader()
        self._api_key = _read_env("EMBEDDING_API_KEY", "")
        self._base_url = _read_env("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self._model = _read_env("EMBEDDING_MODEL", "text-embedding-v2")

    async def synthesize(
        self,
        specialist_outputs: dict[str, Any],
        debate_output: dict[str, Any] | None = None,
    ) -> SynthesisResult:
        results = await asyncio.gather(
            *[self._single_synthesis(specialist_outputs, debate_output) for _ in range(3)],
            return_exceptions=True,
        )

        logger = logging.getLogger(__name__)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Synthesis sample %d failed: %s: %s", i, type(result).__name__, result)

        valid_results = [result for result in results if isinstance(result, SynthesisResult)]
        if not valid_results:
            first_error = next((result for result in results if isinstance(result, Exception)), None)
            if first_error is not None:
                raise first_error
            raise RuntimeError("Synthesis failed without a valid result or captured exception.")
        if len(valid_results) == 1:
            return valid_results[0]
        return self._select_most_consistent(valid_results)

    async def _single_synthesis(
        self,
        specialist_outputs: dict[str, Any],
        debate_output: dict[str, Any] | None,
    ) -> SynthesisResult:
        system = self.prompts.render("synthesis/system.j2", current_year=__import__("datetime").date.today().year)
        user = self.prompts.render(
            "synthesis/user.j2",
            specialists=specialist_outputs,
            debate_output=debate_output or {"debate_occurred": False},
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            model=self.llm.config.light_model,
            temperature=0.7,
        )
        result = self.llm.chat_structured(request, SynthesisResult)

        if not result.synthesis_text:
            extra = result.__pydantic_extra__ or {}
            parts = []
            for _, value in extra.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if isinstance(subval, dict):
                            text = subval.get("content") or subval.get("核心观点") or subval.get("text") or ""
                            if text:
                                parts.append(f"{subkey}: {text}")
                        elif isinstance(subval, str) and len(subval) > 10:
                            parts.append(f"{subkey}: {subval}")
                elif isinstance(value, str) and len(value) > 10:
                    parts.append(value)
            if parts:
                result.synthesis_text = "\n\n".join(parts)
        return result

    def _select_most_consistent(self, results: list[SynthesisResult]) -> SynthesisResult:
        if len(results) < 2:
            return results[0]

        texts = [result.synthesis_text for result in results]
        embeddings = [self._get_embedding(text) for text in texts]

        if any(embedding is None for embedding in embeddings):
            best = max(results, key=lambda result: result.confidence)
            best.consistency_score = 0.5
            return best

        if len(results) == 2:
            sim_ab = self._cosine_similarity(embeddings[0], embeddings[1])
            if sim_ab > 0.9:
                best = results[0]
                best.consistency_score = round(sim_ab, 3)
                return best
            best = results[0] if results[0].confidence >= results[1].confidence else results[1]
            best.consistency_score = round(sim_ab, 3)
            return best

        sim_ab = self._cosine_similarity(embeddings[0], embeddings[1])
        sim_ac = self._cosine_similarity(embeddings[0], embeddings[2])
        sim_bc = self._cosine_similarity(embeddings[1], embeddings[2])
        avg_similarity = (sim_ab + sim_ac + sim_bc) / 3.0

        if sim_ab > 0.9 and sim_ac > 0.9 and sim_bc > 0.9:
            best = results[0]
            best.consistency_score = round(avg_similarity, 3)
            return best

        pairs = [(sim_ab, 0, 1), (sim_ac, 0, 2), (sim_bc, 1, 2)]
        _, idx_a, idx_b = max(pairs, key=lambda pair: pair[0])
        best = results[idx_a] if results[idx_a].confidence >= results[idx_b].confidence else results[idx_b]
        best.consistency_score = round(avg_similarity, 3)
        return best

    def _get_embedding(self, text: str) -> list[float] | None:
        if not self._api_key:
            return self._deterministic_embedding(text)

        text = text[:2000]
        payload = json.dumps({"model": self._model, "input": text}).encode("utf-8")
        req = urllib_request.Request(
            f"{self._base_url.rstrip('/')}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["embedding"]
        except Exception:
            return self._deterministic_embedding(text)

    def _deterministic_embedding(self, text: str) -> list[float]:
        dims = 1536
        values = [0.0] * dims
        for i, char in enumerate(text):
            bucket = (ord(char) + i) % dims
            values[bucket] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
