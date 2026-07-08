"""Grok backend via xAI's OpenAI-compatible API.

Requires `pip install openai` and XAI_API_KEY in the environment. xAI exposes
Grok through the OpenAI SDK by setting base_url to https://api.x.ai/v1.

The default model is Grok Build because agentloop's workers are usually doing
coding-agent tasks. Override `model` with e.g. `grok-4.3` for general chat.
"""

from __future__ import annotations

import os
from typing import Optional

from ..agent import Agent, AgentRequest, AgentResponse


class GrokAgent(Agent):
    def __init__(
        self,
        model: str = "grok-build-0.1",
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
        timeout: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        client=None,
    ):
        self.model = model
        self.max_output_tokens = max_output_tokens
        if client is not None:
            self.client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "GrokAgent requires the OpenAI SDK: pip install openai"
            ) from exc

        kw = {
            "api_key": api_key or os.environ.get("XAI_API_KEY"),
            "base_url": base_url,
        }
        if timeout is not None:
            kw["timeout"] = timeout
        self.client = OpenAI(**kw)

    def run(self, request: AgentRequest) -> AgentResponse:
        system = request.system
        if request.expects_json:
            # Reinforce the format contract; extract_json still tolerates stray prose.
            system += "\n\nReturn ONLY the JSON value, no prose, no code fences."

        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": request.prompt},
            ],
        }
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens

        response = self.client.responses.create(**kwargs)
        return AgentResponse(text=_response_text(response), raw=response)


def _response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text is not None:
        return text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) in ("output_text", "text"):
                value = getattr(part, "text", None)
                if value is not None:
                    chunks.append(value)
    return "".join(chunks)
