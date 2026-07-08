from __future__ import annotations

from types import SimpleNamespace

from agentloop.adapters.grok import GrokAgent
from agentloop.agent import AgentRequest


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="done")


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_grok_agent_uses_responses_api():
    client = FakeClient()
    agent = GrokAgent(client=client, model="grok-4.3", max_output_tokens=123)

    out = agent.run(AgentRequest(role="subagent", system="SYS", prompt="BODY"))

    assert out.text == "done"
    call = client.responses.calls[0]
    assert call["model"] == "grok-4.3"
    assert call["max_output_tokens"] == 123
    assert call["input"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "BODY"},
    ]


def test_grok_agent_reinforces_json_contract():
    client = FakeClient()
    agent = GrokAgent(client=client)

    agent.run(AgentRequest(role="reviewer", system="SYS", prompt="BODY", expects_json=True))

    system = client.responses.calls[0]["input"][0]["content"]
    assert "Return ONLY the JSON value" in system
