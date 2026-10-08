from geoscope import agent
from geoscope.agent import AISettings, execute_tool, list_ollama_models


def test_ai_endpoint_and_disabled_configuration():
    settings = AISettings(base_url="http://localhost:11434/v1", model="test", mode="provider")
    assert settings.enabled
    assert settings.endpoint == "http://localhost:11434/v1/chat/completions"


def test_agent_rejects_unknown_tool():
    assert "Unknown tool" in execute_tool("not_a_tool", {}, lambda _: None)


def test_research_uses_model_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOSCOPE_DATA_DIR", str(tmp_path))
    replies = iter([
        {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {"name": "local_intel", "arguments": '{"query":"test"}'}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "No matching local intelligence."}}]},
    ])

    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return next(replies)

    monkeypatch.setattr(agent.requests, "post", lambda *args, **kwargs: Response())
    activity = []
    answer = agent.research("What is stored?", AISettings(base_url="http://local/v1", model="model", mode="provider"), activity.append)
    assert answer == "No matching local intelligence."
    assert any("local intelligence" in item for item in activity)


def test_ollama_model_discovery(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return {"models": [{"name": "qwen3:latest", "size": 123, "details": {"family": "qwen", "parameter_size": "8B"}}]}
    monkeypatch.setattr(agent.requests, "get", lambda *args, **kwargs: Response())
    assert list_ollama_models() == [{"name": "qwen3:latest", "size": 123, "family": "qwen", "parameters": "8B"}]


def test_provider_key_is_not_sent_to_local_ollama(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def post(*args, **kwargs):
        calls.append(kwargs["headers"])
        return Response()

    monkeypatch.setattr(agent.requests, "post", post)
    settings = AISettings(model="local-model", mode="local", provider_api_key="provider-secret")
    assert agent.research("hello", settings, lambda _: None) == "ok"
    assert "Authorization" not in calls[0]
