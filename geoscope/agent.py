"""Optional OpenAI-compatible tool-using research agent."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable

import requests

from geoscope.browser import BrowserError, read, search
from geoscope.collectors import earthquakes, weather
from geoscope.agent_reach import AgentReachError, search_twitter
from geoscope.storage import list_items

QWEN_TINY_MODEL = "qwen3:0.6b-q4_K_M"


@dataclass
class AISettings:
    base_url: str = os.environ.get("GEOSCOPE_AI_BASE_URL", "")
    provider_api_key: str = os.environ.get("GEOSCOPE_PROVIDER_API_KEY", os.environ.get("GEOSCOPE_AI_API_KEY", ""))
    model: str = os.environ.get("GEOSCOPE_AI_MODEL", "")
    mode: str = "provider"
    ollama_url: str = os.environ.get("GEOSCOPE_OLLAMA_URL", "http://127.0.0.1:11434")

    @property
    def active_base_url(self) -> str:
        if self.mode == "local":
            return self.ollama_url.rstrip("/") + "/v1"
        return self.base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.active_base_url and self.model)

    @property
    def endpoint(self) -> str:
        base = self.active_base_url
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


class ModelDiscoveryError(RuntimeError):
    pass


def list_ollama_models(base_url: str = "http://127.0.0.1:11434") -> list[dict]:
    """Return models already downloaded into the local Ollama library."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=4)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (requests.RequestException, ValueError) as exc:
        raise ModelDiscoveryError(f"Could not reach Ollama at {base_url}: {exc}") from exc
    return [{"name": item.get("name") or item.get("model", "unknown"), "size": item.get("size", 0),
             "family": item.get("details", {}).get("family", ""),
             "parameters": item.get("details", {}).get("parameter_size", "")} for item in models]


TOOLS = [
    {"type": "function", "function": {"name": "x_search", "description": "Read recent public X/Twitter posts through the active Agent Reach backend. Geoscope will show the source posts separately without rewriting them.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search current public web sources for a research question.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 8}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_url", "description": "Open and extract readable text from a public HTTP(S) URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "weather", "description": "Get current weather for a location from Open-Meteo.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "earthquakes", "description": "Get recent public USGS earthquake events.", "parameters": {"type": "object", "properties": {"minimum_magnitude": {"type": "number", "default": 4.5}}}}},
    {"type": "function", "function": {"name": "local_intel", "description": "Search information already stored in the local Geoscope workspace.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "category": {"type": "string"}}, "required": ["query"]}}},
]


def _compact(value, limit: int = 8) -> str:
    if isinstance(value, list):
        value = value[:limit]
    return json.dumps(value, ensure_ascii=False, default=str)[:14_000]


def _fallback_x_query(prompt: str) -> str:
    """Recover a concise X query when the tiny model omits its tool argument."""
    match = re.search(
        r"\b(?:search|find|look up)\s+(?:on\s+)?(?:x|twitter)\s+(?:for|about)\s+(.+?)"
        r"(?:\s+(?:and|then)\s+(?:summari[sz]e|tell|report|show|explain|list)\b|[?.!]|$)",
        prompt,
        flags=re.IGNORECASE,
    )
    if match:
        query = match.group(1).strip(" \t\r\n'\"`.,?!")
        if query:
            return query
    return prompt.strip()


def _qwen_text_tool_call(content: str, prompt: str) -> list[dict]:
    """Translate Qwen 0.6B's textual <search> envelope into its one allowed tool."""
    match = re.search(
        r"<(?P<tag>x_search|search)>\s*(?P<body>\{.*?\})\s*</(?P=tag)>",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    try:
        payload = json.loads(match.group("body"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    arguments = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    arguments = dict(arguments)
    arguments["query"] = str(arguments.get("query") or _fallback_x_query(prompt))
    try:
        limit = int(arguments.get("limit", 10) or 10)
    except (TypeError, ValueError):
        limit = 10
    arguments["limit"] = max(1, min(limit, 10))
    return [{
        "type": "function",
        "function": {"name": "x_search", "arguments": arguments},
    }]


def execute_tool(name: str, arguments: dict, emit: Callable[[str], None]) -> str:
    """Run an approved local module; models never receive shell access."""
    try:
        if name == "x_search":
            query = str(arguments["query"])
            emit(f"AGENT REACH  X search: {query}")
            rows = search_twitter(query, int(arguments.get("limit", 10)), emit=emit)
            return _compact([{
                "author": item["author"], "text": item["body"], "url": item["source_url"],
                "published_at": item["published_at"], "backend": item["backend"],
            } for item in rows], limit=10)
        if name == "web_search":
            query = arguments["query"]
            emit(f"SEARCH  {query}")
            return _compact(search(query, int(arguments.get("limit", 8)), progress=lambda text: emit(f"SOURCE  {text}")))
        if name == "open_url":
            url = arguments["url"]
            emit(f"OPEN    {url}")
            return _compact(read(url, progress=lambda text: emit(f"SOURCE  {text}")))
        if name == "weather":
            location = arguments["location"]
            emit(f"MODULE  weather: {location}")
            return _compact(weather(location))
        if name == "earthquakes":
            magnitude = float(arguments.get("minimum_magnitude", 4.5))
            emit(f"MODULE  earthquakes ≥ M{magnitude}")
            return _compact(earthquakes(magnitude))
        if name == "local_intel":
            emit("MODULE  local intelligence search")
            return _compact(list_items(arguments.get("category"), arguments["query"], 20))
        return json.dumps({"error": f"Unknown tool: {name}"})
    except (AgentReachError, BrowserError, KeyError, ValueError, requests.RequestException) as exc:
        emit(f"ERROR   {name}: {exc}")
        return json.dumps({"error": str(exc)})


def research(prompt: str, settings: AISettings, emit: Callable[[str], None]) -> str:
    """Ask a compatible model to research with Geoscope's constrained tools."""
    if not settings.enabled:
        emit("SEARCH  AI is off; running a direct web search.")
        results = search(prompt, progress=lambda text: emit(f"SOURCE  {text}"))
        return "\n".join(f"{item['title']}\n{item['url']}\n{item['snippet']}" for item in results)

    native_qwen = settings.mode == "local" and settings.model.startswith("qwen3:")
    emit(f"AI      contacting {settings.mode} model ({settings.model})")
    if native_qwen:
        system_prompt = (
            "You are Geoscope's small local helper. Keep replies short and handle basic questions and "
            "Geoscope help. For current X discussion, call x_search with a concise, nonempty query. "
            "Never invent live facts or claim "
            "you accessed a source you did not use. The TUI presents retrieved posts as source text; "
            "your reply is a separate explanation and must not be presented as the original post."
        )
        available_tools = [TOOLS[0]]
    else:
        system_prompt = (
            "You are Geoscope, a cautious research assistant. Use tools for current facts. State sources, "
            "distinguish evidence from inference, and never claim to have accessed a source you did not use."
        )
        available_tools = TOOLS
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    headers = {"Content-Type": "application/json"}
    if settings.mode == "provider" and settings.provider_api_key:
        headers["Authorization"] = f"Bearer {settings.provider_api_key}"
    for _ in range(2 if native_qwen else 6):
        try:
            if native_qwen:
                endpoint = settings.ollama_url.rstrip("/") + "/api/chat"
                payload = {"model": settings.model, "messages": messages, "tools": available_tools,
                           "stream": False, "think": False, "options": {"temperature": 0.2}}
            else:
                endpoint = settings.endpoint
                payload = {"model": settings.model, "messages": messages, "tools": available_tools,
                           "tool_choice": "auto", "temperature": 0.2}
            response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            message = result.get("message", {}) if native_qwen else result["choices"][0]["message"]
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            return f"AI connection failed: {exc}"
        calls = message.get("tool_calls") or []
        if native_qwen and not calls:
            calls = _qwen_text_tool_call(str(message.get("content") or ""), prompt)
            if calls:
                # Convert Qwen's textual action envelope to Ollama's native
                # assistant/tool message sequence before continuing the turn.
                message = {"role": "assistant", "tool_calls": calls}
        messages.append(message)
        if not calls:
            return message.get("content") or "The model returned no response."
        for call in calls:
            function = call.get("function", {})
            raw_arguments = function.get("arguments") or {}
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                try:
                    arguments = json.loads(raw_arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
            if native_qwen and function.get("name") == "x_search" and not arguments.get("query"):
                arguments["query"] = _fallback_x_query(prompt)
            output = execute_tool(function.get("name", ""), arguments, emit)
            if native_qwen:
                if function.get("name") == "x_search":
                    try:
                        tool_result = json.loads(output)
                    except json.JSONDecodeError:
                        tool_result = None
                    if isinstance(tool_result, dict) and tool_result.get("error"):
                        return f"X search failed: {tool_result['error']}"
                messages.append({"role": "tool", "tool_name": function.get("name", "tool"), "content": output})
            else:
                messages.append({"role": "tool", "tool_call_id": call.get("id", function.get("name", "tool")), "content": output})
    return "Research stopped after the maximum number of tool rounds. Refine the request to continue."
