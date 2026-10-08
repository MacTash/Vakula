"""The model's role: explain intelligence, never produce it.

Everything upstream of this module is arithmetic over stored evidence. The model
sits strictly downstream and receives an already-measured picture. It writes
sentences. It does not decide what happened, how confident anyone should be,
what the probability is, or whether a contradiction is settled.

The interface is deliberately tiny, and Ollama is one implementation of it rather
than the assumption behind it. A different local model, or any
OpenAI-compatible endpoint, becomes a drop-in replacement without touching the
intelligence layer.

There is no code path from model output to anything executable. Text returned by
a model is prose: it is rendered into a briefing and stored in the assessments
table, which is separate from source evidence. No subprocess is spawned, no
shell is invoked, no path is built from it, and no database row it could
influence is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import requests

from vidur.agent import AISettings, list_ollama_models

DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 90
PROBE_TIMEOUT = 4


class ModelUnavailableError(RuntimeError):
    """The model cannot be reached or has not been selected.

    Raised rather than swallowed so a caller can fall back deliberately instead
    of receiving an empty string it might mistake for a real answer.
    """


@runtime_checkable
class IntelligenceModel(Protocol):
    """What the intelligence layer needs from any model."""

    name: str

    def is_available(self) -> bool:
        """Whether this model can answer right now."""

    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = DEFAULT_TEMPERATURE) -> str:
        """Return model prose for a prompt. Raise if it cannot be produced."""


@dataclass
class OllamaModel:
    """Local Ollama backend, driven by the application's existing settings.

    Reuses :class:`~vidur.agent.AISettings` and the existing model listing so
    nothing about the Research agent's configuration changes. It never pulls
    weights: a missing model is an error the caller reports, never a download
    triggered behind their back.
    """

    settings: AISettings
    name: str = "ollama"
    timeout: int = DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        self._reachable: bool | None = None

    @property
    def model_name(self) -> str:
        return self.settings.model

    @property
    def base_url(self) -> str:
        return (self.settings.ollama_url or "http://127.0.0.1:11434").rstrip("/")

    def is_available(self) -> bool:
        """A model must be selected and Ollama must answer. No downloads occur."""
        if not self.settings.model:
            return False
        if self._reachable is None:
            try:
                requests.get(f"{self.base_url}/api/tags", timeout=PROBE_TIMEOUT).raise_for_status()
                self._reachable = True
            except requests.RequestException:
                self._reachable = False
        return self._reachable

    def installed_models(self) -> list[str]:
        """Models already present locally, for reporting what could be chosen."""
        try:
            return [entry["name"] for entry in list_ollama_models(self.base_url)]
        except Exception:
            return []

    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = DEFAULT_TEMPERATURE) -> str:
        """Ask Ollama for prose. Tools are never offered and weights never pulled."""
        if not self.settings.model:
            raise ModelUnavailableError("No model is selected.")
        if not self.is_available():
            raise ModelUnavailableError(
                f"Ollama is not reachable at {self.base_url}.")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": float(temperature)},
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload,
                                     headers={"Content-Type": "application/json"},
                                     timeout=self.timeout)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
        except (requests.RequestException, ValueError, AttributeError) as exc:
            raise ModelUnavailableError(f"Ollama did not return a usable reply: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelUnavailableError("The model returned no text.")
        return content.strip()


class NullModel:
    """A model that is never available, used to force the deterministic path."""

    name = "none"

    def is_available(self) -> bool:
        return False

    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = DEFAULT_TEMPERATURE) -> str:
        raise ModelUnavailableError("No model is configured.")


class StubModel:
    """A scripted model for tests and for exercising the fallback deliberately.

    Accepts whatever reply it was given and records the prompts it received, so a
    test can assert on exactly what the intelligence layer handed over.
    """

    name = "stub"

    def __init__(self, reply: str = "", *, available: bool = True) -> None:
        self.reply = reply
        self._available = available
        self.prompts: list[dict] = []
        self.systems: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = DEFAULT_TEMPERATURE) -> str:
        if not self._available:
            raise ModelUnavailableError("Stub model is marked unavailable.")
        self.prompts.append({"prompt": prompt, "temperature": temperature})
        self.systems.append(system)
        return self.reply


def model_from_settings(settings: AISettings) -> IntelligenceModel:
    """Build the backend matching the current configuration.

    Local mode uses Ollama. Provider mode keeps its key in memory for the
    session only, as the Research agent already does, and never writes it down.
    """
    if settings.mode == "local":
        return OllamaModel(settings)
    return _ProviderModel(settings)


class _ProviderModel:
    """OpenAI-compatible endpoint, used only in explicit provider mode."""

    name = "provider"

    def __init__(self, settings: AISettings, *, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.settings = settings
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.settings.enabled)

    def complete(self, prompt: str, *, system: str = "",
                 temperature: float = DEFAULT_TEMPERATURE) -> str:
        if not self.settings.enabled:
            raise ModelUnavailableError("No provider model is configured.")
        headers = {"Content-Type": "application/json"}
        # The key is read from settings for this request and never persisted.
        if self.settings.provider_api_key:
            headers["Authorization"] = f"Bearer {self.settings.provider_api_key}"
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": prompt}]
        payload = {"model": self.settings.model, "messages": messages,
                   "temperature": float(temperature)}
        try:
            response = requests.post(self.settings.endpoint, json=payload,
                                     headers=headers, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"].get("content", "")
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(f"Provider did not return a usable reply: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelUnavailableError("The model returned no text.")
        return content.strip()
