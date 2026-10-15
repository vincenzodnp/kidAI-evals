"""Model providers: Gemini (the app's default engine), Claude, and a fake one for offline tests.

Request parameters mirror the app's GeminiProvider.kt and ClaudeProvider.kt
(max 2048 output tokens; Claude with low effort to favour latency).
API keys come from the environment: GEMINI_API_KEY, ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .model import Case
from .prompts import PromptPair

MAX_OUTPUT_TOKENS = 2048


@dataclass
class Completion:
    text: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider(Protocol):
    name: str
    model: str

    def complete(self, prompt: PromptPair, image_path: Path | None, case: Case) -> Completion: ...


def _jpeg_bytes(image_path: Path) -> bytes:
    return image_path.read_bytes()


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str):
        from google import genai  # imported lazily so tests don't need the SDK configured

        self.model = model
        self._genai = genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def complete(self, prompt: PromptPair, image_path: Path | None, case: Case) -> Completion:
        types = self._genai.types
        parts = []
        if image_path is not None:
            parts.append(types.Part.from_bytes(data=_jpeg_bytes(image_path), mime_type="image/jpeg"))
        parts.append(prompt.user)
        start = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=prompt.system, max_output_tokens=MAX_OUTPUT_TOKENS
            ),
        )
        latency = time.perf_counter() - start
        usage = getattr(resp, "usage_metadata", None)
        return Completion(
            text=resp.text or "",
            latency_s=latency,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str, effort: str | None = "low"):
        import anthropic

        self.model = model
        self.effort = effort
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, prompt: PromptPair, image_path: Path | None, case: Case) -> Completion:
        content: list[dict] = []
        if image_path is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(_jpeg_bytes(image_path)).decode(),
                },
            })
        content.append({"type": "text", "text": prompt.user})
        extra = {"output_config": {"effort": self.effort}} if self.effort else None
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=prompt.system,
            messages=[{"role": "user", "content": content}],
            extra_body=extra,
        )
        latency = time.perf_counter() - start
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return Completion(
            text=text,
            latency_s=latency,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class FakeProvider:
    """Deterministic stand-in, like FakeEngine.kt: lets the whole pipeline run without keys."""

    name = "fake"

    def __init__(self, model: str = "fake-1"):
        self.model = model

    def complete(self, prompt: PromptPair, image_path: Path | None, case: Case) -> Completion:
        if case.expect == "not_confident":
            text = "SOGGETTO: soggetto incerto | FIDUCIA: bassa"
        else:
            text = (
                f"SOGGETTO: {case.subject} | FIDUCIA: alta\n"
                f"Guarda bene: questo è {case.subject}. È un luogo famoso e molto amato. "
                "Tante persone vengono qui ogni anno per vederlo. Si racconta che chi lo osserva "
                "con attenzione scopra sempre un dettaglio nuovo."
            )
        return Completion(text=text, latency_s=0.0, input_tokens=0, output_tokens=0)


def make_provider(name: str, model: str, **kwargs) -> Provider:
    if name == "gemini":
        return GeminiProvider(model)
    if name == "claude":
        return ClaudeProvider(model, effort=kwargs.get("effort", "low"))
    if name == "fake":
        return FakeProvider(model)
    raise ValueError(f"Unknown provider: {name}")
