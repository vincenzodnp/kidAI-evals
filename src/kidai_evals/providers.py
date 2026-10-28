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
from .parser import HEADER_RE
from .prompts import PromptPair

MAX_OUTPUT_TOKENS = 2048


@dataclass
class Completion:
    text: str
    latency_s: float                     # time until the whole answer has arrived
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_s: float | None = None          # time to the first streamed text
    header_s: float | None = None        # time until the SOGGETTO/FIDUCIA line is complete
    first_sentence_s: float | None = None  # time until the app's TTS could start speaking
    # Why the model stopped, in the provider's own words: Gemini "STOP", "MAX_TOKENS", "SAFETY", ...;
    # Claude "end_turn", "max_tokens", "refusal". Anything but the normal stop explains a cut-off guide.
    stop_reason: str | None = None
    stop_detail: str | None = None       # e.g. the safety categories that blocked the answer
    # Gemini only: tokens spent "thinking" before answering. They count towards max_output_tokens
    # but are not in output_tokens, so a long think can leave too little room for the guide.
    thinking_tokens: int | None = None


def first_sentence(text: str) -> str | None:
    """Return the first complete sentence, using the same rules as SentenceSegmenter.kt.

    A sentence ends at . ! ? … or a newline, but only if followed by whitespace or the end
    of the text, so "1.5" or "S.Maria" don't split. Empty sentences (no letters/digits) are skipped.
    In a stream the end of the buffer is temporary, so callers only check after a chunk arrives,
    exactly like the app does.
    """
    start = 0
    for i, ch in enumerate(text):
        if ch in ".!?…\n":
            end = i + 1
            closed = ch == "\n" or end >= len(text) or text[end].isspace()
            if closed:
                sentence = text[start:end].strip()
                if any(c.isalnum() for c in sentence):
                    return sentence
                start = end
    return None


class StreamTimer:
    """Records when the milestones that matter to the listener happen while text streams in.

    Mirrors the app: StreamParser waits for the first line (header); if confidence is "alta",
    the rest is fed to SentenceSegmenter and the TTS starts on the first complete sentence.
    """

    def __init__(self):
        self.start = time.perf_counter()
        self.text = ""
        self.ttft_s = self.header_s = self.first_sentence_s = None

    def feed(self, chunk: str, now: float | None = None) -> None:
        if not chunk:
            return
        elapsed = (now if now is not None else time.perf_counter()) - self.start
        self.text += chunk
        if self.ttft_s is None:
            self.ttft_s = elapsed
        newline = self.text.find("\n")
        if newline < 0:
            return
        if self.header_s is None:
            self.header_s = elapsed
        header = HEADER_RE.search(self.text[:newline])
        if self.first_sentence_s is None and header and header.group(2).lower() == "alta":
            if first_sentence(self.text[newline + 1:].lstrip("\r\n")):
                self.first_sentence_s = elapsed

    def finish(self, input_tokens=None, output_tokens=None) -> "Completion":
        total = time.perf_counter() - self.start
        if self.header_s is None and self.text:
            self.header_s = total  # single-line answer: the header is complete when the stream ends
        return Completion(self.text, total, input_tokens, output_tokens,
                          self.ttft_s, self.header_s, self.first_sentence_s)


def _name(value) -> str:
    """Enum member to its name (FinishReason.SAFETY -> "SAFETY"); plain strings pass through."""
    return getattr(value, "name", None) or str(value)


def gemini_stop(chunk) -> tuple[str | None, str | None]:
    """(stop_reason, stop_detail) from one streamed Gemini chunk, or (None, None) if it has none.

    Only the last chunk carries a finish reason. If the request itself was blocked there is no
    candidate at all, just a prompt_feedback with a block reason.
    """
    feedback = getattr(chunk, "prompt_feedback", None)
    if feedback is not None and feedback.block_reason:
        return f"PROMPT_BLOCKED_{_name(feedback.block_reason)}", feedback.block_reason_message
    for candidate in chunk.candidates or []:
        if candidate.finish_reason:
            blocked = [_name(r.category) for r in candidate.safety_ratings or [] if r.blocked]
            detail = ", ".join(blocked) or candidate.finish_message or None
            return _name(candidate.finish_reason), detail
    return None, None


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
        timer = StreamTimer()
        usage = None
        stop_reason = stop_detail = None
        # Streaming, like the app, so we can time the first sentence the child would hear.
        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=prompt.system, max_output_tokens=MAX_OUTPUT_TOKENS
            ),
        ):
            timer.feed(chunk.text or "")
            usage = getattr(chunk, "usage_metadata", None) or usage  # the last chunk carries the totals
            reason, detail = gemini_stop(chunk)
            if reason:
                stop_reason, stop_detail = reason, detail
        completion = timer.finish(
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )
        completion.stop_reason, completion.stop_detail = stop_reason, stop_detail
        completion.thinking_tokens = getattr(usage, "thoughts_token_count", None)
        return completion


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
        timer = StreamTimer()
        # Streaming, like the app (ClaudeProvider.kt uses SSE), to time the first sentence.
        with self._client.messages.stream(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=prompt.system,
            messages=[{"role": "user", "content": content}],
            extra_body=extra,
        ) as stream:
            for text in stream.text_stream:
                timer.feed(text)
            final = stream.get_final_message()
        completion = timer.finish(final.usage.input_tokens, final.usage.output_tokens)
        completion.stop_reason = final.stop_reason
        return completion


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
        timer = StreamTimer()
        for word in text.split(" "):  # stream word by word so the timing code runs offline too
            timer.feed(word + " ")
        completion = timer.finish(input_tokens=0, output_tokens=0)
        completion.text = text  # keep the exact text (the loop adds a trailing space)
        completion.stop_reason = "STOP"
        return completion


def make_provider(name: str, model: str, **kwargs) -> Provider:
    if name == "gemini":
        return GeminiProvider(model)
    if name == "claude":
        return ClaudeProvider(model, effort=kwargs.get("effort", "low"))
    if name == "fake":
        return FakeProvider(model)
    raise ValueError(f"Unknown provider: {name}")
