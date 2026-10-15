"""LLM-as-judge with Claude, scoring what deterministic checks cannot.

Calibrate before trusting it: score a sample of guides by hand with the same rubric
and check how often the judge agrees with you (see "Next steps" in the README).
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from .model import Case, Profile, Style

RUBRIC_PATH = Path(__file__).resolve().parents[2] / "rubric" / "judge.md"
CRITERIA = ["factual_accuracy", "legend_labelling", "style_adherence", "age_appropriateness", "safety"]


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Judge returned no JSON: {text[:200]!r}")
    data = json.loads(m.group(0))
    for c in CRITERIA:
        if not isinstance(data.get(c), int) or not 1 <= data[c] <= 5:
            raise ValueError(f"Judge score for {c!r} missing or out of range: {data.get(c)!r}")
    return data


class ClaudeJudge:
    def __init__(self, model: str):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()
        self._rubric = RUBRIC_PATH.read_text(encoding="utf-8")

    def score(self, guide: str, case: Case, profile: Profile, style: Style, image_path: Path | None) -> dict:
        content: list[dict] = []
        if image_path is not None:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.b64encode(image_path.read_bytes()).decode()},
            })
        content.append({"type": "text", "text": (
            f"Soggetto reale: {case.subject}\n"
            f"Ascoltatore: {profile.age} anni\n"
            f"Stile richiesto: {style.label} — {style.instructions}\n\n"
            f"Testo della guida:\n<<<\n{guide}\n>>>"
        )})
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._rubric,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _extract_json(text)
