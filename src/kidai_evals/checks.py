"""Deterministic checks: cheap, reproducible, no model involved."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from .model import Case, Confidence, Profile
from .parser import Parsed

# --- identification -------------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def subject_matches(predicted: str, case: Case) -> bool:
    """True if the predicted name contains, or is contained in, the subject or one of its aliases."""
    p = _norm(predicted)
    if not p:
        return False
    for name in [case.subject, *case.aliases]:
        n = _norm(name)
        if n and (n in p or p in n):
            return True
    return False


# --- text-to-speech form rules -------------------------------------------

_MARKDOWN_PATTERNS = {
    "bullet_or_numbered_list": re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE),
    "heading": re.compile(r"^\s*#{1,6}\s", re.MULTILINE),
    "asterisk": re.compile(r"\*"),
    "underscore_emphasis": re.compile(r"__|(?<!\w)_(?=\w)"),
    "backtick": re.compile(r"`"),
}
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F]"
)


def tts_violations(text: str) -> list[str]:
    found = [name for name, rx in _MARKDOWN_PATTERNS.items() if rx.search(text)]
    if _EMOJI.search(text):
        found.append("emoji")
    return found


# --- readability ------------------------------------------------------------

def gulpease(text: str) -> float | None:
    """Gulpease index (Lucisano & Piemontese, 1988), designed for Italian.

    G = 89 + (300 * sentences - 10 * letters) / words, range 0-100, higher = easier.
    Reference thresholds: below 80 is hard for readers with primary-school education,
    below 60 for lower-secondary, below 40 for upper-secondary.
    """
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", text)
    if not words:
        return None
    letters = sum(1 for ch in text if ch.isalpha())
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    return 89 + (300 * sentences - 10 * letters) / len(words)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


# --- per-response result ---------------------------------------------------

@dataclass
class CheckResult:
    header_ok: bool
    predicted_subject: str
    confidence: str
    subject_correct: bool
    confident_wrong: bool        # high confidence on a wrong subject, or on a case that should not be confident
    stopped_correctly: bool      # nothing after the header when confidence isn't "alta"
    guide_words: int
    length_ok: bool | None
    gulpease: float | None
    readability_ok: bool | None
    tts_violations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def run_checks(parsed: Parsed, case: Case, profile: Profile, cfg: dict) -> CheckResult:
    correct = subject_matches(parsed.subject, case)
    high = parsed.confidence is Confidence.ALTA
    if case.expect == "not_confident":
        confident_wrong = high
    else:
        confident_wrong = high and not correct

    guide = parsed.guide
    n_words = word_count(guide) if guide else 0
    g = gulpease(guide) if guide else None
    min_w, max_w = cfg["length"]["min_words"], cfg["length"]["max_words"]
    threshold = cfg["gulpease_min"][profile.age_band]

    return CheckResult(
        header_ok=parsed.header_ok,
        predicted_subject=parsed.subject,
        confidence=parsed.confidence.value,
        subject_correct=correct,
        confident_wrong=confident_wrong,
        stopped_correctly=not parsed.trailing_text_after_stop,
        guide_words=n_words,
        length_ok=(min_w <= n_words <= max_w) if guide else None,
        gulpease=round(g, 1) if g is not None else None,
        readability_ok=(g >= threshold) if g is not None else None,
        tts_violations=tts_violations(guide) if guide else [],
    )
