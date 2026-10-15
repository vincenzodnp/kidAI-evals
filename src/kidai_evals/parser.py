"""Parse a complete model response with the same rules as StreamParser.kt.

The app streams, but its decisions only depend on the first line, so a
non-streaming port is equivalent for evaluation purposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Confidence

HEADER_RE = re.compile(r"^\s*SOGGETTO:\s*(.+?)\s*\|\s*FIDUCIA:\s*(alta|media|bassa)\s*$", re.IGNORECASE)
HEADER_LIMIT = 400  # same as LIMITE_INTESTAZIONE


@dataclass
class Parsed:
    subject: str
    confidence: Confidence
    header_ok: bool          # the first line matched the contract
    guide: str               # text the app would read aloud ("" if it stops)
    trailing_text_after_stop: bool  # model kept writing although confidence wasn't "alta"


def parse(output: str) -> Parsed:
    newline = output.find("\n")
    if newline < 0:
        first, rest = output, ""
        if len(first) > HEADER_LIMIT:
            return Parsed(first.strip()[:120], Confidence.BASSA, False, "", False)
    else:
        first, rest = output[:newline], output[newline + 1:]
        if newline > HEADER_LIMIT:
            return Parsed(output[:HEADER_LIMIT].strip()[:120], Confidence.BASSA, False, "", False)

    m = HEADER_RE.search(first)
    if m is None:
        # App behaviour: never pretend to know; treat as low-confidence identification.
        return Parsed(first.strip()[:120], Confidence.BASSA, False, "", bool(rest.strip()))

    confidence = Confidence(m.group(2).lower())
    subject = m.group(1).strip()
    if confidence is not Confidence.ALTA:
        return Parsed(subject, confidence, True, "", bool(rest.strip()))
    return Parsed(subject, confidence, True, rest.lstrip("\r\n").strip(), False)
