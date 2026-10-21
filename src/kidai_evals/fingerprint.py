"""Fingerprints: short hashes that say which data, prompt and code produced a result.

They make runs comparable over time. Two runs with the same `dataset_id` were scored on the
same photos, labels and profiles, so their numbers can sit side by side in a chart.

- dataset_id  every case in cases.yaml (labels, GPS, ...), every photo's bytes and every profile
- input_id    exactly what the model received for one answer: the prompt text and the photo
- config_id   the thresholds in config/eval.yaml used by the checks
- git_commit  the version of this code, with "-dirty" if src/ has uncommitted changes
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from .model import Case, Profile
from .prompts import PromptPair


def _sha(data: bytes) -> str:
    """First 12 hex characters of SHA-256: short enough to read, long enough to never collide here."""
    return hashlib.sha256(data).hexdigest()[:12]


def _json_bytes(value) -> bytes:
    # sort_keys makes the text, and so the hash, independent of field order.
    # default=str turns enums (e.g. Sex) into plain text.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


def image_hash(path: Path) -> str:
    """Hash of a photo's bytes, or "missing" if the file isn't there."""
    return _sha(path.read_bytes()) if path.exists() else "missing"


def dataset_id(cases: list[Case], profiles: list[Profile], images_dir: Path) -> str:
    """Fingerprint of the whole dataset. Editing a note doesn't change it; editing a label or a photo does."""
    h = hashlib.sha256()
    for case in cases:
        fields = asdict(case)
        del fields["notes"]  # free text for humans, never used by the prompt or the checks
        h.update(_json_bytes(fields))
        h.update(image_hash(images_dir / case.image).encode())
    for profile in profiles:
        h.update(_json_bytes(asdict(profile)))
    return h.hexdigest()[:12]


def input_id(prompt: PromptPair, image_path: Path | None) -> str:
    """Fingerprint of what the model saw. If it changes, an old answer can't be re-scored."""
    image = image_hash(image_path) if image_path else "none"
    return _sha(_json_bytes([prompt.system, prompt.user, image]))


def config_id(cfg: dict) -> str:
    return _sha(_json_bytes(cfg))


def git_commit(repo: Path) -> str | None:
    """Current commit of the code, or None outside a git checkout."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.strip()
        # Only code changes count as "dirty": data changes are already captured by dataset_id.
        changed = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=repo,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if changed else "")
