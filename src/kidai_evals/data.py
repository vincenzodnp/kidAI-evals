"""Load cases, profiles and config from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from .model import Case, Position, Profile, Sex


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_profiles(path: str | Path) -> list[Profile]:
    return [
        Profile(id=p["id"], name=p["name"], sex=Sex[p["sex"]], age=int(p["age"]))
        for p in load_yaml(path)["profiles"]
    ]


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    for c in load_yaml(path)["cases"]:
        pos = c.get("position")
        cases.append(Case(
            id=c["id"],
            image=c["image"],
            subject=c["subject"],
            aliases=c.get("aliases", []),
            expect=c.get("expect", "identify"),
            kind=c.get("kind", "standard"),
            position=Position(pos["lat"], pos["lon"]) if pos else None,
            nearby_pois=c.get("nearby_pois", []),
            notes=c.get("notes", ""),
        ))
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate case ids: {sorted(dupes)}")
    return cases
