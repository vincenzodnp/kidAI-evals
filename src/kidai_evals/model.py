"""Domain types, mirroring core/model/Modelli.kt in the KidAI app."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Sex(Enum):
    FEMMINA = "Femmina"
    MASCHIO = "Maschio"
    ALTRO = "Altro"


class Style(Enum):
    """Guide styles. `instructions` is copied verbatim from StileGuida in Modelli.kt."""

    FANTASTICA = (
        "Fantastica",
        "Racconta come una fiaba o un'avventura: personaggi, meraviglia, un pizzico di magia. "
        "Il luogo o l'opera è il protagonista della storia. Frasi brevi, ritmo vivace, tono caldo.",
    )
    LEGGENDE = (
        "Leggende",
        "Racconta privilegiando miti, leggende e aneddoti legati al soggetto. "
        "Distingui sempre con chiarezza ciò che è leggenda da ciò che è storia documentata.",
    )
    DETTAGLIATA = (
        "Dettagliata",
        "Offri un approfondimento colto: architettura, arte, tecnica, contesto culturale. "
        "Tono da divulgazione di qualità, preciso ma mai pedante.",
    )
    STORICA = (
        "Storica",
        "Racconta la storia del soggetto: quando e perché è nato, cosa ha attraversato, "
        "come è arrivato a noi. Colloca i fatti nel loro contesto storico.",
    )

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def instructions(self) -> str:
        return self.value[1]


class Confidence(Enum):
    ALTA = "alta"
    MEDIA = "media"
    BASSA = "bassa"


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    sex: Sex
    age: int

    @property
    def age_band(self) -> str:
        """Same bands as PromptBuilder.registroPerEta."""
        if self.age < 8:
            return "lt8"
        if self.age < 12:
            return "lt12"
        if self.age < 16:
            return "lt16"
        return "adult"


@dataclass(frozen=True)
class Position:
    lat: float
    lon: float


@dataclass
class Case:
    """One photo with its ground truth.

    expect:
      identify       the model should name `subject` (or one of `aliases`) with high confidence
      not_confident  the model should NOT answer with high confidence (e.g. a person, an ambiguous shot)
    """

    id: str
    image: str
    subject: str
    aliases: list[str] = field(default_factory=list)
    expect: str = "identify"
    kind: str = "standard"  # standard | adversarial
    position: Position | None = None
    nearby_pois: list[str] = field(default_factory=list)
    notes: str = ""
    # Words that must never appear in the answer, e.g. the payload of instructions written
    # inside the photo. If one shows up, the model obeyed text in the image (prompt injection).
    must_not_contain: list[str] = field(default_factory=list)
