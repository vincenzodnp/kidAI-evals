"""Prompt variants.

`v1` is a faithful Python port of PromptBuilder.kt in the KidAI app (commit a5dad70).
Keep it byte-for-byte equivalent: the point of the evals is to measure what the app
really sends. New variants (e.g. `v2` with extra guardrails) go in VARIANTS below.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from .model import Case, Profile, Sex, Style

SUBJECT_MARKER = "SOGGETTO:"
CONFIDENCE_MARKER = "FIDUCIA:"


@dataclass(frozen=True)
class PromptPair:
    system: str
    user: str


def _describe_listener(p: Profile) -> str:
    gender = {
        Sex.FEMMINA: "una bambina/ragazza",
        Sex.MASCHIO: "un bambino/ragazzo",
        Sex.ALTRO: "una persona",
    }[p.sex]
    if p.age >= 18:
        return f"{p.name}, una persona adulta"
    return f"{p.name}, {gender} di {p.age} anni"


def _register_for_age(age: int) -> str:
    if age < 8:
        return (
            f"Adatta tutto a un ascoltatore di {age} anni: parole semplicissime, frasi molto brevi, "
            "tono giocoso, rivolgiti direttamente a lui/lei con il tu. Esempio di tono: "
            '"Guarda lassù! Quella torre ha quasi mille anni: quando è nata, non esistevano né auto né lampadine!"'
        )
    if age < 12:
        return (
            f"Adatta il racconto a un ascoltatore di {age} anni: linguaggio chiaro, paragoni concreti "
            "con cose che conosce (scuola, sport, videogiochi), qualche domanda retorica per tenere viva l'attenzione."
        )
    if age < 16:
        return (
            f"Adatta il racconto a un ascoltatore di {age} anni: tono da coetaneo brillante, "
            "niente infantilismi, curiosità sorprendenti e collegamenti con il presente."
        )
    return "L'ascoltatore è adulto: registro maturo, ricco ma scorrevole."


def system_v1(profile: Profile, style: Style) -> str:
    return "\n".join([
        "Sei la voce di un'audioguida turistica personale per famiglie, in italiano.",
        f"Il tuo ascoltatore è {_describe_listener(profile)}.",
        "",
        f'REGISTRO NARRATIVO (stile "{style.label}"): {style.instructions}',
        _register_for_age(profile.age),
        "",
        "REGOLE DI FORMA (il testo viene letto ad alta voce da una sintesi vocale):",
        "- Solo prosa continua: niente elenchi puntati, niente titoli, niente markdown, niente emoji, niente asterischi.",
        "- Frasi complete e ben punteggiate; evita parentesi lunghe e sigle non spiegate.",
        "- Durata dell'ascolto: circa un minuto e mezzo, due minuti al massimo.",
        "",
        "REGOLE DI VERITÀ:",
        "- Non inventare date, nomi o fatti. Se non sei sicuro di un dettaglio, ometti il dettaglio",
        '  oppure presentalo esplicitamente come leggenda o ipotesi ("si racconta che...", "forse...").',
        "- Non attribuire mai a un'opera fatti che appartengono a un'altra.",
        "- Contenuti sempre adatti a un ascolto in famiglia: niente violenza esplicita o temi non adatti all'età.",
    ])


def user_photo_v1(case: Case) -> str:
    context = ""
    if case.position is not None:
        context += (
            f"La foto è stata scattata alle coordinate GPS {case.position.lat}, {case.position.lon}. "
        )
    else:
        context += "Non è disponibile la posizione GPS. "
    if case.nearby_pois:
        context += "Luoghi d'interesse nelle vicinanze: " + ", ".join(case.nearby_pois) + ". "
    return "\n".join([
        "Osserva la foto e identifica il soggetto principale (monumento, statua, palazzo, opera d'arte, luogo).",
        context,
        "",
        "Rispondi ESATTAMENTE in questo formato:",
        f"- Prima riga, da sola: {SUBJECT_MARKER} nome del soggetto | {CONFIDENCE_MARKER} alta oppure media oppure bassa",
        "- Poi, dalla riga successiva, la guida nello stile richiesto, SOLO se la fiducia è alta.",
        "  Se la fiducia è media o bassa, scrivi solo la prima riga e fermati.",
        "",
        'Valuta la fiducia con onestà: "alta" solo se sei davvero sicuro dell\'identificazione,',
        '"media" se è un\'ipotesi plausibile, "bassa" se stai tirando a indovinare.',
    ])


def build_v1(case: Case, profile: Profile, style: Style) -> PromptPair:
    return PromptPair(system=system_v1(profile, style), user=user_photo_v1(case))


def build_v1_nogps(case: Case, profile: Profile, style: Style) -> PromptPair:
    """Ablation: v1 exactly, but as if the phone had no location (no GPS, no nearby places).

    Comparing v1 with v1-nogps on the same photos shows how much an answer depends on the
    location rather than on what the image shows.
    """
    return build_v1(replace(case, position=None, nearby_pois=[]), profile, style)


# Register new variants here, e.g. "v2": build_v2
VARIANTS: dict[str, Callable[[Case, Profile, Style], PromptPair]] = {
    "v1": build_v1,
    "v1-nogps": build_v1_nogps,
}
