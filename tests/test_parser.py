"""Port of StreamParserTest.kt (non-streaming)."""
from kidai_evals.model import Confidence
from kidai_evals.parser import parse


def test_high_confidence_then_guide():
    p = parse("SOGGETTO: Tempio di Apollo | FIDUCIA: alta\nC'era una volta un tempio.")
    assert (p.subject, p.confidence, p.header_ok) == ("Tempio di Apollo", Confidence.ALTA, True)
    assert p.guide == "C'era una volta un tempio."


def test_medium_confidence_stops():
    p = parse("SOGGETTO: Forse il Duomo | FIDUCIA: media")
    assert p.confidence is Confidence.MEDIA and p.guide == "" and not p.trailing_text_after_stop


def test_text_after_low_confidence_is_flagged():
    p = parse("SOGGETTO: Boh | FIDUCIA: bassa\nComunque ecco una storia...")
    assert p.guide == "" and p.trailing_text_after_stop


def test_unexpected_format_is_low_confidence():
    p = parse("Ecco cosa vedo: una chiesa.\nÈ molto antica.")
    assert p.confidence is Confidence.BASSA and not p.header_ok
    assert p.subject.startswith("Ecco cosa vedo")


def test_case_insensitive_header():
    assert parse("soggetto: Colosseo | fiducia: ALTA\nRoma ti guarda.").guide == "Roma ti guarda."


def test_overlong_header_is_malformed():
    p = parse("x" * 500 + "\nresto")
    assert p.confidence is Confidence.BASSA and not p.header_ok
