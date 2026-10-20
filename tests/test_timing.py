"""Streaming milestones and the consistency metric."""
from kidai_evals.cli import consistency
from kidai_evals.providers import StreamTimer, first_sentence


def test_first_sentence_follows_the_app_rules():
    assert first_sentence("Ciao Sofia! Guarda qui") == "Ciao Sofia!"
    assert first_sentence("Costruita nel 1.5 secolo") is None      # "1.5" does not end a sentence
    assert first_sentence("Santa Maria. Poi") == "Santa Maria."
    assert first_sentence("\n\nCiao\nresto") == "Ciao"             # blank lines skipped, newline ends a sentence
    assert first_sentence("frase senza fine") is None


def _timer(chunks):
    t = StreamTimer()
    t.start = 0.0
    for when, chunk in chunks:
        t.feed(chunk, now=when)
    return t


def test_milestones_for_a_confident_answer():
    t = _timer([(1.0, "SOGG"), (2.0, "ETTO: Big Ben | FIDUCIA: alta\n"), (3.0, "\nCiao Sofia"),
                (4.0, "! Guarda"), (5.0, " la torre.")])
    assert (t.ttft_s, t.header_s, t.first_sentence_s) == (1.0, 2.0, 4.0)


def test_no_first_sentence_when_not_confident():
    t = _timer([(1.0, "SOGGETTO: forse un ponte | FIDUCIA: media\n"), (2.0, "Comunque ecco. Altro.")])
    assert t.header_s == 1.0 and t.first_sentence_s is None


def _row(case, correct, conf):
    return {"case_id": case, "checks": {"subject_correct": correct, "confidence": conf}}


def test_consistency_counts_photos_with_one_outcome():
    rows = [_row("a", True, "alta"), _row("a", True, "alta"),     # consistent
            _row("b", False, "alta"), _row("b", True, "media"),   # same photo, different outcome
            _row("c", True, "alta")]                              # asked once: not counted
    assert consistency(rows) == (1, 2)
