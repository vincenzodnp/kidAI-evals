"""Port of PromptBuilderTest.kt, so the Python prompts stay aligned with the app."""
from kidai_evals.model import Case, Position, Profile, Sex, Style
from kidai_evals.prompts import CONFIDENCE_MARKER, SUBJECT_MARKER, system_v1, user_photo_v1

CHILD = Profile("p", "Sofia", Sex.FEMMINA, 6)
ADULT = Profile("a", "Vincenzo", Sex.MASCHIO, 45)


def test_system_has_age_register_and_style():
    p = system_v1(CHILD, Style.FANTASTICA)
    assert "6 anni" in p and "Fantastica" in p and "fiaba" in p.lower()


def test_system_has_anti_invention_rule():
    p = system_v1(ADULT, Style.STORICA)
    assert "Non inventare" in p and "leggenda" in p


def test_system_forbids_markdown_for_tts():
    p = system_v1(ADULT, Style.DETTAGLIATA)
    assert "markdown" in p and "alta voce" in p


def test_adult_listener_description():
    assert "Vincenzo, una persona adulta" in system_v1(ADULT, Style.STORICA)


def test_user_photo_includes_gps_and_pois():
    case = Case("c", "x.jpg", "Duomo", position=Position(40.6263, 14.3757),
                nearby_pois=["Duomo di Amalfi", "Chiostro del Paradiso"])
    p = user_photo_v1(case)
    assert "40.6263" in p and "Duomo di Amalfi" in p
    assert SUBJECT_MARKER in p and CONFIDENCE_MARKER in p


def test_user_photo_without_gps_says_so():
    assert "Non è disponibile la posizione GPS" in user_photo_v1(Case("c", "x.jpg", "x"))
