from kidai_evals.fingerprint import dataset_id, input_id
from kidai_evals.model import Case, Position, Profile, Sex, Style
from kidai_evals.prompts import build_v1

KID = Profile("k", "Sofia", Sex.FEMMINA, 6)


def _case(**fields) -> Case:
    return Case("x", "x.jpg", "Big Ben", **fields)


def test_dataset_id_ignores_notes_but_not_labels(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"photo")
    base = dataset_id([_case()], [KID], tmp_path)
    assert dataset_id([_case()], [KID], tmp_path) == base  # stable
    assert dataset_id([_case(notes="taken at dusk")], [KID], tmp_path) == base
    assert dataset_id([_case(aliases=["Elizabeth Tower"])], [KID], tmp_path) != base


def test_dataset_id_changes_with_the_photo(tmp_path):
    photo = tmp_path / "x.jpg"
    photo.write_bytes(b"photo")
    before = dataset_id([_case()], [KID], tmp_path)
    photo.write_bytes(b"another photo")
    assert dataset_id([_case()], [KID], tmp_path) != before


def test_input_id_follows_the_prompt_not_the_labels():
    def iid(case):
        return input_id(build_v1(case, KID, Style.STORICA), None)

    # The labels are never shown to the model, so changing them keeps the same input...
    assert iid(_case()) == iid(_case(aliases=["Elizabeth Tower"], expect="not_confident"))
    # ...but the GPS position is part of the prompt.
    assert iid(_case()) != iid(_case(position=Position(51.5007, -0.1246)))
