"""End-to-end run with the fake provider: no API keys, no images."""
import json

from kidai_evals.cli import main


def test_fake_run_and_report(tmp_path, capsys):
    assert main(["run", "--provider", "fake", "--model", "fake-1", "--out", str(tmp_path)]) == 0
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert rows and all(r["checks"]["header_ok"] for r in rows)
    blurry = [r for r in rows if r["case_id"] == "mummy-head-blurry"]
    assert blurry and not any(r["checks"]["confident_wrong"] for r in blurry)

    assert main(["report", "--results", str(tmp_path)]) == 0
    assert "| fake | fake-1 | v1 |" in capsys.readouterr().out


class _Flaky:
    """Fails `fail_times` times, then answers like the fake provider."""
    name, model = "flaky", "flaky-1"

    def __init__(self, fail_times):
        from kidai_evals.providers import FakeProvider
        self.fail_times, self.calls, self._ok = fail_times, 0, FakeProvider()

    def complete(self, prompt, image_path, case):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("429 rate limited")
        return self._ok.complete(prompt, image_path, case)


def _run_with(provider, tmp_path, monkeypatch, *extra):
    import kidai_evals.cli as cli
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: provider)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    assert cli.main(["run", "--provider", "fake", "--model", "fake-1", "--limit", "1",
                     "--combos-per-case", "1", "--out", str(tmp_path), *extra]) == 0
    return [json.loads(l) for l in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]


def test_transient_errors_are_retried(tmp_path, monkeypatch):
    flaky = _Flaky(fail_times=2)
    rows = _run_with(flaky, tmp_path, monkeypatch)
    assert flaky.calls == 3 and "error" not in rows[0] and rows[0]["checks"]["header_ok"]


def test_persistent_error_is_recorded_and_reported(tmp_path, monkeypatch, capsys):
    rows = _run_with(_Flaky(fail_times=99), tmp_path, monkeypatch, "--retries", "1")
    assert rows[0]["error"].startswith("RuntimeError")
    from kidai_evals.cli import main
    assert main(["report", "--results", str(tmp_path)]) == 0
    assert f"| fake | fake-1 | v1 | {rows[0]['dataset_id']} | 0 | 1 |" in capsys.readouterr().out


def test_rescore_applies_new_labels_and_skips_changed_inputs(tmp_path):
    import yaml
    from kidai_evals.cli import main

    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"photo a")  # the fake provider never opens the photos
    (images / "b.jpg").write_bytes(b"photo b")
    cases = tmp_path / "cases.yaml"

    def write_cases(subject_a, lat_b):
        cases.write_text(yaml.safe_dump({"cases": [
            {"id": "a", "image": "a.jpg", "subject": subject_a},
            {"id": "b", "image": "b.jpg", "subject": "Colosseo", "position": {"lat": lat_b, "lon": 12.0}},
        ]}), encoding="utf-8")

    data = ["--cases", str(cases), "--images", str(images)]
    out = tmp_path / "results"
    write_cases("Torre Eiffel", 41.0)
    assert main(["run", "--provider", "fake", "--model", "fake-1", "--out", str(out), *data]) == 0
    path = next(out.glob("*.jsonl"))
    before = [json.loads(l) for l in path.read_text().splitlines()]
    assert all(r["checks"]["subject_correct"] for r in before)

    # Relabel a (the saved answer still says "Torre Eiffel") and move b's GPS (a different prompt).
    write_cases("Pantheon", 42.0)
    assert main(["rescore", "--results", str(out), *data]) == 0
    after = [json.loads(l) for l in path.read_text().splitlines()]

    a = [r for r in after if r["case_id"] == "a"]
    assert a and all(r["checks"]["confident_wrong"] and not r["checks"]["subject_correct"] for r in a)
    assert all(r["dataset_id"] != before[0]["dataset_id"] and r["rescored_at"] for r in a)
    # b's model saw other coordinates: its rows must stay exactly as they were.
    assert [r for r in after if r["case_id"] == "b"] == [r for r in before if r["case_id"] == "b"]
