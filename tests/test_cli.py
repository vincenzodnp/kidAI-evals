"""End-to-end run with the fake provider: no API keys, no images."""
import json

from kidai_evals.cli import main


def test_fake_run_and_report(tmp_path, capsys):
    assert main(["run", "--provider", "fake", "--model", "fake-1", "--out", str(tmp_path)]) == 0
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert rows and all(r["checks"]["header_ok"] for r in rows)
    blurry = [r for r in rows if r["case_id"] == "blurry-facade"]
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
    assert "| fake | fake-1 | v1 | 0 | 1 |" in capsys.readouterr().out
