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
