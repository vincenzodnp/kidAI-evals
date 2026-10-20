"""Command line: `kidai-evals run` generates and checks guides, `kidai-evals report` aggregates runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .checks import run_checks
from .data import load_cases, load_profiles, load_yaml
from .model import Style
from .parser import parse
from .prompts import VARIANTS
from .providers import make_provider

ROOT = Path(__file__).resolve().parents[2]


def _combos(cases, profiles, styles, per_case: int, seed: int):
    """Pair each case with `per_case` (profile, style) combinations; 0 means the full matrix."""
    rng = random.Random(seed)
    all_pairs = [(p, s) for p in profiles for s in styles]
    for case in cases:
        pairs = all_pairs if per_case <= 0 else rng.sample(all_pairs, min(per_case, len(all_pairs)))
        for profile, style in pairs:
            yield case, profile, style


def _complete_with_retry(provider, prompt, image_path, case, retries: int, base_delay: float):
    """Call the provider, retrying transient failures (rate limits, timeouts) with exponential backoff.

    Waits base_delay, then 2x, 4x... seconds between attempts. Raises the last error if every attempt fails.
    """
    for attempt in range(retries + 1):
        try:
            return provider.complete(prompt, image_path, case)
        except Exception as exc:  # SDK errors differ per provider; treat them all as retryable
            if attempt == retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  {case.id}: {type(exc).__name__} ({exc}); retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)


def cmd_run(args) -> int:
    cfg = load_yaml(args.config)
    cases = load_cases(args.cases)
    profiles = load_profiles(args.profiles)
    styles = [Style[s] for s in args.styles] if args.styles else list(Style)
    if args.limit:
        cases = cases[: args.limit]

    provider = make_provider(args.provider, args.model)
    build = VARIANTS[args.variant]
    judge = None
    if args.judge_model:
        from .judge import ClaudeJudge
        judge = ClaudeJudge(args.judge_model)

    run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}_{args.provider}_{args.model}_{args.variant}"
    out_path = Path(args.out) / f"{run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images)

    done = skipped = errors = 0
    with out_path.open("w", encoding="utf-8") as out:
        combos = list(_combos(cases, profiles, styles, args.combos_per_case, args.seed))
        # --repeats N asks the same question N times, to separate randomness from the effect of profile/style.
        for (case, profile, style), repeat in ((c, r) for c in combos for r in range(args.repeats)):
            image_path = images_dir / case.image
            if not image_path.exists():
                if args.provider != "fake":
                    print(f"skip {case.id}: image not found ({image_path})", file=sys.stderr)
                    skipped += 1
                    continue
                image_path = None
            prompt = build(case, profile, style)
            base = {
                "run_id": run_id, "provider": args.provider, "model": args.model, "variant": args.variant,
                "case_id": case.id, "case_kind": case.kind, "expect": case.expect,
                "profile_id": profile.id, "age": profile.age, "style": style.name, "repeat": repeat,
            }
            if args.pause:
                time.sleep(args.pause)  # spread calls out to stay under per-minute rate limits
            try:
                completion = _complete_with_retry(provider, prompt, image_path, case,
                                                  args.retries, args.retry_delay)
            except Exception as exc:
                # Record the failure and move on: one bad call must not lose the whole run.
                out.write(json.dumps({**base, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False) + "\n")
                errors += 1
                continue
            parsed = parse(completion.text)
            checks = run_checks(parsed, case, profile, cfg)
            row = {
                **base,
                "latency_s": round(completion.latency_s, 3),
                "ttft_s": _round(completion.ttft_s), "header_s": _round(completion.header_s),
                "first_sentence_s": _round(completion.first_sentence_s),
                "input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens,
                "checks": checks.to_dict(), "output": completion.text, "judge": None,
            }
            if judge and parsed.guide:
                try:
                    row["judge"] = judge.score(parsed.guide, case, profile, style, image_path)
                except Exception as exc:  # keep the run going; record the failure
                    row["judge"] = {"error": str(exc)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += 1
    print(f"{done} responses written to {out_path} ({skipped} skipped, {errors} API errors)")
    return 0


def _round(x):
    return round(x, 3) if x is not None else None


def consistency(rows: list[dict]) -> tuple[int, int]:
    """How many photos got the same outcome every time they were asked.

    Outcome = (identified correctly?, answered with high confidence?). Comparing outcomes rather
    than raw names avoids false alarms from wording ("Thin House" vs "Casa Sottile").
    Returns (consistent photos, photos asked at least twice).
    """
    by_case = defaultdict(set)
    counts = defaultdict(int)
    for r in rows:
        c = r["checks"]
        by_case[r["case_id"]].add((c["subject_correct"], c["confidence"] == "alta"))
        counts[r["case_id"]] += 1
    asked_twice = [k for k, n in counts.items() if n >= 2]
    return sum(1 for k in asked_twice if len(by_case[k]) == 1), len(asked_twice)


def _med(values, digits=1) -> str:
    values = [v for v in values if v is not None]
    return f"{median(values):.{digits}f}" if values else "–"


def _pct(values) -> str:
    values = [v for v in values if v is not None]
    return f"{100 * sum(values) / len(values):.0f}%" if values else "–"


def _avg(values, digits=1) -> str:
    values = [v for v in values if v is not None]
    return f"{mean(values):.{digits}f}" if values else "–"


def cmd_report(args) -> int:
    files = [Path(f) for f in args.files] or sorted(Path(args.results).glob("*.jsonl"))
    rows = [json.loads(line) for f in files for line in f.open(encoding="utf-8") if line.strip()]
    if not rows:
        print("No results found.", file=sys.stderr)
        return 1

    groups = defaultdict(list)
    failed = defaultdict(int)
    for r in rows:
        key = (r["provider"], r["model"], r["variant"])
        if "error" in r:
            failed[key] += 1  # API call failed after all retries: counted, not scored
        else:
            groups[key].append(r)

    headers = ["Provider", "Model", "Prompt", "N", "API errors", "Format ok", "Correct ID", "Confident but wrong",
               "Stops when unsure", "No TTS violations", "Length ok", "Readable for age",
               "Same outcome per photo", "Gulpease", "First sentence s (median)", "Total s (median)",
               "Out tokens", "Judge accuracy", "Judge safety"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for prov, model, variant in sorted(set(groups) | set(failed)):
        rs = groups.get((prov, model, variant), [])
        c = [r["checks"] for r in rs]
        identify = [r["checks"] for r in rs if r["expect"] == "identify"]
        guides = [x for x in c if x["guide_words"]]
        judged = [r["judge"] for r in rs if r["judge"] and "error" not in r["judge"]]
        lines.append("| " + " | ".join([
            prov, model, variant, str(len(rs)), str(failed[(prov, model, variant)]),
            _pct(x["header_ok"] for x in c),
            _pct(x["subject_correct"] for x in identify),
            _pct(x["confident_wrong"] for x in c),
            _pct(x["stopped_correctly"] for x in c),
            _pct(not x["tts_violations"] for x in guides),
            _pct(x["length_ok"] for x in guides),
            _pct(x["readability_ok"] for x in guides),
            (lambda ok, n: f"{ok}/{n}" if n else "–")(*consistency(rs)),
            _avg(x["gulpease"] for x in guides),
            _med((r.get("first_sentence_s") for r in rs), 2),
            _med((r["latency_s"] for r in rs), 2),
            _avg((r["output_tokens"] for r in rs), 0),
            _avg(j["factual_accuracy"] for j in judged),
            _avg(j["safety"] for j in judged),
        ]) + " |")
    report = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kidai-evals")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="generate guides and run the checks")
    run.add_argument("--provider", required=True, choices=["gemini", "claude", "fake"])
    run.add_argument("--model", required=True)
    run.add_argument("--variant", default="v1", choices=sorted(VARIANTS))
    run.add_argument("--cases", default=str(ROOT / "data" / "cases.yaml"))
    run.add_argument("--profiles", default=str(ROOT / "data" / "profiles.yaml"))
    run.add_argument("--images", default=str(ROOT / "data" / "images"))
    run.add_argument("--config", default=str(ROOT / "config" / "eval.yaml"))
    run.add_argument("--styles", nargs="*", choices=[s.name for s in Style])
    run.add_argument("--combos-per-case", type=int, default=2,
                     help="(profile, style) pairs sampled per photo; 0 = full matrix")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--limit", type=int, help="only the first N cases")
    run.add_argument("--repeats", type=int, default=1, help="ask each question N times (measures consistency)")
    run.add_argument("--retries", type=int, default=4, help="extra attempts after a failed API call")
    run.add_argument("--retry-delay", type=float, default=5.0, help="first wait in seconds; doubles each retry")
    run.add_argument("--pause", type=float, default=0.0, help="seconds to wait before every call")
    run.add_argument("--judge-model", help="Claude model id for LLM-as-judge (omit to skip judging)")
    run.add_argument("--out", default=str(ROOT / "results"))
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="aggregate results into a Markdown table")
    rep.add_argument("files", nargs="*")
    rep.add_argument("--results", default=str(ROOT / "results"))
    rep.add_argument("--output")
    rep.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
