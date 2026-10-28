# KidAI evals

An evaluation harness for **KidAI**, an Android audio-guide app for families. You point the
phone at a monument, the app identifies it with a multimodal model and tells its story in a
style chosen for the listener: a fairy tale for a six-year-old, a short essay for a parent.

The guide goes straight from the model to a child's ear, with no human review in between.
These evals measure whether that is safe and good enough, and what changes when the prompt
or the model changes.

## What it measures

The app already asks the model for a strict output contract and a set of rules
(`PromptBuilder.kt`). The question is how often those rules actually hold.

| Area | What | How |
|---|---|---|
| Output contract | First line `SOGGETTO: … \| FIDUCIA: alta/media/bassa`; nothing more unless confidence is `alta` | deterministic |
| Identification | Correct subject, and the **confident-but-wrong rate** (high confidence on a wrong answer) | ground-truth labels with aliases |
| Text-to-speech form | No markdown, lists, headings, asterisks or emoji | deterministic |
| Complete guide | The story isn't cut off mid-sentence; the provider's stop reason and Gemini's thinking tokens are saved to explain why | deterministic |
| Length | About 1.5–2 minutes of listening | word count (thresholds in `config/eval.yaml`) |
| Age register | Language suited to the listener's age band | [Gulpease index](https://it.wikipedia.org/wiki/Indice_Gulpease), a readability formula designed for Italian |
| Truthfulness and style | Facts correct, legends labelled as legends, style followed | Claude as judge with a written rubric (`rubric/judge.md`) |
| Hard cases | Instructions written inside the photo, ambiguous shots, places with violent histories told to young children | purpose-built adversarial cases |
| Waiting time | Seconds until the first sentence could be spoken, the wait a child actually notices | streaming, same sentence rules as the app's `SentenceSegmenter.kt` |
| Consistency | Whether the same photo gets the same outcome (right or wrong, confident or not) every time | compared across calls; `--repeats N` asks each question N times |

The confident-but-wrong rate matters most: an app that tells a child the wrong story with
full confidence loses the family's trust in every guide that follows.

## How it works

```
data/cases.yaml      photos + ground truth (subject, aliases, GPS, expected behaviour)
data/profiles.yaml   listener profiles: ages 6, 10, 14 and an adult
src/kidai_evals/
  prompts.py         v1 = faithful port of the app's PromptBuilder.kt; add variants here
  parser.py          same parsing rules as the app's StreamParser.kt
  providers.py       Gemini (the app's current engine), Claude, and a fake provider
  checks.py          deterministic checks
  judge.py           LLM-as-judge
  fingerprint.py     hashes that record which data, prompt and code produced each answer
  cli.py             `kidai-evals run`, `kidai-evals rescore` and `kidai-evals report`
```

Each photo is paired with a sample of (profile, style) combinations, so a run stays cheap.
Results are written as JSONL, one line per response, with the raw output kept for review.

## Quick start

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q

# Offline dry run, no API keys needed
kidai-evals run --provider fake --model fake-1
kidai-evals report
```

With real models (keys in `.env`, see `.env.example`):

```sh
set -a; source .env; set +a
kidai-evals run --provider gemini --model gemini-3.8-flash
kidai-evals run --provider claude --model claude-sonnet-5-5 --judge-model claude-opus-5-5
kidai-evals report --output results/report.md
```

Useful options: `--variant v1`, `--combos-per-case 0` (full matrix), `--limit 5`,
`--styles FANTASTICA STORICA`. Failed API calls are retried with exponential backoff
(`--retries`, `--retry-delay`); a call that still fails is recorded as an error and counted in the
report instead of stopping the run. On a rate-limited free tier, `--pause 5` spreads the calls out.

### Comparing runs over time

Every answer is saved with fingerprints: `dataset_id` (all cases, photos and profiles),
`input_id` (the exact prompt and photo the model received), `config_id` (the thresholds) and
`git_commit`. The report groups results by dataset, so runs on different data are never mixed.

After fixing a label or a threshold, re-check the saved answers instead of calling the models again:

```sh
kidai-evals rescore    # updates the files in results/ in place
kidai-evals report
```

An answer is re-scored only if the model would receive exactly the same input today. If the
prompt changed (for example a case's GPS position), the answer is left as it was.

## Results so far

Baseline: the app's current prompt (`v1`) on `gemini-3.8-flash`, 38 photos, 76 answers
(two profile/style pairs per photo), run on 21 October 2026. Full table in `results/report.md`.

| Check | Result |
|---|---|
| Output contract followed (header, stop when unsure) | 100% |
| Correct identification | 97% |
| Confident but wrong | 3% (2 answers, both on the same photo) |
| Obeyed instructions written in the photo | 0% (2 of 2 resisted) |
| Same outcome on every call | 37 of 38 photos |
| First sentence ready to speak (median) | 4.8 s, almost all of it before the first token |

What it shows so far:

- **One systematic error.** A dinosaur skeleton at the Natural History Museum is twice named as
  "Hope", the museum's blue whale, with high confidence. The photo carries the museum's GPS
  position; the `v1-nogps` variant tests whether the location outweighs the image.
- **The language barely changes with age.** Mean Gulpease index 59.5 for a 6-year-old, 52.0 at 10,
  49.2 at 14 and 48.3 for an adult; no guide for a child reaches the primary-school threshold. This is
  the main target for a `v2` prompt.
- **The first run found label errors, not just model errors.** Three of its four "confident but
  wrong" answers were correct, and the labels were fixed.

Limits: two answers per photo is a small sample, the LLM judge has not been calibrated against human
scores yet, and the length and readability thresholds are stated assumptions.

## Photos and privacy

Only photos taken by the author are used, and none show identifiable people. Add a JPEG to
`data/images/` and an entry to `data/cases.yaml`. The example entries in `cases.yaml` still need
their photos.

## Next steps

1. Baseline: run the app's current prompt (`v1`) on Gemini, its default engine (`gemini-3.8-flash`).
2. Calibrate the judge: score a sample of guides by hand with the same rubric and measure agreement.
3. Add guardrails as a `v2` prompt and measure the difference.
4. Compare Gemini and Claude on cost, latency and quality, and publish the results whatever they show.

Developed by Vincenzo De Naro Papa with the help of Claude Code.

## Licence

Code released under the [MIT Licence](LICENSE). Photos in `data/images/` remain the author's and are
not covered by it unless stated otherwise.
