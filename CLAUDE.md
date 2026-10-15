# kidai-evals — notes for Claude

Evaluation harness for KidAI (private repo `vincenzodnp/kidai`, Android/Kotlin).

## Ground rules
- `src/kidai_evals/prompts.py` `v1` must stay a faithful port of `core/.../cloud/PromptBuilder.kt`
  and `parser.py` of `StreamParser.kt`. If the app changes, update the port and note the app commit.
- Never invent results. README tables are filled only from real runs in `results/`.
- No API keys in code, commits or logs. Keys come from `GEMINI_API_KEY` and `ANTHROPIC_API_KEY`.
- Photos in `data/images/` must be the author's own, with no identifiable people.
- Public repo: write in UK English. Prompts and rubric stay in Italian, like the app.
- The owner wants to understand every line: prefer plain, commented code over clever code.

## Commands
- `pytest -q` — unit tests plus an offline end-to-end run with the fake provider
- `kidai-evals run --provider fake --model fake-1` then `kidai-evals report`
