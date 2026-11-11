# Contributing

Corrections are the most valuable contribution: a wrong statement, a number the code does not reproduce,
an exercise that does not work as written, a place where you got lost. Open an issue with what you read,
what you ran, and what you saw.

For code changes:

- `uv sync`, then `uv run pytest` (about 20 seconds). Everything must stay numpy-only and run on a laptop
  CPU without downloads or API keys.
- If a change moves a number that a walkthrough prints, update the walkthrough — and
  `tests/test_published_numbers.py` will tell you which ones moved.
- A new behaviour needs a test that fails without it. `uv run python scripts/mutate.py` shows whether the
  existing tests notice deliberate bugs; add a mutant for anything important you add.
- Keep the code readable over clever: these modules are meant to be read in an evening.

Translations of the walkthroughs (the Chinese README has chapter summaries; the chapters themselves are
in English) are welcome too.
