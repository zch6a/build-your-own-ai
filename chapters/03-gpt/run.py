"""Chapter 3 demo: train a small GPT on the stories, and read what it writes.

    uv run python chapters/03-gpt/run.py

Training takes a minute or two on a laptop CPU; the trained model is cached in checkpoints/ for the
next chapters, keyed by the settings and the source code — change gpt.py for an exercise and it trains
again by itself.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from byoai.corpus import stories
from byoai.gpt import GPT, GPTConfig, evaluate
from byoai.zoo import CONFIG, EOT, STEPS, story_ids, story_model, story_tokenizer

# Print UTF-8 even when the output is piped or redirected (Windows would otherwise use its legacy code page).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tok = story_tokenizer()
data = story_ids(tok)
held_out = story_ids(tok, n=300, seed=123)  # stories the model never trains on
eot = tok.special[EOT]
print(f"{len(data):,} training tokens (vocabulary {tok.vocab_size}), {len(held_out):,} held out")

untrained = GPT(GPTConfig(vocab_size=tok.vocab_size, **CONFIG))
print(f"\na GPT with {untrained.num_params():,} parameters: {CONFIG['n_layer']} blocks, {CONFIG['n_head']} heads, width {CONFIG['n_embd']}, context {CONFIG['block_size']} tokens")
print(f"untrained, held-out loss {evaluate(untrained, held_out):.2f} (uniform guessing: ln {tok.vocab_size} = {np.log(tok.vocab_size):.2f})")
sample = untrained.generate(tok.encode("One day,"), 25, temperature=1.0, rng=np.random.default_rng(0))
print(f"untrained, it writes: {tok.decode(sample)!r}")

print(f"\ntraining for {STEPS} steps (batches of 32 windows of {CONFIG['block_size']} tokens):")
t0 = time.perf_counter()
model, _ = story_model(verbose=True)
took = time.perf_counter() - t0
print(f"  ({took:.0f} s{', from the cache in checkpoints/' if took < 5 else ''})")
print(f"\nheld-out loss {evaluate(model, held_out):.2f} — on stories it never saw")


def write(prompt: str, temperature: float, seed: int) -> str:
    ids = model.generate(tok.encode(prompt), 80, temperature=temperature, rng=np.random.default_rng(seed))
    ids = ids[: ids.index(eot, len(tok.encode(prompt)))] if eot in ids[len(tok.encode(prompt)) :] else ids
    return tok.decode(ids)


print("\nwhat it writes now (temperature 0.8):")
training_text = stories(3000, seed=0)
for i, prompt in enumerate(["One day,", "At night, Ada", "It was rainy and"]):
    story = write(prompt, 0.8, seed=i)
    print(f"\n  {story}")
    # Is it copying? Check every sentence against the training text.
    sentences = [s.strip() for s in story.replace('"', "").split(".") if len(s.strip()) > 10]
    new = [s for s in sentences if s not in training_text]
    print(f"  ({len(sentences) - len(new)} of {len(sentences)} sentences appear in the training text; the story as a whole: {'copied' if story in training_text else 'new'})")
