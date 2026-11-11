"""Chapter 3, the longer run: watch the model learn to copy — suddenly, after a plateau.

    uv run python chapters/03-gpt/longer.py        # about 15 minutes on a laptop CPU

Every 250 steps it measures the held-out loss and how often a generated story's second sentence names
the animal of its first ("with a fox. It was windy and the fox was..."). A random pick among the 10
animals would match 10% of the time.
"""

from __future__ import annotations

import re
import time

import numpy as np

from byoai.corpus import _ANIMALS
from byoai.generate import Engine
from byoai.gpt import GPT, GPTConfig, evaluate, train
from byoai.zoo import CONFIG, story_ids, story_tokenizer

tok = story_tokenizer()
data = story_ids(tok)
held_out = story_ids(tok, n=300, seed=123)
animals = "|".join(_ANIMALS)
prompt = tok.encode("One day,")


def animal_kept(model: GPT, n: int = 100) -> float:
    engine = Engine(model)  # chapter 4's KV cache: 100 stories in a few seconds
    kept = counted = 0
    for seed in range(n):
        text = tok.decode(engine.generate(prompt, 30, temperature=0.8, seed=seed))
        first = re.search(rf"with an? ({animals})\.", text)
        second = re.search(rf"and the ({animals}) was", text)
        if first and second:
            counted += 1
            kept += first.group(1) == second.group(1)
    return kept / max(counted, 1)


model = GPT(GPTConfig(vocab_size=tok.vocab_size, **CONFIG))
steps, every = 2500, 250
print(f"training {steps} steps; every {every}: held-out loss, and how often the animal is kept (chance: 10%)")
t0 = time.perf_counter()
done = 0
while done < steps:
    # One schedule over the whole run would be the usual choice; chunks keep the measurements in between.
    train(model, data, steps=every, batch_size=32, lr=3e-3 if done == 0 else 1.5e-3, warmup=50 if done == 0 else 0, seed=done)
    done += every
    loss, kept = evaluate(model, held_out), animal_kept(model)
    print(f"  step {done:>5}: loss {loss:.3f}   animal kept {kept:4.0%}  {'#' * round(kept * 40)}   ({time.perf_counter() - t0:.0f} s)", flush=True)
