"""Chapter 3, the longer run: watch the model learn to copy — suddenly, after a plateau.

    uv run python chapters/03-gpt/longer.py        # about 13 minutes on a laptop CPU

Every 250 steps it measures the held-out loss and how often a generated story's second sentence names
the animal of its first ("with a fox. It was windy and the fox was..."). A random pick among the 10
animals would match 10% of the time.
"""

from __future__ import annotations

import math
import re
import sys
import time

import numpy as np

from byoai.autograd import Adam
from byoai.corpus import _ANIMALS
from byoai.generate import Engine
from byoai.gpt import GPT, GPTConfig, batches, clip_grad_norm, evaluate
from byoai.zoo import CONFIG, story_ids, story_tokenizer

# Print UTF-8 even when the output is piped or redirected (Windows would otherwise use its legacy code page).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


SEED = 0  # change it to see the jump come at a different step
model = GPT(GPTConfig(vocab_size=tok.vocab_size, **CONFIG), seed=SEED)
steps, every, lr, warmup = 2500, 250, 3e-3, 50
print(f"training {steps} steps; every {every}: held-out loss, and how often the animal is kept (chance: 10%)")

# One optimizer and one learning-rate schedule for the whole run (the same recipe as `train`), with the
# measurements taken in between. (A first version called `train` once per 250 steps, which restarted
# Adam and the schedule at every measurement — a jump right after a restart would prove nothing.)
opt = Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
rng = np.random.default_rng(SEED)
t0 = time.perf_counter()
for step, (x, y) in zip(range(steps), batches(data, model.config.block_size, 32, rng)):
    opt.lr = lr * (step + 1) / warmup if step < warmup else lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))
    opt.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    clip_grad_norm(opt.params, 1.0)
    opt.step()
    if (step + 1) % every == 0:
        loss_now, kept = evaluate(model, held_out), animal_kept(model)
        print(f"  step {step + 1:>5}: loss {loss_now:.3f}   animal kept {kept:4.0%}  {'#' * round(kept * 40)}   ({time.perf_counter() - t0:.0f} s)", flush=True)