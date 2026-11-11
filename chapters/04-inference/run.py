"""Chapter 4 demo: a KV cache, and what each sampling strategy does to the text.

    uv run python chapters/04-inference/run.py

Uses the story model from chapter 3 (trained on first use, then loaded from checkpoints/).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from byoai.corpus import stories
from byoai.generate import Engine
from byoai.zoo import EOT, story_model

# Print UTF-8 even when the output is piped or redirected (Windows would otherwise use its legacy code page).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

model, tok = story_model(verbose=True)
engine = Engine(model)
eot = tok.special[EOT]
prompt = tok.encode("One day,")

# ---- 1. The KV cache --------------------------------------------------------------------------------------
print("generating 60 tokens, greedy:")
for use_cache in (False, True):
    t0 = time.perf_counter()
    out = engine.generate(prompt, 60, temperature=0, use_cache=use_cache)
    dt = time.perf_counter() - t0
    print(f"  {'with' if use_cache else 'without'} a KV cache: {dt * 1000:6.0f} ms  ({60 / dt:5.0f} tokens/s)")
print(f"  same text either way: {engine.generate(prompt, 60, temperature=0) == engine.generate(prompt, 60, temperature=0, use_cache=False)}")

# ---- 2. Sampling strategies --------------------------------------------------------------------------------
training_sentences = {s.strip() for s in stories(3000, seed=0).replace('"', "").split(".") if s.strip()}


def score(setting: dict, n: int = 40) -> tuple[float, float, str]:
    """Fluency: the share of generated sentences that are real sentences of the story grammar
    (all of them appear in the training text). Variety: the share of distinct stories."""
    texts, good, total = [], 0, 0
    for seed in range(n):
        ids = list(engine.stream(prompt, 60, stop={eot}, seed=seed, **setting))
        text = tok.decode([i for i in ids if i != eot])
        texts.append(text)
        sentences = [s.strip() for s in ("One day," + text).replace('"', "").split(".") if s.strip()]
        complete = sentences[:-1] if not text.rstrip().endswith((".", '"')) else sentences  # an unfinished last sentence does not count
        good += sum(s in training_sentences for s in complete)
        total += len(complete)
    return good / max(total, 1), len(set(texts)) / n, "One day," + texts[0]


print("\nsampling strategies (40 stories each, prompt 'One day,'):")
print(f"  {'setting':<28} {'fluent':>7} {'distinct':>9}   first story")
for name, setting in [
    ("greedy", dict(temperature=0)),
    ("temperature 0.5", dict(temperature=0.5)),
    ("temperature 1.0", dict(temperature=1.0)),
    ("temperature 1.5", dict(temperature=1.5)),
    ("temperature 2.5", dict(temperature=2.5)),
    ("top-k 3, temperature 1.5", dict(temperature=1.5, top_k=3)),
    ("top-p 0.9, temperature 1.5", dict(temperature=1.5, top_p=0.9)),
]:
    fluent, distinct, first = score(setting)
    shown = first.replace(chr(10), ' / ')[:66]
    print(f"  {name:<28} {fluent:>6.0%} {distinct:>8.0%}   {shown}{'…' if len(first) > 66 else ''}")

print("\nstreaming, token by token (temperature 0.8, top-p 0.9):\n  ", end="")
for token in engine.stream(tok.encode("At night, Lena"), 60, temperature=0.8, top_p=0.9, stop={eot}, seed=7):
    if token != eot:
        print(tok.decode([token]), end="", flush=True)
print()
