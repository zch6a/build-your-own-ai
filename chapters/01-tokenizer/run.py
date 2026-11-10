"""Chapter 1 demo: train a byte-level BPE tokenizer and look at what it learned.

    uv run python chapters/01-tokenizer/run.py
"""

from __future__ import annotations

import time

from byoai.corpus import stories
from byoai.tokenizer import Tokenizer

text = stories(2000, seed=0)
print(f"training text: {len(text):,} characters, {len(text.encode('utf-8')):,} bytes\n")

t0 = time.perf_counter()
tok = Tokenizer.train(text, vocab_size=1024, special=["<|endoftext|>"])
print(f"trained {len(tok.merges)} merges in {time.perf_counter() - t0:.2f} s")
if tok.vocab_size < 1024:
    print(f"(asked for 1024 tokens, got {tok.vocab_size}: after {len(tok.merges)} merges every word of these stories"
          " is a single token, and there is nothing left to merge)\n")

print("the first merges, in the order they were learned:")
for (a, b), new in list(tok.merges.items())[:12]:
    print(f"  {tok.token_text(a)!r:>10} + {tok.token_text(b)!r:<10} -> {new}: {tok.token_text(new)!r}")
print("  ...")
print("and some of the last ones (whole words and pieces of phrases):")
for (a, b), new in list(tok.merges.items())[-6:]:
    print(f"  {new}: {tok.token_text(new)!r}")

sentence = 'One day, Mia went to the old house with a curious fox. "Look!" said Leo.'
ids = tok.encode(sentence)
print(f"\n{sentence!r}\n-> {len(ids)} tokens:")
print("  " + "|".join(tok.token_text(i) for i in ids))

unseen = "Zoë found a 🦊 near 北京 — it was 37°C."
ids = tok.encode(unseen)
print(f"\ntext it never saw: {unseen!r}")
print(f"  {len(unseen.encode('utf-8'))} bytes -> {len(ids)} tokens; decodes back exactly: {tok.decode(ids) == unseen}")

held_out = stories(200, seed=99)
print("\nvocabulary size vs. compression (bytes per token) on stories it did not train on:")
for size in (256, 300, 400, 600, 1024, 2048):
    t = tok if size == 1024 else Tokenizer.train(text, vocab_size=size)
    n = len(t.encode(held_out))
    got = f" (got {t.vocab_size})" if t.vocab_size < size - 1 else ""
    print(f"  {size:>5} tokens{got}: {len(held_out.encode('utf-8')) / n:5.2f} bytes per token")

doc = "Mia slept.<|endoftext|>Leo woke."
print(f"\nspecial tokens are opt-in: {doc!r}")
print(f"  as text:         {tok.encode(doc)}")
print(f"  allowed:         {tok.encode(doc, allowed_special='all')}  (<|endoftext|> = {tok.special['<|endoftext|>']})")
