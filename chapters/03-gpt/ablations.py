"""Chapter 3, two textbook claims tested: are position embeddings and weight tying needed here?

    uv run python chapters/03-gpt/ablations.py        # three trainings of 500 steps, ~6 minutes

Trains the chapter's model three ways with the same seed and data, one after another (running them in
parallel makes numpy's threads fight over the CPU), and prints the held-out loss of each.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from byoai.autograd import Tensor
from byoai.gpt import GPT, GPTConfig, evaluate, train
from byoai.zoo import CONFIG, story_ids, story_tokenizer

# Print UTF-8 even when the output is piped or redirected (Windows would otherwise use its legacy code page).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Untied(GPT):
    """The same model with its own output matrix instead of the embedding table, transposed."""

    def __init__(self, config: GPTConfig, seed: int = 0):
        super().__init__(config, seed)
        rng = np.random.default_rng(seed + 1)
        self.params["wout"] = Tensor((rng.standard_normal((config.n_embd, config.vocab_size)) * 0.02).astype(np.float32), requires_grad=True)

    def __call__(self, idx, targets=None):
        idx = np.asarray(idx)
        idx = idx[None, :] if idx.ndim == 1 else idx
        p, T = self.params, idx.shape[1]
        x = p["wte"][idx] + p["wpe"][np.arange(T)]
        for i in range(self.config.n_layer):
            x = x + self._attention(x.layer_norm(p[f"h{i}.ln1.w"], p[f"h{i}.ln1.b"]), i)
            x = x + self._mlp(x.layer_norm(p[f"h{i}.ln2.w"], p[f"h{i}.ln2.b"]), i)
        logits = x.layer_norm(p["lnf.w"], p["lnf.b"]) @ p["wout"]
        return logits, (logits.cross_entropy(targets) if targets is not None else None)


tok = story_tokenizer()
data, held_out = story_ids(tok), story_ids(tok, n=300, seed=123)
cfg = GPTConfig(vocab_size=tok.vocab_size, **CONFIG)
print(f"{'model':<34} {'parameters':>10} {'held-out loss':>14}")
for name in ("as in the chapter", "no position embedding", "untied output layer"):
    model = Untied(cfg, seed=0) if name == "untied output layer" else GPT(cfg, seed=0)
    if name == "no position embedding":
        model.params["wpe"] = Tensor(np.zeros_like(model.params["wpe"].data))  # zeros, not trained
    t0 = time.perf_counter()
    train(model, data, steps=500, batch_size=32, lr=3e-3, warmup=50, seed=0)
    trainable = sum(t.data.size for t in model.params.values() if t.requires_grad)
    print(f"{name:<34} {trainable:>10,} {evaluate(model, held_out):>14.3f}   ({time.perf_counter() - t0:.0f} s)", flush=True)
