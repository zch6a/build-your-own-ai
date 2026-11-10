"""Chapter 4 — inference: generating text fast, and choosing what to say.

Training needs autograd; generating does not. This module runs the chapter-3 GPT with plain numpy and
two ideas that every inference engine is built on:

* **A KV cache.** Generating token 50 the naive way re-runs the whole model over tokens 0–49, although
  their keys and values in every layer are exactly what they were one step ago. Keep them: the prompt
  is processed once ("prefill"), and each new token then costs one position's worth of work ("decode").
* **Sampling.** The model gives a probability for every token; how you pick from it — greedy,
  temperature, top-k, top-p — decides whether the text is repetitive, varied or nonsense.

    >>> from byoai.generate import Engine
    >>> engine = Engine(model)
    >>> for token in engine.stream(prompt_ids, max_new_tokens=50, temperature=0.8, top_p=0.9):
    ...     print(tok.decode([token]), end="")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from byoai.gpt import GPT


# ---- sampling ---------------------------------------------------------------------------------------------
def sample(logits: np.ndarray, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None, rng: np.random.Generator | None = None) -> int:
    """Pick the next token from one row of logits.

    * `temperature`: divide the logits by it. Below 1 sharpens the distribution (more predictable), above
      1 flattens it (more surprising); 0 means always the most likely token (greedy).
    * `top_k`: keep only the k most likely tokens.
    * `top_p` (nucleus): keep the smallest set of tokens whose probabilities add up to p — few tokens when
      the model is sure, many when it is not.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0:
        return int(logits.argmax())
    rng = rng or np.random.default_rng()
    logits = logits / temperature
    if top_k is not None and top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        # Keep tokens up to and including the one that crosses p (at least one).
        cut = int(np.searchsorted(cumulative, top_p)) + 1
        keep = np.zeros_like(probs, dtype=bool)
        keep[order[:cut]] = True
        probs = np.where(keep, probs, 0.0)
        probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


# ---- the engine -------------------------------------------------------------------------------------------
@dataclass
class Cache:
    """Keys and values of every position seen so far, per layer: (heads, positions, head_dim) each."""

    k: list[np.ndarray] = field(default_factory=list)
    v: list[np.ndarray] = field(default_factory=list)

    @property
    def length(self) -> int:
        return 0 if not self.k else self.k[0].shape[1]


class Engine:
    """The chapter-3 model, run with numpy only (no autograd graph), with a KV cache."""

    def __init__(self, model: GPT):
        self.cfg = model.config
        self.p = {k: t.data for k, t in model.params.items()}  # plain arrays: nothing to differentiate

    def _ln(self, x: np.ndarray, name: str) -> np.ndarray:
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * self.p[name + ".w"] + self.p[name + ".b"]

    def forward(self, ids: list[int], cache: Cache) -> np.ndarray:
        """Run `ids` (new tokens) through the model after whatever `cache` holds; return their logits
        (len(ids), vocab) and extend the cache. With an empty cache this is the prefill."""
        cfg, p = self.cfg, self.p
        start = cache.length
        T = len(ids)
        if start + T > cfg.block_size:
            raise ValueError(f"{start + T} positions exceed block_size {cfg.block_size}")
        C, H = cfg.n_embd, cfg.n_head
        D = C // H
        x = p["wte"][ids] + p["wpe"][start : start + T]  # (T, C): new positions only
        for i in range(cfg.n_layer):
            h = self._ln(x, f"h{i}.ln1")
            qkv = h @ p[f"h{i}.attn.w"] + p[f"h{i}.attn.b"]
            q, k, v = (qkv[:, j * C : (j + 1) * C].reshape(T, H, D).transpose(1, 0, 2) for j in range(3))
            # The cache: keys and values of all earlier positions, plus the new ones.
            if len(cache.k) <= i:
                cache.k.append(k)
                cache.v.append(v)
            else:
                cache.k[i] = np.concatenate([cache.k[i], k], axis=1)
                cache.v[i] = np.concatenate([cache.v[i], v], axis=1)
            K, V = cache.k[i], cache.v[i]  # (H, start+T, D)
            scores = q @ K.transpose(0, 2, 1) / math.sqrt(D)  # (H, T, start+T)
            # New position t (absolute start+t) may see positions 0..start+t.
            future = np.arange(start + T)[None, :] > (start + np.arange(T))[:, None]
            scores = np.where(future, -1e9, scores)
            w = np.exp(scores - scores.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            out = (w @ V).transpose(1, 0, 2).reshape(T, C)
            x = x + out @ p[f"h{i}.attn.proj.w"] + p[f"h{i}.attn.proj.b"]
            h = self._ln(x, f"h{i}.ln2")
            a = h @ p[f"h{i}.mlp.w"] + p[f"h{i}.mlp.b"]
            a = 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * a * (1 + 0.044715 * a * a)))
            x = x + a @ p[f"h{i}.mlp.proj.w"] + p[f"h{i}.mlp.proj.b"]
        return self._ln(x, "lnf") @ p["wte"].T

    def stream(
        self,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop: set[int] | None = None,
        seed: int | None = None,
        use_cache: bool = True,
    ) -> Iterator[int]:
        """Yield new tokens one at a time (stop after a token in `stop`, which is yielded too).

        When the context fills up (`block_size` positions — the most this model's position table
        covers), the cache is rebuilt from the most recent half of the context and generation goes on.
        """
        rng = np.random.default_rng(seed)
        stop = stop or set()
        tokens = list(prompt)
        window = self.cfg.block_size
        cache = Cache()
        logits = self.forward(tokens[-window:], cache)[-1]
        for _ in range(max_new_tokens):
            nxt = sample(logits, temperature, top_k, top_p, rng)
            tokens.append(nxt)
            yield nxt
            if nxt in stop:
                return
            if not use_cache:
                # The naive way, for comparison: every step re-runs the whole context.
                cache = Cache()
                logits = self.forward(tokens[-window:], cache)[-1]
            elif cache.length >= window:
                cache = Cache()
                logits = self.forward(tokens[-window // 2 :], cache)[-1]
            else:
                logits = self.forward([nxt], cache)[-1]

    def generate(self, prompt: list[int], max_new_tokens: int, **kwargs) -> list[int]:
        """The prompt followed by up to `max_new_tokens` new tokens."""
        return list(prompt) + list(self.stream(prompt, max_new_tokens, **kwargs))
