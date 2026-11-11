"""Chapter 3 — GPT: a decoder-only transformer, built from chapter 2's autograd.

A language model answers one question over and over: given these tokens, what is the next one? GPT
answers it with a stack of identical blocks. Each block lets every position look at the positions
before it (attention) and then think about what it saw (a small MLP). The output is, for every
position, a score for every token in the vocabulary; training pushes up the score of the token that
actually came next.

    from byoai.gpt import GPT, GPTConfig
    model = GPT(GPTConfig(vocab_size=520, block_size=64, n_layer=2, n_head=4, n_embd=64))
    logits, loss = model(idx, targets)       # idx, targets: (batch, time) integer arrays

The architecture is GPT-2's: token and position embeddings, pre-norm blocks with residual
connections, GELU, and an output layer that shares its weights with the token embedding.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from byoai.autograd import Adam, Tensor, no_grad


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 64  # the longest context the model can see
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64  # width of every position's vector
    dtype: str = "float32"


class GPT:
    """Parameters live in `self.params` (name → Tensor); the forward pass is plain functions of them."""

    def __init__(self, config: GPTConfig, seed: int = 0):
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        rng = np.random.default_rng(seed)
        dt = np.dtype(config.dtype)
        C, L = config.n_embd, config.n_layer

        def normal(*shape: int, std: float = 0.02) -> Tensor:
            return Tensor((rng.standard_normal(shape) * std).astype(dt), requires_grad=True)

        def const(value: float, *shape: int) -> Tensor:
            return Tensor(np.full(shape, value, dtype=dt), requires_grad=True)

        p: dict[str, Tensor] = {"wte": normal(config.vocab_size, C), "wpe": normal(config.block_size, C)}
        # GPT-2's initialization: small weights, and the layers that write into the residual stream
        # scaled down by sqrt(2·n_layer), so the stream does not grow with depth.
        proj_std = 0.02 / math.sqrt(2 * L)
        for i in range(L):
            p[f"h{i}.ln1.w"], p[f"h{i}.ln1.b"] = const(1.0, C), const(0.0, C)
            p[f"h{i}.attn.w"], p[f"h{i}.attn.b"] = normal(C, 3 * C), const(0.0, 3 * C)
            p[f"h{i}.attn.proj.w"], p[f"h{i}.attn.proj.b"] = normal(C, C, std=proj_std), const(0.0, C)
            p[f"h{i}.ln2.w"], p[f"h{i}.ln2.b"] = const(1.0, C), const(0.0, C)
            p[f"h{i}.mlp.w"], p[f"h{i}.mlp.b"] = normal(C, 4 * C), const(0.0, 4 * C)
            p[f"h{i}.mlp.proj.w"], p[f"h{i}.mlp.proj.b"] = normal(4 * C, C, std=proj_std), const(0.0, C)
        p["lnf.w"], p["lnf.b"] = const(1.0, C), const(0.0, C)
        self.params = p
        # Position i may attend to positions 0..i: True above the diagonal means "masked".
        self._mask = np.triu(np.ones((config.block_size, config.block_size), dtype=bool), k=1)

    def parameters(self) -> list[Tensor]:
        return list(self.params.values())

    def num_params(self) -> int:
        return sum(t.data.size for t in self.params.values())

    # ---- the forward pass --------------------------------------------------------------------------
    def __call__(self, idx: np.ndarray, targets: np.ndarray | None = None) -> tuple[Tensor, Tensor | None]:
        """Logits (batch, time, vocab) for every position, and the loss if `targets` are given."""
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx[None, :]
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence of {T} tokens is longer than block_size {self.config.block_size}")
        p = self.params
        # Each position starts as "which token" + "where": two lookups, added.
        x = p["wte"][idx] + p["wpe"][np.arange(T)]
        for i in range(self.config.n_layer):
            x = x + self._attention(x.layer_norm(p[f"h{i}.ln1.w"], p[f"h{i}.ln1.b"]), i)
            x = x + self._mlp(x.layer_norm(p[f"h{i}.ln2.w"], p[f"h{i}.ln2.b"]), i)
        x = x.layer_norm(p["lnf.w"], p["lnf.b"])
        # Weight tying: the output layer is the embedding table, transposed. A token's input vector
        # and the vector that predicts it are the same: fewer parameters. (Not better loss here —
        # chapter 3 measures an untied output reaching 0.52 vs 0.62 after 500 steps.)
        logits = x @ p["wte"].T
        loss = logits.cross_entropy(targets) if targets is not None else None
        return logits, loss

    def _attention(self, x: Tensor, i: int) -> Tensor:
        B, T, C = x.shape
        H = self.config.n_head
        D = C // H
        p = self.params
        qkv = x @ p[f"h{i}.attn.w"] + p[f"h{i}.attn.b"]  # (B, T, 3C): queries, keys, values for every position
        # Split into heads: (B, T, C) -> (B, H, T, D). Each head attends with its own slice of the vector.
        q, k, v = (qkv[:, :, j * C : (j + 1) * C].reshape(B, T, H, D).transpose(0, 2, 1, 3) for j in range(3))
        # How much each position (row) should look at each earlier position (column).
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(D))  # (B, H, T, T)
        scores = scores.masked_fill(self._mask[:T, :T], -1e9)  # no looking at the future
        weights = scores.softmax(axis=-1)
        out = weights @ v  # (B, H, T, D): a weighted mix of the values
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)  # heads side by side again
        return out @ p[f"h{i}.attn.proj.w"] + p[f"h{i}.attn.proj.b"]

    def _mlp(self, x: Tensor, i: int) -> Tensor:
        p = self.params
        h = (x @ p[f"h{i}.mlp.w"] + p[f"h{i}.mlp.b"]).gelu()
        return h @ p[f"h{i}.mlp.proj.w"] + p[f"h{i}.mlp.proj.b"]

    # ---- generating (chapter 4 makes this fast and adds better sampling) ------------------------------
    def generate(self, idx: list[int], max_new_tokens: int, temperature: float = 1.0, rng: np.random.Generator | None = None) -> list[int]:
        """Append tokens one at a time: run the model, sample from the last position's distribution."""
        if not idx:
            raise ValueError("generate needs a prompt of at least one token (e.g. the end-of-text token)")
        rng = rng or np.random.default_rng()
        out = list(idx)
        with no_grad():
            for _ in range(max_new_tokens):
                context = np.array(out[-self.config.block_size :])
                logits, _ = self(context)
                last = logits.data[0, -1].astype(np.float64)
                if temperature == 0:
                    out.append(int(last.argmax()))
                    continue
                probs = np.exp((last - last.max()) / temperature)
                out.append(int(rng.choice(len(probs), p=probs / probs.sum())))
        return out

    # ---- saving --------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save to exactly `path` (numpy alone would append ".npz" to a name without it, and `load`
        would then not find the file)."""
        with open(path, "wb") as f:
            np.savez(f, __config__=np.array(json.dumps(asdict(self.config))), **{k: t.data for k, t in self.params.items()})

    @classmethod
    def load(cls, path: str | Path) -> GPT:
        with np.load(path) as f:
            model = cls(GPTConfig(**json.loads(str(f["__config__"]))))
            for k, t in model.params.items():
                t.data = f[k].astype(t.data.dtype)
        return model


# ---- training ---------------------------------------------------------------------------------------------
def clip_grad_norm(params: list[Tensor], max_norm: float) -> float:
    """Scale all gradients down together if their overall size exceeds `max_norm`: one unlucky batch
    should not throw the weights far away. Returns the norm before clipping."""
    norm = math.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in params if p.grad is not None))
    if norm > max_norm:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * (max_norm / norm)  # a new array: gradients may be shared, never edit in place
    return norm

def batches(data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator):
    """Random windows of the token stream: input `x`, and `y` = the same window shifted by one."""
    if len(data) <= block_size + 1:
        raise ValueError(f"need more than block_size + 1 = {block_size + 1} tokens of training data, got {len(data)}")
    while True:
        starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([data[s : s + block_size] for s in starts])
        y = np.stack([data[s + 1 : s + block_size + 1] for s in starts])
        yield x, y


def train(
    model: GPT,
    data: np.ndarray,
    steps: int,
    batch_size: int = 32,
    lr: float = 3e-3,
    warmup: int = 50,
    seed: int = 0,
    log: Callable[[int, float, float], None] | None = None,
    log_every: int = 50,
) -> list[float]:
    """Train with AdamW on random windows of `data` (a 1-D array of token ids). Returns the losses.

    The learning rate warms up linearly (Adam's early estimates are noisy) and then decays along a
    cosine to a tenth of its peak.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    losses: list[float] = []
    t0 = time.perf_counter()
    for step, (x, y) in zip(range(steps), batches(data, model.config.block_size, batch_size, rng)):
        if step < warmup:
            opt.lr = lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, steps - warmup)
            opt.lr = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        clip_grad_norm(opt.params, 1.0)
        opt.step()
        losses.append(loss.item())
        if log and (step % log_every == 0 or step == steps - 1):
            log(step, loss.item(), time.perf_counter() - t0)
    return losses


def evaluate(model: GPT, data: np.ndarray, batches_n: int = 20, batch_size: int = 32, seed: int = 1) -> float:
    """Mean loss on random windows of `data`, without gradients."""
    rng = np.random.default_rng(seed)
    total = 0.0
    with no_grad():
        for _, (x, y) in zip(range(batches_n), batches(data, model.config.block_size, batch_size, rng)):
            total += model(x, y)[1].item()
    return total / batches_n
