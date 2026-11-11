"""The story tokenizer and model the later chapters share — trained once, then cached.

Chapter 3 trains a small GPT on the generated stories; chapters 4–7 use it (sampling, embeddings,
retrieval, an agent). Training takes a minute or two on a CPU, so the result is kept in
`checkpoints/` (or `$BYOAI_CHECKPOINTS`), keyed by everything that determines it: corpus size,
tokenizer size, model shape, training steps and seed — and the source code of the modules that build
it. Change any of them (an exercise that removes the position embedding, say) and it is trained again
instead of a stale model being loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from byoai.corpus import stories
from byoai.gpt import GPT, GPTConfig, train
from byoai.tokenizer import Tokenizer

N_STORIES = 3000
VOCAB = 1024
EOT = "<|endoftext|>"
CONFIG = dict(block_size=64, n_layer=3, n_head=4, n_embd=96)
STEPS = 500


def checkpoints() -> Path:
    d = Path(os.environ.get("BYOAI_CHECKPOINTS", Path(__file__).resolve().parent.parent / "checkpoints"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _code(*modules: str) -> str:
    """A fingerprint of the source files a checkpoint depends on."""
    here = Path(__file__).resolve().parent
    # Line endings normalized: a Windows checkout with CRLF must find the same checkpoint.
    return hashlib.sha256(b"".join((here / m).read_bytes().replace(b"\r\n", b"\n") for m in modules)).hexdigest()[:12]


def _key(**parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def story_tokenizer() -> Tokenizer:
    """BPE trained on the stories, with `<|endoftext|>` between them."""
    key = _key(n=N_STORIES, vocab=VOCAB, eot=EOT, code=_code("corpus.py", "tokenizer.py"))
    path = checkpoints() / f"stories-{key}.tok.json"
    if path.exists():
        return Tokenizer.load(path)
    tok = Tokenizer.train(stories(N_STORIES, seed=0), vocab_size=VOCAB, special=[EOT])
    _replace_old(path, "stories-*.tok.json")
    tok.save(path)
    return tok


def story_ids(tok: Tokenizer, n: int = N_STORIES, seed: int = 0) -> np.ndarray:
    """The stories as one stream of token ids, each story followed by `<|endoftext|>`."""
    eot = tok.special[EOT]
    ids: list[int] = []
    for line in stories(n, seed=seed).splitlines():
        ids += tok.encode(line)
        ids.append(eot)
    return np.array(ids, dtype=np.int64)


def story_model(steps: int = STEPS, seed: int = 0, verbose: bool = False) -> tuple[GPT, Tokenizer]:
    """The chapter-3 GPT trained on the stories (from the cache when it is there)."""
    tok = story_tokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size, **CONFIG)
    key = _key(cfg=asdict(cfg), steps=steps, seed=seed, n=N_STORIES, code=_code("corpus.py", "tokenizer.py", "autograd.py", "gpt.py"))
    path = checkpoints() / f"story-gpt-{key}.npz"
    if path.exists():
        return GPT.load(path), tok
    data = story_ids(tok)
    model = GPT(cfg, seed=seed)
    log = (lambda s, loss, t: print(f"  step {s:>4}: loss {loss:.3f}  ({t:.0f} s)")) if verbose else None
    train(model, data, steps=steps, batch_size=32, lr=3e-3, warmup=50, seed=seed, log=log)
    _replace_old(path, "story-gpt-*.npz")
    model.save(path)
    return model, tok


def _replace_old(path: Path, pattern: str) -> None:
    """A new checkpoint replaces older ones of the same kind (made by earlier code or settings), so
    editing `gpt.py` for an exercise does not leave a pile of stale models behind."""
    for old in path.parent.glob(pattern):
        if old != path:
            old.unlink()
