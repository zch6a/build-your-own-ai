"""Chapter 1 — a byte-level BPE tokenizer, the kind GPT models use.

A language model reads integers, not text. The tokenizer decides which integers: too few kinds (one
per byte) and every sentence becomes a long sequence the model must attend over; too many (one per
word) and it never sees most of them often enough to learn them. Byte-pair encoding (BPE) finds the
middle: start from the 256 byte values, and repeatedly give the most frequent *pair* of adjacent
tokens an id of its own.

Three decisions shape it:

* **Bytes, not characters.** Text is UTF-8 encoded first, so any input — emoji, Chinese, a binary
  blob — has a lossless encoding: the 256 byte tokens are always there to fall back on. There is no
  "unknown" token.
* **Pre-tokenization.** Before counting pairs, text is split into word-like chunks (`" the"`,
  `"'s"`, `" 123"`). Merges never cross a chunk boundary, so the vocabulary is spent on pieces of
  words rather than on accidents like `"e t"`.
* **Special tokens are opt-in.** `<|endoftext|>` in a document is just text unless the caller allows
  special tokens — otherwise anyone who can put text in front of the model could inject a control
  token.

    >>> from byoai.tokenizer import Tokenizer
    >>> tok = Tokenizer.train("the cat sat on the mat. the cat ate.", vocab_size=270)
    >>> tok.decode(tok.encode("the cat")) == "the cat"
    True
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Word-like chunks: contractions, letters (with one leading space), up to 3 digits, punctuation runs,
# and whitespace. `[^\W\d_]` is "a letter" in any script (Python's `re` has no `\p{L}`). The underscore
# is a "word" character to `re` but neither a letter nor a digit, so it must be named in the punctuation
# branch — without it, `snake_case` silently lost its underscores (a reviewer found it; a test now
# checks that the chunks always join back to the input).
PRETOKENIZE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")

Pair = tuple[int, int]


class Tokenizer:
    """Byte-level BPE: `merges[(a, b)] = id` in the order they were learned."""

    def __init__(self, merges: dict[Pair, int] | None = None, special: dict[str, int] | None = None):
        self.merges: dict[Pair, int] = dict(merges or {})
        self.special: dict[str, int] = dict(special or {})
        # The bytes each id stands for: 0..255 are themselves, every merge is its two parts joined.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new] = self.vocab[a] + self.vocab[b]
        for text, i in self.special.items():
            self.vocab[i] = text.encode("utf-8")
        self._cache: dict[str, list[int]] = {}
        self._special_re = re.compile("|".join(re.escape(s) for s in sorted(self.special, key=len, reverse=True))) if self.special else None

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    # ---- training ------------------------------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int, special: list[str] | None = None, verbose: bool = False) -> Tokenizer:
        """Learn `vocab_size - 256 - len(special)` merges from `text`.

        Each distinct chunk is a word with a count; a pair's frequency is the sum over the words it
        occurs in. Ties go to the smallest pair, so training is deterministic. Training stops early when
        there is nothing left to merge — every chunk is a single token — so ocab_size is an upper bound.
        """
        special = special or []
        n_merges = vocab_size - 256 - len(special)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special)}")

        counts = Counter(PRETOKENIZE.findall(text))
        words: list[list[int]] = [list(w.encode("utf-8")) for w in counts]
        freq: list[int] = list(counts.values())

        # Pair frequencies, and which words each pair occurs in — so a merge only revisits those.
        pair_count: Counter[Pair] = Counter()
        where: defaultdict[Pair, set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_count[p] += freq[wi]
                where[p].add(wi)

        merges: dict[Pair, int] = {}
        for step in range(n_merges):
            if not pair_count:
                break  # every word is a single token already
            best = max(pair_count, key=lambda p: (pair_count[p], (-p[0], -p[1])))
            if pair_count[best] < 1:
                break
            new = 256 + step
            merges[best] = new
            for wi in list(where.pop(best, ())):
                w = words[wi]
                # Take the word's pairs out of the counts, merge, put the new pairs back in.
                for p in zip(w, w[1:]):
                    pair_count[p] -= freq[wi]
                    if pair_count[p] <= 0:
                        del pair_count[p]
                    where[p].discard(wi)
                w = words[wi] = _merge(w, best, new)
                for p in zip(w, w[1:]):
                    pair_count[p] += freq[wi]
                    where[p].add(wi)
            if verbose and step % 100 == 0:
                print(f"merge {step}: {best} -> {new}")

        # Special tokens come after every merge.
        return cls(merges, {s: 256 + len(merges) + i for i, s in enumerate(special)})

    # ---- encoding ------------------------------------------------------------------------------
    def encode(self, text: str, allowed_special: str | set[str] = "none") -> list[int]:
        """Text to ids. Special tokens are only recognised if allowed (`"all"`, or a collection of them)."""
        if isinstance(allowed_special, str) and allowed_special not in ("all", "none"):
            raise ValueError(f"allowed_special must be \"all\", \"none\" or a collection of tokens, not {allowed_special!r}")
        allowed = set(self.special) if allowed_special == "all" else (set() if allowed_special == "none" else set(allowed_special))
        ids: list[int] = []
        if not allowed or not self._special_re:
            return self._encode_ordinary(text)
        pos = 0
        for m in self._special_re.finditer(text):
            if m.group() not in allowed:
                continue
            ids += self._encode_ordinary(text[pos : m.start()])
            ids.append(self.special[m.group()])
            pos = m.end()
        return ids + self._encode_ordinary(text[pos:])

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in PRETOKENIZE.findall(text):
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                if len(self._cache) < 100_000:
                    self._cache[chunk] = cached
            out += cached
        return out

    def _encode_chunk(self, data: bytes) -> list[int]:
        """Apply merges in the order they were learned: always the earliest-learned pair first."""
        ids = list(data)
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, 1 << 60))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    # ---- decoding ------------------------------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        """Ids to text. A sequence cut in the middle of a UTF-8 character decodes to U+FFFD there."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def decode_bytes(self, ids: list[int]) -> bytes:
        try:
            return b"".join(self.vocab[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"unknown token id {e.args[0]}") from None

    def token_text(self, i: int) -> str:
        """What a single token looks like (for display), with unprintable bytes escaped."""
        b = self.vocab[i]
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.hex(" ")

    # ---- saving --------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {"version": 1, "merges": [[a, b] for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])], "special": self.special}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError(f"unsupported tokenizer file version {data.get('version')}")
        merges = {(a, b): 256 + i for i, (a, b) in enumerate(data["merges"])}
        return cls(merges, data.get("special", {}))


def _merge(ids: list[int], pair: Pair, new: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new`, left to right."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
