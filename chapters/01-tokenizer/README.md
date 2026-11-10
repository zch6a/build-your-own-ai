# Chapter 1 — A tokenizer: byte-level BPE

> **Code:** [`byoai/tokenizer.py`](../../byoai/tokenizer.py) (about 200 lines) · **Tests:** [`tests/test_tokenizer.py`](../../tests/test_tokenizer.py) · **Run:** `uv run python chapters/01-tokenizer/run.py`

A language model never sees text. It sees a sequence of integers, predicts the next integer, and the
tokenizer turns those integers back into text. Which integers it uses is a real design decision:

* **One token per byte** (256 tokens): nothing is ever unknown, but a sentence becomes a long
  sequence, and attention cost grows with the square of the length.
* **One token per word**: short sequences, but a vocabulary of hundreds of thousands, most of them too
  rare to learn — and a word the tokenizer never saw has no token at all.

Byte-pair encoding finds the middle, and it does so from data.

## The algorithm, on one word

Start with the bytes. Count every pair of adjacent tokens. Give the most frequent pair a new id.
Repeat.

```text
a a a b d a a a b a c        pairs: (a,a)×4  (a,b)×2  (b,d) (d,a) (b,a) (a,c)
                             merge (a,a) → 256
256 a b d 256 a b a c        pairs: (256,a)×2  (a,b)×2 ...   a tie: the smallest pair wins
                             merge (97,98) "ab" → 257
256 257 d 256 257 a c        ...
```

Each merge makes the text shorter. The list of merges *is* the tokenizer: encoding new text means
applying the same merges, in the order they were learned. That order matters — `"ab"` must be merged
before a later merge that builds on it.

The first test in [`tests/test_tokenizer.py`](../../tests/test_tokenizer.py) is exactly this example.

## Three decisions that make it work in practice

**Bytes, not characters.** The text is UTF-8 encoded before anything else, so the base vocabulary is
the 256 byte values — and *every* possible input has an encoding, however strange. Run the demo: a
fox emoji and two Chinese characters the tokenizer never saw still round-trip exactly, they just cost
more tokens (one per byte). A character-level tokenizer would need an "unknown" token, and would lose
information there.

**Pre-tokenization.** Counting pairs over raw text would happily merge `"e"` + `" t"` because
`"the tree"` is common. That spends vocabulary on accidents of word order. So the text is first split
into word-like chunks with a regular expression:

```python
PRETOKENIZE.findall("Mia's 2024 kites, flew!")
# ['Mia', "'s", ' 202', '4', ' kites', ',', ' flew', '!']
```

Merges happen only *inside* a chunk. Note the leading space belongs to the word (`' kites'`): the
model learns `" kites"` as one token, which is why GPT vocabularies are full of tokens starting with a
space. Numbers are split into groups of at most three digits so that `"2024"` and `"1999"` share
pieces instead of each becoming a token of its own.

**Special tokens are opt-in.** Models use control tokens like `<|endoftext|>` to mark document
boundaries. If the tokenizer turned that *string* into the control token wherever it appears, anyone
who can put text in front of the model — a web page, a file, a tool result — could inject one. So
`encode()` treats special tokens as ordinary text unless the caller passes `allowed_special`.

## Making training fast

The naive loop recounts every pair in the whole corpus after every merge. Two ideas make it fast
enough to train in a fraction of a second here:

1. **Count words, not positions.** After pre-tokenization, the corpus is a few hundred *distinct*
   chunks with counts (`" the"` × 4,821). A pair's frequency is the sum over the chunks it occurs in.
2. **Update incrementally.** Keep, for each pair, the set of chunks it occurs in. A merge only touches
   those chunks: subtract their old pairs, merge, add their new pairs. Everything else stays counted.

```python
for wi in list(where.pop(best, ())):     # only the words that contain the pair
    for p in zip(w, w[1:]): pair_count[p] -= freq[wi]   # their old pairs out
    w = words[wi] = _merge(w, best, new)
    for p in zip(w, w[1:]): pair_count[p] += freq[wi]   # their new pairs in
```

## What the demo shows

* The first merges are the commonest pieces of English: `" t"`, `"he"`, `" the"`, `" and"`.
* The last ones are whole words of the stories: `" shell"`, `" map"`, `" ball"`.
* Training stops early: these generated stories use about a hundred distinct words, so after 264
  merges every word is a single token and there is nothing left to merge. `vocab_size` is an upper
  bound. A real tokenizer is trained on far more varied text and uses 50,000–200,000 tokens.
* Compression levels off at the same point: about 3.9 bytes per token once every word is one token.

## Exercises

1. Change the pre-tokenization pattern so digits are split one by one. What happens to the
   compression of `"in 2024 and 2025"`? (Some modern tokenizers do exactly this, for arithmetic.)
2. Train on `stories(2000)` and encode a sentence of plain English prose that is *not* about these
   stories. Which words fall apart into small pieces?
3. Write `encode` without the cache. How much slower is encoding the training text?
4. A tokenizer can be attacked: find a string that encodes to many more tokens than its length in
   bytes would suggest. (Hint: there is a hard upper bound — why?)

## Where real tokenizers differ

GPT-2's tokenizer maps bytes to printable Unicode characters so its merges file is readable text;
`tiktoken` implements the same algorithm in Rust with a pattern that uses Unicode classes Python's
`re` does not support (`\p{L}`), splits numbers differently, and ships hand-picked special tokens.
SentencePiece (used by many open models) treats spaces as a visible character and can train with a
unigram model instead of merges. The core idea — a learned vocabulary of byte sequences — is the same.

**Next:** [Chapter 2 — Autograd](../02-autograd/README.md): before we can build a model that predicts
the next token, we need a way to compute gradients.
