"""Chapter 6 — RAG: retrieve the right passages, then answer from them.

A model only knows what was in its training data. Retrieval-augmented generation looks the answer up
first: split the documents into chunks, find the chunks most relevant to the question, and put them in
the prompt. Everything here is measured against questions whose answers are known, because "it seems to
work" is how most RAG systems ship broken.

    from byoai.rag import BM25, chunk_documents, RAG
    chunks = chunk_documents(documents, max_words=40)
    rag = RAG(chunks, BM25([c.text for c in chunks]))
    rag.ask("Who is the mayor of Nustorford?")

Retrievers: `BM25` (keywords, weighted), `LSA` (dense vectors from a factorized word matrix, searched
with chapter 5's index), and `Hybrid` (both, merged by reciprocal rank fusion).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from byoai.vectordb import FlatIndex

_WORD = re.compile(r"[^\W_]+")  # letters and digits of any script: "Zoë" stays one word
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> list[str]:
    """Lowercase words and numbers; "25,700" becomes "25700" so the comma does not split it."""
    return _WORD.findall(re.sub(r"(?<=\d),(?=\d)", "", text.lower()))


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


# ---- chunking ---------------------------------------------------------------------------------------------
@dataclass
class Chunk:
    doc: int  # index of the document it came from
    title: str
    text: str  # what is indexed and shown to the model (with the title in front, if `header`)


def chunk_documents(documents: list[dict], max_words: int = 40, overlap: int = 1, header: bool = True) -> list[Chunk]:
    """Split each document into chunks of whole sentences, at most `max_words` words each (a longer
    sentence is a chunk of its own). Consecutive chunks share `overlap` sentences, so a fact that
    needs two neighbouring sentences is not always cut in half.

    `header=True` puts the document's title in front of every chunk. Without it, a chunk that says
    "Its mayor is Cleo Ashby." has lost the one word a search for the town could find it by.
    """
    chunks = []
    for d, doc in enumerate(documents):
        sents = sentences(doc["text"])
        start = 0
        while start < len(sents):
            end, n = start, 0
            while end < len(sents) and (end == start or n + len(sents[end].split()) <= max_words):
                n += len(sents[end].split())
                end += 1
            body = " ".join(sents[start:end])
            chunks.append(Chunk(d, doc["title"], f"{doc['title']}: {body}" if header else body))
            if end == len(sents):
                break
            start = max(end - overlap, start + 1)
    return chunks


# ---- retrievers -------------------------------------------------------------------------------------------
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """The k best chunk ids with their scores, best first."""
        ...


class BM25:
    """Okapi BM25: the keyword ranking that search engines used for decades, and still the baseline
    any RAG retriever has to beat.

    score(chunk) = Σ over query words w of  idf(w) · tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))

    * idf: a word in few chunks ("Nustorford") counts for much more than one in all ("the").
    * tf saturates: the 2nd mention of a word adds less than the 1st, the 10th almost nothing (`k1`).
    * long chunks are penalized a little (`b`): they match more words just by being long.
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        counts = [Counter(words(t)) for t in texts]
        self.n = len(texts)
        self.lengths = np.array([sum(c.values()) for c in counts], dtype=np.float64)
        self.avg_len = self.lengths.mean() if self.n else 0.0
        df = Counter(w for c in counts for w in c)
        self.idf = {w: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        # An inverted index: word -> (chunk ids, term counts). A query only touches its own words' lists.
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(counts):
            for w, tf in c.items():
                postings.setdefault(w, []).append((i, tf))
        self.postings = {w: (np.array([i for i, _ in p]), np.array([tf for _, tf in p], dtype=np.float64)) for w, p in postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_len, 1e-9))
        for w in set(words(query)):
            if w in self.postings:
                ids, tf = self.postings[w]
                s[ids] += self.idf[w] * tf * (self.k1 + 1) / (tf + norm[ids])
        return s

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]


class LSA:
    """Dense retrieval without a neural network: latent semantic analysis (Deerwester et al., 1990).

    Build the chunk × word TF-IDF matrix, factorize it with an SVD, and keep the `dim` strongest
    directions. Each chunk becomes a `dim`-dimensional vector; words that occur in the same chunks end
    up pointing the same way, so a query can match a chunk that shares none of its words. The vectors
    go into chapter 5's `FlatIndex`.
    """

    def __init__(self, texts: list[str], dim: int = 64):
        counts = [Counter(words(t)) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for c in counts for w in c}))}
        df = np.zeros(len(self.vocab))
        for c in counts:
            for w in c:
                df[self.vocab[w]] += 1
        self.idf = np.log(len(texts) / df) + 1
        X = np.stack([self._tfidf(c) for c in counts])
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        self.basis = vt[: min(dim, len(vt))].T  # (vocab, dim): word space -> concept space
        self.index = FlatIndex(self.basis.shape[1])
        self.index.add(X @ self.basis)

    def _tfidf(self, counts: Counter) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, tf in counts.items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(tf)) * self.idf[self.vocab[w]]
        return v

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = self._tfidf(Counter(words(query))) @ self.basis
        if not q.any():
            return []
        ids, scores = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids, scores)]


class Hybrid:
    """Reciprocal rank fusion (Cormack et al., 2009): score = Σ 1 / (c + rank) over the retrievers.

    BM25 scores and cosine similarities are on different scales, so adding them needs tuning; ranks
    are on the same scale by construction. A chunk ranked well by either retriever rises; one ranked
    well by both rises most. `c = 60` is the paper's constant and the usual default.
    """

    def __init__(self, *retrievers: Retriever, c: int = 60, depth: int = 50):
        self.retrievers, self.c, self.depth = retrievers, c, depth

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for r in self.retrievers:
            for rank, (i, _) in enumerate(r.search(query, self.depth)):
                fused[i] = fused.get(i, 0.0) + 1 / (self.c + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


# ---- measuring retrieval ----------------------------------------------------------------------------------
def evaluate_retrieval(retriever: Retriever, chunks: list[Chunk], questions: list[dict], k: int = 5) -> dict[str, float]:
    """A retrieved chunk is relevant if it comes from the question's document *and* contains the answer.

    * hit@1, hit@k: the share of questions with a relevant chunk first / in the top k.
    * MRR: the mean of 1/rank of the first relevant chunk (0 if none in the top k) — rewards ranking it
      high, not just somewhere.
    """
    if not questions:
        raise ValueError("no questions to evaluate on")
    hit1 = hitk = rr = 0.0
    for q in questions:
        for rank, (i, _) in enumerate(retriever.search(q["question"], k), start=1):
            if chunks[i].doc == q["doc"] and q["answer"] in chunks[i].text:
                hit1 += rank == 1
                hitk += 1
                rr += 1 / rank
                break
    n = len(questions)
    return {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "MRR": rr / n}


# ---- answering --------------------------------------------------------------------------------------------
def build_prompt(question: str, passages: list[str]) -> list[dict]:
    """Chat messages for a model: numbered passages, the question, and the rules that make the answer
    checkable (cite the passage; say so when the passages do not contain the answer)."""
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return [
        {"role": "system", "content": (
            "Answer the question using only the numbered passages. Cite the passage you used, like [2]. "
            "If the passages do not contain the answer, reply exactly: I don't know.")},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
    ]


class SentenceReader:
    """A stand-in for the language model that writes the answer: it answers with the single sentence
    from the passages that best matches the question, and its passage number — or "I don't know."

    Two rules, both things a careful reader does:

    * **Stay on topic.** The names in the question must appear in the passage. A passage about another
      town is not evidence, however well its words match. Names are the known document titles found in
      the question, in any case ("nustorford" too), plus capitalized words after the first. (The first
      version used capitalization alone: lowercase questions silently turned the rule off and the
      reader went from 0% to 61% wrong answers — found by an independent review.)
    * **Match on the rest.** Among those passages' sentences, take the one sharing the most question
      words (weighted by idf), not counting the names — "It was founded in 1690." does not repeat the
      town, and should not lose to a sentence that merely does.

    Word weights are idf over *sentences* (a word in every document can still be rare among sentences:
    "mayor"), and question words ("how", "which") are ignored — "How many people live in X?" must not
    match "Visitors remark on how quiet the evenings are." Below `min_score` it says it does not know.

    It is literal — it cannot tell that "the first houses were built" answers "founded" — which is
    exactly the gap a real model fills (see `byoai.llm`).
    """

    QUESTION_WORDS = {"who", "what", "which", "when", "where", "why", "how", "many", "much", "does", "do", "did", "is", "are", "was", "the", "a", "an", "of", "in", "on", "for"}

    def __init__(self, texts: list[str], min_score: float = 1.0, names: list[str] = ()):
        self.known_names = {w for n in names for w in words(n)}
        all_sentences = [set(words(s)) for t in texts for s in sentences(t)]
        df = Counter(w for s in all_sentences for w in s)
        self.idf = {w: math.log(len(all_sentences) / f) for w, f in df.items()}
        self.min_score = min_score

    def answer(self, question: str, passages: list[str], titles: list[str] | None = None) -> tuple[str, int | None]:
        """`titles`, if known, are the passages' document titles: then a passage must be *about* the
        names in the question (its title names one), not merely mention them — "The nearest town is
        Nustorford." in the article on Tamgalwick says nothing about Nustorford's mayor."""
        asked = set(words(question))
        names = (asked & self.known_names) | {w.lower() for w in re.findall(r"\b[A-Z][a-z]+", question)[1:]}
        q = set(asked) - names - self.QUESTION_WORDS
        best, best_score, source = "I don't know.", self.min_score, None
        for n, passage in enumerate(passages, start=1):
            if not names <= set(words(passage)):
                continue
            if titles is not None and names and not names & set(words(titles[n - 1])):
                continue
            for s in sentences(passage.split(": ", 1)[-1]):
                score = sum(self.idf.get(w, 0.0) for w in q & set(words(s)))
                if score > best_score:
                    best, best_score, source = s, score, n
        return best, source


class RAG:
    """Retrieve `k` chunks, then answer from them — with `llm` (anything with `.chat(messages)`, see
    `byoai.llm`) if given, otherwise with the `SentenceReader` stand-in."""

    def __init__(self, chunks: list[Chunk], retriever: Retriever, k: int = 4, llm=None):
        self.chunks, self.retriever, self.k, self.llm = chunks, retriever, k, llm
        self.reader = SentenceReader([c.text for c in chunks], names=sorted({c.title for c in chunks}))

    def ask(self, question: str) -> dict:
        hits = self.retriever.search(question, self.k)
        passages = [self.chunks[i].text for i, _ in hits]
        if self.llm is not None:
            answer = self.llm.chat(build_prompt(question, passages))["content"]
            m = re.search(r"\[(\d+)\]", answer)
            source = int(m.group(1)) if m else None
        else:
            answer, source = self.reader.answer(question, passages, [self.chunks[i].title for i, _ in hits])
        cited = hits[source - 1][0] if source and 0 < source <= len(hits) else None
        return {"answer": answer, "source": cited, "passages": passages}
