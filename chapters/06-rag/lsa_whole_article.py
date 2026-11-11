"""Chapter 6: why LSA's hit@5 looks better than BM25's — it tends to return a whole article.

    uv run python chapters/06-rag/lsa_whole_article.py        # ~2 seconds

For each question, take as many results as the right article has chunks (three, at most), and count how
often *all* of them come from that article.
"""

from __future__ import annotations

from collections import Counter

from byoai.corpus import knowledge_base
from byoai.rag import BM25, LSA, chunk_documents

documents, questions = knowledge_base()
chunks = chunk_documents(documents, max_words=40)
texts = [c.text for c in chunks]
per_article = Counter(c.doc for c in chunks)
for name, retriever in (("BM25", BM25(texts)), ("LSA, 64 dimensions", LSA(texts, dim=64)), ("LSA, 128 dimensions", LSA(texts, dim=128))):
    whole = sum(all(chunks[i].doc == q["doc"] for i, _ in retriever.search(q["question"], per_article[q["doc"]])) for q in questions)
    print(f"  {name:<20} top results all from the right article: {whole / len(questions):.0%}")
