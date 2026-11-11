"""Chapter 6 demo: chunking, three retrievers, and answering from what was retrieved — all measured.

    uv run python chapters/06-rag/run.py

With BYOAI_LLM_BASE_URL / BYOAI_LLM_MODEL / BYOAI_LLM_API_KEY set (see byoai/llm.py), part 4 also
answers 30 questions with that model. Without them nothing leaves your machine.
"""

from __future__ import annotations

import sys

from byoai.corpus import knowledge_base
from byoai.llm import OpenAICompatible
from byoai.rag import BM25, LSA, RAG, Hybrid, chunk_documents, evaluate_retrieval

# Print UTF-8 even when the output is piped or redirected (Windows would otherwise use its legacy code page).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IDK = "I don't know."
documents, questions = knowledge_base()
print(f"knowledge base: {len(documents)} articles about towns that do not exist, {len(questions)} questions\n")
print("  " + documents[0]["text"][:300] + "…")
print("  e.g. " + " / ".join(q["question"] for q in questions[:3]))


def row(name: str, m: dict) -> str:
    return f"  {name:<36} {m['hit@1']:>6.2f} {m['hit@5']:>6.2f} {m['MRR']:>6.2f}"


# ---- 1. Chunking --------------------------------------------------------------------------------------------
print(f"\n1. chunking (BM25; a hit = a chunk from the right article that contains the answer)\n  {'':<36} {'hit@1':>6} {'hit@5':>6} {'MRR':>6}")
for max_words in (1000, 40, 20, 10):
    for header in (True, False):
        chunks = chunk_documents(documents, max_words=max_words, header=header)
        label = f"{'whole articles' if max_words == 1000 else f'≤{max_words} words'}, {'title on each' if header else 'no title'} ({len(chunks)})"
        print(row(label, evaluate_retrieval(BM25([c.text for c in chunks]), chunks, questions)))

# ---- 2. Retrievers ------------------------------------------------------------------------------------------
chunks = chunk_documents(documents, max_words=40)
texts = [c.text for c in chunks]
print(f"\n2. retrievers ({len(chunks)} chunks of ≤40 words, titled)\n  {'':<36} {'hit@1':>6} {'hit@5':>6} {'MRR':>6}")
bm25, lsa = BM25(texts), LSA(texts, dim=64)
for name, retriever in [("BM25", bm25), ("LSA, 64 dimensions", lsa), ("hybrid (reciprocal rank fusion)", Hybrid(bm25, lsa))]:
    print(row(name, evaluate_retrieval(retriever, chunks, questions)))
for dim in (16, 32, 128):
    print(row(f"LSA, {dim} dimensions", evaluate_retrieval(LSA(texts, dim=dim), chunks, questions)))

# ---- 3. Answering -------------------------------------------------------------------------------------------
print("\n3. answering with the literal stand-in reader (BM25, ≤40-word chunks)")
print(f"  {'':<40} {'correct':>8} {'I dont know':>12} {'wrong':>6}")


def tally(answers: list[str]) -> str:
    n = len(questions)
    right = sum(q["answer"] in a for q, a in zip(questions, answers))
    idk = sum(a == IDK for a in answers)
    return f"{right / n:>8.0%} {idk / n:>12.0%} {(n - right - idk) / n:>6.0%}"


for k in (1, 4):
    rag = RAG(chunks, bm25, k=k)
    print(f"  {f'top {k} chunk' + ('s' if k > 1 else ''):<40} {tally([rag.ask(q['question'])['answer'] for q in questions])}")
rag = RAG(chunks, bm25, k=4)
untitled = [rag.reader.answer(q["question"], [chunks[i].text for i, _ in bm25.search(q["question"], 4)])[0] for q in questions]
print(f"  {'top 4, without the on-topic rule':<40} {tally(untitled)}")
whole = chunk_documents(documents, max_words=1000)
oracle = [RAG(whole, bm25).reader.answer(q["question"], [whole[q["doc"]].text], [whole[q["doc"]].title])[0] for q in questions]
print(f"  {'the right article, handed over (ceiling)':<40} {tally(oracle)}")

print("\n  some answers:")
for q in questions[:6]:
    out = rag.ask(q["question"])
    print(f"    {q['question']:<52} → {out['answer']}" + ("" if q["answer"] in out["answer"] else f"   (answer: {q['answer']})"))

# ---- 4. A real model, if configured --------------------------------------------------------------------------
llm = OpenAICompatible.from_env()
if llm is None:
    print("\n4. a real model: not configured (set BYOAI_LLM_BASE_URL, BYOAI_LLM_MODEL, BYOAI_LLM_API_KEY to try one)")
else:
    sample = questions[::12]
    rag = RAG(chunks, bm25, k=4, llm=llm)
    answers = [rag.ask(q["question"])["answer"] for q in sample]
    right = sum(q["answer"].lower() in a.lower() for q, a in zip(sample, answers))
    print(f"\n4. {llm.model}: {right}/{len(sample)} correct with the same retrieval")
    for q, a in list(zip(sample, answers))[:5]:
        print(f"    {q['question']:<52} → {a.strip()[:80]}")
