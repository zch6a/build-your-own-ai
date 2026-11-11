"""The numbers the walkthroughs print, pinned: a code change that moves one fails here, and the text gets
updated with it instead of silently going stale. (Only the fast ones; the trained model's numbers are
checked by running the demos.)"""

import pytest

from byoai.agent import Agent
from byoai.corpus import knowledge_base, stories
from byoai.planner import Noisy, TownPlanner, agent_tasks, is_correct, town_tools
from byoai.rag import BM25, RAG, chunk_documents, evaluate_retrieval
from byoai.tokenizer import Tokenizer
from byoai.vectordb import FlatIndex, HNSWIndex, clustered_vectors, recall_at_k


def test_chapter_1_merges():
    assert len(Tokenizer.train(stories(2000, seed=0), vocab_size=1024).merges) == 264


@pytest.fixture(scope="module")
def kb():
    return knowledge_base()


def test_chapter_6_tables(kb):
    documents, questions = kb
    titled = chunk_documents(documents, max_words=40)
    bare = chunk_documents(documents, max_words=40, header=False)
    m = evaluate_retrieval(BM25([c.text for c in titled]), titled, questions)
    assert (round(m["hit@1"], 2), round(m["hit@5"], 2), round(m["MRR"], 2)) == (0.73, 0.93, 0.81)
    assert round(evaluate_retrieval(BM25([c.text for c in bare]), bare, questions)["hit@5"], 2) == 0.76
    rag = RAG(titled, BM25([c.text for c in titled]), k=4)
    answers = [rag.ask(q["question"])["answer"] for q in questions]
    correct = sum(q["answer"] in a for q, a in zip(questions, answers)) / len(questions)
    idk = sum(a == "I don't know." for a in answers) / len(questions)
    assert (round(correct, 2), round(idk, 2)) == (0.69, 0.31)


def test_chapter_7_tables(kb):
    documents, questions = kb
    tasks, tools = agent_tasks(documents, questions, 200), town_tools(documents)
    assert len(tasks) == 199
    rate = lambda agent: sum(is_correct(agent.run(t["task"]).answer, t) for t in tasks) / len(tasks)  # noqa: E731
    assert rate(Agent(TownPlanner(), tools)) == 1.0
    assert round(rate(Agent(Noisy(TownPlanner(), 0.3, seed=1), tools)), 2) == 0.99
    assert round(rate(Agent(Noisy(TownPlanner(), 0.3, seed=1), tools, feedback=False)), 2) == 0.54


def test_chapter_5_heuristic():
    data, queries = clustered_vectors(2000, 32, seed=1), clustered_vectors(100, 32, seed=2)
    flat = FlatIndex(32)
    flat.add(data)
    exact = [flat.search(q, 10)[0] for q in queries]
    recalls = []
    for nearest_only in (True, False):
        index = HNSWIndex(32, M=12, ef_construction=80, seed=0)
        if nearest_only:
            index._select_neighbors = lambda q, cands, m: [c for _, c in cands[:m]]
        index.add(data)
        recalls.append(round(recall_at_k([index.search(q, 10, ef=32)[0] for q in queries], exact), 3))
    assert recalls == [0.719, 0.927]
