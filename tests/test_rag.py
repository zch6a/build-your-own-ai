import math

import pytest

from byoai.corpus import knowledge_base
from byoai.llm import ScriptedLLM
from byoai.rag import BM25, LSA, RAG, Hybrid, SentenceReader, build_prompt, chunk_documents, evaluate_retrieval, sentences, words


def test_words_keeps_numbers_whole():
    assert words("About 25,700 people, in 1690!") == ["about", "25700", "people", "in", "1690"]


# ---- chunking ----------------------------------------------------------------------------------------------
DOC = {"title": "Testford", "text": "One two three. Four five six seven. Eight nine. Ten eleven twelve thirteen fourteen. Fifteen."}


def test_chunks_cover_every_sentence_and_respect_the_size():
    chunks = chunk_documents([DOC], max_words=6, overlap=0, header=False)
    assert " ".join(c.text for c in chunks) == DOC["text"]
    for c in chunks:
        assert len(c.text.split()) <= 6 or len(sentences(c.text)) == 1  # only a lone long sentence may exceed


def test_overlap_repeats_the_last_sentence():
    chunks = chunk_documents([DOC], max_words=7, overlap=1, header=False)
    for a, b in zip(chunks, chunks[1:]):
        assert sentences(a.text)[-1] == sentences(b.text)[0]


def test_chunking_always_terminates():
    # Overlap as large as a whole chunk must still move forward.
    chunks = chunk_documents([DOC], max_words=1, overlap=5, header=False)
    assert len(chunks) == len(sentences(DOC["text"]))


def test_header_names_the_document():
    chunks = chunk_documents([DOC], max_words=6)
    assert all(c.text.startswith("Testford: ") and c.doc == 0 for c in chunks)


# ---- BM25 --------------------------------------------------------------------------------------------------
def test_bm25_matches_the_formula():
    texts = ["the cat sat", "the dog sat on the mat", "a bird"]
    bm = BM25(texts, k1=1.5, b=0.75)
    avg = (3 + 6 + 2) / 3
    idf_cat = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    idf_the = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = idf_cat * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 3 / avg)) + idf_the * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 3 / avg))
    assert bm.scores("cat the")[0] == pytest.approx(expected)


def test_bm25_rare_words_count_more_and_repeats_saturate():
    bm = BM25(["alpha common", "beta common", "gamma common", "alpha alpha alpha alpha alpha alpha common"])
    assert bm.idf["alpha"] > bm.idf["common"]
    s = bm.scores("alpha")
    assert s[3] > s[0] and s[3] < 2 * s[0]  # six mentions are worth less than two documents' worth
    assert bm.search("nothing matches this", 3) == []


def test_bm25_penalizes_length():
    bm = BM25(["key", "key filler filler filler filler filler filler filler"])
    s = bm.scores("key")
    assert s[0] > s[1]


# ---- LSA and fusion ----------------------------------------------------------------------------------------
def test_lsa_matches_without_shared_words():
    texts = ["cat feline", "feline whiskers", "dog canine", "canine bark"]
    ids = [i for i, _ in LSA(texts, dim=2).search("cat", 4)]
    assert ids[0] == 0 and ids.index(1) < ids.index(2) and ids.index(1) < ids.index(3)
    assert 1 not in [i for i, _ in BM25(texts).search("cat", 4)]  # keywords cannot find it at all


class Fixed:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, k=5):
        return [(i, 1.0) for i in self.ranking[:k]]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = Hybrid(Fixed([7, 2, 3]), Fixed([5, 2, 9])).search("q", 5)
    assert fused[0][0] == 2  # second in both beats first in one
    assert fused[0][1] == pytest.approx(2 / 62)


# ---- evaluation --------------------------------------------------------------------------------------------
def test_evaluate_retrieval_counts_rank_and_answer():
    docs = [{"title": "A", "text": "The answer is x1."}, {"title": "B", "text": "The answer is y2."}]
    chunks = chunk_documents(docs)
    questions = [{"question": "q", "answer": "x1", "doc": 0}, {"question": "q", "answer": "y2", "doc": 1}]
    m = evaluate_retrieval(Fixed([1, 0]), chunks, questions, k=2)
    assert m == {"hit@1": 0.5, "hit@2": 1.0, "MRR": 0.75}


@pytest.fixture(scope="module")
def kb():
    return knowledge_base()


def test_bm25_finds_the_facts(kb):
    docs, questions = kb
    chunks = chunk_documents(docs, max_words=40)
    assert evaluate_retrieval(BM25([c.text for c in chunks]), chunks, questions)["hit@5"] > 0.9


def test_the_title_header_matters(kb):
    docs, questions = kb
    with_header = chunk_documents(docs, max_words=40)
    without = chunk_documents(docs, max_words=40, header=False)
    a = evaluate_retrieval(BM25([c.text for c in with_header]), with_header, questions)["hit@5"]
    b = evaluate_retrieval(BM25([c.text for c in without]), without, questions)["hit@5"]
    assert a > b + 0.1, (a, b)


# ---- answering ---------------------------------------------------------------------------------------------
PASSAGES = [
    "Kelby: The mayor of Kelby is Ada Venn. It was founded in 1700.",
    "Morton: Morton was founded in 1500. Its mayor is Bram Wren. Visitors remark on how quiet it is.",
    "Pellgate: The first houses were built in 1650. About 3,000 people live in Pellgate.",
]
READER = SentenceReader(PASSAGES)


def test_reader_answers_literally_and_cites():
    assert READER.answer("Who is the mayor of Morton?", PASSAGES) == ("Its mayor is Bram Wren.", 2)
    assert READER.answer("When was Kelby founded?", PASSAGES) == ("It was founded in 1700.", 1)


def test_reader_stays_on_topic_and_admits_ignorance():
    # Pellgate's passage says "first houses were built", not "founded"; Kelby's and Morton's
    # "founded" sentences are about other towns and must not be used.
    assert READER.answer("When was Pellgate founded?", PASSAGES) == ("I don't know.", None)
    # "how" is a question word, not evidence.
    assert READER.answer("How many people live in Morton?", PASSAGES)[0] == "I don't know."


def test_a_passage_must_be_about_the_name_not_just_mention_it():
    passages = ["Kelby: Kelby lies on the Rell River.", "Morton: The mayor of Morton is Bram Wren. The nearest town is Kelby."]
    reader = SentenceReader(passages)
    question = "Who is the mayor of Kelby?"
    assert reader.answer(question, passages)[0] == "The mayor of Morton is Bram Wren."  # the trap, without titles
    assert reader.answer(question, passages, titles=["Kelby", "Morton"]) == ("I don't know.", None)


def test_prompt_numbers_passages_and_sets_the_rules():
    messages = build_prompt("Who?", ["first", "second"])
    assert "[1] first" in messages[1]["content"] and "[2] second" in messages[1]["content"]
    assert "I don't know" in messages[0]["content"] and "Question: Who?" in messages[1]["content"]


def test_rag_with_a_model_parses_the_citation(kb):
    docs, questions = kb
    chunks = chunk_documents(docs, max_words=40)
    llm = ScriptedLLM(["Cleo Ashby is the mayor [2]."])
    rag = RAG(chunks, BM25([c.text for c in chunks]), k=4, llm=llm)
    out = rag.ask(questions[0]["question"])
    hits = rag.retriever.search(questions[0]["question"], 4)
    assert out["answer"].startswith("Cleo Ashby") and out["source"] == hits[1][0]
    sent = llm.requests[0]["messages"][1]["content"]
    assert all(p in sent for p in out["passages"])


@pytest.mark.parametrize("case", [str, str.lower])
def test_rag_without_a_model_answers_most_literal_questions(kb, case):
    # Lowercase questions too: capitalization must not be what keeps the reader on topic.
    docs, questions = kb
    chunks = chunk_documents(docs, max_words=40)
    rag = RAG(chunks, BM25([c.text for c in chunks]))
    answers = [rag.ask(case(q["question"]))["answer"] for q in questions]
    correct = sum(q["answer"] in a for q, a in zip(questions, answers)) / len(questions)
    wrong = sum(q["answer"] not in a and a != "I don't know." for q, a in zip(questions, answers)) / len(questions)
    assert correct > 0.6 and wrong < 0.1, (correct, wrong)
