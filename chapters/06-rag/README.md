# Chapter 6 — RAG: look it up, then answer

> **Code:** [`byoai/rag.py`](../../byoai/rag.py) (about 280 lines), [`byoai/llm.py`](../../byoai/llm.py) (an optional real model) · **Tests:** [`tests/test_rag.py`](../../tests/test_rag.py), [`tests/test_llm.py`](../../tests/test_llm.py) · **Run:** `uv run python chapters/06-rag/run.py` (two seconds)

A language model knows what was in its training data, as of when it was trained, approximately.
Retrieval-augmented generation (RAG) stops asking it to remember: find the passages that answer the
question, put them in the prompt, and have the model answer *from them*, citing which one it used.

Every RAG system is the same four steps — chunk, retrieve, build a prompt, answer — and every step can
quietly lose the answer. This chapter builds each one and measures what it costs, on questions whose
answers are known.

## Data nobody could have memorized

`knowledge_base()` in [`corpus.py`](../../byoai/corpus.py) writes 60 encyclopedia articles about towns
that do not exist — Nustorford, Senzimgate, … — each with a mayor, a river, a famous product, a
founding year, a population and a festival month, and 360 questions about them. Because the towns are
invented, no model can answer from memory; the only way to get it right is to retrieve.

Two details make it realistic, and both turn out to matter:

* **Each fact is written in one of three ways.** "The mayor of Nustorford is Cleo Ashby", "Its mayor
  is Cleo Ashby", or "Nustorford is governed by Cleo Ashby". The questions always use one wording.
* **Most sentences do not repeat the name.** "It was founded in 1690." — as in real articles.

## Chunking

Real documents are long and prompts are not, so documents are split into chunks; the retriever ranks
chunks, not documents. `chunk_documents` cuts at sentence boundaries (never mid-sentence), keeps chunks
under `max_words`, and repeats one sentence between neighbouring chunks so a fact that spans two
sentences is not always cut apart.

Then the problem: the chunk "Its mayor is Cleo Ashby. Every September the town holds a harvest
festival." no longer says which town. A search for "mayor of Nustorford" cannot find it — the one word
that identified it stayed in another chunk. The fix is one line, putting the document's title in front
of every chunk:

```text
                          hit@1  hit@5    MRR
≤40 words, title on each   0.73   0.93   0.81
≤40 words, no title        0.62   0.76   0.68
≤10 words, title on each   0.66   0.74   0.70
≤10 words, no title        0.37   0.38   0.37
```

(A *hit* is a chunk from the right article that contains the answer; MRR is the average of 1/rank of
the first hit.) Without titles, small chunks lose almost two thirds of the answers. Production systems
go further — "contextual retrieval" has a model write a sentence of context for every chunk — but the
title alone recovers most of it here.

Smaller chunks are worse on every row of this data because the articles are tiny (76 words); whole
articles score 0.99. With real documents of thousands of words the trade-off turns around: a whole
document does not fit the prompt, and a long chunk buries the one sentence that matters.

## BM25: keywords, weighted

```text
score(chunk) = Σ over query words w:  idf(w) · tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))
```

Three ideas in one line. **idf**: a word in few chunks (`nustorford`) counts for much more than a
word in all (`the`). **Saturation**: the second mention of a word adds less than the first and the
tenth almost nothing (`k1 = 1.5`). **Length**: a long chunk matches more words just by being long, so it
is marked down a little (`b = 0.75`). The index is inverted — word → the chunks containing it — so a
query only touches the lists of its own words. BM25 is from the 1990s and still the baseline every
retriever has to beat; for names, numbers and exact terms it is very hard to beat.

## LSA: dense vectors without a neural network

Chapter 5 found that the chapter-3 model's hidden states encode sentence *shape*, not content — useless
for retrieval. So the dense retriever here is latent semantic analysis: build the chunk × word TF-IDF
matrix, factorize it with an SVD, keep the strongest 64 directions. Words that occur in the same
chunks end up pointing the same way, so a query can match a chunk with none of its words (the test
`test_lsa_matches_without_shared_words` finds "feline whiskers" for "cat"). The vectors go into chapter
5's `FlatIndex`.

```text
                                    hit@1  hit@5    MRR
BM25                                 0.73   0.93   0.81
LSA, 64 dimensions                   0.73   0.98   0.84
hybrid (reciprocal rank fusion)      0.74   0.95   0.83
LSA, 16 dimensions                   0.19   0.44   0.27
```

LSA's better hit@5 is not understanding. Checking what it returns: its top results are *all* chunks of
the right article 72% of the time, against 56% for BM25 — it latches on to the town (every chunk of an
article shares the name and its vocabulary), and an article is only three chunks long. Its hit@1, where
it would have to pick the right chunk *within* the article, is the same as BM25's. And with too few
dimensions it cannot even keep 60 town names apart (0.19 at 16).

## Hybrid: merge the rankings, not the scores

BM25 scores here run from 0 to about 13 (and that depends on the corpus); cosines from −1 to 1. Adding them needs a weight that is wrong for
the next query. Reciprocal rank fusion ignores scores: each retriever contributes 1/(60 + rank) for
every chunk it returns. Second place in both lists beats first place in one. On this data it lands
between its parts, as it usually does when one retriever is not finding things the other misses —
hybrid pays off when the two fail on *different* questions, which is why measuring per retriever comes
first.

## Answering: the stand-in reader, and its two rules

The answer is written by a model reading the prompt (`build_prompt`: numbered passages, the question,
"cite the passage you used", "if the passages do not contain the answer, reply exactly: I don't
know."). So that everything runs offline, the default is a stand-in: `SentenceReader` answers with the
best-matching sentence from the passages — or "I don't know." Two rules make it trustworthy:

1. **Stay on topic.** A passage counts only if it is *about* the names in the question. Without this
   rule, "How many people live in Nustorford?" was answered from the article on Tamgalwick, which
   mentions Nustorford as its neighbour — and whose "About 19,300 people live in Tamgalwick." matches
   the question's words beautifully. Which words are names is decided against the known article
   titles, in any case: a first version used capital letters, and a lowercase question ("who is the
   mayor of nustorford?") silently switched the rule off — 61% wrong answers. A test now asks every
   question in lowercase too.
2. **Match on the rest.** Score sentences by the question's *other* words, weighted by rarity among
   sentences, ignoring question words — otherwise "How many…" matched "Visitors remark on *how* quiet
   the evenings are."

```text
                                          correct  I dont know  wrong
top 1 chunk                                   61%          39%     0%
top 4 chunks                                  69%          31%     0%
top 4, without the on-topic rule              66%          27%     6%
the right article, handed over (ceiling)      69%          31%     0%
```

Read the table from the bottom. Handed the right article, the reader gets 69%: it is literal, and
"What is Nustorford famous for?" against "It is best known for honey cakes." shares no word it can use,
so it says it does not know. (About two thirds of the fact phrasings share a word with the question;
69% is that number.) With four retrieved chunks it reaches the same 69% — **retrieval loses nothing at
k = 4; every remaining miss is the reader's**. With one chunk, 8 points are lost to retrieval.

And the most important column is the last one. A wrong answer stated confidently is worse than "I don't
know", and the on-topic rule is what turns 6% wrong into 0%. It even adds 3 points of correct
answers: the right sentence no longer loses to a better-matching one about the neighbour.

## A real model (optional)

The gap from 69% to 100% is exactly what a language model fills: knowing that "best known for" answers
"famous for". [`byoai/llm.py`](../../byoai/llm.py) is a ~120-line client for any OpenAI-compatible
chat endpoint — DeepSeek, OpenAI, a local server — using only the standard library: one POST to
`/chat/completions`, retries on rate limits (429) and server errors but not on bad requests, and tool
calls parsed for chapter 7. Its tests run against a fake server on localhost. To try one:

```bash
export BYOAI_LLM_BASE_URL=https://api.deepseek.com BYOAI_LLM_MODEL=deepseek-chat BYOAI_LLM_API_KEY=...
uv run python chapters/06-rag/run.py      # part 4 now asks it 30 questions
```

Nothing is sent anywhere unless `BYOAI_LLM_BASE_URL` is set — an API key in your environment alone does
not switch it on.

## Exercises

1. Set `overlap=0`. Which question types lose hits, and why those?
2. Remove the title header but prepend the first sentence of the article to each chunk instead. How
   close to the title does that get?
3. Make a question set with paraphrased questions ("Who runs Nustorford?", "What does Nustorford
   make?"). Which retriever degrades least? Which reader?
4. Add a typo ("Nustorferd"). BM25 and LSA both fail; add character trigrams as extra "words" and fuse.
5. With a real model configured, remove the "say I don't know" instruction and ask about a town that
   does not exist in the knowledge base. What does it say?

## Where real systems differ

Embedding models trained for retrieval replace LSA, rerankers (a model that reads query and chunk
together) re-order the top 50, chunks get model-written context, and queries are rewritten before
search. Evaluation grows too — answer faithfulness, citation accuracy — often judged by another model.
But the questions stay the ones measured here: did the answer reach the prompt, and did the model use
it or make something up?

**Next:** [Chapter 7 — Agent](../07-agent/README.md): a model that decides for itself which tool to call — retrieval among
them — in a loop, until the task is done.
