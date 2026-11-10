# Chapter 3 — GPT: a transformer you can train on a laptop

> **Code:** [`byoai/gpt.py`](../../byoai/gpt.py) (about 250 lines) · **Tests:** [`tests/test_gpt.py`](../../tests/test_gpt.py) · **Run:** `uv run python chapters/03-gpt/run.py` (about two minutes)

A language model answers one question again and again: *given these tokens, what comes next?* This
chapter builds the architecture behind GPT-2, GPT-3 and most models since, from the operations of
chapter 2, and trains it on the stories of chapter 1 — 392,000 parameters, two minutes on a CPU.

## The shape of the computation

```text
token ids (batch, time)
  → look up a vector per token, add a vector per position      (batch, time, C)
  → N × block:
        x = x + attention(layer_norm(x))     # look at earlier positions
        x = x + mlp(layer_norm(x))           # think about what was seen
  → layer_norm
  → x @ token_embedding.T                    (batch, time, vocab)   a score for every token
```

Every position gets a prediction, so one forward pass over 64 tokens is 64 training examples: "after
`One`, predict `day`", "after `One day`, predict `,`", and so on.

## Embeddings: "which token" + "where"

The model starts by looking up a learned vector for each token (`wte`) and a learned vector for each
position (`wpe`), and adds them. Attention itself has no notion of order — it is a weighted sum over a
set — so without the position vector, "Mia gave Leo" and "Leo gave Mia" would look the same.

## Attention: every position asks a question

Each position produces three vectors from its own: a **query** (what am I looking for?), a **key**
(what do I contain?) and a **value** (what will I pass on?). The score between position *i* and *j*
is the dot product of *i*'s query and *j*'s key; a softmax turns a row of scores into weights; the
output for *i* is the weighted mix of the values.

```python
scores = (q @ k.transpose(0, 1, 3, 2)) / sqrt(D)          # (B, H, T, T)
scores = scores.masked_fill(future, -1e9)                 # no peeking ahead
out = scores.softmax(axis=-1) @ v
```

Three details carry most of the weight:

* **The causal mask.** A position may only attend to itself and earlier positions; otherwise training
  would be trivial (just look at the answer). The test `test_the_future_does_not_change_the_past`
  changes token 10 and checks that predictions 0–9 do not move.
* **Scaling by √D.** Dot products of D-dimensional vectors grow like √D; without the division the
  softmax saturates and its gradient vanishes.
* **Multiple heads.** The vector is split into `n_head` slices that attend independently — one head
  can track the animal while another tracks punctuation — and are joined again.

## The MLP, residuals and layer norm

After attention gathers information, a two-layer MLP (width 4C, GELU) processes each position on its
own. Both sublayers *add* their output to the running vector instead of replacing it: the residual
stream. Gradients flow straight down that stream, which is what lets transformers be deep. Layer norm
before each sublayer ("pre-norm", as in GPT-2) keeps the scale of what each sublayer sees stable.

Initialization follows GPT-2: weights drawn with standard deviation 0.02, and the two projections that
write into the residual stream scaled down by √(2·layers), so the stream does not grow with depth.

## Weight tying

The final layer turns a vector into a score per token: a `(C, vocab)` matrix. GPT-2 uses the
transpose of the embedding table for it. The vector that represents a token on the way in is the same
one that predicts it on the way out — fewer parameters, better training, and a token that is only ever
a *target* still trains its embedding (there is a test for exactly that).

## Training

`train()` takes random 64-token windows of the token stream; the target is the window shifted by one.
AdamW with β₂ = 0.95 and weight decay 0.1, a learning rate that warms up for 50 steps and then follows
a cosine down to a tenth of its peak, and the gradient's overall norm clipped to 1 so that one odd
batch cannot throw the weights far.

The loss starts at ln 520 ≈ 6.25 — the model guesses uniformly over the vocabulary — and falls below
1 within about 100 steps.

## Making it fast enough

The first version took 290 ms per step for the smallest model; two profiles found why:

* **A float64 hiding in float32.** A constant like `1 / sqrt(D)`, wrapped in a `Tensor`, became a
  0-dimensional *float64 array* — and NumPy promotes float32 to float64 for that (a bare Python number
  it would not). Most of the forward and backward pass silently ran in double precision. Constants now
  take the dtype of the tensor they meet.
* **Batched products that did not need to be.** `(batch, time, C) @ (C, 3C)` is one big matrix product
  in disguise; reshaping to 2-D makes it a single BLAS call, and the weight's gradient comes out summed
  over the batch instead of as a `(batch, C, 3C)` array summed afterwards.

Together with smaller things (no float powers in GELU, no gradient copies), the step went from 1.47 s
to 0.32 s for the largest configuration tried.

## What the demo shows

Untrained, the model writes byte soup. After 500 steps it writes grammatical stories that are *new* —
the demo checks every sentence against the training text — on held-out loss 0.62.

What it does not do yet is stay consistent: the dog of the first sentence becomes a bird in the third.
That is measurable. Over 60 generated stories, the second sentence names the animal of the first in
22% of them after 500 steps and 9% after 1,000 — a random pick among the 10 animals would give 10%.
The model has learned *where* an animal goes, not *which* one.

Keeping it would mean attending back to the first mention and copying it — a two-step circuit (one
head finds "the token after the earlier `with a`", another copies it) that small transformers are known
to form suddenly, after a plateau, sometimes much later in training. The plateau is visible here: the
loss flattens at about 0.61 from step 150 on. What copying would be worth is computable: the animal
appears about three times per ~50-token story, so knowing it saves about 3 × ln 10 / 50 ≈ 0.14 nats per
token — the loss would drop to about 0.47. That is a clear target for the exercises below.

## Exercises

1. Remove the position embedding. Train again: what happens to the loss, and to the stories?
2. Set `n_head=1` with the same width. Does it learn as fast?
3. Remove the √D scaling and train with `n_embd=256`. Watch the loss.
4. Untie the output layer (give it its own `(C, vocab)` matrix). How many parameters does that add,
   and does the held-out loss change?
5. Make the model keep its animal. Train much longer, with a larger batch, or with more heads, and watch
   for the loss to drop below the plateau of ~0.61 towards ~0.47 — then measure the animal again.

## Where real models differ

Scale, mostly: GPT-2 small is 124 million parameters with 12 layers, 12 heads and width 768; models
today are thousands of times larger and train on trillions of tokens on thousands of GPUs. Newer
architectures replace learned position vectors with rotary embeddings (RoPE), layer norm with RMSNorm,
GELU with SwiGLU, and share keys and values across heads (grouped-query attention) — refinements of
the same block.

**Next:** Chapter 4 — Inference: generating text one token at a time is slow done naively; a KV cache
makes it fast, and the sampling strategy decides what the model says.
