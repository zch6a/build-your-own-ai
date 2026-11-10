# Chapter 2 — Autograd: gradients for free

> **Code:** [`byoai/autograd.py`](../../byoai/autograd.py) (about 350 lines) · **Tests:** [`tests/test_autograd.py`](../../tests/test_autograd.py) · **Run:** `uv run python chapters/02-autograd/run.py`

Training a network means repeating one step: compute the loss, find out how the loss changes when
each weight changes a little — the gradient — and move every weight a little in the direction that
lowers it. A GPT has millions of weights. Working out their derivatives by hand is hopeless; computing
them one at a time by nudging each weight would take millions of forward passes.

Reverse-mode automatic differentiation gets *all* of them for roughly the cost of one more forward
pass. PyTorch's `loss.backward()` is this idea, and so is this chapter.

## The chain rule, as a graph

Take one neuron, `y = tanh(w·x + b)`. Name the intermediate step: `z = w·x + b`, `y = tanh(z)`. The
chain rule says

```text
dy/dw = dy/dz · dz/dw = (1 − tanh²(z)) · x
```

Autograd does exactly this, mechanically. Every operation records its inputs and knows its own local
derivative. `tanh` knows that its gradient is `1 − tanh²`; `*` knows that the gradient with respect
to `w` is `x`. Calling `backward()` on `y` starts with `dy/dy = 1` and hands each operation the
gradient that arrived at its output; the operation multiplies by its local derivative and passes the
result on to its inputs.

```python
def tanh(self):
    out = np.tanh(self.data)
    return Tensor._make(out, (self,), "tanh",
                        lambda g: self._accumulate(g * (1 - out**2)))
```

That lambda is the whole contribution of `tanh` to backpropagation. The demo checks the result
against the formula worked out by hand.

## Why "reverse"

You could also push derivatives *forward*, input to output: that tells you how every output changes
with *one* input. Training has the opposite shape — one output (the loss), millions of inputs (the
weights) — so going backwards from the loss gets all the gradients in one pass.

## The three subtle parts

**Gradients add up.** If a tensor is used twice — `a * a.tanh() + a` — each use sends back its own
gradient and the true gradient is their sum. That is why tensors `_accumulate` instead of assigning,
and why an optimizer must zero gradients before the next step.

**Order matters.** A tensor may only pass its gradient on once *all* its uses have reported in.
`backward()` therefore sorts the graph topologically first — iteratively, because a deep network
would overflow Python's recursion limit (there is a test with a chain of 5,000 operations).

**Broadcasting runs backwards as a sum.** Adding a bias of shape `(3,)` to activations of shape
`(4, 3)` uses each bias element 4 times, so its gradient is the sum of 4 gradients.
`_unbroadcast` does that for every operation, which is why `b + W @ x` just works.

## What a transformer needs

Matrix products with batch dimensions (attention multiplies `(batch, heads, time, dim)` arrays),
reshape and transpose (to split into heads), indexing with integer arrays (the embedding table is
looked up with token ids — and a token that appears twice must get both gradients, hence
`np.add.at`), a mask (a token may not attend to the future), softmax, layer norm, GELU, and the loss:

```python
def cross_entropy(self, targets):
    ...  # log-softmax and the negative log-likelihood in one step
    grad = softmax(logits) - one_hot(targets)
```

Fusing them matters: computing `softmax` first and then `log` rounds tiny probabilities to zero and
produces `-inf`. The test feeds it logits of ±1000.

`layer_norm` needs no gradient of its own: it is written with `mean`, `-`, `*`, `**` and `/`, and
autograd differentiates the composition. Most new operations could be built that way; the ones with
their own backward (`softmax`, `cross_entropy`) are there for numerical stability and speed.

## Trust, but check: finite differences

A wrong backward function does not crash — it silently trains worse. So every operation is checked
against the definition of a derivative:

```text
df/dx ≈ (f(x + ε) − f(x − ε)) / 2ε
```

`gradcheck` does this for every element of every input and compares. It is slow (two forward passes
per number) but it is how you know. Every operation in the module has a test that runs it; the demo
shows a relative error around 10⁻⁸.

## Adam

Plain gradient descent moves each weight by `lr · gradient`. Adam keeps two running averages per
weight — of the gradient (momentum: keep going in a consistent direction) and of its square (scale:
take smaller steps where the gradient is large and noisy) — and divides one by the square root of the
other. The bias correction `/ (1 − β^t)` matters for the first steps, when the averages start at zero.
Weight decay is applied directly to the weights ("AdamW") and only to matrices, not biases.

## What the demo shows

1. The gradients of one neuron, by autograd and by hand — identical.
2. A finite-difference check of softmax.
3. A two-hidden-layer network learning three interleaved spirals, which no straight line can
   separate, in about a second — and a map of how it divides the plane.

## Exercises

1. Add `sigmoid` with its own backward, and a `gradcheck` test for it.
2. Remove the topological sort and walk the graph depth-first instead. Which test fails, and why?
3. Replace `np.add.at` in `__getitem__` with `full[index] += g`. Find the input that makes the
   gradient wrong.
4. Train the spiral network with plain gradient descent (`p.data -= lr * p.grad`). How much more
   tuning does it take to reach 100%?

## Where real frameworks differ

PyTorch records the same kind of graph, but its operations run in C++/CUDA, it frees intermediate
results as soon as the backward pass no longer needs them, and it can recompute instead of storing
(gradient checkpointing). JAX takes the other road: it transforms the Python function itself into its
derivative. The mathematics is the same.

**Next:** [Chapter 3 — GPT](../03-gpt/README.md): a transformer built from these operations, trained on the stories from
chapter 1.
