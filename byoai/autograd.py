"""Chapter 2 — autograd: reverse-mode automatic differentiation on numpy arrays.

Training a network means nudging every weight in the direction that lowers the loss — which needs
the derivative of the loss with respect to every weight. Writing those derivatives by hand for a
transformer is hopeless; autograd builds them from the derivatives of a few dozen simple operations.

The idea fits in one sentence: every operation remembers its inputs and how to send a gradient back
to them, and `backward()` walks that graph from the loss to the inputs in reverse order, adding up
the gradients that arrive at each tensor (the chain rule).

    >>> from byoai.autograd import Tensor
    >>> x = Tensor([2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()          # y = x0² + x1²
    >>> y.backward()
    >>> x.grad                     # dy/dx = 2x
    array([4., 6.])

Everything chapter 3's GPT needs is here: matrix products with batch dimensions, broadcasting,
reshaping and transposing, embedding lookup, masking, softmax, cross-entropy, layer norm, GELU —
and Adam to do the nudging.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Iterable, Sequence

import numpy as np

_grad_enabled = True


@contextmanager
def no_grad():
    """Inside this block, operations build no graph (for evaluation and generation: less memory, faster)."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` down to `shape`: the gradient of a broadcast is the sum over the broadcast dimensions.

    If `b` of shape (3,) was added to `a` of shape (4, 3), each element of `b` influenced 4 outputs,
    so its gradient is the sum of those 4.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, n in enumerate(shape):
        if n == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """An array that remembers how it was computed, so gradients can flow back through it."""

    __array_priority__ = 100  # so `ndarray + Tensor` calls Tensor.__radd__

    def __init__(self, data, requires_grad: bool = False, _parents: tuple[Tensor, ...] = (), _op: str = ""):
        arr = np.asarray(data)
        # Floats keep their precision (float32 for speed, float64 for gradient checks); anything else
        # (Python numbers, integers, booleans) becomes float64.
        self.data = arr if arr.dtype.kind == "f" else arr.astype(np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward: Callable[[np.ndarray], None] = lambda g: None
        self._op = _op

    # ---- the graph ---------------------------------------------------------------------------------
    @staticmethod
    def _make(data: np.ndarray, parents: tuple[Tensor, ...], op: str, backward: Callable[[np.ndarray], None]) -> Tensor:
        """A result tensor. It joins the graph only if gradients are on and some input needs one."""
        needs = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs, _parents=parents if needs else (), _op=op)
        if needs:
            out._backward = backward
        return out

    def _accumulate(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return
        g = _unbroadcast(np.asarray(g, dtype=self.data.dtype), self.data.shape)
        self.grad = g.copy() if self.grad is None else self.grad + g

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Gradients of this tensor (a scalar loss, usually) with respect to everything it came from."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() without a gradient needs a scalar; pass grad= for anything else")
            grad = np.ones_like(self.data)
        # Topological order: a tensor's gradient is complete only after every use of it has sent its share.
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:  # iterative depth-first search (a deep network would overflow Python's recursion limit)
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if id(p) not in seen)
        self.grad = grad if self.grad is None else self.grad + grad
        for node in reversed(order):
            if node.grad is not None and node._parents:
                node._backward(node.grad)

    def zero_grad(self) -> None:
        self.grad = None

    # ---- shape and conveniences ----------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        return float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}{', requires_grad=True' if self.requires_grad else ''})"

    def __len__(self) -> int:
        return len(self.data)

    # ---- elementwise arithmetic ----------------------------------------------------------------------
    def __add__(self, other) -> Tensor:
        o = _t(other)
        return Tensor._make(self.data + o.data, (self, o), "+", lambda g: (self._accumulate(g), o._accumulate(g)))

    def __mul__(self, other) -> Tensor:
        o = _t(other)
        return Tensor._make(self.data * o.data, (self, o), "*", lambda g: (self._accumulate(g * o.data), o._accumulate(g * self.data)))

    def __pow__(self, k: float) -> Tensor:
        return Tensor._make(self.data**k, (self,), f"**{k}", lambda g: self._accumulate(g * k * self.data ** (k - 1)))

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other) -> Tensor:
        return self + (-_t(other))

    def __truediv__(self, other) -> Tensor:
        o = _t(other)
        return Tensor._make(self.data / o.data, (self, o), "/", lambda g: (self._accumulate(g / o.data), o._accumulate(-g * self.data / o.data**2)))

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other) -> Tensor:
        return _t(other) - self

    def __rtruediv__(self, other) -> Tensor:
        return _t(other) / self

    # ---- matrix product ------------------------------------------------------------------------------
    def __matmul__(self, other) -> Tensor:
        """`A @ B`, with numpy's batching: (..., n, k) @ (..., k, m) -> (..., n, m)."""
        o = _t(other)
        a, b = self.data, o.data

        def backward(g: np.ndarray) -> None:
            # d(A@B)/dA = g @ Bᵀ, d(A@B)/dB = Aᵀ @ g — on the last two axes; batch axes broadcast.
            if b.ndim == 1:
                self._accumulate(np.multiply.outer(g, b))
                o._accumulate(np.tensordot(a, g, axes=(tuple(range(a.ndim - 1)), tuple(range(g.ndim)))))
                return
            self._accumulate(g @ np.swapaxes(b, -1, -2))
            o._accumulate(np.swapaxes(a, -1, -2) @ g if a.ndim > 1 else np.multiply.outer(a, g))

        return Tensor._make(a @ b, (self, o), "@", backward)

    # ---- reductions ----------------------------------------------------------------------------------
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        def backward(g: np.ndarray) -> None:
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.data.shape))

        return Tensor._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", backward)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        n = self.data.size if axis is None else math.prod(self.data.shape[a] for a in ((axis,) if isinstance(axis, int) else axis))
        return self.sum(axis=axis, keepdims=keepdims) / n

    def max(self, axis: int, keepdims: bool = False) -> Tensor:
        """The maximum along `axis`; the gradient goes to the (first) position that had it."""
        idx = self.data.argmax(axis=axis)
        out = np.take_along_axis(self.data, np.expand_dims(idx, axis), axis)

        def backward(g: np.ndarray) -> None:
            full = np.zeros_like(self.data)
            np.put_along_axis(full, np.expand_dims(idx, axis), g if keepdims else np.expand_dims(g, axis), axis)
            self._accumulate(full)

        return Tensor._make(out if keepdims else out.squeeze(axis), (self,), "max", backward)

    # ---- shape -----------------------------------------------------------------------------------------
    def reshape(self, *shape: int) -> Tensor:
        return Tensor._make(self.data.reshape(*shape), (self,), "reshape", lambda g: self._accumulate(g.reshape(self.data.shape)))

    def transpose(self, *axes: int) -> Tensor:
        """Permute axes (`transpose(0, 2, 1, 3)`); with no arguments, swap the last two."""
        perm = axes or tuple(range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2)
        inverse = np.argsort(perm)
        return Tensor._make(self.data.transpose(perm), (self,), "transpose", lambda g: self._accumulate(g.transpose(inverse)))

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __getitem__(self, index) -> Tensor:
        """Slicing (`x[:, -1]`) and indexing with integer arrays (an embedding lookup: `table[ids]`)."""
        if isinstance(index, Tensor):
            index = index.data.astype(np.int64)

        def backward(g: np.ndarray) -> None:
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)  # add.at: an index used twice gets both gradients
            self._accumulate(full)

        return Tensor._make(self.data[index], (self,), "getitem", backward)

    # ---- elementwise functions -------------------------------------------------------------------------
    def exp(self) -> Tensor:
        out = np.exp(self.data)
        return Tensor._make(out, (self,), "exp", lambda g: self._accumulate(g * out))

    def log(self) -> Tensor:
        return Tensor._make(np.log(self.data), (self,), "log", lambda g: self._accumulate(g / self.data))

    def tanh(self) -> Tensor:
        out = np.tanh(self.data)
        return Tensor._make(out, (self,), "tanh", lambda g: self._accumulate(g * (1 - out**2)))

    def relu(self) -> Tensor:
        return Tensor._make(np.maximum(self.data, 0), (self,), "relu", lambda g: self._accumulate(g * (self.data > 0)))

    def gelu(self) -> Tensor:
        """GELU, the tanh approximation GPT-2 uses: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))."""
        x = self.data
        c = math.sqrt(2 / math.pi)
        inner = c * (x + 0.044715 * x**3)
        th = np.tanh(inner)
        out = 0.5 * x * (1 + th)

        def backward(g: np.ndarray) -> None:
            d_inner = c * (1 + 3 * 0.044715 * x**2)
            self._accumulate(g * (0.5 * (1 + th) + 0.5 * x * (1 - th**2) * d_inner))

        return Tensor._make(out, (self,), "gelu", backward)

    def masked_fill(self, mask: np.ndarray, value: float) -> Tensor:
        """`value` where `mask` is true (no gradient there), the tensor elsewhere — e.g. a causal mask."""
        mask = np.broadcast_to(mask, self.data.shape)
        return Tensor._make(np.where(mask, value, self.data), (self,), "masked_fill", lambda g: self._accumulate(np.where(mask, 0.0, g)))

    # ---- the functions a transformer is made of ------------------------------------------------------
    def softmax(self, axis: int = -1) -> Tensor:
        """exp(x) / Σ exp(x), computed after subtracting the max so large inputs do not overflow."""
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        out = e / e.sum(axis=axis, keepdims=True)

        def backward(g: np.ndarray) -> None:
            # The Jacobian of softmax applied to g, without building the Jacobian: s · (g − Σ g·s).
            self._accumulate(out * (g - (g * out).sum(axis=axis, keepdims=True)))

        return Tensor._make(out, (self,), "softmax", backward)

    def log_softmax(self, axis: int = -1) -> Tensor:
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))

        def backward(g: np.ndarray) -> None:
            self._accumulate(g - np.exp(out) * g.sum(axis=axis, keepdims=True))

        return Tensor._make(out, (self,), "log_softmax", backward)

    def cross_entropy(self, targets) -> Tensor:
        """Mean negative log-likelihood of `targets` (integer class ids) under these logits.

        Fused: softmax and log in one step. The gradient is the famous `softmax(logits) − one_hot`,
        and nothing is computed as a probability that could round to zero first.
        """
        t = np.asarray(targets.data if isinstance(targets, Tensor) else targets, dtype=np.int64)
        logits = self.data.reshape(-1, self.data.shape[-1])
        flat = t.reshape(-1)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        n = flat.shape[0]
        loss = -log_probs[np.arange(n), flat].mean()

        def backward(g: np.ndarray) -> None:
            grad = np.exp(log_probs)
            grad[np.arange(n), flat] -= 1
            self._accumulate((g * grad / n).reshape(self.data.shape))

        return Tensor._make(np.asarray(loss), (self,), "cross_entropy", backward)

    def layer_norm(self, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
        """Normalize the last axis to mean 0 and variance 1, then scale and shift: built from the ops above."""
        mean = self.mean(axis=-1, keepdims=True)
        centered = self - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        return centered / (var + eps) ** 0.5 * weight + bias


def _t(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def concatenate(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """Join tensors along `axis`; each gets its slice of the gradient back."""
    sizes = [t.data.shape[axis] for t in tensors]
    bounds = np.cumsum([0, *sizes])

    def backward(g: np.ndarray) -> None:
        for t, lo, hi in zip(tensors, bounds[:-1], bounds[1:]):
            t._accumulate(np.take(g, np.arange(lo, hi), axis=axis))

    return Tensor._make(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors), "concat", backward)


# ---- training -------------------------------------------------------------------------------------------
class Adam:
    """Adam with decoupled weight decay (AdamW): per-weight step sizes from running averages of the
    gradient (momentum) and of its square (scale)."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1**self.t)  # early steps: the averages start at 0, so correct for it
            v_hat = v / (1 - b2**self.t)
            if self.weight_decay and p.ndim >= 2:  # decay weight matrices, not biases or norms
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def gradcheck(f: Callable[..., Tensor], *inputs: Tensor, eps: float = 1e-6, tol: float = 1e-5) -> float:
    """Compare autograd's gradients with finite differences, (f(x+ε) − f(x−ε)) / 2ε, for every input
    element. Returns the largest relative error; raises if it is above `tol`."""
    for x in inputs:
        x.grad = None
    f(*inputs).backward()
    worst = 0.0
    for x in inputs:
        analytic = x.grad if x.grad is not None else np.zeros_like(x.data)
        numeric = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = x.data[i]
            x.data[i] = old + eps
            up = f(*inputs).item()
            x.data[i] = old - eps
            down = f(*inputs).item()
            x.data[i] = old
            numeric[i] = (up - down) / (2 * eps)
        err = np.abs(analytic - numeric) / np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
        worst = max(worst, float(err.max()) if err.size else 0.0)
    if worst > tol:
        raise AssertionError(f"gradient check failed: relative error {worst:.2e} > {tol:.0e}")
    return worst
