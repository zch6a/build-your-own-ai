"""Chapter 2 demo: gradients by hand vs. by autograd, then a network that learns spirals.

    uv run python chapters/02-autograd/run.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from byoai.autograd import Adam, Tensor, gradcheck

# ---- 1. One neuron: compare with the derivative worked out by hand ---------------------------------------
w, x, b = Tensor(0.5, requires_grad=True), Tensor(2.0, requires_grad=True), Tensor(-0.3, requires_grad=True)
y = (w * x + b).tanh()
y.backward()
z = 0.5 * 2.0 - 0.3
print("y = tanh(w·x + b) at w=0.5, x=2, b=-0.3")
print(f"  autograd:  dy/dw = {w.grad:.6f}   dy/dx = {x.grad:.6f}   dy/db = {b.grad:.6f}")
d = 1 - math.tanh(z) ** 2  # d tanh(z)/dz
print(f"  by hand:   dy/dw = {d * 2.0:.6f}   dy/dx = {d * 0.5:.6f}   dy/db = {d:.6f}   (chain rule: tanh'(z) times dz/d·)")

# ---- 2. The check every new operation gets: finite differences ---------------------------------------------
a = Tensor(np.random.default_rng(0).standard_normal((3, 4)), requires_grad=True)
err = gradcheck(lambda t: (t.softmax() * Tensor(np.arange(4.0))).sum(), a)
print(f"\nsoftmax gradient vs. finite differences: largest relative error {err:.1e}")

# ---- 3. Three interleaved spirals: no straight line separates them ----------------------------------------
rng = np.random.default_rng(42)
n_per, classes = 150, 3
xs, ys = [], []
for c in range(classes):
    r = np.linspace(0.05, 1, n_per)
    theta = np.linspace(c * 2 * np.pi / classes, c * 2 * np.pi / classes + 3.2, n_per) + rng.normal(0, 0.12, n_per)
    xs.append(np.c_[r * np.sin(theta), r * np.cos(theta)])
    ys.append(np.full(n_per, c))
X, Y = np.concatenate(xs), np.concatenate(ys)

hidden = 64
W1 = Tensor(rng.standard_normal((2, hidden)) * 0.8, requires_grad=True)
b1 = Tensor(np.zeros(hidden), requires_grad=True)
W2 = Tensor(rng.standard_normal((hidden, hidden)) / np.sqrt(hidden), requires_grad=True)
b2 = Tensor(np.zeros(hidden), requires_grad=True)
W3 = Tensor(rng.standard_normal((hidden, classes)) / np.sqrt(hidden), requires_grad=True)
b3 = Tensor(np.zeros(classes), requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]


def model(inp: np.ndarray) -> Tensor:
    h = (Tensor(inp) @ W1 + b1).tanh()
    h = (h @ W2 + b2).tanh()
    return h @ W3 + b3


opt = Adam(params, lr=0.01)
print(f"\ntraining a 2-{hidden}-{hidden}-3 network on {len(X)} points of three spirals:")
t0 = time.perf_counter()
for step in range(601):
    opt.zero_grad()
    logits = model(X)
    loss = logits.cross_entropy(Y)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        acc = (logits.data.argmax(axis=1) == Y).mean()
        print(f"  step {step:>4}: loss {loss.item():.3f}, accuracy {acc:.0%}")
print(f"  ({time.perf_counter() - t0:.1f} s)")

# What it learned: the class it predicts at each point of the plane, and the training points on top.
print("\nthe plane, as the network divides it — its regions · - ° for classes A B C, with training points on top:")
W, H = 64, 28
gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, W), np.linspace(1.1, -1.1, H))
regions = model(np.c_[gx.ravel(), gy.ravel()]).data.argmax(axis=1).reshape(H, W)
grid = [["·-°"[c] for c in row] for row in regions]
for (px, py), c in zip(X[::3], Y[::3]):
    col, row = int((px + 1.1) / 2.2 * (W - 1)), int((1.1 - py) / 2.2 * (H - 1))
    grid[row][col] = "ABC"[c]
print("\n".join("  " + "".join(r) for r in grid))
