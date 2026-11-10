import numpy as np
import pytest

from byoai.autograd import Adam, Tensor, concatenate, gradcheck, no_grad

rng = np.random.default_rng(0)


def T(*shape: int, scale: float = 1.0, positive: bool = False) -> Tensor:
    x = rng.standard_normal(shape) * scale
    return Tensor(np.abs(x) + 0.5 if positive else x, requires_grad=True)


# ---- every operation against finite differences --------------------------------------------------------
@pytest.mark.parametrize(
    "name, f, shapes",
    [
        ("add, broadcast", lambda a, b: (a + b).sum(), [(4, 3), (3,)]),
        ("sub, broadcast the other way", lambda a, b: ((a - b) ** 2).sum(), [(1, 3), (4, 1)]),
        ("mul", lambda a, b: (a * b).sum(), [(2, 3), (2, 3)]),
        ("div", lambda a, b: (a / b).sum(), [(2, 3), "positive (2, 3)"]),
        ("pow", lambda a: (a**3).sum(), [(5,)]),
        ("matmul 2-D", lambda a, b: (a @ b).sum(), [(3, 4), (4, 2)]),
        ("matmul batched", lambda a, b: ((a @ b) ** 2).sum(), [(2, 3, 4), (4, 5)]),
        ("matmul batched both", lambda a, b: ((a @ b) ** 2).sum(), [(2, 2, 3, 4), (2, 2, 4, 3)]),
        ("sum axis", lambda a: (a.sum(axis=1) ** 2).sum(), [(3, 4)]),
        ("sum keepdims", lambda a: (a.sum(axis=0, keepdims=True) * a).sum(), [(3, 4)]),
        ("mean", lambda a: (a.mean(axis=-1) ** 2).sum(), [(2, 5)]),
        ("max", lambda a: (a.max(axis=1) ** 2).sum(), [(3, 6)]),
        ("reshape", lambda a: (a.reshape(2, 6) ** 2 * Tensor(np.arange(12.0).reshape(2, 6))).sum(), [(3, 4)]),
        ("transpose", lambda a: (a.transpose(1, 0, 2) * Tensor(np.arange(24.0).reshape(3, 2, 4))).sum(), [(2, 3, 4)]),
        ("slice", lambda a: (a[:, 1:3] ** 2).sum(), [(4, 5)]),
        ("exp", lambda a: a.exp().sum(), [(3, 3)]),
        ("log", lambda a: a.log().sum(), ["positive (3, 3)"]),
        ("tanh", lambda a: a.tanh().sum(), [(3, 3)]),
        ("gelu", lambda a: (a.gelu() ** 2).sum(), [(4, 4)]),
        ("softmax", lambda a: (a.softmax() * Tensor(np.arange(5.0))).sum(), [(3, 5)]),
        ("log_softmax", lambda a: (a.log_softmax() * Tensor(np.arange(5.0))).sum(), [(3, 5)]),
        ("layer_norm", lambda a, w, b: (a.layer_norm(w, b) * Tensor(np.arange(6.0))).sum(), [(4, 6), (6,), (6,)]),
        ("masked_fill", lambda a: (a.masked_fill(np.triu(np.ones((4, 4), bool), 1), -1e9).softmax() * Tensor(np.arange(4.0))).sum(), [(4, 4)]),
        ("concatenate", lambda a, b: (concatenate([a, b], axis=1) * Tensor(np.arange(15.0).reshape(3, 5))).sum(), [(3, 2), (3, 3)]),
        ("a tensor used twice", lambda a: (a * a.tanh() + a).sum(), [(3,)]),
    ],
)
def test_gradients_match_finite_differences(name, f, shapes):
    inputs = [T(*eval(s.split(" ", 1)[1]), positive=True) if isinstance(s, str) else T(*s) for s in shapes]
    gradcheck(f, *inputs)


def test_relu_gradient():
    # Away from 0 (the kink), where finite differences are honest.
    x = Tensor(np.array([-2.0, -0.5, 0.5, 2.0]), requires_grad=True)
    gradcheck(lambda a: (a.relu() ** 2).sum(), x)


def test_cross_entropy_matches_its_definition():
    logits = T(6, 5)
    targets = np.array([0, 4, 2, 2, 1, 3])
    gradcheck(lambda x: x.cross_entropy(targets), logits)
    manual = -logits.log_softmax().data[np.arange(6), targets].mean()
    assert np.isclose(logits.cross_entropy(targets).item(), manual)
    # 3-D logits (batch, time, vocab), as a language model has them.
    gradcheck(lambda x: x.cross_entropy(np.array([[1, 2, 0], [3, 3, 4]])), T(2, 3, 5))


def test_cross_entropy_is_stable_for_large_logits():
    logits = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    loss = logits.cross_entropy([0])
    loss.backward()
    assert np.isfinite(loss.item()) and loss.item() < 1e-6
    assert np.all(np.isfinite(logits.grad))


def test_embedding_lookup_adds_gradients_of_repeated_ids():
    table = T(5, 3)
    ids = np.array([[1, 1, 4], [0, 1, 2]])
    gradcheck(lambda t: (t[ids] ** 2).sum(), table)
    table.grad = None
    table[ids].sum().backward()
    assert np.allclose(table.grad[1], 3.0), "id 1 was used three times"
    assert np.allclose(table.grad[3], 0.0), "id 3 was not used"


def test_backward_needs_a_scalar_and_accumulates():
    x = T(3)
    with pytest.raises(ValueError):
        (x * 2).backward()
    (x * 2).sum().backward()
    (x * 3).sum().backward()
    assert np.allclose(x.grad, 5.0), "gradients add up until zeroed"


def test_no_grad_builds_no_graph():
    x = T(3)
    with no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad and y._parents == ()
    z = (x * 2).sum()
    assert z.requires_grad


def test_deep_graphs_do_not_hit_the_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.backward()
    assert np.isclose(x.grad, 1.0001**5000)


def test_adam_finds_a_minimum():
    w = Tensor(np.array([5.0, -3.0]), requires_grad=True)
    opt = Adam([w], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        loss = ((w - Tensor(np.array([1.0, 2.0]))) ** 2).sum()
        loss.backward()
        opt.step()
    assert np.allclose(w.data, [1.0, 2.0], atol=1e-3)


def test_a_two_layer_network_learns_xor():
    r = np.random.default_rng(1)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    w1, b1 = Tensor(r.standard_normal((2, 8)), requires_grad=True), Tensor(np.zeros(8), requires_grad=True)
    w2, b2 = Tensor(r.standard_normal((8, 2)), requires_grad=True), Tensor(np.zeros(2), requires_grad=True)
    opt = Adam([w1, b1, w2, b2], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        logits = (Tensor(x) @ w1 + b1).tanh() @ w2 + b2
        loss = logits.cross_entropy(y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert list(logits.data.argmax(axis=1)) == [0, 1, 1, 0]


def test_float32_stays_float32():
    x = Tensor(np.ones((2, 3), dtype=np.float32), requires_grad=True)
    y = (x @ Tensor(np.ones((3, 4), dtype=np.float32))).gelu().sum()
    y.backward()
    assert x.data.dtype == np.float32 and x.grad.dtype == np.float32
