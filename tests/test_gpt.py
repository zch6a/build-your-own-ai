import math

import numpy as np

from byoai.autograd import gradcheck
from byoai.gpt import GPT, GPTConfig, evaluate, train

SMALL = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=16)


def test_shapes_and_loss_at_initialization():
    model = GPT(SMALL)
    r = np.random.default_rng(0)
    idx, targets = r.integers(0, 50, size=(3, 16)), r.integers(0, 50, size=(3, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 16, 50)
    # Untrained, every token is about equally likely: the loss is about ln(vocab_size). (Not with
    # targets = inputs: with tied weights an untrained model leans toward the token it is reading.)
    assert abs(loss.item() - math.log(50)) < 0.1


def test_the_future_does_not_change_the_past():
    model = GPT(SMALL, seed=1)
    a = np.random.default_rng(2).integers(0, 50, size=16)
    b = a.copy()
    b[10] = (b[10] + 1) % 50  # change one token
    la, lb = model(a)[0].data[0], model(b)[0].data[0]
    assert np.allclose(la[:10], lb[:10], atol=1e-6), "positions before 10 must not see position 10"
    assert not np.allclose(la[10:], lb[10:]), "positions from 10 on do see it"


def test_gradients_of_every_parameter_are_right():
    # A tiny model in float64, so finite differences are accurate.
    model = GPT(GPTConfig(vocab_size=7, block_size=4, n_layer=1, n_head=2, n_embd=4, dtype="float64"), seed=3)
    idx = np.array([[1, 2, 3, 4], [6, 5, 0, 1]])
    targets = np.array([[2, 3, 4, 5], [5, 0, 1, 2]])
    names = list(model.params)
    tensors = [model.params[n] for n in names]

    def loss(*_):
        return model(idx, targets)[1]

    gradcheck(loss, *tensors, tol=1e-4)


def test_weight_tying_trains_the_embedding_from_both_ends():
    model = GPT(SMALL)
    idx = np.array([[1, 2, 3]])
    model(idx, np.array([[2, 3, 4]]))[1].backward()
    g = model.params["wte"].grad
    assert np.abs(g[4]).sum() > 0, "token 4 is never an input, but it is a target: the output layer reaches it"


def test_it_memorizes_a_short_sequence():
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=2, n_head=2, n_embd=32))
    data = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3] * 8)
    losses = train(model, data, steps=300, batch_size=16, lr=1e-2, warmup=20)
    assert losses[-1] < 0.25, losses[-1]
    # Given the start, it continues the sequence it learned.
    assert model.generate([3, 1, 4, 1, 5], 5, temperature=0)[5:] == [9, 2, 6, 5, 3]


def test_save_and_load_give_the_same_model(tmp_path):
    model = GPT(SMALL, seed=5)
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = GPT.load(path)
    assert loaded.config == model.config
    idx = np.array([[1, 2, 3, 4]])
    assert np.array_equal(loaded(idx)[0].data, model(idx)[0].data)


def test_too_long_a_sequence_is_an_error():
    model = GPT(SMALL)
    try:
        model(np.zeros((1, 17), dtype=int))
    except ValueError as e:
        assert "block_size" in str(e)
    else:
        raise AssertionError("expected an error")


def test_evaluate_is_lower_after_training():
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, n_embd=16))
    data = np.tile(np.arange(20), 30)
    before = evaluate(model, data, batches_n=5)
    train(model, data, steps=100, batch_size=16, lr=1e-2, warmup=10)
    assert evaluate(model, data, batches_n=5) < before / 3
