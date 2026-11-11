import time

import numpy as np
import pytest

from byoai.autograd import no_grad
from byoai.generate import Cache, Engine, sample
from byoai.gpt import GPT, GPTConfig, batches

CFG = GPTConfig(vocab_size=60, block_size=32, n_layer=2, n_head=4, n_embd=32)


def full_logits(model: GPT, ids: list[int]) -> np.ndarray:
    with no_grad():
        return model(np.array(ids))[0].data[0]


def test_engine_matches_the_training_model():
    model = GPT(CFG, seed=1)
    ids = list(np.random.default_rng(0).integers(0, 60, size=20))
    assert np.allclose(Engine(model).forward(ids, Cache()), full_logits(model, ids), atol=1e-5)


def test_the_cache_gives_exactly_what_a_full_pass_gives():
    model = GPT(CFG, seed=2)
    engine = Engine(model)
    ids = list(np.random.default_rng(1).integers(0, 60, size=24))
    cache = Cache()
    first = engine.forward(ids[:10], cache)  # prefill
    assert np.allclose(first, full_logits(model, ids[:10]), atol=1e-5)
    for t in range(10, 24):  # then one token at a time
        step = engine.forward([ids[t]], cache)[-1]
        assert np.allclose(step, full_logits(model, ids[: t + 1])[-1], atol=1e-5), f"position {t}"
    assert cache.length == 24


def test_greedy_generation_matches_chapter_3():
    model = GPT(CFG, seed=3)
    prompt = [5, 6, 7]
    assert Engine(model).generate(prompt, 20, temperature=0) == model.generate(prompt, 20, temperature=0)


def test_generation_continues_past_the_context_window():
    model = GPT(CFG, seed=4)
    out = Engine(model).generate([1, 2, 3], 60, temperature=0)
    assert len(out) == 63
    # Without the cache it must make the same choices up to where the window first fills.
    naive = Engine(model).generate([1, 2, 3], 60, temperature=0, use_cache=False)
    assert out[:32] == naive[:32]


def test_stop_tokens_end_the_stream():
    model = GPT(CFG, seed=5)
    engine = Engine(model)
    first = next(engine.stream([1], 5, temperature=0))
    out = list(engine.stream([1], 50, temperature=0, stop={first}))
    assert out == [first]


def test_sampling_rules():
    logits = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
    rng = np.random.default_rng(0)
    assert sample(logits, temperature=0) == 0
    assert all(sample(logits, top_k=1, rng=rng) == 0 for _ in range(50))
    assert set(sample(logits, top_k=2, rng=rng) for _ in range(300)) == {0, 1}
    # top_p: probabilities are ~[.61, .22, .14, .03, .004]; p = 0.8 keeps the first two.
    assert set(sample(logits, top_p=0.8, rng=rng) for _ in range(500)) == {0, 1}
    # Temperature: low makes the top token dominate, high spreads the choices.
    low = [sample(logits, temperature=0.2, rng=rng) for _ in range(500)]
    high = [sample(logits, temperature=3.0, rng=rng) for _ in range(500)]
    assert low.count(0) > 480 and len(set(high)) == 5


def test_same_seed_same_text():
    model = GPT(CFG, seed=6)
    a = Engine(model).generate([1, 2], 30, temperature=1.0, top_p=0.9, seed=42)
    b = Engine(model).generate([1, 2], 30, temperature=1.0, top_p=0.9, seed=42)
    assert a == b


def test_the_cache_is_faster():
    model = GPT(GPTConfig(vocab_size=200, block_size=128, n_layer=3, n_head=4, n_embd=96), seed=7)
    engine = Engine(model)
    t0 = time.perf_counter()
    engine.generate([1, 2, 3], 100, temperature=0)
    cached = time.perf_counter() - t0
    t0 = time.perf_counter()
    engine.generate([1, 2, 3], 100, temperature=0, use_cache=False)
    naive = time.perf_counter() - t0
    assert naive > 2 * cached, (naive, cached)


# ---- edge cases (from an independent review) --------------------------------------------------------------
def test_top_k_keeps_exactly_k_even_with_ties():
    rng = np.random.default_rng(0)
    picks = {sample(np.array([1.0, 1.0, 1.0, 0.0]), top_k=1, rng=rng) for _ in range(50)}
    assert len(picks) == 1


@pytest.mark.parametrize("kwargs", [dict(temperature=-1.0), dict(top_k=0), dict(top_p=0.0), dict(top_p=1.5)])
def test_invalid_sampling_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        sample(np.zeros(5), **kwargs)


def test_empty_inputs_are_clear_errors():
    model = GPT(CFG, seed=0)
    engine = Engine(model)
    for call in (lambda: model.generate([], 3), lambda: engine.generate([], 3), lambda: engine.embed([])):
        with pytest.raises(ValueError, match="prompt|empty"):
            call()
    with pytest.raises(ValueError, match="more than block_size"):
        next(batches(np.arange(10), block_size=32, batch_size=2, rng=np.random.default_rng(0)))


def test_save_writes_the_path_it_was_given(tmp_path):
    model = GPT(CFG, seed=0)
    model.save(tmp_path / "model")  # no ".npz": numpy alone would have added it
    loaded = GPT.load(tmp_path / "model")
    assert np.array_equal(loaded.params["wte"].data, model.params["wte"].data)


def test_a_full_window_is_rebuilt_from_its_recent_half():
    # When the cache reaches block_size, generation goes on from the most recent block_size // 2 tokens:
    # record what the engine is fed.
    engine = Engine(GPT(CFG, seed=3))
    fed = []
    forward = engine.forward
    engine.forward = lambda ids, cache: (fed.append(list(ids)), forward(ids, cache))[1]
    prompt = [1, 2, 3]
    window = CFG.block_size
    out = engine.generate(prompt, window + 2 - len(prompt), temperature=0)
    assert fed[0] == prompt and all(len(f) == 1 for f in fed[1 : window - len(prompt) + 1])
    assert fed[window - len(prompt) + 1] == out[: window + 1][-(window // 2):]