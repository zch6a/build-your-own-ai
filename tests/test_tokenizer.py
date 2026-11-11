import random
import time

import pytest

from byoai.corpus import stories
from byoai.tokenizer import PRETOKENIZE, Tokenizer


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train(stories(300, seed=1), vocab_size=512, special=["<|endoftext|>"])


def test_merges_follow_the_counts():
    # One chunk "aaabdaaabac": the pair (a, a) occurs 4 times, more than any other.
    t = Tokenizer.train("aaabdaaabac", vocab_size=259)
    first = next(iter(t.merges))
    assert first == (97, 97)
    assert t.merges[first] == 256
    assert len(t.merges) == 3
    assert t.decode(t.encode("aaabdaaabac")) == "aaabdaaabac"
    assert len(t.encode("aaabdaaabac")) < len("aaabdaaabac")


def test_training_is_deterministic():
    text = stories(50, seed=3)
    assert Tokenizer.train(text, 400).merges == Tokenizer.train(text, 400).merges


def test_vocab_size_is_exact(tok: Tokenizer):
    assert tok.vocab_size == 512
    assert len(tok.vocab) == 512
    assert tok.special["<|endoftext|>"] == 511


def test_merges_never_cross_chunks():
    # A space may lead a token (" cat"), never sit inside one ("t c"): merges stay within a chunk.
    t = Tokenizer.train(stories(100, seed=6), vocab_size=600)
    for i in range(256, 256 + len(t.merges)):
        text = t.token_text(i)
        assert " " not in text[1:], repr(text)


@pytest.mark.parametrize("seed", range(20))
def test_any_text_round_trips(tok: Tokenizer, seed: int):
    # Text the tokenizer never saw: other scripts, emoji, control characters, odd whitespace.
    r = random.Random(seed)
    alphabet = "abc XYZ 123 ,.!?'\"\n\t\r é ß 中文 日本語 한국어 😀🚀 \u200d \u0301 \x00 \x7f"
    text = "".join(r.choice(alphabet) for _ in range(r.randint(0, 200)))
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize("seed", range(20))
def test_pretokenizer_loses_nothing(seed: int):
    # Every character must land in some chunk. `_` is a regex "word" character that is neither a
    # letter nor a digit, and once fell through every branch; superscripts, roman numerals, combining
    # marks and other scripts are the next places to look.
    r = random.Random(seed)
    alphabet = "a_b __init__ x1_2 ²³ Ⅷ ٣ é" + chr(0x301) + chr(0x200D) + " 中_文 ' 's -_- \t\n"
    text = "".join(r.choice(alphabet) for _ in range(r.randint(0, 120)))
    assert "".join(PRETOKENIZE.findall(text)) == text


def test_underscores_round_trip(tok: Tokenizer):
    for text in ("snake_case_name", "__init__", "search_towns(read_file)", "_"):
        assert tok.decode(tok.encode(text)) == text


def test_ids_stay_in_the_vocabulary(tok: Tokenizer):
    ids = tok.encode(stories(20, seed=9) + "😀 中文")
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_it_compresses_text_like_its_training_text(tok: Tokenizer):
    text = stories(100, seed=2)  # same grammar, different stories
    ratio = len(text.encode("utf-8")) / len(tok.encode(text))
    assert ratio > 3.0, ratio


def test_special_tokens_are_opt_in(tok: Tokenizer):
    text = "<|endoftext|>Mia went home."
    eot = tok.special["<|endoftext|>"]
    assert eot not in tok.encode(text), "by default a special token in text is just text"
    assert tok.encode(text, allowed_special="all")[0] == eot
    assert tok.encode(text, allowed_special={"<|endoftext|>"})[0] == eot
    assert tok.decode(tok.encode(text, allowed_special="all")) == text


def test_save_and_load(tok: Tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special == tok.special
    text = stories(5, seed=4)
    assert loaded.encode(text) == tok.encode(text)


def test_pretokenizer_splits_words_numbers_and_punctuation():
    assert PRETOKENIZE.findall("Mia's 2024 kites, flew!") == ["Mia", "'s", " 202", "4", " kites", ",", " flew", "!"]


def test_unknown_ids_are_an_error(tok: Tokenizer):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size + 5])


def test_training_and_encoding_are_fast_enough():
    text = stories(2000, seed=5)  # ~600 KB
    t0 = time.perf_counter()
    t = Tokenizer.train(text, vocab_size=1000)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    t.encode(text)
    encode_s = time.perf_counter() - t0
    assert train_s < 20, train_s
    assert encode_s < 10, encode_s
