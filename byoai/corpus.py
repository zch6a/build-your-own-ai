"""A small training corpus, generated — not copied from anywhere.

The chapters need text to train on: enough structure for a tokenizer to find useful merges and for a
tiny transformer to learn something you can see, small enough to train on a laptop CPU in minutes.
A seeded grammar of short stories gives exactly that, with no licensing questions and the same text
on every machine.

    >>> from byoai.corpus import stories
    >>> text = stories(200, seed=0)
"""

from __future__ import annotations

import random

_NAMES = ["Mia", "Leo", "Ada", "Sam", "Noor", "Ravi", "Lena", "Tom", "Yuki", "Omar", "Ivy", "Ben"]
_ANIMALS = ["cat", "dog", "fox", "owl", "rabbit", "horse", "bird", "turtle", "goat", "bear"]
_PLACES = ["the garden", "the forest", "the river", "the hill", "the market", "the old house", "the beach", "the field"]
_THINGS = ["a red ball", "a small box", "a blue kite", "an apple", "a map", "a lamp", "a key", "a book", "a hat", "a shell"]
_FEELINGS = ["happy", "tired", "curious", "brave", "hungry", "sleepy", "excited", "calm"]
_WEATHER = ["sunny", "rainy", "windy", "cold", "warm", "cloudy"]
_TIMES = ["In the morning", "One day", "After lunch", "In the evening", "At night", "On Sunday"]


def _story(r: random.Random) -> str:
    name, friend = r.sample(_NAMES, 2)
    animal = r.choice(_ANIMALS)
    place, other = r.sample(_PLACES, 2)
    thing = r.choice(_THINGS)
    feeling = r.choice(_FEELINGS)
    sentences = [
        f"{r.choice(_TIMES)}, {name} went to {place} with a {animal}.",
        f"It was {r.choice(_WEATHER)} and the {animal} was {feeling}.",
        r.choice([
            f"{name} found {thing} under a tree.",
            f"The {animal} found {thing} near {other}.",
            f"{friend} gave {name} {thing}.",
        ]),
        r.choice([
            f"{name} said, \"Let us go to {other}.\"",
            f"\"Look!\" said {friend}. \"The {animal} is {feeling}.\"",
            f"{name} and {friend} played until it was dark.",
        ]),
        f"Then they went home, and the {animal} slept.",
    ]
    return " ".join(sentences)


def stories(n: int, seed: int = 0) -> str:
    """`n` short stories, one per line; the same `seed` gives the same text everywhere."""
    r = random.Random(seed)
    return "\n".join(_story(r) for _ in range(n)) + "\n"
