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


# ---- a knowledge base for chapter 6 ---------------------------------------------------------------------------
# Facts about towns that do not exist, so no model can know them from training: the only way to answer a
# question about them is to retrieve the right passage.
_SYLLABLES = ["bar", "wen", "dor", "lis", "tam", "ker", "vol", "mir", "sen", "gal", "rho", "pel", "nus", "tor", "fen", "qua", "zim", "har", "lo", "bri"]
_ENDINGS = ["ford", "wick", "mere", "holt", "by", "ton", "stead", "gate", "burn", "dale"]
_FIRST = ["Ada", "Bram", "Cleo", "Dov", "Esme", "Finn", "Greta", "Hugo", "Iris", "Jude", "Kai", "Lior", "Maren", "Nils", "Opal", "Pia"]
_LAST = ["Venn", "Holloway", "Marsh", "Quill", "Ashby", "Thorne", "Calder", "Wren", "Fairley", "Okafor", "Lindqvist", "Moreau"]
_PRODUCTS = ["blue pottery", "smoked cheese", "copper bells", "wool blankets", "pear cider", "glass beads", "rope", "honey cakes", "clocks", "paper lanterns", "salt", "violins"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
_FILLER = [
    "The streets are narrow and paved with grey stone.",
    "Winters there are long, and the summers short and bright.",
    "Visitors often remark on how quiet the evenings are.",
    "A covered market stands in the main square.",
    "Most houses have red roofs and small gardens.",
    "The old bridge was rebuilt after a flood.",
    "Children walk to school along the water.",
    "There is a bakery on almost every corner.",
]


def _town_name(r: random.Random, taken: set[str]) -> str:
    while True:
        name = (r.choice(_SYLLABLES) + r.choice(_SYLLABLES) + r.choice(_ENDINGS)).capitalize()
        if name not in taken:
            taken.add(name)
            return name


def knowledge_base(n_towns: int = 60, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Encyclopedia articles about invented towns, and questions about them with their answers.

    Returns `(documents, questions)`: documents are `{"title", "text"}`; questions are `{"question",
    "answer", "doc", "field"}` with `doc` the index of the article that answers it. Each fact is written in one of
    several phrasings, and only some sentences repeat the town's name ("Its mayor is …"), as in real text.
    """
    r = random.Random(seed)
    taken: set[str] = set()
    names = [_town_name(r, taken) for _ in range(n_towns)]
    rivers = [_town_name(r, taken).replace("ford", "").replace("by", "") + " River" for _ in range(n_towns // 3)]
    documents, questions = [], []
    for i, town in enumerate(names):
        mayor = f"{r.choice(_FIRST)} {r.choice(_LAST)}"
        river = r.choice(rivers)
        product = r.choice(_PRODUCTS)
        founded = r.randint(1150, 1890)
        people = r.randint(8, 400) * 100
        month = r.choice(_MONTHS)
        neighbour = names[(i + r.randint(1, n_towns - 1)) % n_towns]
        facts = [
            r.choice([f"{town} lies on the {river}.", f"The {river} runs through {town}.", f"It was built where the {river} bends."]),
            r.choice([f"The mayor of {town} is {mayor}.", f"Its mayor is {mayor}.", f"{town} is governed by {mayor}."]),
            r.choice([f"{town} is famous for its {product}.", f"It is best known for {product}.", f"Its workshops make {product}, sold across the region."]),
            r.choice([f"{town} was founded in {founded}.", f"It was founded in {founded}.", f"The first houses were built in {founded}."]),
            r.choice([f"About {people:,} people live in {town}.", f"Its population is about {people:,}.", f"The town has about {people:,} inhabitants."]),
            r.choice([f"The harvest festival of {town} is held every {month}.", f"Every {month} the town holds a harvest festival.", f"Its harvest festival takes place in {month}."]),
            f"The nearest town is {neighbour}.",
        ]
        body = facts[:1] + r.sample(facts[1:] + r.sample(_FILLER, 3), len(facts) - 1 + 3)
        documents.append({"title": town, "text": f"{town} is a town in the valley. " + " ".join(body)})
        questions += [
            {"question": f"Who is the mayor of {town}?", "answer": mayor, "doc": i, "field": "mayor"},
            {"question": f"Which river does {town} lie on?", "answer": river, "doc": i, "field": "river"},
            {"question": f"What is {town} famous for?", "answer": product, "doc": i, "field": "product"},
            {"question": f"When was {town} founded?", "answer": str(founded), "doc": i, "field": "founded"},
            {"question": f"How many people live in {town}?", "answer": f"{people:,}", "doc": i, "field": "population"},
            {"question": f"In which month is the harvest festival of {town}?", "answer": month, "doc": i, "field": "festival"},
        ]
    return documents, questions
