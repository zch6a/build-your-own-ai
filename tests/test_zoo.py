from byoai import zoo


def test_tokenizer_is_cached_and_stories_end_with_eot(tmp_path, monkeypatch):
    monkeypatch.setenv("BYOAI_CHECKPOINTS", str(tmp_path))
    tok = zoo.story_tokenizer()
    assert len(list(tmp_path.glob("*.tok.json"))) == 1
    again = zoo.story_tokenizer()
    assert again.merges == tok.merges, "loaded from the cache"
    ids = zoo.story_ids(tok, n=5, seed=1)
    eot = tok.special[zoo.EOT]
    assert ids[-1] == eot and (ids == eot).sum() == 5, "one <|endoftext|> after each story"
    assert tok.decode(list(ids[: list(ids).index(eot)])).startswith(("In the", "One day", "After", "At night", "On Sunday"))
