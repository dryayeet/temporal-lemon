from facts import format_user_facts


def test_empty_facts_returns_empty_string():
    assert format_user_facts({}) == ""


def test_facts_block_is_wrapped_in_tags():
    out = format_user_facts({"city": "Bangalore"})
    assert out.startswith("<user_facts>")
    assert out.endswith("</user_facts>")


def test_facts_block_lists_each_pair():
    out = format_user_facts({"city": "Bangalore", "pet": "pickle"})
    assert "city: Bangalore" in out
    assert "pet: pickle" in out
