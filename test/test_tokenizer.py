# test/test_tokenizer.py

import pytest

from src.tokenizer import CharTokenizer


def test_vocab_size_equals_number_of_unique_chars():
    text = "hello"
    tokenizer = CharTokenizer(text)

    assert tokenizer.vocab_size == len(set(text))


def test_encode_returns_list_of_ints_with_same_length_as_text():
    text = "hello world"
    tokenizer = CharTokenizer(text)

    ids = tokenizer.encode(text)

    assert isinstance(ids, list)
    assert len(ids) == len(text)
    assert all(isinstance(i, int) for i in ids)


def test_decode_restores_original_text():
    text = "hello world"
    tokenizer = CharTokenizer(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_same_character_maps_to_same_id():
    text = "banana"
    tokenizer = CharTokenizer(text)

    ids = tokenizer.encode(text)

    assert ids[1] == ids[3] == ids[5]  # "a"
    assert ids[2] == ids[4]            # "n"


def test_unknown_character_raises_keyerror():
    tokenizer = CharTokenizer("hello")

    with pytest.raises(KeyError):
        tokenizer.encode("hello!")