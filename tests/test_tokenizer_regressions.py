"""Boundary, rank, and serialization checks beyond the supplied tokenizer tests."""

import json
import random

import pytest
import regex as re

from cs336_basics.bpe_tokenizer import PAT
from cs336_basics.tokenizer import Tokenizer


def make_tokenizer(special_tokens=None, cache_size=32768):
    vocab = {index: bytes([index]) for index in range(256)}
    merges = [
        (b"t", b"h"), (b"th", b"e"), (b"a", b"a"), (b"aa", b"aa"),
        (b"'", b"l"), (b"'l", b"l"), (b"'", b"r"), (b"'r", b"e"),
        (b" ", b"t"), (b"\xc3", b"\xa9"),
    ]
    for length in (1, 2, 4, 8, 16, 32, 64):
        merges.append((b"-" * length, b"-" * length))
    for first, second in merges:
        vocab[len(vocab)] = first + second
    return Tokenizer(vocab, merges, special_tokens, cache_size=cache_size)


def reference_encode(tokenizer, text):
    """Slow byte-based oracle: replace only the earliest-ranked leftmost pair."""
    ranks = {pair: index for index, pair in enumerate(tokenizer.merges)}
    token_ids = {spelling: index for index, spelling in tokenizer.vocab.items()}
    result = []
    for match in re.finditer(PAT, text):
        pieces = [bytes([value]) for value in match.group().encode("utf-8")]
        while True:
            options = [(ranks[pair], index) for index, pair in enumerate(zip(pieces, pieces[1:])) if pair in ranks]
            if not options:
                break
            _, index = min(options)
            pieces[index:index + 2] = [pieces[index] + pieces[index + 1]]
        result.extend(token_ids[piece] for piece in pieces)
    return result


@pytest.mark.parametrize("text", [
    "the cat ate", "the'", "we'll they're", "\n   the\r\n é", "the\t\t  cat ",
    "!!!'ll?!", "a" * 137, "-" * 1025, "é" * 70,
])
def test_streaming_at_every_split(text):
    tokenizer = make_tokenizer()
    expected = tokenizer.encode(text)
    assert expected == reference_encode(tokenizer, text)
    assert list(tokenizer.encode_iterable(iter(text))) == expected
    for split in range(len(text) + 1):
        assert list(tokenizer.encode_iterable((text[:split], "", text[split:]))) == expected
    assert tokenizer.decode(expected) == text


@pytest.mark.parametrize("text", [
    "the<e>longthe<e>é", "<e>long<e><e>long", "abc<e", "<e>lon", "a <e>long\n  the",
    "abENDabcabENDx", "the\r\n<S>the",
])
def test_streaming_split_and_overlapping_specials(text):
    tokenizer = make_tokenizer(["<e>", "<e>long", "abEND", "\r\n<S>"])
    expected = tokenizer.encode(text)
    assert tokenizer.decode(expected) == text
    assert list(tokenizer.encode_iterable(iter(text))) == expected
    for split in range(len(text) + 1):
        assert list(tokenizer.encode_iterable((text[:split], text[split:]))) == expected


def test_random_stream_partitions():
    rng = random.Random(17)
    tokenizer = make_tokenizer(["<e>", "<e>long"])
    alphabet = ["the", "a", "-", "'", "l", "re", " ", "\n", "\r", "\t", "é", "🙃", "<e>", "<e>long"]
    for _ in range(150):
        text = "".join(rng.choice(alphabet) for _ in range(40))
        chunks = []
        start = 0
        while start < len(text):
            end = start + rng.randint(1, 11)
            chunks.append(text[start:end])
            start = end
        assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)


def test_earlier_rank_beats_more_frequent_pair():
    vocab = {index: bytes([index]) for index in range(256)}
    vocab.update({256: b"ab", 257: b"bc"})
    tokenizer = Tokenizer(vocab, [(b"a", b"b"), (b"b", b"c")])
    assert [tokenizer.vocab[i] for i in tokenizer.encode("abcbcbc")] == [b"ab", b"c", b"bc", b"bc"]


def test_long_heap_matches_byte_oracle():
    rng = random.Random(29)
    tokenizer = make_tokenizer()
    for _ in range(40):
        text = "".join(rng.choice("athe") for _ in range(200)) + "-" * rng.randint(65, 300)
        assert tokenizer.encode(text) == reference_encode(tokenizer, text)


def test_utf8_combined_before_decoding_and_errors_replaced():
    tokenizer = make_tokenizer()
    assert tokenizer.decode([0xC3, 0xA9]) == "é"
    assert tokenizer.decode([0xC3]) == "�"
    assert tokenizer.decode([0xFF, 0x61]) == "�a"


def test_hex_json_roundtrip_and_missing_specials(tmp_path):
    original = make_tokenizer()
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.json"
    vocab_path.write_text(json.dumps({str(k): v.hex() for k, v in original.vocab.items()}))
    merges_path.write_text(json.dumps([[a.hex(), b.hex()] for a, b in original.merges]))
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<e>", "<e>long", "<e>"])
    assert len(tokenizer.vocab) == len(original.vocab) + 2
    assert len(original.vocab) == len(json.loads(vocab_path.read_text()))
    text = "the<e>longé<e>"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert [tokenizer.vocab[i] for i in tokenizer.encode("<e>long")] == [b"<e>long"]


def test_cache_is_bounded_clearable_and_unused_by_streaming():
    tokenizer = make_tokenizer(cache_size=4)
    tokenizer.encode("one two three four five six")
    assert tokenizer.cache_info().currsize <= 4
    tokenizer.cache_clear()
    assert tokenizer.cache_info().currsize == 0
    assert tokenizer.decode(tokenizer.encode_iterable(["the ", "cat"])) == "the cat"
    assert tokenizer.cache_info().currsize == 0


def test_empty_special_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        make_tokenizer([""])
