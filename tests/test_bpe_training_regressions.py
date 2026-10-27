"""Regression tests for exact merges while optimizing BPE training."""

import random
from collections import Counter

import pytest
import regex

from cs336_basics.bpe_tokenizer import PAT, _pretokenize, train_bpe


def _reference_pretokens(text, special_tokens):
    if special_tokens:
        pattern = "|".join(regex.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True))
        documents = regex.split(pattern, text)
    else:
        documents = [text]
    return [match.group() for document in documents for match in regex.finditer(PAT, document)]


def _reference_train(text, vocab_size, special_tokens):
    """Small oracle: recount and rewrite every occurrence after each merge."""
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        encoded = token.encode("utf-8")
        if encoded not in vocab.values():
            vocab[len(vocab)] = encoded

    words = [
        [bytes([byte]) for byte in pretoken.encode("utf-8")]
        for pretoken in _reference_pretokens(text, special_tokens)
    ]

    merges = []
    while len(vocab) < vocab_size:
        counts = Counter(pair for word in words for pair in zip(word, word[1:]))
        if not counts:
            break
        pair = max(counts, key=lambda candidate: (counts[candidate], candidate))
        merged = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = merged
        for index, word in enumerate(words):
            result = []
            position = 0
            while position < len(word):
                if tuple(word[position : position + 2]) == pair:
                    result.append(merged)
                    position += 2
                else:
                    result.append(word[position])
                    position += 1
            words[index] = result
    return vocab, merges


@pytest.mark.parametrize(
    ("text", "special_tokens"),
    [
        ("", []),
        ("a", []),
        ("aaaa aaa aaaaa ababab baba", []),
        ("banana " * 19 + "bandana " * 3 + "ananas", []),
        (" café 我😊\n\n naïve\tこんにちは café 我😊", []),
        ("hello<e>world<e><e>banana", ["<e>", "<e>"]),
        ("hello<e>longworld<e>banana", ["<e>", "<e>long"]),
        ("<e><e>", ["<e>"]),
    ],
)
def test_training_matches_reference(tmp_path, text, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    assert train_bpe(path, 310, special_tokens) == _reference_train(text, 310, special_tokens)


@pytest.mark.parametrize("seed", range(6))
def test_randomized_merges_match_reference(tmp_path, seed):
    rng = random.Random(seed)
    vocabulary = ["aaaa", "abab", "banana", "cab", "café", "我", "😊", "'ll", "123", "!!"]
    separators = [" ", "\n", "\t", "<e>"]
    text = "".join(rng.choice(vocabulary) + rng.choice(separators) for _ in range(60))
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    assert train_bpe(path, 300, ["<e>"]) == _reference_train(text, 300, ["<e>"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ab ac", [(b"a", b"c")]),
        # The ID assigned to b"aa" exceeds the ID of b"z", but its byte
        # spelling sorts earlier. Numeric-ID tie-breaking would be incorrect.
        ("aaa zz", [(b"a", b"a"), (b"z", b"z")]),
    ],
)
def test_frequency_ties_use_largest_byte_pair(tmp_path, text, expected):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    _, merges = train_bpe(path, 256 + len(expected), [])
    assert merges == expected


def test_overlapping_adjacencies_merge_without_overlap(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("aaaa", encoding="utf-8")
    _, merges = train_bpe(path, 258, [])
    assert merges == [(b"a", b"a"), (b"aa", b"aa")]


@pytest.mark.parametrize("chunk_size", [1, 7, 31])
def test_document_chunks_and_workers_preserve_merges(tmp_path, chunk_size):
    # Small target chunks exercise document partitioning and worker reduction.
    # The shorter special token must not consume the longer token's prefix.
    specials = ["<e>", "<e>long", "<文>"]
    text = ("我😊 café<e>longbanana\n\n<文>aaaa<e> hello ") * 4 + "tail😊"
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    expected = _reference_train(text, 310, specials)
    for workers in (1, 2):
        assert train_bpe(path, 310, specials, num_workers=workers, chunk_size=chunk_size) == expected


@pytest.mark.parametrize("special_tokens", [[], ["<absent>"]])
def test_corpus_without_delimiters_preserves_pretokens(tmp_path, special_tokens):
    text = "verylongword" * 10 + " café\n\n   我😊\tend"
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    assert train_bpe(path, 310, special_tokens, num_workers=2, chunk_size=7) == _reference_train(
        text, 310, special_tokens
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_delimiters_cross_scanner_read_boundaries(tmp_path, workers):
    block_size = 64 * 1024
    # The complete short marker fits in the first read, but its longer matching
    # suffix requires the next read. A Chinese code point spans the next boundary.
    prefix_length = block_size - len(b"<e>")
    data = b"ab " * (prefix_length // 3) + b" " * (prefix_length % 3) + b"<e>long"
    padding = 2 * block_size - 2 - len(data)
    data += b"cd " * (padding // 3) + b" " * (padding % 3)
    data += "<文>tail<e>end".encode("utf-8")
    path = tmp_path / "corpus.txt"
    path.write_bytes(data)
    specials = ("<e>long", "<e>", "<文>")
    expected = Counter(_reference_pretokens(data.decode("utf-8"), specials))
    assert _pretokenize(path, specials, num_workers=workers, chunk_size=4096) == expected


@pytest.mark.parametrize("special_tokens", [["<e>"], ["\n<e>\n"], ["\r\n"]])
def test_universal_newlines_and_newline_delimiters(tmp_path, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_bytes("hello\r\nworld\rnext\r\n<e>\r\nnaïve\r我".encode("utf-8"))
    # The original text-mode reader normalizes CRLF and bare CR before matching
    # special tokens. Delimiters containing newlines must preserve that behavior.
    expected = _reference_train(path.read_text(encoding="utf-8"), 300, special_tokens)
    assert train_bpe(path, 300, special_tokens, num_workers=2, chunk_size=7) == expected


@pytest.mark.parametrize("special_tokens", [[""], ["<e>", ""]])
def test_empty_special_token_is_rejected(tmp_path, special_tokens):
    path = tmp_path / "corpus.txt"
    path.write_text("abc", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        train_bpe(path, 300, special_tokens)


@pytest.mark.parametrize(("workers", "chunk_size"), [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_nonpositive_preprocessing_configuration_is_rejected(tmp_path, workers, chunk_size):
    path = tmp_path / "corpus.txt"
    path.write_text("abc", encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        train_bpe(path, 300, [], num_workers=workers, chunk_size=chunk_size)
