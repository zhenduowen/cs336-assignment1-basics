import heapq
import os
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pair = tuple[int, int]


class _MergeCandidate:
    """Reverse heap ordering: largest count, then largest pair of byte spellings."""

    __slots__ = ("count", "pair", "spellings")

    def __init__(self, count: int, pair: Pair, vocab: dict[int, bytes]):
        self.count = count
        self.pair = pair
        self.spellings = (vocab[pair[0]], vocab[pair[1]])

    def __lt__(self, other: "_MergeCandidate") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.spellings > other.spellings


def _file_ranges(input_path: str | os.PathLike, special_tokens: tuple[str, ...], chunk_size: int) -> Iterator[Pair]:
    """Yield ranges ending after complete special tokens, preserving regex boundaries.

    Read with overlap so a delimiter can straddle a read boundary. Matching all
    delimiters together also preserves left-to-right, longest-first matching when
    delimiters overlap. A document longer than chunk_size remains a single range.
    """
    file_size = os.path.getsize(input_path)
    # Text-mode newline normalization can change delimiter positions. Keep such
    # inputs whole, as well as inputs without a safe document delimiter.
    if not special_tokens or file_size <= chunk_size or any("\r" in t or "\n" in t for t in special_tokens):
        if file_size:
            yield 0, file_size
        return

    delimiters = [token.encode("utf-8") for token in special_tokens]
    pattern = re.compile(b"|".join(re.escape(token) for token in delimiters))
    max_length = max(map(len, delimiters))
    read_size = max(chunk_size, max_length, 64 * 1024)
    carry = b""
    buffer_start = range_start = 0
    with open(input_path, "rb") as file:
        while True:
            block = file.read(read_size)
            buffer = carry + block
            # Defer starts lacking enough lookahead to rule out a longer token.
            safe_end = max(0, len(buffer) - max_length + 1) if block else len(buffer)
            consumed = safe_end
            for match in pattern.finditer(buffer):
                if match.start() >= safe_end:
                    break
                consumed = max(consumed, match.end())
                end = buffer_start + match.end()
                if end - range_start >= chunk_size:
                    yield range_start, end
                    range_start = end
            carry = buffer[consumed:]
            buffer_start += consumed
            if not block:
                break
    if range_start < file_size:
        yield range_start, file_size


def _count_pretokens(job: tuple[str | os.PathLike, int, int, tuple[str, ...]]) -> Counter[str]:
    """Read a document-aligned range and count strings before allocating token lists."""
    input_path, start, end, special_tokens = job
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")
    # Match the original trainer's open(..., encoding="utf-8") newline behavior.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(PAT)
    counts: Counter[str] = Counter()
    previous = 0
    if special_tokens:
        special_pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
        for match in special_pattern.finditer(text):
            counts.update(piece.group() for piece in pattern.finditer(text, previous, match.start()))
            previous = match.end()
    counts.update(piece.group() for piece in pattern.finditer(text, previous))
    return counts


def _pretokenize(
    input_path: str | os.PathLike, special_tokens: tuple[str, ...], num_workers: int, chunk_size: int
) -> Counter[str]:
    jobs = ((input_path, start, end, special_tokens) for start, end in _file_ranges(input_path, special_tokens, chunk_size))
    counts: Counter[str] = Counter()
    if num_workers == 1:
        for job in jobs:
            counts.update(_count_pretokens(job))
        return counts

    # Bound submitted work: completed Counters must not accumulate for the entire
    # corpus while the parent is still combining earlier results.
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pending = deque()
        for _ in range(2 * num_workers):
            job = next(jobs, None)
            if job is None:
                break
            pending.append(pool.submit(_count_pretokens, job))
        while pending:
            counts.update(pending.popleft().result())
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(_count_pretokens, job))
    return counts


def _merge_word(word: list[int], pair: Pair, merged_id: int) -> list[int]:
    result: list[int] = []
    first, second = pair
    i, length = 0, len(word)
    while i < length:
        if i + 1 < length and word[i] == first and word[i + 1] == second:
            result.append(merged_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return result


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int = 1,
    chunk_size: int = 4 * 1024 * 1024,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE using incremental counts and an affected-word index.

    Args:
        input_path: UTF-8 training corpus.
        vocab_size: Target vocabulary size, including bytes and special tokens.
        special_tokens: Strings kept out of ordinary pre-tokenization and merges.
            Overlapping special tokens are matched longest-first.
        num_workers: Number of preprocessing processes (1 avoids process overhead).
        chunk_size: Target bytes per preprocessing job. Jobs end at special-token
            boundaries; an individual document may exceed this size. With no safe
            delimiters, the entire file is processed together.
        **kwargs: Reserved for compatibility with the assignment adapter.

    Returns:
        An ID-to-bytes vocabulary and byte-pair merges in learning order. Frequency
        ties choose the lexicographically largest pair of byte spellings.
    """
    if num_workers < 1 or chunk_size < 1:
        raise ValueError("num_workers and chunk_size must be positive")
    if any(not token for token in special_tokens):
        raise ValueError("Special tokens must not be empty")

    vocab = {i: bytes([i]) for i in range(256)}
    token_ids = {token: token_id for token_id, token in vocab.items()}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in token_ids:
            token_ids[token_bytes] = len(vocab)
            vocab[len(vocab)] = token_bytes

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    specials = tuple(sorted(set(special_tokens), key=lambda token: (-len(token), token)))
    pretoken_counts = _pretokenize(input_path, specials, num_workers, chunk_size)
    words: list[list[int]] = []
    frequencies: list[int] = []
    pair_counts: dict[Pair, int] = defaultdict(int)
    pair_words: dict[Pair, set[int]] = defaultdict(set)
    # Consume strings as they are converted: each distinct pre-token is encoded
    # once, and its original frequency remains fixed throughout training.
    while pretoken_counts:
        text, frequency = pretoken_counts.popitem()
        word = list(text.encode("utf-8"))
        word_id = len(words)
        words.append(word)
        frequencies.append(frequency)
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency
            pair_words[pair].add(word_id)
    del pretoken_counts

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], (vocab[pair[0]], vocab[pair[1]])),
        )
        spellings = (vocab[best_pair[0]], vocab[best_pair[1]])
        merges.append(spellings)
        merged_bytes = spellings[0] + spellings[1]
        next_token_id = len(vocab)
        # Identical byte spellings must share an internal identity, even if two
        # different merge paths produce them. Keep the public vocabulary ordering.
        merged_id = token_ids.setdefault(merged_bytes, next_token_id)
        vocab[next_token_id] = merged_bytes

        # Snapshot memberships before updating them. Only these words can change.
        for word_id in tuple(pair_words[best_pair]):
            word = words[word_id]
            old_pairs = Counter(zip(word, word[1:]))
            new_word = _merge_word(word, best_pair, merged_id)
            new_pairs = Counter(zip(new_word, new_word[1:]))
            words[word_id] = new_word
            frequency = frequencies[word_id]
            for pair in old_pairs.keys() | new_pairs.keys():
                before, after = old_pairs[pair], new_pairs[pair]
                delta = (after - before) * frequency
                if delta:
                    count = pair_counts.get(pair, 0) + delta
                    if count:
                        pair_counts[pair] = count
                    else:
                        pair_counts.pop(pair, None)
                if before and not after:
                    locations = pair_words[pair]
                    locations.remove(word_id)
                    if not locations:
                        del pair_words[pair]
                elif after and not before:
                    pair_words[pair].add(word_id)


    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
