"""Byte-level BPE encoding using a trained vocabulary and ranked merges."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from functools import lru_cache
import heapq
import json
from pathlib import Path

import regex as re

from cs336_basics.bpe_tokenizer import PAT


class Tokenizer:
    """Encode with learned merge ranks; preserve special tokens and raw text.

    The optional bounded cache is used by ``encode``. ``encode_iterable`` avoids
    growing this cache so streaming memory depends on unfinished input, rather
    than the number of distinct pre-tokens encountered.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        *,
        cache_size: int = 32768,
    ):
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = tuple(sorted(set(special_tokens or ()), key=lambda s: (-len(s), s)))
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens must not be empty")
        self._token_ids: dict[bytes, int] = {}
        for token_id, spelling in sorted(self.vocab.items()):
            self._token_ids.setdefault(spelling, token_id)
        # Assign new IDs in the caller's order, independently of matching order.
        for token in special_tokens or ():
            spelling = token.encode("utf-8")
            if spelling not in self._token_ids:
                token_id = max(self.vocab, default=-1) + 1
                self.vocab[token_id] = spelling
                self._token_ids[spelling] = token_id
        self._special_ids = {token: self._token_ids[token.encode("utf-8")] for token in self.special_tokens}
        self._special_pattern = (
            re.compile("|".join(re.escape(token) for token in self.special_tokens)) if self.special_tokens else None
        )
        self._max_special_length = max(map(len, self.special_tokens), default=1)
        self._pattern = re.compile(PAT)
        self._byte_ids = tuple(self._token_ids.get(bytes([value])) for value in range(256))
        self._rules: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right) in enumerate(self.merges):
            try:
                pair = (self._token_ids[left], self._token_ids[right])
                merged_id = self._token_ids[left + right]
            except KeyError as error:
                raise ValueError("Every merge component and result must exist in the vocabulary") from error
            self._rules.setdefault(pair, (rank, merged_id))
        self._encode_cached = lru_cache(maxsize=cache_size)(self._encode_pretoken)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, *, cache_size: int = 32768):
        """Load the byte-safe hexadecimal JSON written by our BPE trainer."""
        vocab = {
            int(token_id): bytes.fromhex(spelling)
            for token_id, spelling in json.loads(Path(vocab_filepath).read_text(encoding="utf-8")).items()
        }
        merges = [
            (bytes.fromhex(left), bytes.fromhex(right))
            for left, right in json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        ]
        return cls(vocab, merges, special_tokens, cache_size=cache_size)

    def cache_clear(self) -> None:
        """Clear pre-token encodings, e.g. before a cold-cache throughput run."""
        self._encode_cached.cache_clear()

    def cache_info(self):
        return self._encode_cached.cache_info()

    def _encode_pretoken(self, text: str) -> tuple[int, ...]:
        word = [self._byte_ids[value] for value in text.encode("utf-8")]
        if None in word:
            raise ValueError("The input contains a byte missing from the vocabulary")
        if len(word) > 64:
            return self._merge_long_word(word)
        rules = self._rules
        while len(word) > 1:
            best_rank = len(self.merges)
            best_pair = None
            merged_id = None
            for first, second in zip(word, word[1:]):
                rule = rules.get((first, second))
                if rule is not None and rule[0] < best_rank:
                    best_rank, merged_id = rule
                    best_pair = (first, second)
            if best_pair is None:
                break
            updated = []
            index = 0
            while index < len(word):
                if index + 1 < len(word) and (word[index], word[index + 1]) == best_pair:
                    updated.append(merged_id)
                    index += 2
                else:
                    updated.append(word[index])
                    index += 1
            word = updated
        return tuple(word)

    def _merge_long_word(self, word: list[int]) -> tuple[int, ...]:
        """Use adjacent links and a rank/position heap for unusually long pieces."""
        size = len(word)
        previous = list(range(-1, size - 1))
        following = list(range(1, size)) + [-1]
        heap = []
        rules = self._rules

        def candidate(left: int, right: int):
            if left == -1 or right == -1:
                return None
            rule = rules.get((word[left], word[right]))
            if rule is None:
                return None
            rank, merged_id = rule
            return rank, left, right, word[left], word[right], merged_id

        for left in range(size - 1):
            entry = candidate(left, left + 1)
            if entry is not None:
                heap.append(entry)
        heapq.heapify(heap)
        while heap:
            _, left, right, left_id, right_id, merged_id = heapq.heappop(heap)
            if following[left] != right or word[left] != left_id or word[right] != right_id:
                continue
            word[left] = merged_id
            word[right] = -1
            after = following[right]
            following[left] = after
            if after != -1:
                previous[after] = left
            following[right] = previous[right] = -1
            for first, second in ((previous[left], left), (left, after)):
                entry = candidate(first, second)
                if entry is not None:
                    heapq.heappush(heap, entry)
        return tuple(token_id for token_id in word if token_id != -1)

    def encode(self, text: str) -> list[int]:
        """Return token IDs without changing whitespace or newline characters."""
        result: list[int] = []
        previous = 0
        if self._special_pattern is not None:
            for special in self._special_pattern.finditer(text):
                for piece in self._pattern.finditer(text, previous, special.start()):
                    result.extend(self._encode_cached(piece.group()))
                result.append(self._special_ids[special.group()])
                previous = special.end()
        for piece in self._pattern.finditer(text, previous):
            result.extend(self._encode_cached(piece.group()))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode concatenated strings while preserving regex boundaries.

        Retain the last two ordinary matches: a trailing apostrophe plus letters
        can become one contraction, and trailing whitespace needs lookahead.
        Special-token starts also need enough lookahead to select the longest
        spelling. An arbitrarily long unfinished pre-token may require a large
        buffer, but completed text and encoded IDs are not retained.
        """
        buffer = ""
        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk
            safe_start_limit = max(0, len(buffer) - self._max_special_length + 1)
            consumed = 0
            if self._special_pattern is not None:
                for special in self._special_pattern.finditer(buffer):
                    if special.start() >= safe_start_limit:
                        break
                    for piece in self._pattern.finditer(buffer, consumed, special.start()):
                        yield from self._encode_pretoken(piece.group())
                    yield self._special_ids[special.group()]
                    consumed = special.end()
            pending = deque()
            for piece in self._pattern.finditer(buffer, consumed):
                pending.append(piece)
                if len(pending) > 2:
                    ready = pending.popleft()
                    if ready.end() > safe_start_limit:
                        break
                    yield from self._encode_pretoken(ready.group())
                    consumed = ready.end()
            buffer = buffer[consumed:]
        # At EOF, every remaining special/ordinary token boundary is known.
        previous = 0
        if self._special_pattern is not None:
            for special in self._special_pattern.finditer(buffer):
                for piece in self._pattern.finditer(buffer, previous, special.start()):
                    yield from self._encode_pretoken(piece.group())
                yield self._special_ids[special.group()]
                previous = special.end()
        for piece in self._pattern.finditer(buffer, previous):
            yield from self._encode_pretoken(piece.group())

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
