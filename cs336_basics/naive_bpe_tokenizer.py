import os
from collections import Counter

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str], 
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input data.

    Args:
        input_path: str | os.PathLike,
            The path to the input text file.
        vocab_size: vocab_size: int,
            The desired size of the vocabulary (including special tokens).
        special_tokens: list[str],
            A list of special tokens to include in the vocabulary.
        **kwargs: Additional keyword arguments for training.

    Returns:
        vocab: dict[int, bytes] 
            The tokenizer vocabulary, 
            a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes,bytes]]
            A list of BPE merges produced from training. 
            Each list item is a tuple of bytes (<token1>, <token2>), 
            representing that <token1> was merged with <token2>. 
            The merges should be ordered by order of creation.
    """
    # Initialize vocabulary with all single-byte tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add special tokens to the vocabulary after the 256 byte tokens.
    next_token_id = 256
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_token_id] = token_bytes
            next_token_id += 1

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on special tokens before pre-tokenization so no merge can cross a
    # special-token boundary, e.g. across <|endoftext|>.
    if special_tokens:
        special_token_pattern = "|".join(re.escape(token) for token in special_tokens)
        chunks = re.split(special_token_pattern, text)
    else:
        chunks = [text]

    # Pre-tokenize the corpus. We store each pre-token as a tuple of bytes.
    # Counter lets us aggregate repeated pre-tokens efficiently.
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pretoken = match.group()
            byte_tuple = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            pretoken_counts[byte_tuple] += 1

    def get_pair_counts(
        counts: Counter[tuple[bytes, ...]],
    ) -> Counter[tuple[bytes, bytes]]:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for tokens, frequency in counts.items():
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += frequency
        return pair_counts

    def merge_pair(
        tokens: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        merged_tokens: list[bytes] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return tuple(merged_tokens)

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(pretoken_counts)
        if not pair_counts:
            break

        # Tie-break rule: choose the lexicographically largest pair among pairs
        # with maximum frequency. This is the convention used in many CS336 tests.
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best_pair)

        vocab[next_token_id] = best_pair[0] + best_pair[1]
        next_token_id += 1

        new_pretoken_counts: Counter[tuple[bytes, ...]] = Counter()
        for tokens, frequency in pretoken_counts.items():
            new_pretoken_counts[merge_pair(tokens, best_pair)] += frequency
        pretoken_counts = new_pretoken_counts

    return vocab, merges

if __name__ == "__main__":
    local_test_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = train_bpe(
        input_path=local_test_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
