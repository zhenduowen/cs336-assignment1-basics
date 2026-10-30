"""Compare saved TinyStories and OpenWebText vocabulary types, not compression.

Run after both training jobs have completed:
    .venv/bin/python scripts/compare_bpe_vocabularies.py

Inputs use the training runner's JSON format: decimal token ID -> hex bytes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
SPECIAL = b"<|endoftext|>"
CANDIDATES = (
    b" princess", b" dragon", b" fairy", b" forest", b" kitten", b" happily",
    b" accomplishment", b" disappointment", b" responsibility",
    b" government", b" political", b" software", b" internet", b" technology",
    b" scientific", b" legislation", b" algorithm", b" Python",
)


def load_vocabulary(path: Path) -> dict[int, bytes]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    vocab = {int(key): bytes.fromhex(value) for key, value in raw.items()}
    if len(vocab) != len(raw) or set(vocab) != set(range(len(vocab))):
        raise ValueError(f"Vocabulary IDs must be unique and contiguous: {path}")
    if any(vocab.get(index) != bytes([index]) for index in range(256)):
        raise ValueError(f"Expected the 256 byte tokens at IDs 0..255: {path}")
    if vocab.get(256) != SPECIAL:
        raise ValueError(f"Expected <|endoftext|> at ID 256: {path}")
    return vocab


def describe_token(token: bytes) -> dict:
    return {"hex": token.hex(), "bytes_repr": repr(token),
            "text": token.decode("utf-8", errors="replace")}


def summarize(path: Path, vocab: dict[int, bytes]) -> dict:
    learned = {index: token for index, token in vocab.items() if index >= 257}
    lengths = [len(token) for token in learned.values()]
    maximum = max(lengths, default=None)
    return {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "vocabulary_entries": len(vocab),
        "learned_entries": len(learned),
        "unique_learned_byte_spellings": len(set(learned.values())),
        "learned_token_byte_lengths": {
            "mean": statistics.mean(lengths) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "maximum": maximum,
        },
        "longest_learned_tokens": [
            {"id": index, "byte_length": len(token), **describe_token(token)}
            for index, token in sorted(learned.items()) if len(token) == maximum
        ],
    }


def compare(tinystories_path: Path, openwebtext_path: Path) -> dict:
    vocabularies = {
        "tinystories": load_vocabulary(tinystories_path),
        "openwebtext": load_vocabulary(openwebtext_path),
    }
    learned_sets = {
        name: {token for index, token in vocab.items() if index >= 257}
        for name, vocab in vocabularies.items()
    }
    tiny, owt = learned_sets["tinystories"], learned_sets["openwebtext"]
    shared = len(tiny & owt)
    reverse = {}
    for name, vocab in vocabularies.items():
        reverse[name] = defaultdict(list)
        for index, token in sorted(vocab.items()):
            reverse[name][token].append(index)
    return {
        "method": {
            "population": "Learned vocabulary entries at IDs >=257; exclude 256 byte tokens and <|endoftext|>.",
            "length_statistics": "Unweighted over learned vocabulary entries; not corpus bytes per encoded token.",
            "overlap": "Set intersection of distinct learned byte spellings; fractions use each set's size.",
            "interpretation": "Both corpus and vocabulary budget differ; this is not a corpus-only comparison.",
        },
        "tinystories": summarize(tinystories_path, vocabularies["tinystories"]),
        "openwebtext": summarize(openwebtext_path, vocabularies["openwebtext"]),
        "overlap": {
            "shared_unique_learned_tokens": shared,
            "fraction_of_tinystories_unique_learned_tokens": shared / len(tiny) if tiny else None,
            "fraction_of_openwebtext_unique_learned_tokens": shared / len(owt) if owt else None,
            "jaccard": shared / len(tiny | owt) if tiny | owt else None,
        },
        "representative_candidates": [
            {**describe_token(token), **{f"{name}_ids": ids.get(token, []) for name, ids in reverse.items()}}
            for token in CANDIDATES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tinystories-vocab", type=Path,
                        default=ROOT / "artifacts/tokenizers/tinystories/model/vocab.json")
    parser.add_argument("--openwebtext-vocab", type=Path,
                        default=ROOT / "artifacts/tokenizers/openwebtext/model/vocab.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/tokenizers/openwebtext/comparison.json")
    args = parser.parse_args()
    result = compare(args.tinystories_vocab.resolve(), args.openwebtext_vocab.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Saved vocabulary comparison to {output}")


if __name__ == "__main__":
    main()
