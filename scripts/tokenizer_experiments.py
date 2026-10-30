"""Run assignment 2.7 using the project's trained byte-level BPE tokenizers.

    .venv/bin/python scripts/tokenizer_experiments.py --phase measure
    .venv/bin/python scripts/tokenizer_experiments.py --phase encode --workers 4 --resume

Small measurements live under artifacts/tokenizer_experiments. Full uint16
arrays live under data/tokenized, which is excluded from version control.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import _file_ranges

SPECIAL = "<|endoftext|>"
DELIMITER = SPECIAL.encode("utf-8")
HEADER_BYTES = 256
DATASETS = {
    "tinystories_train": ("TinyStoriesV2-GPT4-train.txt", "tinystories"),
    "tinystories_valid": ("TinyStoriesV2-GPT4-valid.txt", "tinystories"),
    "owt_train": ("owt_train.txt", "openwebtext"),
    "owt_valid": ("owt_valid.txt", "openwebtext"),
}
_WORKER_TOKENIZER = None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_paths(models_dir: Path, name: str) -> tuple[Path, Path]:
    directory = models_dir / name / "model"
    return directory / "vocab.json", directory / "merges.json"


def load_tokenizer(paths: tuple[Path, Path]):
    from cs336_basics.tokenizer import Tokenizer

    return Tokenizer.from_files(*paths, special_tokens=[SPECIAL])


def iter_documents(path: Path, read_size: int = 4 * 1024 * 1024):
    """Yield raw document bodies, preserving whitespace and omitting delimiters."""
    pending = b""
    index = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(read_size), b""):
            parts = (pending + block).split(DELIMITER)
            for body in parts[:-1]:
                yield index, body
                index += 1
            pending = parts[-1]
    if pending:
        yield index, pending


def sample_documents(path: Path, count: int, seed: int):
    """Reservoir sampling gives every non-whitespace document equal probability."""
    rng = random.Random(seed)
    sample = []
    eligible = 0
    for index, body in iter_documents(path):
        if not body.decode("utf-8").strip():
            continue
        eligible += 1
        item = (index, body)
        if len(sample) < count:
            sample.append(item)
        else:
            replacement = rng.randrange(eligible)
            if replacement < count:
                sample[replacement] = item
    if eligible < count:
        raise ValueError(f"Only {eligible} nonempty documents in {path}; requested {count}")
    return sorted(sample), eligible


def encode_sample(tokenizer, sample):
    documents = []
    for index, body in sample:
        text = body.decode("utf-8")
        ids = tokenizer.encode(text)
        if tokenizer.decode(ids).encode("utf-8") != body:
            raise AssertionError(f"Sample document {index} failed exact round trip")
        documents.append({
            "document_index": index, "utf8_bytes": len(body), "tokens": len(ids),
            "sha256": hashlib.sha256(body).hexdigest(), "text": text, "ids": ids,
        })
    byte_count = sum(document["utf8_bytes"] for document in documents)
    token_count = sum(document["tokens"] for document in documents)
    return {"documents": documents, "total_utf8_bytes": byte_count, "total_tokens": token_count,
            "bytes_per_token": byte_count / token_count, "roundtrip_verified": True}


def benchmark_encoding(paths, input_path: Path, target_bytes: int, repeats: int):
    start, end = next(_file_ranges(input_path, (SPECIAL,), target_bytes))
    if start != 0:
        raise AssertionError("Benchmark prefix must start at the beginning of the file")
    with input_path.open("rb") as source:
        raw = source.read(end)
    text = raw.decode("utf-8")
    samples = []
    reference_ids = None
    for repetition in range(repeats):
        tokenizer = load_tokenizer(paths)
        started = time.perf_counter()
        ids = tokenizer.encode(text)
        elapsed = time.perf_counter() - started
        if tokenizer.decode(ids).encode("utf-8") != raw:
            raise AssertionError("Benchmark failed exact round trip")
        if reference_ids is not None and reference_ids != ids:
            raise AssertionError("Repeated benchmark token IDs differ")
        reference_ids = ids if reference_ids is None else reference_ids
        samples.append(elapsed)
        print(f"Benchmark {input_path.name}, run {repetition + 1}: {elapsed:.3f} s", flush=True)
    median = statistics.median(samples)
    rate = len(raw) / median
    return {
        "input_path": str(input_path), "start_offset": start, "end_offset": end,
        "utf8_bytes": len(raw), "tokens": len(reference_ids),
        "sha256": hashlib.sha256(raw).hexdigest(), "seconds": samples,
        "median_seconds": median, "bytes_per_second": rate,
        "pile_bytes": 825_000_000_000, "pile_estimated_hours": 825_000_000_000 / rate / 3600,
        "method": "Single-process encode only on a document-aligned validation prefix; fresh tokenizer and empty cache each repetition. Model loading, disk I/O, and round-trip verification excluded.",
    }


def measure(args):
    samples = {}
    populations = {}
    for name, filename in (("tinystories", "TinyStoriesV2-GPT4-valid.txt"),
                           ("openwebtext", "owt_valid.txt")):
        path = args.data_dir / filename
        selected, eligible = sample_documents(path, args.sample_size, args.seed)
        samples[name] = selected
        populations[name] = {"input_path": str(path), "eligible_documents": eligible,
                             "input_sha256": sha256_file(path)}
    tiny_paths = model_paths(args.models_dir, "tinystories")
    owt_paths = model_paths(args.models_dir, "openwebtext")
    rows = {
        "tinystories_with_tinystories": encode_sample(load_tokenizer(tiny_paths), samples["tinystories"]),
        "openwebtext_with_openwebtext": encode_sample(load_tokenizer(owt_paths), samples["openwebtext"]),
        "openwebtext_with_tinystories": encode_sample(load_tokenizer(tiny_paths), samples["openwebtext"]),
    }
    throughput = {
        name: benchmark_encoding(model_paths(args.models_dir, name), args.data_dir / filename,
                                 args.benchmark_bytes, args.repeats)
        for name, filename in (("tinystories", "TinyStoriesV2-GPT4-valid.txt"),
                               ("openwebtext", "owt_valid.txt"))
    }
    return {
        "seed": args.seed, "documents_per_corpus": args.sample_size,
        "sampling_method": "Uniform reservoir sample of non-whitespace validation documents, split at literal <|endoftext|>. Document-body whitespace is retained; delimiter bytes and IDs are excluded from samples only.",
        "compression_method": "Sum of UTF-8 bytes divided by sum of encoded token counts; not the mean of per-document ratios.",
        "populations": populations, "samples": rows, "throughput": throughput,
    }


def initialize_worker(vocab_path: str, merges_path: str):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_tokenizer((Path(vocab_path), Path(merges_path)))


def encode_range(job):
    path, start, end = job
    with open(path, "rb") as source:
        source.seek(start)
        raw = source.read(end - start)
    if len(raw) != end - start:
        raise AssertionError("Short input read")
    ids = _WORKER_TOKENIZER.encode(raw.decode("utf-8"))
    minimum, maximum = (min(ids), max(ids)) if ids else (0, 0)
    if minimum < 0 or maximum >= 65536:
        raise ValueError("Token ID cannot be represented by uint16")
    if b"".join(_WORKER_TOKENIZER.vocab[token_id] for token_id in ids) != raw:
        raise AssertionError(f"Full-corpus round trip failed at offsets {start}:{end}")
    array = np.asarray(ids, dtype="<u2")
    special_id = next(index for index, token in _WORKER_TOKENIZER.vocab.items() if token == DELIMITER)
    special_count = int(np.count_nonzero(array == special_id))
    if special_count != raw.count(DELIMITER):
        raise AssertionError("Document boundary count changed during encoding")
    return array, {"start": start, "end": end, "utf8_bytes": len(raw), "tokens": len(ids),
                   "maximum_id": maximum, "special_tokens": special_count,
                   "input_sha256": hashlib.sha256(raw).hexdigest()}


def write_npy_header(destination, count: int):
    """Reserve a constant 256-byte NumPy v2 header so final shape can be patched."""
    header = repr({"descr": "<u2", "fortran_order": False, "shape": (count,)}).encode("ascii")
    padding = HEADER_BYTES - 12 - len(header) - 1
    if padding < 0:
        raise ValueError("Array shape does not fit the reserved header")
    destination.seek(0)
    destination.write(np.lib.format.magic(2, 0))
    destination.write(struct.pack("<I", HEADER_BYTES - 12))
    destination.write(header + b" " * padding + b"\n")


def dataset_identity(path: Path, paths, encoder_hash: str):
    stat = path.stat()
    return {
        "input_path": str(path), "input_bytes": stat.st_size, "input_mtime_ns": stat.st_mtime_ns,
        "input_sha256": sha256_file(path), "encoder_sha256": encoder_hash,
        "pretokenizer_sha256": sha256_file(ROOT / "cs336_basics/bpe_tokenizer.py"),
        "vocab_sha256": sha256_file(paths[0]), "merges_sha256": sha256_file(paths[1]),
    }


def encode_dataset(args, name: str, encoder_hash: str, previous=None):
    filename, model = DATASETS[name]
    path = args.data_dir / filename
    paths = model_paths(args.models_dir, model)
    print(f"Checking input and model hashes for {name}.", flush=True)
    identity = dataset_identity(path, paths, encoder_hash)
    output = args.tokenized_dir / f"{name}.npy"
    if args.resume and previous and previous.get("complete") and previous.get("identity") == identity:
        if output.is_file() and sha256_file(output) == previous.get("output_sha256"):
            print(f"Verified existing complete array: {output}", flush=True)
            return previous
    args.tokenized_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npy.partial")
    jobs = ((str(path), start, end) for start, end in _file_ranges(path, (SPECIAL,), args.chunk_size))
    chunks = []
    next_offset = total_tokens = special_tokens = max_id = 0
    started = last_progress = time.perf_counter()
    print(f"Encoding {name}: {identity['input_bytes']:,} bytes, {args.workers} workers", flush=True)
    with temporary.open("w+b") as destination:
        write_npy_header(destination, 0)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                 initargs=tuple(str(p) for p in paths)) as pool:
            pending = deque()
            for _ in range(2 * args.workers):
                job = next(jobs, None)
                if job is None:
                    break
                pending.append(pool.submit(encode_range, job))
            while pending:
                array, record = pending.popleft().result()
                if record["start"] != next_offset or record["end"] - record["start"] != record["utf8_bytes"]:
                    raise AssertionError("Input ranges were reordered, omitted, or duplicated")
                next_offset = record["end"]
                total_tokens += len(array)
                special_tokens += record["special_tokens"]
                max_id = max(max_id, record["maximum_id"])
                array.tofile(destination)
                chunks.append(record)
                if time.perf_counter() - last_progress >= 20:
                    elapsed = time.perf_counter() - started
                    print(f"{name}: {next_offset / identity['input_bytes']:.1%}, {elapsed:.1f} s, "
                          f"{total_tokens:,} tokens", flush=True)
                    last_progress = time.perf_counter()
                job = next(jobs, None)
                if job is not None:
                    pending.append(pool.submit(encode_range, job))
        if next_offset != identity["input_bytes"]:
            raise AssertionError("Not all source bytes were encoded")
        if path.stat().st_size != identity["input_bytes"] or path.stat().st_mtime_ns != identity["input_mtime_ns"]:
            raise RuntimeError("Input file changed during encoding")
        write_npy_header(destination, total_tokens)
        destination.flush()
    elapsed = time.perf_counter() - started
    array = np.load(temporary, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype("<u2") or array.shape != (total_tokens,):
        raise AssertionError("Saved NumPy shape or dtype is incorrect")
    if temporary.stat().st_size != HEADER_BYTES + total_tokens * 2:
        raise AssertionError("Saved NumPy payload size is incorrect")
    del array
    temporary.replace(output)
    result = {
        "complete": True, "identity": identity, "tokenizer": model,
        "output_path": str(output), "output_sha256": sha256_file(output),
        "dtype": "uint16", "byte_order": "little", "shape": [total_tokens],
        "token_count": total_tokens, "maximum_token_id": max_id,
        "special_token_count": special_tokens, "source_bytes_verified": next_offset,
        "output_bytes": output.stat().st_size, "npy_header_bytes": HEADER_BYTES,
        "elapsed_seconds": elapsed, "bytes_per_second": identity["input_bytes"] / elapsed,
        "workers": args.workers, "chunk_size_target": args.chunk_size,
        "verification": "Every chunk reconstructed byte-for-byte before casting; ID bounds checked before uint16 conversion; contiguous ordered ranges cover the full source; special-token counts preserved; saved NumPy header, shape and file size checked.",
        "timing_method": "Full parallel encoding, per-chunk round-trip verification, process startup, IPC and writing included; input/model hashing and final output hashing excluded.",
        "chunks": chunks,
    }
    print(f"Finished {name}: {total_tokens:,} tokens in {elapsed:.2f} s; {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("measure", "encode", "all"), default="all")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "artifacts/tokenizers")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "artifacts/tokenizer_experiments")
    parser.add_argument("--tokenized-dir", type=Path, default=ROOT / "data/tokenized")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--benchmark-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.chunk_size, args.sample_size, args.repeats, args.benchmark_bytes) < 1:
        parser.error("Worker, chunk, sample, repeat, and benchmark sizes must be positive")
    for key in ("data_dir", "models_dir", "results_dir", "tokenized_dir"):
        setattr(args, key, getattr(args, key).resolve())
    encoder_hash = sha256_file(ROOT / "cs336_basics/tokenizer.py")
    pretokenizer_hash = sha256_file(ROOT / "cs336_basics/bpe_tokenizer.py")
    model_hashes = {name: {path.name: sha256_file(path) for path in model_paths(args.models_dir, name)}
                    for name in ("tinystories", "openwebtext")}
    locations = {key: str(getattr(args, key)) for key in ("data_dir", "models_dir", "tokenized_dir")}
    result_path = args.results_dir / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    if (results.get("encoder_sha256") not in (None, encoder_hash)
            or results.get("pretokenizer_sha256") not in (None, pretokenizer_hash)
            or results.get("model_hashes", model_hashes) != model_hashes
            or results.get("locations", locations) != locations):
        results = {}
    results.update({"complete": False, "encoder_sha256": encoder_hash,
                    "pretokenizer_sha256": pretokenizer_hash, "locations": locations,
                    "environment": {"python": sys.version, "platform": platform.platform()},
                    "model_hashes": model_hashes})
    write_json(result_path, results)
    if args.phase in ("measure", "all"):
        results["measurements"] = measure(args)
        write_json(result_path, results)
    if args.phase in ("encode", "all"):
        encoded = results.setdefault("encoded_datasets", {})
        for name in args.datasets:
            encoded[name] = encode_dataset(args, name, encoder_hash, encoded.get(name))
            write_json(result_path, results)
    results["complete"] = "measurements" in results and set(DATASETS) <= set(results.get("encoded_datasets", {}))
    write_json(result_path, results)
    print(f"Saved experiment results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
