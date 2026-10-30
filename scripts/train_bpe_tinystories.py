"""Train, serialize, measure, and optionally profile a BPE tokenizer.

Run from the repository root:
    .venv/bin/python scripts/train_bpe_tinystories.py

TinyStories defaults can be overridden for other corpora with --input-path,
--vocab-size, and --output-dir. Use --skip-profile for a training-only run.

The timing run uses four preprocessing workers by default. A separate, serial
cProfile run exposes preprocessing work that a parent-only parallel profile
would miss. Both runs must learn exactly the same vocabulary and ordered merges.
Vocabulary values and merge components are hexadecimal byte strings in JSON.
"""

from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import subprocess
import sys
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import train_bpe


SPECIAL = "<|endoftext|>"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(directory: Path):
    vocab = {int(key): bytes.fromhex(value) for key, value in json.loads((directory / "vocab.json").read_text()).items()}
    merges = [tuple(bytes.fromhex(part) for part in pair) for pair in json.loads((directory / "merges.json").read_text())]
    return vocab, merges


def longest_tokens(vocab: dict[int, bytes], *, exclude_special: bool = False):
    candidates = [(key, value) for key, value in vocab.items() if not exclude_special or value != SPECIAL.encode()]
    longest = max(len(value) for _, value in candidates)
    return [
        {"id": key, "byte_length": len(value), "hex": value.hex(), "bytes_repr": repr(value),
         "text": value.decode("utf-8", errors="replace")}
        for key, value in candidates if len(value) == longest
    ]


def validate_model(vocab, merges, target: int) -> None:
    assert len(vocab) == target, (len(vocab), target)
    assert len(merges) == target - 257
    assert set(vocab) == set(range(target))
    assert all(vocab[index] == bytes([index]) for index in range(256))
    assert vocab[256] == SPECIAL.encode()
    known = set(vocab[index] for index in range(257))
    for index, (first, second) in enumerate(merges, 257):
        assert first in known and second in known
        assert vocab[index] == first + second
        assert SPECIAL.encode() not in (first, second)
        known.add(vocab[index])


def profile_report(profiler: cProfile.Profile, directory: Path):
    profiler.dump_stats(str(directory / "training.prof"))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative").print_stats(30)
    stats.sort_stats("tottime").print_stats(30)
    (directory / "training_profile.txt").write_text(stream.getvalue())
    rows = []
    for (filename, line, function), (primitive, calls, self_time, cumulative, _) in stats.stats.items():
        rows.append({"filename": filename, "line": line, "function": function,
                     "primitive_calls": primitive, "calls": calls,
                     "self_seconds": self_time, "cumulative_seconds": cumulative})
    rows.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
    return {"total_calls": stats.total_calls, "total_profile_seconds": stats.total_tt,
            "functions": rows}


def worker(args) -> None:
    directory = args.output_dir / ("profile_serial" if args.stage == "profile" else "model")
    directory.mkdir(parents=True, exist_ok=True)
    workers = 1 if args.stage == "profile" else args.num_workers
    profiler = cProfile.Profile() if args.stage == "profile" else None
    print(f"Starting {args.stage}: {args.input_path.name}, vocab={args.vocab_size}, workers={workers}", flush=True)
    if profiler:
        profiler.enable()
    started = time.perf_counter()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, [SPECIAL],
                              num_workers=workers, chunk_size=args.chunk_size)
    elapsed = time.perf_counter() - started
    if profiler:
        profiler.disable()
    print(f"Finished {args.stage}: {elapsed:.3f} s; {len(vocab):,} tokens, {len(merges):,} merges", flush=True)
    validate_model(vocab, merges, args.vocab_size)
    write_json(directory / "vocab.json", {str(key): value.hex() for key, value in vocab.items()})
    write_json(directory / "merges.json", [[first.hex(), second.hex()] for first, second in merges])
    assert load_model(directory) == (vocab, merges)
    result = {
        "stage": args.stage, "workers": workers, "training_seconds": elapsed,
        "training_hours": elapsed / 3600, "vocabulary_size": len(vocab), "merge_count": len(merges),
        "longest_tokens": longest_tokens(vocab),
        "longest_non_special_tokens": longest_tokens(vocab, exclude_special=True),
        "serialization_roundtrip_verified": True,
        "self_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    }
    if profiler:
        result["profile"] = profile_report(profiler, directory)
    write_json(directory / "measurement.json", result)


def measure_stage(args, stage: str):
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
               "--input-path", str(args.input_path), "--output-dir", str(args.output_dir),
               "--num-workers", str(args.num_workers), "--chunk-size", str(args.chunk_size),
               "--vocab-size", str(args.vocab_size)]
    child = subprocess.Popen(command, cwd=ROOT)
    process = psutil.Process(child.pid)
    peak_rss = peak_processes = samples = 0
    last_progress = time.monotonic()
    started = last_progress
    while child.poll() is None:
        rss = count = 0
        try:
            tree = [process, *process.children(recursive=True)]
        except psutil.NoSuchProcess:
            tree = []
        for member in tree:
            try:
                rss += member.memory_info().rss
                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        peak_rss = max(peak_rss, rss)
        peak_processes = max(peak_processes, count)
        samples += 1
        if time.monotonic() - last_progress >= 20:
            print(f"{stage}: {time.monotonic() - started:.0f} s elapsed; "
                  f"current RSS sum {rss / 2**20:.1f} MiB; peak {peak_rss / 2**20:.1f} MiB", flush=True)
            last_progress = time.monotonic()
        time.sleep(0.05)
    if child.returncode:
        raise subprocess.CalledProcessError(child.returncode, command)
    directory = args.output_dir / ("profile_serial" if stage == "profile" else "model")
    result = json.loads((directory / "measurement.json").read_text())
    result.update({"sampled_peak_process_tree_rss_bytes": peak_rss,
                   "sampled_peak_process_count": peak_processes, "memory_samples": samples,
                   "process_lifetime_seconds": time.monotonic() - started})
    write_json(directory / "measurement.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=ROOT / "data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tokenizers/tinystories")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target vocabulary size, including special tokens")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--stage", choices=("timing", "profile"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.input_path = args.input_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.num_workers < 1 or args.chunk_size < 1 or args.vocab_size < 257:
        parser.error("workers/chunk size must be positive and vocabulary size must be at least 257")
    if args.stage:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "cs336_basics/bpe_tokenizer.py"
    (args.output_dir / "trainer_source.py").write_bytes(source.read_bytes())
    print("Hashing input before timing (this can warm filesystem caches).", flush=True)
    summary = {
        "complete": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(args.input_path), "bytes": args.input_path.stat().st_size,
                  "sha256": sha256_file(args.input_path)},
        "trainer_sha256": sha256_file(source),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "logical_cpus": os.cpu_count(), "physical_memory_bytes": psutil.virtual_memory().total},
        "configuration": {"vocab_size": args.vocab_size, "special_tokens": [SPECIAL],
                          "num_workers": args.num_workers, "chunk_size": args.chunk_size},
        "serialization": {"encoding": "hex", "vocab.json": "JSON object: decimal token ID string -> hexadecimal bytes",
                          "merges.json": "JSON array in learning order: [left bytes as hex, right bytes as hex]"},
        "method": {"timing": "One unprofiled training call in a fresh subprocess; imports, validation and serialization excluded.",
                   "memory": "Approximate peak sum of trainer and descendant RSS, sampled every 50 ms over subprocess lifetime; supervisor excluded. Shared pages may be counted more than once and brief peaks may be missed.",
                   "profile": ("Not run (--skip-profile)." if args.skip_profile else
                               "Separate full-input serial cProfile run, so preprocessing CPU work is visible; not used as production timing."),
                   "cache": "Input hashed before timing; no cold-cache claim."},
    }
    result_path = args.output_dir / "results.json"
    write_json(result_path, summary)
    summary["timing"] = measure_stage(args, "timing")
    write_json(result_path, summary)
    if not args.skip_profile:
        summary["serial_profile"] = measure_stage(args, "profile")
        assert load_model(args.output_dir / "model") == load_model(args.output_dir / "profile_serial")
        summary["parallel_serial_outputs_identical"] = True
    summary["complete"] = True
    write_json(result_path, summary)
    print(f"Saved tokenizer and measurements to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
