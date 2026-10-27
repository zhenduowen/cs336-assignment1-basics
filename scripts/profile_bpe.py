"""Compare naive/current BPE training with wall timing and separate cProfile runs.

Example (run from the repository root):
    .venv/bin/python scripts/profile_bpe.py --baseline cs336_basics/naive_bpe_tokenizer.py --no-heap-ablation

results.json contains metadata and a workloads list. Each workload has timings
(samples_seconds/median_seconds), profiles (raw/text paths, counts and function
rows with cc/nc/tt/ct), and output equality checks. Runs execute sequentially.
"""

import argparse
import cProfile
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import train_bpe as optimized_train


WORKLOADS = [
    ("corpus", "tests/fixtures/corpus.en", 500),
    ("tinystories_5m", "tests/fixtures/tinystories_sample_5M.txt", 1000),
    ("tinystories_valid_10k", "data/TinyStoriesV2-GPT4-valid.txt", 10000),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.train_bpe


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("No-heap ablation source anchor changed: " + repr(old[:100]))
    return source.replace(old, new, 1)


def make_no_heap(source):
    """Retain incremental updates but replace all heap work with a full max scan."""
    source = replace_once(
        source,
        "    heap = [_MergeCandidate(count, pair, vocab) for pair, count in pair_counts.items()]\n"
        "    heapq.heapify(heap)\n",
        "",
    )
    source = replace_once(
        source,
        "        while heap:\n"
        "            candidate = heapq.heappop(heap)\n"
        "            if pair_counts.get(candidate.pair, 0) == candidate.count:\n"
        "                break\n"
        "        else:\n"
        "            break\n\n"
        "        best_pair = candidate.pair\n"
        "        merges.append(candidate.spellings)\n"
        "        merged_bytes = candidate.spellings[0] + candidate.spellings[1]\n",
        "        if not pair_counts:\n"
        "            break\n"
        "        best_pair = max(\n"
        "            pair_counts,\n"
        "            key=lambda pair: (pair_counts[pair], (vocab[pair[0]], vocab[pair[1]])),\n"
        "        )\n"
        "        spellings = (vocab[best_pair[0]], vocab[best_pair[1]])\n"
        "        merges.append(spellings)\n"
        "        merged_bytes = spellings[0] + spellings[1]\n",
    )
    source = replace_once(source, "        changed_pairs: set[Pair] = set()\n", "")
    source = replace_once(source, "                    changed_pairs.add(pair)\n", "")
    source = replace_once(
        source,
        "        # Counts are authoritative. Old entries are discarded lazily when popped;\n"
        "        # an old entry with the same count is equivalent to the current candidate.\n"
        "        for pair in changed_pairs:\n"
        "            count = pair_counts.get(pair, 0)\n"
        "            if count:\n"
        "                heapq.heappush(heap, _MergeCandidate(count, pair, vocab))\n"
        "        if len(heap) > 4 * max(len(pair_counts), 1):\n"
        "            heap = [_MergeCandidate(count, pair, vocab) for pair, count in pair_counts.items()]\n"
        "            heapq.heapify(heap)\n",
        "",
    )
    return source


def save_results(path, results):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_timing(workload, variant, elapsed):
    timing = workload["timings"].setdefault(variant, {"samples_seconds": []})
    timing["samples_seconds"].append(elapsed)
    timing["median_seconds"] = statistics.median(timing["samples_seconds"])


def profile_training(function, kwargs, workload, output_dir, variant):
    stem = f"{workload['name']}_{variant}"
    profile_path = output_dir / f"{stem}.prof"
    text_path = output_dir / f"{stem}.txt"
    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = profiler.runcall(
        function, Path(workload["path"]), workload["vocab_size"], workload["special_tokens"], **kwargs
    )
    elapsed = time.perf_counter() - start
    profiler.dump_stats(profile_path)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stream.write("Sorted by cumulative time (nested rows overlap):\n")
    stats.sort_stats("cumulative").print_stats(25)
    stream.write("\nSorted by self time (excludes time in callees):\n")
    stats.sort_stats("tottime").print_stats(25)
    text_path.write_text(stream.getvalue(), encoding="utf-8")
    functions = [
        {"filename": filename, "line": line, "function": name, "cc": cc, "nc": nc, "tt": tt, "ct": ct}
        for (filename, line, name), (cc, nc, tt, ct, _callers) in stats.stats.items()
    ]
    functions.sort(key=lambda row: row["ct"], reverse=True)
    data = {
        "profile_path": str(profile_path),
        "text_path": str(text_path),
        "wall_seconds": elapsed,
        "total_tt": stats.total_tt,
        "total_calls": stats.total_calls,
        "primitive_calls": stats.prim_calls,
        "functions": functions,
    }
    return data, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Earlier naive trainer Python source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/bpe_profile")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Skip the full TinyStories validation workload")
    parser.add_argument("--workloads", nargs="+", choices=[entry[0] for entry in WORKLOADS])
    parser.add_argument("--no-heap-ablation", action="store_true", help="Also time incremental BPE with max selection")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_path = ROOT / "cs336_basics/bpe_tokenizer.py"
    baseline_source = args.baseline.read_text(encoding="utf-8")
    current_source = current_path.read_text(encoding="utf-8")
    baseline_snapshot = output_dir / "naive_source.py"
    current_snapshot = output_dir / "optimized_source.py"
    baseline_snapshot.write_text(baseline_source, encoding="utf-8")
    current_snapshot.write_text(current_source, encoding="utf-8")
    naive_train = load_source(baseline_snapshot, "bpe_profile_naive")
    variants = {"naive": (naive_train, {}), "optimized": (optimized_train, {"num_workers": 1})}
    ablation = None
    if args.no_heap_ablation:
        no_heap_source = make_no_heap(current_source)
        no_heap_path = output_dir / "no_heap_source.py"
        no_heap_path.write_text(no_heap_source, encoding="utf-8")
        ablation = load_source(no_heap_path, "bpe_profile_no_heap")

    results = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "baseline_original_path": str(args.baseline.resolve()),
            "baseline_snapshot": str(baseline_snapshot),
            "baseline_sha256": sha256_file(baseline_snapshot),
            "optimized_original_path": str(current_path),
            "optimized_snapshot": str(current_snapshot),
            "optimized_sha256": sha256_file(current_snapshot),
            "repeats": args.repeats,
        },
        "methodology": {
            "timing": "perf_counter around train_bpe only; median of alternating-order runs",
            "warmup": "Both trainers run once on the small corpus; each input is hashed before timing",
            "profile": "Separate cProfile runs, both single process; profiler overhead is not production speedup",
            "parallel": "Four-worker wall time only; parent cProfile would omit child-process work",
            "equality": "Exact vocabulary dictionary and ordered merge list comparison after every measured call",
            "memory": "Peak memory is not measured",
        },
        "workloads": [],
    }
    if ablation:
        results["metadata"]["no_heap_snapshot"] = str(no_heap_path)
        results["metadata"]["no_heap_sha256"] = sha256_file(no_heap_path)
    git_source = subprocess.run(
        ["git", "show", "HEAD:cs336_basics/bpe_tokenizer.py"], cwd=ROOT, capture_output=True, check=False
    )
    if git_source.returncode == 0:
        head_source_hash = hashlib.sha256(git_source.stdout).hexdigest()
        results["metadata"]["git_head_trainer_sha256"] = head_source_hash
        results["metadata"]["baseline_matches_git_head"] = head_source_hash == results["metadata"]["baseline_sha256"]
    results_path = output_dir / "results.json"
    save_results(results_path, results)

    print("Warmup: both trainers on corpus.en, vocabulary 500", flush=True)
    small_path = ROOT / WORKLOADS[0][1]
    warm_naive = naive_train(small_path, 500, ["<|endoftext|>"])
    warm_optimized = optimized_train(small_path, 500, ["<|endoftext|>"], num_workers=1)
    assert warm_naive == warm_optimized, "Warmup results differ"
    if ablation:
        assert ablation(small_path, 500, ["<|endoftext|>"], num_workers=1) == warm_naive

    for name, relative_path, vocab_size in WORKLOADS:
        if args.quick and name == "tinystories_valid_10k":
            continue
        if args.workloads and name not in args.workloads:
            continue
        path = ROOT / relative_path
        workload = {
            "name": name,
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "vocab_size": vocab_size,
            "special_tokens": ["<|endoftext|>"],
            "timings": {},
            "profiles": {},
            "checks": {"equal_all_completed_runs": True, "completed_comparisons": 0},
        }
        results["workloads"].append(workload)
        save_results(results_path, results)
        reference = None

        def check(actual):
            nonlocal reference
            if reference is None:
                reference = actual
            else:
                assert actual == reference, f"Vocabulary/merge mismatch for {name}"
                workload["checks"]["completed_comparisons"] += 1
            workload["actual_vocab_size"] = len(actual[0])
            workload["merge_count"] = len(actual[1])

        for repeat in range(args.repeats):
            order = ("naive", "optimized") if repeat % 2 == 0 else ("optimized", "naive")
            for variant in order:
                function, kwargs = variants[variant]
                print(f"[{name}] timing {variant} {repeat + 1}/{args.repeats}: start", flush=True)
                start = time.perf_counter()
                actual = function(path, vocab_size, workload["special_tokens"], **kwargs)
                elapsed = time.perf_counter() - start
                check(actual)
                record_timing(workload, variant, elapsed)
                save_results(results_path, results)
                print(f"[{name}] timing {variant}: {elapsed:.6f}s, exact output match", flush=True)

        for variant, (function, kwargs) in variants.items():
            print(f"[{name}] cProfile {variant}: start", flush=True)
            profile_data, actual = profile_training(function, kwargs, workload, output_dir, variant)
            check(actual)
            workload["profiles"][variant] = profile_data
            save_results(results_path, results)
            print(
                f"[{name}] cProfile {variant}: {profile_data['wall_seconds']:.6f}s, "
                f"{profile_data['total_calls']:,} calls, exact output match",
                flush=True,
            )

        extras = []
        if ablation and name != "corpus":
            # Pair the ablation with a fresh heap control to reduce order and
            # temperature effects after the much longer naive profiling run.
            for repeat in range(args.repeats):
                order = [("optimized_heap_control", optimized_train), ("optimized_no_heap", ablation)]
                if repeat % 2:
                    order.reverse()
                for variant, function in order:
                    print(f"[{name}] timing {variant} {repeat + 1}/{args.repeats}: start", flush=True)
                    start = time.perf_counter()
                    actual = function(path, vocab_size, workload["special_tokens"], num_workers=1)
                    elapsed = time.perf_counter() - start
                    check(actual)
                    record_timing(workload, variant, elapsed)
                    save_results(results_path, results)
                    print(f"[{name}] timing {variant}: {elapsed:.6f}s, exact output match", flush=True)
        if name == "tinystories_valid_10k":
            extras.append(("optimized_4_workers", optimized_train, {"num_workers": 4}))
        for variant, function, kwargs in extras:
            for repeat in range(args.repeats):
                print(f"[{name}] timing {variant} {repeat + 1}/{args.repeats}: start", flush=True)
                start = time.perf_counter()
                actual = function(path, vocab_size, workload["special_tokens"], **kwargs)
                elapsed = time.perf_counter() - start
                check(actual)
                record_timing(workload, variant, elapsed)
                save_results(results_path, results)
                print(f"[{name}] timing {variant}: {elapsed:.6f}s, exact output match", flush=True)
        workload["speedup"] = (
            workload["timings"]["naive"]["median_seconds"] / workload["timings"]["optimized"]["median_seconds"]
        )
        workload["complete"] = True
        save_results(results_path, results)
        print(f"[{name}] complete: median speedup {workload['speedup']:.2f}x", flush=True)

    results["complete"] = True
    save_results(results_path, results)
    print(f"Results saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
