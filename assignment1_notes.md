---
title: Stanford CS336 Assignment 1 - Basics
description: Student version of Assignment 1 for Stanford CS336 - Language Modeling From Scratch
date: 2025-11-14
author: Zhenduo
categories: [Deep Learning, Language Models from Scratch]
tags: [Deep Learning, CUDA, PyTorch, Language Models]
published: true
---
## Preliminaries

- My repository forked from the course: [https://github.com/zhenduowen/cs336-assignment1-basics](https://github.com/zhenduowen/cs336-assignment1-basics)

- <a href="{{ site.baseurl }}/assets/files/notes/cs336_spring2025_assignment1_basics.pdf">The problem descriptions for Assignment 1 can also be found within this Blog.</a>

- For setting up a local environment, [another post]({{ site.baseurl }}/posts/windows-sub-linux/) may be useful.

- This project's dependencies are managed by `uv`. To use `uv` to set up dependencies locally:
    ```terminal
    # Navigate to the cloned repository
    cd your-repo-name

    # Install dependencies from pyproject.toml
    uv sync

    # Activate the virtual environment created by uv
    source .venv/bin/activate  # On Unix/macOS
    # OR
    .venv\Scripts\activate     # On Windows
    ```
`uv sync` creates a local environment for this particular project and then installs dependencies according to `pyproject.toml`.

An additional file, `uv.lock`, records the exact dependency versions that should be installed. This file usually imposes stricter version requirements than `pyproject.toml`.

For further reading about the features of `uv`, see [https://docs.astral.sh/uv/getting-started/features/#projects](https://docs.astral.sh/uv/getting-started/features/#projects).

## Overview of the Assignment

- `/cs336_basics/` is where we write all the implementations.
- `/tests/` includes all the unit tests, and `adapters.py` connects these unit tests with our implementations in `/cs336_basics/`. `pyproject.toml` is the `uv` project file that contains all the dependencies needed for this project. To run the tests, first use `uv` to set up the dependencies, and then run `uv run pytest`. If a corresponding unit has not been implemented, a `NotImplementedError` will be raised.
- Create a `/data/` folder. `TinyStoriesV2-GPT4.txt` (training and validation) can be found [here](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main), and `owt.txt` (training and validation) can be found [here](https://huggingface.co/datasets/stanford-cs336/owt-sample/tree/main). Download the datasets to `/data/`. Add this folder to `.gitignore` so that Git ignores the datasets.

I am following the spring 2025 version of assignment 1.

## Byte-Pair Encoding Tokenizer

### Unicode Standard and UTF-8 Encoding

#### Intro and Example
Unicode Standard gives every character a unique number called **code point**.

```plain
A        → U+0041
中       → U+4E2D
😊       → U+1F60A
```
But it is not practical to train tokenizers on Unicode code points, the vocabulary would be too large (roughly 150K items in it) and sparse (a lot of rare characters). This is why `UTF-8` encoding was introduced:

`UTF-8` is an encoding method on Unicode code points as bytes. The `8` stands for 1 byte which is 8 bits. The Unicode standard itself defines three encodings: `UTF-8`, `UTF-16`, and `UTF-32`. `UTF-8` is the most prevailing encoding for the Internet, used roughly in over 98% of web pages.

```python
In [9]: test_string = "I 我"

In [10]: utf8_encoded = test_string.encode('utf-8')

In [11]: utf8_encoded
Out[11]: b'I \xe6\x88\x91'
```

In `b'I \xe6\x88\x91'`, the prefix `b` means it is a sequence of raw bytes in Python. After UTF-8 encoding:
```plain
I      → 0x49
space  → 0x20
我     → 0xe6 0x88 0x91
```
Since each byte value is an integer range 0 to 255, the number of codepoints are 154997. 

#### prefix-free and self-synchronizing
A code point could take multiple bytes after encoding, like `我 : 0xe6 0x88 0x91`. UTF-8 is **prefix-free** and **self-synchronizing**. Self-synchronizing means that even if one start reading a UTF-8 byte sequence from the middle, one can still recover the correct character boundaries. And prefix-free means that no encoding is the prefix of some other encodings. The following example explains these in details:
```plain
0xxxxxxx   = 1-byte character (for ASCII)
110xxxxx   = start of a 2-byte character
1110xxxx   = start of a 3-byte character
11110xxx   = start of a 4-byte character
10xxxxxx   = continuation byte
```
Observe that starting bytes never start with `10` and continuation bytes always start with `10`. So if the stream is corrupted and we start in the middle, any byte start with `10` is a continuation byte and we can move on, till we find a starting byte.

Note that Python displays printable ASCII directly:
```plain
0x49 → I
0x20 → space
```
The byte `\xe6` means it is a hexdecimal value is `e6`. So the actual UTF-8 encoding is `49 20 e6 88 91`.

The UTF-8 encoding helps us to focus on byte values which only range within 0 to 255. 

Now a quick summary. The advantages of UTF-8 over UTF 16 and UTF 32 are:
- Compact:
  ```plain
    Character    UTF-8 bytes              UTF-16 bytes        UTF-32 bytes

    I            49                       49 00               49 00 00 00
    space        20                       20 00               20 00 00 00
    我           e6 88 91                 11 62               11 62 00 00
  ```
- ACSII-compatible:
  ```plain
  A → 0x41
  B → 0x42
  space → 0x20
  ```
  English text remains readable at the byte level.
- UTF-8 gives a universal vocabulary of size 256.
Also, note that most training data is already stored in UTF-8.

### Subword Tokenization

#### Intro

Although byte-level tokenization can alleviate the issue of out-of-vocabulary, tokenizing texts by character results in extremely long input sequences. For example, a sentence with 10 words may only be 10 tokens in a word-level language model, but could be 50 or more tokens in a character-level model. 

Subword tokenization is designed to balance such a mismatch. For character level tokenization, byte vocabulary has size 256. A subword tokenizer trades off a larger vocabulary size for better compressing the length of the input bytes sequence. For example, it would be good to add a particular token in the vocabulary for the word "the".

To achieve such a scheme, [Sennrich et al [2016]](https://aclanthology.org/P16-1162/) propose to use byte-pair encoding (BPE). This algorithm iteratively *merges the most frequent pair of bytes with a new unused index*. Tokenizers constructed in such a scheme are called **BPE tokenizers**. The process of building the BPE tokenizer vocabulary is call "training" the BPE tokenizer.

#### Training BPE Tokenizers

The training procedure consists of three steps:
- **Initialization** The tokenizer vocabulary is a bijective mapping from bytestring token to integer ID. For byte level BPE tokenizer, the initial vocabulary is simply the set of all 256 bytes. It contains all possible single-byte values. 
  - A byte have 256 possible values: `0x00, 0x01, ..., 0x7F, 0x80, ..., 0xFF` The ASCII compatible part is `0x00 to 0x7F`, which is 128 values contains English letters, digits, punctuations, spaces, newline, and control characters. The remaining byte values `0x80 to 0xFF` are not ASCII characters by themselves. In UTF-8 they are used as part of multi-byte encodings for non-ASCII characters. For example, `我 → e6 88 91`.
  - **Special tokens** Often, some strings like `<|endoftext|>` are used to encode metadata like boundaries between files. These special tokens should never be split into multiple tokens. Special tokens are often added to the vocabulary in the initialization.
- **Pre-tokenization** Going through the entire text corpus each time we want to merge some bytes can be computationally expensive. Also, directly merging tokens across the corpus may result in tokens that differ only in punctuation, as BPE merges frequent adjacent byte pairs purely by frequency, not by semantic meaning. For example, suppose the corpus contains many examples like:
    ```plain
    dog.
    dog!
    dog?
    ```
    If we directly run BPE over the whole byte stream, the algorithm may see these frequent adjacent pairs:
    ```plain
    d o
    do g
    dog .
    dog !
    dog ?
    ```
    These are different byte strings, so they receive different token IDs:
    ```plain
    "dog." → token ID 1057
    "dog!" → token ID 3092
    "dog?" → token ID 8271
    ```
    To avoid this, we *pre-tokenize* the corpus by manually splitting the text into meaningful pieces. A naive approach can be splitting by spaces, like `I love dogs!` to `["I", " love", "dogs", "!"]`. 
    
    Pre-tokenization also improves efficiency in a simple way. Suppose the word `dogs` appear 10 times, we can directly add character pair frequence with `(d,o) += 10, (o,g) += 10, (g,s) += 10`, which makes character pairs counting more efficient.

    In GPT-2 (and in this assignment), regex-based pre-tokenizer is applied. See [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) from `github.com/openai/tiktoken/pull/234/files:`
    ```python
    # split by contractions | words | numbers | punctuation | trailing spaces | other spaces
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    ```

- **Compute BPE Merges** After splitting text into pre-tokens, each pre-token is represented as a sequence of UTF-8 Bytes. The BPE Algorithm then iteratively counts every pair of bytes, merges the most frequent pairs, and add to the vocabulary.

    - We **do not consider pairs across the pre-token boundaries**. If we broke `dog.` into `dog` + `.` in pre-tokenization, then such a scheme means that `g` will never merge with the `.`. And when `dog` appear enough times, this word will be added to the vocabulary.

### Experimenting with BPE Tokenizer Training

Now we train a byte-level BPE tokenizer on the [TinySories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). For local tests, use the validation set for debugging.

#### General suggestions for implementation
Before jumping into implementation, a few things to keep in mind:
- **Profiling** Use profiling tools to identify the bottlenecks in our implementation. For this assignment, built-in deterministic profiler `cProfile` is recommended.
- **Parallelizing pre-tokenization** A major bottleneck is the pre-tokenization step. We can chuck the corpus by the location of the special token indicating end of the document (the chunking is always valid, as we never want to merge across document boundaries), and use the built-in library `multiprocessing`. The chunking does not need to be perfect and there will be edge cases of receiving a very large corpus that does not contain `<|endoftext|>`.
- **Remove special token before pre-tokenization** Before running pre-tokenization with the regex pattern, we should strip out all special tokens from the corpus. For example, if you have a corpus (or chunk) like `[Doc 1]<|endoftext|>[Doc 2]`, you should split on the special token `<|endoftext|>`, and pre-tokenize `[Doc 1]` and `[Doc 2]` separately, so that no merging can occur across the document boundary.
    > A naive idea to split by `<|endoftext|>` is to use `parts = re.split("<|endoftext|>", text)`. But this is **wrong** because in regex, `|` means OR. The regex pattern `<|endoftext|>` means (roughly) `< OR endoftext OR >`, it says here may split on `<`, `endoftext`, or `>` separately. To treat the special token literally, use `re.escape`:

    ```python
    import re

    text = "[Doc 1]<|endoftext|>[Doc 2]"
    special_tokens = ["<|endoftext|>", "<|pad|>"]

    escaped_tokens = [re.escape(tok) for tok in special_tokens] # | is escaped
    delimiter = "|".join(escaped_tokens) # here "|" means OR, combine multiple escaped tokens into one regex pattern.

    parts = re.split(delimiter, text)
    print(parts)
    ```
- **Adaptive merging** A naive implementation of BPE training iterates over all byte pairs to identify the most frequent pair. Note that the only pair counts changes after each merge are those that overlap with the merged pair. This is also why this merging step cannot be easily parallelized: Thread 2's work is depended on the result of Thread 1's work.
  > A remark here is that in CPython, the standard Python implementation, there is **Global Interpreter Lock(GIL)**. It means that, within one Python process, only one thread at a time is allowed to execute Python bytecode. Conceptually, even if we create four threads, the GIL means that they take turns to executing Python bytecode.

#### Python implementation details

The implementation in `cs336_basics/bpe_tokenizer.py` separates training into two stages: **count pre-tokens**, then **learn merges from their weighted token sequences**. Pre-tokenization can run in independent worker processes. The merge decisions remain sequential because each merge changes the frequencies used to choose the next one.

The original implementation already counted repeated pre-tokens, but it still recounted all pairs and rewrote all unique pre-tokens after every merge. The main improvement is to keep an index of where each pair occurs, so we only revisit the pre-tokens affected by the selected merge.

**1. Vocabulary initialization and integer token IDs**

We start with the 256 possible byte values:

```python
vocab = {i: bytes([i]) for i in range(256)}
token_ids = {token: token_id for token_id, token in vocab.items()}
```

`vocab` maps an integer ID to its byte spelling; `token_ids` provides the reverse lookup. Special-token spellings not already present are then appended in the order supplied by the caller. They count toward `vocab_size`, so with one new special token and a target vocabulary of 10,000, we can learn at most `10000 - 256 - 1 = 9743` merges. Training can stop earlier if no adjacent pairs remain.

Internally, each pre-token is a list of integer token IDs. For example, `"the"` initially becomes `[116, 104, 101]`. If a merge assigns ID 257 to `b"th"`, its representation becomes `[257, 101]`. The byte spelling is stored in `vocab[257]`; each occurrence of the new token only needs its ID.

This avoids building and concatenating byte strings throughout the word sequences. When a merge is selected, we construct its combined byte spelling once. The reverse map also lets the internal representation reuse an ID if that byte spelling already exists. The returned vocabulary and merge list still use the byte-based format required by the assignment.

**2. Count strings before converting them to bytes**

`_count_pretokens()` first returns a `Counter[str]`, conceptually:

```python
Counter({" the": 100000, " cat": 8000, "!": 3000})
```

If `" the"` occurs 100,000 times, we count those occurrences while scanning the corpus, but only UTF-8 encode that distinct string once when preparing the merge data structures. The original code constructed a tuple of single-byte objects for every occurrence before adding it to its counter.

Special tokens are matched before applying the GPT-2 pre-tokenization pattern. A separate matching list is deduplicated and sorted longest-first: if both `<e>` and `<e>long` are special tokens, the longer one must be recognized as a whole. This sorting does not change the vocabulary IDs assigned during initialization.

The worker uses compiled patterns and `finditer()`. For each ordinary text region between special-token matches, it passes the region's start and end positions to the pre-tokenization regex. This avoids materializing a list of document substrings with `re.split()`. The generator passed to `Counter.update()` also avoids building a list of all matches.

After combining the counts, `train_bpe()` consumes the counter with `popitem()` while converting each distinct string to an integer list. Consumed string keys can be released as this conversion proceeds, and the counter itself is deleted afterward.

**3. Separate weighted frequencies from pair membership**

The merge stage uses the following data structures. Here, a “word” means a regex pre-token, which may include a leading space or consist entirely of punctuation.

| Structure | Meaning |
| --- | --- |
| `words[word_id]` | Current integer token sequence for one distinct original pre-token |
| `frequencies[word_id]` | Number of occurrences of that pre-token in the corpus |
| `pair_counts[(a, b)]` | Total weighted count of adjacent token IDs `a, b` |
| `pair_words[(a, b)]` | Set of word IDs whose current sequence contains `a, b` |
| `heap` | Candidates for the next merge, including some outdated entries |

The word ID and its frequency remain fixed during training; only the word's token sequence changes.

It is important to distinguish **how often a pair occurs** from **which words contain it**. Suppose the pre-token `aaaa` occurs five times:

```plain
Current sequence:  a a a a
Adjacent pairs:   (a,a), (a,a), (a,a)

Contribution to pair_counts[(a,a)]: 3 × 5 = 15
Membership in pair_words[(a,a)]:   one word ID
```

The global count includes all three adjacency positions and their corpus frequency. The set stores that word ID only once, because we only need to update its stored sequence once per merge.

These structures are initially built in one pass over the unique pre-token sequences. `defaultdict(int)` supplies a zero count for a new pair, and `defaultdict(set)` supplies an empty membership set.

**4. Update only affected words**

Once the best pair has been selected, we obtain the affected words directly:

```python
for word_id in tuple(pair_words[best_pair]):
    ...
```

The `tuple(...)` takes a snapshot of the membership set. This matters because we update the index while processing those words; iterating directly over a changing set would be unsafe.

For each affected word, we count its adjacent pairs before and after applying `_merge_word()`. For every pair appearing in either version, we compute:

```python
delta = (after - before) * frequency
```

For a simple example, suppose a pre-token occurs ten times and we merge `a b`:

```plain
Before: x a b y
After:  x  ab y

(x,a), (a,b), (b,y): each decreases by 10
(x,ab), (ab,y):      each increases by 10
```

Counts elsewhere in the corpus remain unchanged. We remove a pair from `pair_counts` when its global count reaches zero. In `pair_words`, we remove a word ID when that word no longer contains the pair, and add it when a new pair appears in the word.

`_merge_word()` scans left-to-right and advances by two positions after a replacement. This preserves non-overlapping merge semantics:

```plain
Before: a a a a       three (a,a) adjacencies
After:  aa aa         two replacements, one (aa,aa) adjacency
```

The old/new adjacency counters handle this overlap correctly. Counting three adjacent pairs does not mean we can make three replacements.

This implementation still scans the complete sequences of the affected words to compute their old and new counters. It does not maintain a linked list or an index of individual pair positions. The improvement comes from skipping every unaffected word, while keeping the update logic relatively easy to verify.

**5. Use a heap while preserving the exact tie-breaking rule**

The original `max(pair_counts.items(), ...)` searched every distinct pair after each merge. We now initialize a heap with `heapq.heapify()`, then push candidates only for pairs whose counts changed.

Python's `heapq` returns the smallest item. The `_MergeCandidate` class reverses the comparison so that the desired candidate comes first:

```python
def __lt__(self, other):
    if self.count != other.count:
        return self.count > other.count
    return self.spellings > other.spellings
```

Here, `spellings` is the tuple `(vocab[a], vocab[b])`. We prioritize the largest frequency, then the lexicographically largest **pair of byte spellings**. Newly assigned numeric IDs do not follow lexical order, so comparing IDs would change the learned merges on ties. We also compare the two spellings as a pair, rather than comparing only their concatenation.

The class declares `__slots__ = ("count", "pair", "spellings")`. These are the only fields needed by a candidate, so Python does not need a separate instance attribute dictionary for each object.

A pair's priority changes whenever its frequency changes. Instead of locating an existing object inside the heap and modifying it, we push a fresh candidate and leave the old one in place. When popping, we validate its saved count:

```python
if pair_counts.get(candidate.pair, 0) == candidate.count:
    # This candidate has a current priority.
    ...
```

This is **lazy invalidation**. The dictionary is the authoritative source of counts; outdated heap entries are discarded when encountered. This version does not use version numbers. An older entry whose count happens to equal the current count is still safe: it represents the same pair and priority, and we retrieve affected words from the current index.

Two details keep this bookkeeping manageable:

- `changed_pairs` is a set, so a pair changed by several words receives only one new heap entry after the iteration's updates are complete.
- If the heap grows beyond `4 * max(len(pair_counts), 1)`, we rebuild it from the active pair counts. This removes accumulated stale candidates.

**6. Find safe file ranges for preprocessing**

`_file_ranges()` scans the file in binary mode and yields byte ranges whose internal boundaries fall immediately after complete special tokens. The final range may end at EOF. A range can contain several documents. An internal boundary is emitted once the range has reached the requested `chunk_size` and a suitable delimiter has been found.

The scanner must handle delimiters split across its own reads. For example:

```plain
Read 1 ends with:   ...<|endof
Read 2 starts with: text|>...
```

Let `L` be the maximum encoded delimiter length. While more input may follow, the scanner only accepts matches starting before:

```python
safe_end = max(0, len(buffer) - L + 1)
```

Starts at or beyond this point may need more lookahead to distinguish a short delimiter from a longer one with the same prefix. The undecidable suffix is carried into the next read. If an accepted match extends into that suffix, the scanner advances beyond the complete match so it cannot be processed twice. At EOF, all remaining starts can be considered.

Matching all escaped special tokens in one longest-first alternation preserves the same left-to-right interpretation as pre-tokenization. Complete delimiters also give safe UTF-8 boundaries, so workers can decode their ranges strictly without dropping partial characters.

Workers normalize CRLF and bare CR to LF after decoding. This preserves the original trainer's text-mode newline behavior. If a special token itself contains a newline, we conservatively use one whole-file job, since matching normalized text and matching raw byte offsets can otherwise disagree.

There are two important memory limits to keep in mind. `chunk_size` is a target, not a strict ceiling: a document longer than the target remains together, and a file with no usable delimiters is processed as one job. Also, the global counters and pair index still grow with the number and lengths of distinct pre-tokens.

**7. Parallelize preprocessing with bounded pending work**

`_pretokenize()` runs sequentially when `num_workers=1`, which is the default. For larger inputs, we can use `ProcessPoolExecutor`:

```python
from cs336_basics.bpe_tokenizer import train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_workers=4,
        chunk_size=4 * 1024 * 1024,
    )
```

The main guard lets worker processes safely import the module when using a process start method such as `spawn`.

Each job contains the file path, start/end offsets, and special tokens. Workers read their own ranges and return aggregate pre-token counters. We avoid transferring large corpus strings or lists of individual matches between processes.

The parent keeps at most `2 * num_workers` submitted jobs in a `deque`. It consumes a result, adds its counts to the global counter, and submits another job. This bounds outstanding work, including completed results waiting to be combined, instead of queuing results for the entire corpus.

Processes allow the Python counting work to use multiple cores in ordinary GIL-enabled CPython. They also incur startup, serialization, and counter-combination costs, which is why a single worker remains useful for small inputs. Range discovery generally reads the corpus once and the workers read their ranges again: we trade additional I/O for exact boundaries and controlled buffering.

After preprocessing, one process performs the merge loop. Training separate BPE models on different chunks and combining their merge lists would not produce the globally most frequent-pair sequence required here.

**8. Complexity**

At merge iteration `t`, let `L_t` be the total token length of all stored unique pre-tokens, and `P_t` the number of active distinct pairs. The original implementation performs roughly `O(L_t + P_t)` work per iteration: it recounts pairs, selects the maximum, and rewrites all words.

In the new implementation, let `A_t` be the total token length of just the affected words. Recounting and rewriting now scale with `A_t`. Candidate pushes and pops cost logarithmic time in the current heap size, and occasional heap rebuilds are linear in the number of active pairs. Stale entries add extra pops. These estimates treat dictionary operations as constant-time on average and omit the cost of comparing byte spellings on ties.

This is a reduction in unnecessary work, rather than a guarantee that the entire trainer has a particular `O(N log N)` bound. Some merges affect many words, and the inverted index and heap use additional memory.

#### Profiling comparison: naive and optimized trainers

A simple implementation is preserved in `cs336_basics/naive_bpe_tokenizer.py`. The optimized implementation is in `cs336_basics/bpe_tokenizer.py`; both expose `train_bpe()` with the same vocabulary and merge-list return format.

The naive implementation already aggregates repeated pre-tokens. Its main inefficiency is that every merge still calls `get_pair_counts()` over all unique pre-tokens, scans all pair counts with `max()`, and calls `merge_pair()` on every unique pre-token. Profiling lets us check whether removing this repeated work improves actual runtime.

**Measurement method**

The comparison was run on Python 3.12.12 under Linux/WSL2, with 16 logical CPUs visible. For each workload, we used the same input file, vocabulary size, and `["<|endoftext|>"]` special-token list. The direct naive-versus-optimized comparison used one preprocessing worker.

We timed only the training call with `time.perf_counter()`, ran each version three times in alternating order, and reported the median. Imports and output comparisons were outside the timed region. Both trainers received a small warmup, and each input was read to compute its hash before timing, so these measurements reflect warmed filesystem caches rather than cold-disk performance.

Separate `cProfile` runs recorded function calls and hotspots. Every timed and profiled run, including the heap ablation and parallel runs, produced the same vocabulary dictionary and ordered merge list on these inputs. Source snapshots, hashes, raw timing samples, and profile statistics are saved under `benchmarks/bpe_profile/`.

**Timing without profiling**

| Input | Target vocabulary | Naive median | Optimized median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `tests/fixtures/corpus.en` (~130 KiB) | 500 | 1.140 s | 0.080 s | 14.3× |
| `tests/fixtures/tinystories_sample_5M.txt` (5 MiB) | 1,000 | 6.156 s | 0.393 s | 15.7× |
| `data/TinyStoriesV2-GPT4-valid.txt` (21.46 MiB) | 10,000 | 80.740 s | 1.473 s | 54.8× |

Speedups are calculated from the unrounded medians. These results show a larger gain on the validation workload, which learns 9,743 merges. Repeatedly visiting all unique pre-tokens accumulates substantial work over thousands of merge iterations, while the optimized version visits only the words affected by each selected pair. Both corpus size and vocabulary size differ between these workloads, so this table is not an isolated experiment on vocabulary size.

**What cProfile shows**

On the validation file with a 10,000-token vocabulary, the profiles recorded:

| Metric | Naive | Optimized |
| --- | ---: | ---: |
| Total function calls | 1,682,838,513 | 14,492,725 |
| Word-merge calls: `merge_pair()` / `_merge_word()` | 127,740,473 | 68,766 |
| `str.encode()` calls | 5,419,002 | 13,114 |
| Global `get_pair_counts()` calls | 9,743 | 0 |
| Elapsed time with cProfile enabled | 239.487 s | 3.185 s |

The affected-word index reduces word-merge calls by about **1,858×**. Counting pre-token strings before converting them reduces encoding calls by about **413×**. The encoding counts include setup calls as well as pre-token encoding. Zero global recounts means that `get_pair_counts()` has been eliminated; the optimized trainer still constructs old and new adjacency counters for each affected word.

The relevant cumulative times make the shift in work clearer:

| Function or stage | Naive cumulative time | Optimized cumulative time |
| --- | ---: | ---: |
| Word merging | 117.521 s | 0.072 s |
| Global pair recounting | 53.152 s | Removed; local updates remain |
| Optimized `_pretokenize()` | Not separately factored into a function | 2.394 s |

Word merging and global recounting together account for approximately **71%** of the naive profiled runtime. In the optimized trainer, preprocessing accounts for approximately **75%** of profiled runtime. The main bottleneck has moved from repeatedly traversing word sequences to reading and pre-tokenizing the corpus.

In a cProfile report, `tottime` is time spent in the function itself, excluding callees; `cumtime` also includes its callees. We can add the naive merging and recounting times because they are separate stages, but should not add `_pretokenize()` and `_count_pretokens()` cumulative times: the former includes the latter.

Profiler overhead is substantial and differs between implementations. On this workload, cProfile changes the naive elapsed time from about 80.7 s to 239.5 s, and the optimized time from about 1.47 s to 3.19 s. Therefore, the **54.8× unprofiled speedup** is the performance result; the profiles explain which work was avoided.

**How the individual optimizations help**

| Optimization | Work it avoids or controls | Evidence from this comparison |
| --- | --- | --- |
| Affected-word index, `pair_words` | Searching and rewriting words that do not contain the chosen pair | Word-merge calls fall from 127.7 million to 68,766 |
| Incremental weighted pair counts | Recounting all pairs after every merge | 9,743 global recounts disappear; updates are limited to affected words |
| Count strings before UTF-8 conversion | Encoding and constructing initial byte sequences for every occurrence | `str.encode()` calls fall from 5.42 million to 13,114 |
| Integer token IDs | Repeated byte-string concatenation in word sequences; byte spellings are stored in the vocabulary | Included in the overall speedup, but not independently timed |
| Heap selection with lazy invalidation | Scanning every active pair to select each merge | Isolated by the heap experiment below |
| Parallel preprocessing | Performing all independent pre-token counting on one CPU core | Isolated by the worker-count experiment below |
| Generators, safe chunks, and bounded pending jobs | Retaining all document substrings, matches, or pending results at once | Buffering behavior follows from the implementation; peak memory was not measured |
| `changed_pairs`, heap rebuilding, and `__slots__` | Repeated candidate pushes within an iteration, stale-entry accumulation, and per-candidate attribute dictionaries | Included in the implementation; their individual effects were not separately measured |

The call counts provide direct evidence that the implementation performs less work. They are not independent speedup factors that we can multiply together. Apart from the heap and worker-count experiments, the timing comparison measures the optimizations as a group.

**Isolating the heap's contribution**

To test the heap separately, we created a benchmark-only variant that retains the current preprocessing, integer IDs, affected-word index, and incremental counts, but selects the next merge with a full `max()` scan. It removes heap construction, maintenance, and changed-pair bookkeeping, while preserving the same byte-spelling tie-breaking rule.

We alternated this variant with fresh runs of the current heap implementation and compared their outputs:

| Workload | Incremental updates + `max()` | Incremental updates + heap | Heap speedup |
| --- | ---: | ---: | ---: |
| 5 MiB sample, vocabulary 1,000 | 0.662 s | 0.398 s | 1.66× |
| Validation file, vocabulary 10,000 | 9.671 s | 1.414 s | 6.84× |

The heap becomes particularly useful in the larger-vocabulary case: once full word rescans are removed, repeatedly scanning the active pair dictionary is a substantial remaining cost. These are fresh paired measurements, which explains the small differences from the optimized medians in the first timing table.

**Isolating parallel preprocessing**

On the validation file with a 10,000-token vocabulary, the optimized trainer gave:

| Configuration | Median elapsed time |
| --- | ---: |
| One worker, fresh heap control | 1.414 s |
| Four preprocessing workers | 0.785 s |

This is an additional **1.80×** improvement. The timings include starting the process pool and transferring results. The gain is less than fourfold because the merge loop remains serial, and preprocessing also incurs I/O, serialization, and parent-side counter aggregation.

These parallel results use elapsed time only. Profiling just the parent process with cProfile would omit computation inside its worker processes and would not provide a complete cost breakdown.

**Reproducing and inspecting the comparison**

From the repository root, run the benchmark against the preserved naive implementation:

```sh
uv run python scripts/profile_bpe.py \
    --baseline cs336_basics/naive_bpe_tokenizer.py \
    --output-dir benchmarks/bpe_profile/rerun \
    --repeats 3 \
    --no-heap-ablation
```

The separate output directory preserves the original measurements. Add `--quick` to skip the longer full-validation workload. The script saves `.prof` files, readable cumulative/self-time tables, source snapshots, and `results.json`. The large naive profiling run took about four minutes in this session.

The existing [profiling report](benchmarks/bpe_profile/comparison.html) collects the results and links to the raw profiles. [results.json](benchmarks/bpe_profile/results.json) contains the individual timing samples, function metrics, and source/input hashes. These measurements establish performance on the tested corpora and machine; they do not measure peak memory or guarantee the same speedup on other datasets.

The BPE adapter in `tests/adapters.py` calls the optimized trainer. The three provided BPE-training tests and 33 additional regression cases all passed. The regression cases cover frequency weighting, overlapping replacements, byte-level ties, Unicode, special tokens crossing actual scanner read boundaries, newline normalization, and serial/parallel equivalence:

```sh
uv run pytest tests/test_train_bpe.py tests/test_bpe_training_regressions.py
```
