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

#### Notes before coding
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

#### Detailed Implementation

Problem descriptions are in <a href="{{ site.baseurl }}/assets/files/notes/cs336_spring2025_assignment1_basics.pdf">Assignment 1: Section 2.5 and 2.6</a> The code is available in my repo.

A naive implementation of training BPE tokenizer is given in `bpe_tokenizer_naive.py`

Run with `cProfile` over `TinyStoriesV2-GPT4-valid.txt` for local test gives the following diagnosis (only the first few lines):

```terminal
         1682860121 function calls (1682859884 primitive calls) in 198.862 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      6/1    0.000    0.000  198.862  198.862 {built-in method builtins.exec}
        1    0.002    0.002  198.862  198.862 bpe_tokenizer_naive.py:1(<module>)
        1   39.531   39.531  198.851  198.851 bpe_tokenizer_naive.py:8(train_bpe)
127740473   61.590    0.000   94.428    0.000 bpe_tokenizer_naive.py:88(merge_pair)
     9743   39.426    0.004   46.865    0.005 bpe_tokenizer_naive.py:79(get_pair_counts)
877629679/877629665   23.034    0.000   23.034    0.000 {built-in method builtins.len}
311084394   13.299    0.000   13.299    0.000 {method 'append' of 'list' objects}
227410752   10.862    0.000   10.862    0.000 __init__.py:609(__missing__)
15080/15079    5.588    0.000    8.495    0.001 {built-in method builtins.max}
 99657168    2.907    0.000    2.907    0.000 bpe_tokenizer_naive.py:112(<lambda>)
 27562412    1.735    0.000    1.735    0.000 bpe_tokenizer_naive.py:76(<genexpr>)
  5419002    0.358    0.000    0.358    0.000 {method 'encode' of 'str' objects}
  5419001    0.332    0.000    0.332    0.000 {method 'group' of '_regex.Match' objects}
    27631    0.009    0.000    0.139    0.000 regex.py:340(finditer)
    27633    0.029    0.000    0.122    0.000 regex.py:449(_compile)
    55437    0.026    0.000    0.076    0.000 enum.py:1561(__and__)
   166397    0.022    0.000    0.036    0.000 enum.py:1543(_get_value)
    19487    0.016    0.000    0.018    0.000 __init__.py:595(__init__)
        1    0.000    0.000    0.015    0.015 regex.py:314(split)
```

We can see that most time-consuming steps are `merge_pair()` and `get_pair_counts()`.

