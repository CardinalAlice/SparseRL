# SparseRL Minimal Pipeline

This repository is the open-source code release for the SparseRL paper
("Mastering Sparse CUDA Generation Through Pretrained Models and Deep Reinforcement Learning").
It provides a minimal, runnable implementation that reproduces the core ideas for sparse CUDA
code generation.

## Overview

SparseRL treats a pretrained code LLM as a stochastic policy. Given the row/column indices of
non-zero entries in a sparse matrix, the model generates CUDA kernels for sparse operators (e.g.,
SpMV and SpMM). The training loop combines supervised fine-tuning and reinforcement learning
with compiler/executor feedback to optimize both correctness and runtime efficiency.

## How It Works (Pipeline)

- Pre-training: augment a base LLM with CUDA code to build domain knowledge.
- Sparse matrix embedding: encode row/column indices with sinusoidal embeddings to capture
  dynamic sparsity patterns.
- SFT: learn to map embedded sparse matrices to CUDA kernels.
- RL: optimize the generator with a hierarchical reward that includes compilation/correctness and
  execution time feedback from the compiler and executor.
- Currently, we provide the minimal pipeline of SFT and PPO. 


## SFT (from .mtx)

```bash
python -m sparserl.train_sft --dataset-dir dataset --max-nnz 128 --steps 2
```

Optional: provide a JSONL file with `rows`, `cols`, `shape`, `format`, and `code` fields.
`code` should start with `FORMAT: <...>` to match the PPO format-selection logic:

```bash
python -m sparserl.train_sft --data sft_data/sft_samples.jsonl --steps 2
```

## PPO Run

```bash
python -m sparserl.train_minimal --steps 1 --max-nnz 128 \
  --model-path /data02/wangyaoyu/models/Qwen/Qwen3-8B \
  --tokenizer-path /data02/wangyaoyu/models/Qwen/Qwen3-8B
```

Notes:
- `max-nnz` keeps code size manageable for compilation.
- The PPO prompt expects a first line like `FORMAT: CSR` and a kernel matching that format signature.
- If kernel extraction fails, the compile reward will be negative.


## Dataset Statement

The specific pre-training and SFT datasets will be released in future work under the "Qiwu (齐物)"
project.
