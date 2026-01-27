import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .data import csr_to_row_col, load_mtx, sample_nnz
from .embedding import row_col_embedding


@dataclass
class SFTSample:
    name: str
    shape: Tuple[int, int]
    rows: np.ndarray
    cols: np.ndarray
    fmt: str
    code: str


def load_sft_jsonl(path: str) -> List[SFTSample]:
    samples: List[SFTSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(
                SFTSample(
                    name=obj.get("name", "sample"),
                    shape=tuple(obj["shape"]),
                    rows=np.asarray(obj["rows"], dtype=np.int32),
                    cols=np.asarray(obj["cols"], dtype=np.int32),
                    fmt=obj.get("format", "CSR"),
                    code=obj["code"],
                )
            )
    return samples


def build_matrix_embedding(sample: SFTSample, hidden_size: int) -> np.ndarray:
    emb, emb_dim = row_col_embedding(sample.rows, sample.cols, hidden_size)
    if emb_dim != hidden_size:
        pad = hidden_size - emb_dim
        emb = np.pad(emb, ((0, 0), (0, pad)), mode="constant")
    return emb


def build_sft_from_mtx(path: str, max_nnz: int, seed: int, code: str) -> SFTSample:
    csr = load_mtx(path)
    csr = sample_nnz(csr, max_nnz, seed=seed)
    rows, cols = csr_to_row_col(csr)
    return SFTSample(
        name=path.split("/")[-1],
        shape=csr.shape,
        rows=rows,
        cols=cols,
        fmt="CSR",
        code=code,
    )
