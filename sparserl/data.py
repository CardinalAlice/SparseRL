import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.io import mmread
import time


@dataclass
class SparseMatrixCSR:
    shape: Tuple[int, int]
    row_ptr: np.ndarray
    col_idx: np.ndarray
    val: np.ndarray


def load_mtx(path: str, dtype=np.float32) -> SparseMatrixCSR:
    mtx = mmread(path).tocsr().astype(dtype)
    return SparseMatrixCSR(
        shape=mtx.shape,
        row_ptr=mtx.indptr.astype(np.int32),
        col_idx=mtx.indices.astype(np.int32),
        val=mtx.data.astype(dtype),
    )


def list_mtx_files(dataset_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.endswith(".mtx"):
            files.append(os.path.join(dataset_dir, name))
    return files


def sample_nnz(
    csr: SparseMatrixCSR,
    max_nnz: int,
    seed: int = 0,
) -> SparseMatrixCSR:
    nnz = csr.val.shape[0]
    if nnz <= max_nnz:
        return csr

    rng = np.random.default_rng(seed)
    idx = rng.choice(nnz, size=max_nnz, replace=False)
    idx.sort()

    # Rebuild a CSR matrix from sampled entries.
    rows = np.empty_like(idx)
    row_ptr = csr.row_ptr
    row_ids = np.searchsorted(row_ptr[1:], idx, side="right")
    rows[:] = row_ids

    cols = csr.col_idx[idx]
    vals = csr.val[idx]

    # Build CSR.
    m, n = csr.shape
    row_counts = np.bincount(rows, minlength=m)
    new_row_ptr = np.zeros(m + 1, dtype=np.int32)
    new_row_ptr[1:] = np.cumsum(row_counts)

    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    new_col_idx = np.empty_like(cols, dtype=np.int32)
    new_val = np.empty_like(vals)

    cursor = np.zeros(m, dtype=np.int32)
    for r, c, v in zip(rows, cols, vals):
        offset = new_row_ptr[r] + cursor[r]
        new_col_idx[offset] = c
        new_val[offset] = v
        cursor[r] += 1

    return SparseMatrixCSR(
        shape=(m, n),
        row_ptr=new_row_ptr,
        col_idx=new_col_idx,
        val=new_val,
    )


def csr_to_row_col(csr: SparseMatrixCSR) -> Tuple[np.ndarray, np.ndarray]:
    row_ptr = csr.row_ptr
    nnz = csr.val.shape[0]
    rows = np.empty(nnz, dtype=np.int32)
    for r in range(csr.shape[0]):
        start, end = row_ptr[r], row_ptr[r + 1]
        rows[start:end] = r
    return rows, csr.col_idx.copy()


def csr_to_coo(csr: SparseMatrixCSR) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = csr_to_row_col(csr)
    return rows, cols, csr.val.copy()


def csr_to_ell(csr: SparseMatrixCSR) -> Tuple[int, np.ndarray, np.ndarray]:
    m, _ = csr.shape
    row_ptr = csr.row_ptr
    row_lengths = row_ptr[1:] - row_ptr[:-1]
    width = int(row_lengths.max(initial=0))
    if width == 0:
        return 0, np.zeros((m, 0), dtype=np.int32), np.zeros((m, 0), dtype=csr.val.dtype)
    col = -np.ones((m, width), dtype=np.int32)
    val = np.zeros((m, width), dtype=csr.val.dtype)
    for r in range(m):
        start, end = row_ptr[r], row_ptr[r + 1]
        length = end - start
        if length > 0:
            col[r, :length] = csr.col_idx[start:end]
            val[r, :length] = csr.val[start:end]
    return width, col, val


def csr_to_bsr(
    csr: SparseMatrixCSR, block_size: int = 2
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    m, n = csr.shape
    mb = (m + block_size - 1) // block_size
    nb = (n + block_size - 1) // block_size
    blocks = {}
    row_ptr = csr.row_ptr
    for r in range(m):
        br = r // block_size
        for idx in range(row_ptr[r], row_ptr[r + 1]):
            c = csr.col_idx[idx]
            bc = c // block_size
            key = (br, bc)
            if key not in blocks:
                blocks[key] = np.zeros((block_size, block_size), dtype=csr.val.dtype)
            blocks[key][r % block_size, c % block_size] = csr.val[idx]
    row_blocks = [[] for _ in range(mb)]
    for (br, bc), block in blocks.items():
        row_blocks[br].append((bc, block))
    bsr_row_ptr = np.zeros(mb + 1, dtype=np.int32)
    bsr_col_idx = []
    bsr_val = []
    nnzb = 0
    for br in range(mb):
        row_blocks[br].sort(key=lambda x: x[0])
        bsr_row_ptr[br] = nnzb
        for bc, block in row_blocks[br]:
            bsr_col_idx.append(bc)
            bsr_val.extend(block.reshape(-1).tolist())
            nnzb += 1
    bsr_row_ptr[mb] = nnzb
    return (
        mb,
        block_size,
        bsr_row_ptr,
        np.asarray(bsr_col_idx, dtype=np.int32),
        np.asarray(bsr_val, dtype=csr.val.dtype),
    )


def csr_to_sell(
    csr: SparseMatrixCSR, slice_height: int = 4
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    m, _ = csr.shape
    row_ptr = csr.row_ptr
    num_slices = (m + slice_height - 1) // slice_height
    slice_ptr = np.zeros(num_slices + 1, dtype=np.int32)
    col_chunks = []
    val_chunks = []
    offset = 0
    for s in range(num_slices):
        start_row = s * slice_height
        end_row = min(m, start_row + slice_height)
        lengths = row_ptr[start_row + 1 : end_row + 1] - row_ptr[start_row:end_row]
        width = int(lengths.max(initial=0))
        slice_ptr[s] = offset
        for r in range(start_row, end_row):
            row_start, row_end = row_ptr[r], row_ptr[r + 1]
            row_len = row_end - row_start
            cols = csr.col_idx[row_start:row_end]
            vals = csr.val[row_start:row_end]
            if row_len < width:
                pad = width - row_len
                cols = np.pad(cols, (0, pad), constant_values=-1)
                vals = np.pad(vals, (0, pad), constant_values=0.0)
            col_chunks.append(cols)
            val_chunks.append(vals)
        offset += width * (end_row - start_row)
    slice_ptr[num_slices] = offset
    if col_chunks:
        col = np.concatenate(col_chunks).astype(np.int32)
        val = np.concatenate(val_chunks).astype(csr.val.dtype)
    else:
        col = np.zeros(0, dtype=np.int32)
        val = np.zeros(0, dtype=csr.val.dtype)
    return slice_height, slice_ptr, col, val


def convert_format(
    csr: SparseMatrixCSR,
    fmt: str,
    max_nnz: int,
    seed: int,
    block_size: int = 2,
    slice_height: int = 4,
) -> Tuple[dict, float]:
    csr = sample_nnz(csr, max_nnz, seed=seed)
    fmt = fmt.upper()
    start = time.perf_counter()
    payload = {"format": fmt}
    if fmt == "CSR":
        payload.update(
            {
                "row_ptr": csr.row_ptr,
                "col_idx": csr.col_idx,
                "val": csr.val,
            }
        )
    elif fmt == "COO":
        row, col, val = csr_to_coo(csr)
        payload.update({"row_idx": row, "col_idx": col, "val": val})
    elif fmt == "ELL":
        width, col, val = csr_to_ell(csr)
        payload.update({"ell_width": width, "col_idx": col, "val": val})
    elif fmt == "BSR":
        mb, bs, row_ptr, col_idx, val = csr_to_bsr(csr, block_size=block_size)
        payload.update(
            {
                "mb": mb,
                "block_size": bs,
                "row_ptr": row_ptr,
                "col_idx": col_idx,
                "val": val,
            }
        )
    elif fmt == "SELL":
        sh, slice_ptr, col, val = csr_to_sell(csr, slice_height=slice_height)
        payload.update(
            {"slice_height": sh, "slice_ptr": slice_ptr, "col_idx": col, "val": val}
        )
    else:
        payload.update(
            {
                "row_ptr": csr.row_ptr,
                "col_idx": csr.col_idx,
                "val": csr.val,
            }
        )
        fmt = "CSR"
        payload["format"] = "CSR"
    conversion_ms = (time.perf_counter() - start) * 1000.0
    return payload, conversion_ms
