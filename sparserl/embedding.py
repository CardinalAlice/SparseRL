from typing import Tuple

import numpy as np
import torch


def sinusoidal_embedding(indices: np.ndarray, dim: int) -> np.ndarray:
    if dim % 2 != 0:
        raise ValueError("Embedding dimension must be even.")
    indices = indices.astype(np.float32)[:, None]
    half = dim // 2
    div_term = np.exp(
        np.arange(half, dtype=np.float32) * (-np.log(10000.0) / half)
    )
    angles = indices * div_term[None, :]
    emb = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return emb


def row_col_embedding(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    hidden_size: int,
) -> Tuple[np.ndarray, int]:
    half = hidden_size // 2
    if half % 2 != 0:
        half -= 1
    if half <= 0:
        raise ValueError("Hidden size too small for sinusoidal embedding.")
    row_emb = sinusoidal_embedding(row_idx, half)
    col_emb = sinusoidal_embedding(col_idx, half)
    merged = np.concatenate([row_emb, col_emb], axis=1)
    if merged.shape[1] != 2 * half:
        raise ValueError("Unexpected embedding width.")
    return merged, 2 * half


def to_torch(emb: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(emb, device=device, dtype=dtype)
