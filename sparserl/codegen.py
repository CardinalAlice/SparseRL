import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .data import SparseMatrixCSR


@dataclass
class CompileResult:
    success: bool
    log: str
    binary_path: str = ""
    source_path: str = ""


@dataclass
class RunResult:
    success: bool
    correct: bool
    base_ms: float = 0.0
    cand_ms: float = 0.0
    conversion_ms: float = 0.0
    log: str = ""


KERNEL_SIGNATURES = {
    "CSR": "__global__ void spmv_kernel(int m, const int* row_ptr, const int* col_idx, const float* val, const float* x, float* y)",
    "COO": "__global__ void spmv_kernel(int nnz, const int* row_idx, const int* col_idx, const float* val, const float* x, float* y)",
    "ELL": "__global__ void spmv_kernel(int m, int ell_width, const int* col_idx, const float* val, const float* x, float* y)",
    "BSR": "__global__ void spmv_kernel(int mb, int block_size, const int* row_ptr, const int* col_idx, const float* val, const float* x, float* y)",
    "SELL": "__global__ void spmv_kernel(int m, int slice_height, const int* slice_ptr, const int* col_idx, const float* val, const float* x, float* y)",
}


def _canonical_signature(sig: str) -> str:
    return re.sub(r"\s+", "", sig)


def _normalize_param(param: str) -> str:
    param = re.sub(r"\b(__restrict__|__restrict)\b", "", param)
    param = re.sub(r"\s+", " ", param).strip()
    param = re.sub(r"\b[A-Za-z_]\w*(\s*\[[^\]]*\])?\s*$", "", param).strip()
    param = re.sub(r"\s+", " ", param).strip()
    param = param.replace(" *", "*").replace("* ", "*")
    return param


def _extract_param_types(signature: str) -> list:
    start = signature.find("(")
    end = signature.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return []
    params = signature[start + 1 : end]
    parts = [p.strip() for p in params.split(",") if p.strip()]
    return [_normalize_param(p) for p in parts]


def _select_code_block(text: str) -> str:
    code_blocks = re.findall(r"```(?:cuda|cpp|c\+\+)?\n(.*?)```", text, re.S)
    if not code_blocks:
        return text
    for block in code_blocks:
        if "spmv_kernel" in block:
            return block
    return code_blocks[0]


def extract_format(text: str) -> str:
    match = re.search(r"FORMAT\s*:\s*(CSR|COO|ELL|SELL|BSR)", text, re.I)
    if match:
        return match.group(1).upper()
    return "CSR"


def _find_matching_brace(text: str, start: int) -> int:
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def extract_kernel(text: str, fmt: str) -> str:
    text = _select_code_block(text)
    match = re.search(r"__global__\s+void\s+spmv_kernel\s*\(", text)
    if not match:
        return ""

    brace_idx = text.find("{", match.end())
    if brace_idx == -1:
        return ""
    end_idx = _find_matching_brace(text, brace_idx)
    if end_idx == -1:
        return ""

    kernel = text[match.start() : end_idx + 1].strip()
    sig = kernel.split("{", 1)[0]
    param_types = _extract_param_types(sig)
    expected = {
        "CSR": ["int", "const int*", "const int*", "const float*", "const float*", "float*"],
        "COO": ["int", "const int*", "const int*", "const float*", "const float*", "float*"],
        "ELL": ["int", "int", "const int*", "const float*", "const float*", "float*"],
        "BSR": ["int", "int", "const int*", "const int*", "const float*", "const float*", "float*"],
        "SELL": ["int", "int", "const int*", "const int*", "const float*", "const float*", "float*"],
    }.get(fmt.upper(), [])
    if param_types != expected:
        return ""
    if re.search(r"\bint\s+main\s*\(", kernel):
        return ""
    if re.search(r"__global__\s+void\s+(?!spmv_kernel)", kernel):
        return ""
    if "#include" in kernel:
        return ""
    return kernel


def _array_literal(arr: np.ndarray) -> str:
    return ", ".join(str(x) for x in arr.tolist())


def build_program(csr: SparseMatrixCSR, fmt_payload: dict, kernel_code: str) -> str:
    m, n = csr.shape
    nnz = csr.val.shape[0]
    csr_row_ptr = _array_literal(csr.row_ptr)
    csr_col_idx = _array_literal(csr.col_idx)
    csr_val = _array_literal(csr.val)

    x = np.random.default_rng(0).normal(size=n).astype(np.float32)
    x_lit = _array_literal(x)

    kernel_code = kernel_code.strip()
    fmt = fmt_payload.get("format", "CSR").upper()

    host_defs = ""
    device_allocs = ""
    device_copies = ""
    kernel_call = ""
    device_frees = ""

    if fmt == "CSR":
        kernel_call = "spmv_kernel<<<grid, block>>>(m, d_row_ptr, d_col_idx, d_val, d_x, d_y);"
    elif fmt == "COO":
        row_idx = _array_literal(fmt_payload["row_idx"])
        col_idx = _array_literal(fmt_payload["col_idx"])
        val = _array_literal(fmt_payload["val"])
        host_defs = (
            f"\n    static const int h_row_idx[{fmt_payload['row_idx'].size}] = {{{row_idx}}};"
            f"\n    static const int h_coo_col_idx[{fmt_payload['col_idx'].size}] = {{{col_idx}}};"
            f"\n    static const float h_coo_val[{fmt_payload['val'].size}] = {{{val}}};\n"
        )
        device_allocs = (
            "\n    cudaMalloc(&d_row_idx, sizeof(int) * nnz);"
            "\n    cudaMalloc(&d_coo_col_idx, sizeof(int) * nnz);"
            "\n    cudaMalloc(&d_coo_val, sizeof(float) * nnz);\n"
        )
        device_copies = (
            "\n    cudaMemcpy(d_row_idx, h_row_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);"
            "\n    cudaMemcpy(d_coo_col_idx, h_coo_col_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);"
            "\n    cudaMemcpy(d_coo_val, h_coo_val, sizeof(float) * nnz, cudaMemcpyHostToDevice);\n"
        )
        kernel_call = "spmv_kernel<<<grid, block>>>(nnz, d_row_idx, d_coo_col_idx, d_coo_val, d_x, d_y);"
        device_frees = "\n    cudaFree(d_row_idx);\n    cudaFree(d_coo_col_idx);\n    cudaFree(d_coo_val);\n"
    elif fmt == "ELL":
        width = int(fmt_payload["ell_width"])
        col_idx = _array_literal(fmt_payload["col_idx"].reshape(-1))
        val = _array_literal(fmt_payload["val"].reshape(-1))
        total = fmt_payload["col_idx"].size
        host_defs = (
            f"\n    const int ell_width = {width};"
            f"\n    static const int h_ell_col[{total}] = {{{col_idx}}};"
            f"\n    static const float h_ell_val[{total}] = {{{val}}};\n"
        )
        device_allocs = (
            f"\n    cudaMalloc(&d_ell_col, sizeof(int) * {total});"
            f"\n    cudaMalloc(&d_ell_val, sizeof(float) * {total});\n"
        )
        device_copies = (
            f"\n    cudaMemcpy(d_ell_col, h_ell_col, sizeof(int) * {total}, cudaMemcpyHostToDevice);"
            f"\n    cudaMemcpy(d_ell_val, h_ell_val, sizeof(float) * {total}, cudaMemcpyHostToDevice);\n"
        )
        kernel_call = "spmv_kernel<<<grid, block>>>(m, ell_width, d_ell_col, d_ell_val, d_x, d_y);"
        device_frees = "\n    cudaFree(d_ell_col);\n    cudaFree(d_ell_val);\n"
    elif fmt == "BSR":
        mb = int(fmt_payload["mb"])
        bs = int(fmt_payload["block_size"])
        row_ptr = _array_literal(fmt_payload["row_ptr"])
        col_idx = _array_literal(fmt_payload["col_idx"])
        val = _array_literal(fmt_payload["val"])
        row_ptr_size = fmt_payload["row_ptr"].size
        col_idx_size = fmt_payload["col_idx"].size
        val_size = fmt_payload["val"].size
        host_defs = (
            f"\n    const int mb = {mb};"
            f"\n    const int block_size = {bs};"
            f"\n    static const int h_bsr_row_ptr[{row_ptr_size}] = {{{row_ptr}}};"
            f"\n    static const int h_bsr_col_idx[{col_idx_size}] = {{{col_idx}}};"
            f"\n    static const float h_bsr_val[{val_size}] = {{{val}}};\n"
        )
        device_allocs = (
            f"\n    cudaMalloc(&d_bsr_row_ptr, sizeof(int) * {row_ptr_size});"
            f"\n    cudaMalloc(&d_bsr_col_idx, sizeof(int) * {col_idx_size});"
            f"\n    cudaMalloc(&d_bsr_val, sizeof(float) * {val_size});\n"
        )
        device_copies = (
            f"\n    cudaMemcpy(d_bsr_row_ptr, h_bsr_row_ptr, sizeof(int) * {row_ptr_size}, cudaMemcpyHostToDevice);"
            f"\n    cudaMemcpy(d_bsr_col_idx, h_bsr_col_idx, sizeof(int) * {col_idx_size}, cudaMemcpyHostToDevice);"
            f"\n    cudaMemcpy(d_bsr_val, h_bsr_val, sizeof(float) * {val_size}, cudaMemcpyHostToDevice);\n"
        )
        kernel_call = "spmv_kernel<<<grid, block>>>(mb, block_size, d_bsr_row_ptr, d_bsr_col_idx, d_bsr_val, d_x, d_y);"
        device_frees = "\n    cudaFree(d_bsr_row_ptr);\n    cudaFree(d_bsr_col_idx);\n    cudaFree(d_bsr_val);\n"
    elif fmt == "SELL":
        slice_height = int(fmt_payload["slice_height"])
        slice_ptr = _array_literal(fmt_payload["slice_ptr"])
        col_idx = _array_literal(fmt_payload["col_idx"])
        val = _array_literal(fmt_payload["val"])
        slice_ptr_size = fmt_payload["slice_ptr"].size
        col_idx_size = fmt_payload["col_idx"].size
        val_size = fmt_payload["val"].size
        host_defs = (
            f"\n    const int slice_height = {slice_height};"
            f"\n    static const int h_slice_ptr[{slice_ptr_size}] = {{{slice_ptr}}};"
            f"\n    static const int h_sell_col[{col_idx_size}] = {{{col_idx}}};"
            f"\n    static const float h_sell_val[{val_size}] = {{{val}}};\n"
        )
        device_allocs = (
            f"\n    cudaMalloc(&d_slice_ptr, sizeof(int) * {slice_ptr_size});"
            f"\n    cudaMalloc(&d_sell_col, sizeof(int) * {col_idx_size});"
            f"\n    cudaMalloc(&d_sell_val, sizeof(float) * {val_size});\n"
        )
        device_copies = (
            f"\n    cudaMemcpy(d_slice_ptr, h_slice_ptr, sizeof(int) * {slice_ptr_size}, cudaMemcpyHostToDevice);"
            f"\n    cudaMemcpy(d_sell_col, h_sell_col, sizeof(int) * {col_idx_size}, cudaMemcpyHostToDevice);"
            f"\n    cudaMemcpy(d_sell_val, h_sell_val, sizeof(float) * {val_size}, cudaMemcpyHostToDevice);\n"
        )
        kernel_call = "spmv_kernel<<<grid, block>>>(m, slice_height, d_slice_ptr, d_sell_col, d_sell_val, d_x, d_y);"
        device_frees = "\n    cudaFree(d_slice_ptr);\n    cudaFree(d_sell_col);\n    cudaFree(d_sell_val);\n"

    return f"""
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

{kernel_code}

__global__ void baseline_kernel(int m, const int* row_ptr, const int* col_idx,
                               const float* val, const float* x, float* y) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {{
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int idx = start; idx < end; ++idx) {{
            sum += val[idx] * x[col_idx[idx]];
        }}
        y[row] = sum;
    }}
}}

int main() {{
    const int m = {m};
    const int n = {n};
    const int nnz = {nnz};

    static const int h_row_ptr[{csr.row_ptr.size}] = {{{csr_row_ptr}}};
    static const int h_col_idx[{csr.col_idx.size}] = {{{csr_col_idx}}};
    static const float h_val[{csr.val.size}] = {{{csr_val}}};
    static const float h_x[{x.size}] = {{{x_lit}}};
{host_defs}
    float* h_y = new float[m];
    float* h_y_ref = new float[m];
    for (int i = 0; i < m; ++i) {{
        float sum = 0.0f;
        for (int idx = h_row_ptr[i]; idx < h_row_ptr[i + 1]; ++idx) {{
            sum += h_val[idx] * h_x[h_col_idx[idx]];
        }}
        h_y_ref[i] = sum;
    }}

    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    float* d_val = nullptr;
    int* d_row_idx = nullptr;
    int* d_coo_col_idx = nullptr;
    float* d_coo_val = nullptr;
    int* d_ell_col = nullptr;
    float* d_ell_val = nullptr;
    int* d_bsr_row_ptr = nullptr;
    int* d_bsr_col_idx = nullptr;
    float* d_bsr_val = nullptr;
    int* d_slice_ptr = nullptr;
    int* d_sell_col = nullptr;
    float* d_sell_val = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_row_ptr, sizeof(int) * (m + 1));
    cudaMalloc(&d_col_idx, sizeof(int) * nnz);
    cudaMalloc(&d_val, sizeof(float) * nnz);
    cudaMalloc(&d_x, sizeof(float) * n);
    cudaMalloc(&d_y, sizeof(float) * m);

    cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(float) * n, cudaMemcpyHostToDevice);
{device_allocs}
{device_copies}

    dim3 block(256);
    dim3 grid((m + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    baseline_kernel<<<grid, block>>>(m, d_row_ptr, d_col_idx, d_val, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float base_ms = 0.0f;
    cudaEventElapsedTime(&base_ms, start, stop);

    cudaEventRecord(start);
    {kernel_call}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cand_ms = 0.0f;
    cudaEventElapsedTime(&cand_ms, start, stop);

    cudaMemcpy(h_y, d_y, sizeof(float) * m, cudaMemcpyDeviceToHost);

    int correct = 1;
    for (int i = 0; i < m; ++i) {{
        float diff = fabsf(h_y[i] - h_y_ref[i]);
        if (diff > 1e-2f) {{
            correct = 0;
            break;
        }}
    }}

    printf("{{\\\"correct\\\": %d, \\\"base_ms\\\": %.6f, \\\"cand_ms\\\": %.6f}}\\n",
           correct, base_ms, cand_ms);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_val);
{device_frees}
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_y;
    delete[] h_y_ref;

    return 0;
}}
"""


def compile_program(source_code: str, work_dir: str) -> CompileResult:
    src_path = os.path.join(work_dir, "candidate.cu")
    bin_path = os.path.join(work_dir, "candidate")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(source_code)

    cmd = [
        "nvcc",
        src_path,
        "-O3",
        "-std=c++17",
        "-arch=sm_80",
        "-o",
        bin_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return CompileResult(False, proc.stderr, source_path=src_path)
    return CompileResult(True, proc.stdout, binary_path=bin_path, source_path=src_path)


def run_program(binary_path: str, conversion_ms: float) -> RunResult:
    proc = subprocess.run([binary_path], capture_output=True, text=True)
    if proc.returncode != 0:
        return RunResult(False, False, conversion_ms=conversion_ms, log=proc.stderr)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return RunResult(False, False, conversion_ms=conversion_ms, log=proc.stdout)
    return RunResult(
        True,
        bool(payload.get("correct", 0)),
        base_ms=float(payload.get("base_ms", 0.0)),
        cand_ms=float(payload.get("cand_ms", 0.0)),
        conversion_ms=conversion_ms,
        log=proc.stdout,
    )


def compile_and_run(
    csr: SparseMatrixCSR, fmt_payload: dict, kernel_code: str, conversion_ms: float
) -> Tuple[CompileResult, RunResult]:
    if not kernel_code.strip():
        compile_res = CompileResult(False, "empty kernel")
        return compile_res, RunResult(False, False, conversion_ms=conversion_ms, log=compile_res.log)
    with tempfile.TemporaryDirectory(prefix="sparserl_") as tmp:
        program = build_program(csr, fmt_payload, kernel_code)
        compile_res = compile_program(program, tmp)
        if not compile_res.success:
            return compile_res, RunResult(False, False, conversion_ms=conversion_ms, log=compile_res.log)
        run_res = run_program(compile_res.binary_path, conversion_ms)
        return compile_res, run_res
