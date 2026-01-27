import re
from dataclasses import dataclass
from typing import Tuple

from .codegen import CompileResult, RunResult


@dataclass
class RewardComponents:
    correctness: float
    efficiency: float
    memory_penalty: float
    total: float


def estimate_shared_memory_bytes(kernel_code: str) -> int:
    total = 0
    for match in re.finditer(r"__shared__\s+\w+\s+(\w+)\s*\[(\d+)\]", kernel_code):
        size = int(match.group(2))
        total += size * 4
    return total


def compute_reward(
    compile_res: CompileResult,
    run_res: RunResult,
    kernel_code: str,
    reward_compile_success: float,
    reward_compile_failure: float,
    reward_test_success: float,
    reward_test_failure: float,
    efficiency_scale: float,
    memory_penalty: float,
    shared_mem_limit_bytes: int,
) -> RewardComponents:
    if not compile_res.success:
        return RewardComponents(
            correctness=reward_compile_failure,
            efficiency=0.0,
            memory_penalty=0.0,
            total=reward_compile_failure,
        )

    correctness = reward_compile_success
    if run_res.success and run_res.correct:
        correctness += reward_test_success
    else:
        correctness += reward_test_failure

    efficiency = 0.0
    if run_res.success and run_res.correct:
        effective_cand = run_res.cand_ms + run_res.conversion_ms
        if effective_cand > 0:
            efficiency = efficiency_scale * (run_res.base_ms / effective_cand - 1.0)

    penalty = 0.0
    shared_bytes = estimate_shared_memory_bytes(kernel_code)
    if shared_bytes > shared_mem_limit_bytes:
        penalty = memory_penalty

    total = correctness + efficiency - penalty
    return RewardComponents(
        correctness=correctness,
        efficiency=efficiency,
        memory_penalty=penalty,
        total=total,
    )
