from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_path: str = "/data02/wangyaoyu/models/Qwen/Qwen3-8B"
    tokenizer_path: str = "/data02/wangyaoyu/models/Qwen/Qwen3-8B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    prompt: str = (
        "First line must be: FORMAT: <CSR|COO|ELL|SELL|BSR>. "
        "Then output only a single CUDA kernel named spmv_kernel. "
        "Do not include headers, main(), or any extra functions. "
        "Signatures by format: "
        "CSR: __global__ void spmv_kernel(int m, const int* row_ptr, const int* col_idx, "
        "const float* val, const float* x, float* y). "
        "COO: __global__ void spmv_kernel(int nnz, const int* row_idx, const int* col_idx, "
        "const float* val, const float* x, float* y). "
        "ELL: __global__ void spmv_kernel(int m, int ell_width, const int* col_idx, "
        "const float* val, const float* x, float* y). "
        "SELL: __global__ void spmv_kernel(int m, int slice_height, const int* slice_ptr, "
        "const int* col_idx, const float* val, const float* x, float* y). "
        "BSR: __global__ void spmv_kernel(int mb, int block_size, const int* row_ptr, "
        "const int* col_idx, const float* val, const float* x, float* y). "
        "No code fences, no commentary."
    )
    sft_prompt: str = (
        "First line must be: FORMAT: <CSR|COO|ELL|SELL|BSR>. "
        "Then output only a single CUDA kernel named spmv_kernel. "
        "Do not include headers, main(), or any extra functions. "
        "Use the signature for the chosen format (see PPO prompt). "
        "No code fences, no commentary."
    )


@dataclass
class DataConfig:
    dataset_dir: str = "dataset"
    max_nnz: int = 256
    seed: int = 42


@dataclass
class RewardConfig:
    reward_compile_success: float = 0.5
    reward_compile_failure: float = -0.5
    reward_test_success: float = 0.5
    reward_test_failure: float = -0.5
    efficiency_scale: float = 0.2
    memory_penalty: float = 0.2
    shared_mem_limit_bytes: int = 48 * 1024


@dataclass
class TrainConfig:
    total_steps: int = 2
    ppo_epochs: int = 1
    clip_ratio: float = 0.2
    kl_coeff: float = 0.01
    lr: float = 1e-6
    grad_clip: float = 1.0
    log_dir: str = "runs"


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
