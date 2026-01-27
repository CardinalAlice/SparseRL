import argparse
import os
import numpy as np
import torch

from .codegen import compile_and_run, extract_format, extract_kernel
from .config import DataConfig, ModelConfig, RewardConfig, TrainConfig
from .data import convert_format, csr_to_row_col, list_mtx_files, load_mtx, sample_nnz
from .embedding import row_col_embedding
from .model import SparseRLModel
from .reward import compute_reward


def build_matrix_embeds(csr, model) -> torch.Tensor:
    row_idx, col_idx = csr_to_row_col(csr)
    hidden_size = model.model.config.hidden_size
    emb_np, emb_dim = row_col_embedding(row_idx, col_idx, hidden_size)
    if emb_dim != hidden_size:
        pad = hidden_size - emb_dim
        emb_np = np.pad(emb_np, ((0, 0), (0, pad)), mode="constant")
    return model.prepare_matrix_embeds(emb_np)


def ppo_update(
    model,
    prompt_ids,
    gen_ids,
    matrix_embeds,
    reward,
    cfg: TrainConfig,
):
    old_logprobs, old_values = model.score_sequence(matrix_embeds, prompt_ids, gen_ids)
    old_logprobs = old_logprobs.detach()
    old_values = old_values.detach()
    advantages = reward - old_values

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, foreach=False)
    for _ in range(cfg.ppo_epochs):
        logprobs, values = model.score_sequence(matrix_embeds, prompt_ids, gen_ids)
        ratio = torch.exp(logprobs - old_logprobs)
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
        policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))
        value_loss = 0.5 * torch.mean((values - reward) ** 2)

        kl = torch.mean(old_logprobs - logprobs)
        loss = policy_loss + value_loss + cfg.kl_coeff * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    return {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=ModelConfig.model_path)
    parser.add_argument("--tokenizer-path", default=ModelConfig.tokenizer_path)
    parser.add_argument("--dataset-dir", default=DataConfig.dataset_dir)
    parser.add_argument("--max-nnz", type=int, default=DataConfig.max_nnz)
    parser.add_argument("--steps", type=int, default=TrainConfig.total_steps)
    args = parser.parse_args()

    model_cfg = ModelConfig(model_path=args.model_path, tokenizer_path=args.tokenizer_path)
    data_cfg = DataConfig(dataset_dir=args.dataset_dir, max_nnz=args.max_nnz)
    reward_cfg = RewardConfig()
    train_cfg = TrainConfig(total_steps=args.steps)

    try:
        model = SparseRLModel(
            model_cfg.model_path,
            model_cfg.tokenizer_path,
            device=model_cfg.device,
            dtype=model_cfg.dtype,
            use_lora=model_cfg.use_lora,
            lora_r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            lora_target_modules=model_cfg.lora_target_modules,
        )
    except ValueError as exc:
        if "model type `qwen3`" not in str(exc):
            raise
        fallback_env = os.environ.get("SPARSERL_FALLBACK_MODEL")
        fallback_candidates = (
            [fallback_env] if fallback_env else []
        ) + [
            "/data02/wangyaoyu/models/Qwen/Qwen2.5-Coder-7B",
            "/data02/wangyaoyu/models/Qwen/Qwen2.5-7B",
        ]
        fallback = next((p for p in fallback_candidates if p and os.path.isdir(p)), None)
        if not fallback:
            raise
        print(
            "Qwen3 is not supported by the installed transformers version; "
            f"falling back to {fallback}."
        )
        model_cfg.model_path = fallback
        model_cfg.tokenizer_path = fallback
        model = SparseRLModel(
            model_cfg.model_path,
            model_cfg.tokenizer_path,
            device=model_cfg.device,
            dtype=model_cfg.dtype,
            use_lora=model_cfg.use_lora,
            lora_r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            lora_target_modules=model_cfg.lora_target_modules,
        )
    if not getattr(model.model, "peft_config", None):
        for param in model.model.parameters():
            param.requires_grad = False
    files = list_mtx_files(data_cfg.dataset_dir)
    if not files:
        raise RuntimeError("No .mtx files found in dataset directory.")

    for step in range(train_cfg.total_steps):
        path = files[step % len(files)]
        csr = load_mtx(path)
        csr = sample_nnz(csr, data_cfg.max_nnz, seed=data_cfg.seed + step)
        matrix_embeds = build_matrix_embeds(csr, model)

        text, gen_ids = model.generate(
            matrix_embeds,
            model_cfg.prompt,
            max_new_tokens=model_cfg.max_new_tokens,
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
        )

        fmt = extract_format(text)
        fmt_payload, conversion_ms = convert_format(
            csr,
            fmt=fmt,
            max_nnz=data_cfg.max_nnz,
            seed=data_cfg.seed + step,
        )
        kernel_code = extract_kernel(text, fmt)
        compile_res, run_res = compile_and_run(csr, fmt_payload, kernel_code, conversion_ms)
        reward = compute_reward(
            compile_res,
            run_res,
            kernel_code,
            reward_cfg.reward_compile_success,
            reward_cfg.reward_compile_failure,
            reward_cfg.reward_test_success,
            reward_cfg.reward_test_failure,
            reward_cfg.efficiency_scale,
            reward_cfg.memory_penalty,
            reward_cfg.shared_mem_limit_bytes,
        )

        prompt_ids = model.tokenizer(model_cfg.prompt, return_tensors="pt").input_ids.to(
            model.device
        )
        stats = ppo_update(
            model,
            prompt_ids,
            gen_ids,
            matrix_embeds,
            torch.tensor(reward.total, device=model.device),
            train_cfg,
        )

        print(
            f"step={step} file={os.path.basename(path)} reward={reward.total:.4f} "
            f"correctness={reward.correctness:.4f} efficiency={reward.efficiency:.4f} "
            f"penalty={reward.memory_penalty:.4f} compile={compile_res.success} "
            f"correct={run_res.correct} stats={stats}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
