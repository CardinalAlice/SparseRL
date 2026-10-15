import argparse
import os

import torch

from .config import DataConfig, ModelConfig, TrainConfig
from .model import SparseRLModel
from .sft_data import build_matrix_embedding, build_sft_from_mtx, load_sft_jsonl


def build_inputs(model: SparseRLModel, matrix_embeds, prompt: str, code: str):
    prompt_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    code_ids = model.tokenizer(
        code, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)

    prompt_embeds = model.embed_tokens(prompt_ids)
    code_embeds = model.embed_tokens(code_ids)
    inputs_embeds = torch.cat([matrix_embeds, prompt_embeds, code_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)

    ignore = -100
    labels = torch.full((1, inputs_embeds.shape[1]), ignore, device=inputs_embeds.device)
    labels[:, -code_ids.shape[1] :] = code_ids

    return inputs_embeds, attention_mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="")
    parser.add_argument("--model-path", default=ModelConfig.model_path)
    parser.add_argument("--tokenizer-path", default=ModelConfig.tokenizer_path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dataset-dir", default=DataConfig.dataset_dir)
    parser.add_argument("--max-nnz", type=int, default=DataConfig.max_nnz)
    args = parser.parse_args()

    model_cfg = ModelConfig(model_path=args.model_path, tokenizer_path=args.tokenizer_path)
    train_cfg = TrainConfig(total_steps=args.steps)

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, foreach=False)

    if args.data:
        samples = load_sft_jsonl(args.data)
    else:
        from .data import list_mtx_files

        files = list_mtx_files(args.dataset_dir)
        if not files:
            raise RuntimeError("No .mtx files found for SFT.")
        default_code = (
            "FORMAT: CSR\\n"
            "__global__ void spmv_kernel(int m, const int* row_ptr, "
            "const int* col_idx, const float* val, const float* x, float* y) {\\n"
            "    int row = blockIdx.x * blockDim.x + threadIdx.x;\\n"
            "    if (row < m) {\\n"
            "        float sum = 0.0f;\\n"
            "        int start = row_ptr[row];\\n"
            "        int end = row_ptr[row + 1];\\n"
            "        for (int idx = start; idx < end; ++idx) {\\n"
            "            sum += val[idx] * x[col_idx[idx]];\\n"
            "        }\\n"
            "        y[row] = sum;\\n"
            "    }\\n"
            "}\\n"
        )
        samples = [
            build_sft_from_mtx(
                path,
                max_nnz=args.max_nnz,
                seed=step,
                code=default_code,
            )
            for step, path in enumerate(files)
        ]
    if not samples:
        raise RuntimeError("No SFT samples loaded.")

    model.model.train()
    for step in range(train_cfg.total_steps):
        sample = samples[step % len(samples)]
        matrix_np = build_matrix_embedding(sample, model.model.config.hidden_size)
        matrix_embeds = model.prepare_matrix_embeds(matrix_np)

        inputs_embeds, attention_mask, labels = build_inputs(
            model, matrix_embeds, model_cfg.sft_prompt, sample.code
        )
        outputs = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.grad_clip)
        optimizer.step()

        print(
            f"step={step} sample={sample.name} loss={loss.detach().float().item():.6f}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
