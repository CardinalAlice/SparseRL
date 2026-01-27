from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .embedding import to_torch


@dataclass
class GenerationOutput:
    text: str
    token_ids: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor


class SparseRLModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple = (),
    ) -> None:
        super().__init__()
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if use_lora:
            from peft import LoraConfig, get_peft_model

            target_modules = list(lora_target_modules) if lora_target_modules else None
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(device)
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False).to(
            device=self.device, dtype=torch_dtype
        )

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.model.get_input_embeddings()(token_ids)
        return emb

    def build_inputs(
        self,
        matrix_embeds: torch.Tensor,
        prompt_ids: torch.Tensor,
        gen_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = self.embed_tokens(prompt_ids)
        if gen_ids is not None:
            gen_embeds = self.embed_tokens(gen_ids)
            inputs = torch.cat([matrix_embeds, prompt_embeds, gen_embeds], dim=1)
            attention = torch.ones(inputs.shape[:2], device=inputs.device)
            return inputs, attention
        inputs = torch.cat([matrix_embeds, prompt_embeds], dim=1)
        attention = torch.ones(inputs.shape[:2], device=inputs.device)
        return inputs, attention

    def generate(
        self,
        matrix_embeds: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, torch.Tensor]:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        prompt_embeds = self.embed_tokens(prompt_ids)
        inputs = torch.cat([matrix_embeds, prompt_embeds], dim=1)

        generated = []
        for _ in range(max_new_tokens):
            outputs = self.model(inputs_embeds=inputs, return_dict=True)
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(
                -1, torch.multinomial(sorted_probs, num_samples=1)
            )
            generated.append(next_token)

            next_embed = self.embed_tokens(next_token)
            inputs = torch.cat([inputs, next_embed], dim=1)

        gen_ids = (
            torch.cat(generated, dim=1)
            if generated
            else torch.empty((1, 0), device=self.device, dtype=torch.long)
        )
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return text, gen_ids

    def score_sequence(
        self,
        matrix_embeds: torch.Tensor,
        prompt_ids: torch.Tensor,
        gen_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, attention = self.build_inputs(matrix_embeds, prompt_ids, gen_ids)
        outputs = self.model(
            inputs_embeds=inputs,
            attention_mask=attention,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        seq_len = inputs.shape[1]
        prefix_len = matrix_embeds.shape[1]
        prompt_len = prompt_ids.shape[1]
        gen_len = gen_ids.shape[1]

        # Logits are aligned to predict token t at position t-1.
        start = prefix_len + prompt_len - 1
        end = prefix_len + prompt_len + gen_len - 1
        target_logits = logits[:, start:end, :]
        target_ids = gen_ids
        logprobs = torch.log_softmax(target_logits, dim=-1)
        token_logprobs = torch.gather(
            logprobs, -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Value per generated token from last hidden state.
        value_states = hidden_states[:, start:end, :]
        values = self.value_head(value_states).squeeze(-1)
        return token_logprobs, values

    def prepare_matrix_embeds(
        self,
        matrix_np: torch.Tensor,
    ) -> torch.Tensor:
        return to_torch(matrix_np, self.device, self.model.dtype).unsqueeze(0)
