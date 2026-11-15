import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

# reasoning_end: reasoning과 response를 나누는 token
# qwen3는 chat template이 아니라 모델이 reasoning 시작 token을 직접 쓰고, decode해도 남음
# 예시) qwen3 출력 "<think>\n...\n</think>\n\n... \\boxed{3}"
GENERATION_CONFIGS = {
    "Qwen/Qwen3-0.6B": {"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}, "reasoning_end": "</think>"},
    "Qwen/Qwen3-1.7B": {"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}, "reasoning_end": "</think>"},
}


def load_model(
    model_path: str,
) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model


def get_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> Tensor:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True, return_tensors="pt")["input_ids"][0]
