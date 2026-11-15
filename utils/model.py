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


def get_reasoning_ids(
    tokenizer: PreTrainedTokenizerBase,
    reasoning: str,
) -> Tensor:
    return tokenizer(reasoning, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def get_reasoning_hiddens(
    model: PreTrainedModel,
    input_ids: Tensor,
    prompt_len: int,
) -> Tensor:
    # t의 hidden이 t+1 token을 예측하기 때문에 마지막 prompt token부터 끝에서 두 번째 reasoning token까지 유지
    # 예시) 입력 [p0, p1, p2, r0, r1, r2]
    # 예시) 유지 [        p2, r0, r1    ] (각각 r0, r1, r2 예측)
    return model.model(input_ids=input_ids[None, :].to(model.device), use_cache=False).last_hidden_state[0, prompt_len - 1 : -1]
