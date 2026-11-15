from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Literal

from simple_parsing import parse
import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer

from utils.dataset import load_math500
from utils.format import format_duration, format_memory
from utils.hub import download_model, load_result, save_tensors, tensors_exist
from utils.model import get_prompt_ids, get_reasoning_hiddens, get_reasoning_ids, load_model
from utils.paths import make_hub_path


@dataclass
class Config:
    seed: int = 0
    max_generation_len: int = 32768
    kl_batch_len: int = 2048
    # bf16과 비교할 모델
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["gptq", "awq", "mxfp4"] = "mxfp4"
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    experiment_name = "entropy_kl/math500"
    hub_path = make_hub_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    if tensors_exist(experiment_name, hub_path, file_name):
        print(f"{hub_path} already evaluated")
        return
    problems = load_math500()
    bf16_result = load_result("math500", make_hub_path("bf16", config.model_name), file_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bf16_model = load_model(config.model_name)
    quantized_model = load_model(download_model(hub_path))

    # 1. bf16이 맞힌 문제에서 bf16 reasoning 추출
    problem_inputs = [(get_prompt_ids(tokenizer, prompt), get_reasoning_ids(tokenizer, reasoning)) for (prompt, _), reasoning, correct in zip(problems, bf16_result["reasonings"], bf16_result["corrects"]) if correct]
    print(f"{hub_path} problems {len(problem_inputs)} of {len(problems)} positions {sum(len(reasoning_ids) for _, reasoning_ids in problem_inputs)}")

    # 2. reasoning token마다 bf16과 양자화 모델의 다음 token 예측 비교
    tensors = defaultdict(list)
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for input_index, (prompt_ids, reasoning_ids) in enumerate(problem_inputs):
        input_ids = torch.cat([prompt_ids, reasoning_ids])
        bf16_hiddens = get_reasoning_hiddens(bf16_model, input_ids, len(prompt_ids))
        quantized_hiddens = get_reasoning_hiddens(quantized_model, input_ids, len(prompt_ids))
        # logit은 메모리 때문에 kl_batch_len token씩 나눠 계산
        for batch_start in range(0, len(reasoning_ids), config.kl_batch_len):
            bf16_logprobs = log_softmax(bf16_model.lm_head(bf16_hiddens[batch_start : batch_start + config.kl_batch_len]).float(), dim=-1)
            quantized_logprobs = log_softmax(quantized_model.lm_head(quantized_hiddens[batch_start : batch_start + config.kl_batch_len]).float(), dim=-1)
            bf16_top1_ids = bf16_logprobs.argmax(dim=-1)
            tensors["bf16_entropy"].append(-(bf16_logprobs.exp() * bf16_logprobs).sum(dim=-1).cpu())
            tensors["kl"].append((bf16_logprobs.exp() * (bf16_logprobs - quantized_logprobs)).sum(dim=-1).cpu())
            tensors["bf16_top1_ids"].append(bf16_top1_ids.cpu())
            tensors["quantized_top1_ids"].append(quantized_logprobs.argmax(dim=-1).cpu())
            # tail mass: bf16 top-1 token 말고 다른 token이 뽑힐 확률, 양자화 모델도 bf16 top-1 token 기준
            tensors["bf16_tail_mass"].append(1 - bf16_logprobs.gather(-1, bf16_top1_ids[:, None])[:, 0].exp().cpu())
            tensors["quantized_tail_mass"].append(1 - quantized_logprobs.gather(-1, bf16_top1_ids[:, None])[:, 0].exp().cpu())
        tensors["reasoning_ids"].append(reasoning_ids)

        if (input_index + 1) % 50 == 0:
            print(f"{hub_path} problem {input_index + 1}/{len(problem_inputs)} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
            start_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()

    tensors = {name: torch.cat(values) for name, values in tensors.items()}
    print(f"{hub_path} kl {tensors['kl'].mean():.4f} bf16 entropy {tensors['bf16_entropy'].mean():.4f} top1 mismatch {(tensors['bf16_top1_ids'] != tensors['quantized_top1_ids']).float().mean():.2%}")
    save_tensors(experiment_name, hub_path, file_name, tensors)


if __name__ == "__main__":
    main()
