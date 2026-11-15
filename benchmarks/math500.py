from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

from simple_parsing import parse

from utils.dataset import load_math500
from utils.evaluation import evaluate_outputs
from utils.format import format_duration
from utils.generation import generate_vllm, load_vllm
from utils.hub import download_model, result_exists, save_result
from utils.paths import make_hub_path


@dataclass
class Config:
    seeds: tuple[int, ...] = (0,)
    max_generation_len: int = 32768
    # 평가할 모델
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["bf16", "gptq", "awq", "mxfp4"] = "bf16"
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128


def main() -> None:
    config = parse(Config)
    experiment_name = "math500"
    hub_path = make_hub_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
    file_names = {seed: f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds}

    # 1. seed마다 hub result에서 빠진 것 확인, 빠진 것이 없으면 종료
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(experiment_name, hub_path, file_name)]
    if not missing_seeds:
        print(f"{hub_path} already evaluated")
        return
    problems = load_math500()

    # 2. vllm은 띄우는 데 오래 걸리기 때문에 빠진 seed를 모아 한 번에 생성
    model_path = config.model_name if config.method == "bf16" else download_model(hub_path)
    model = load_vllm(model_path, config.model_name)
    tokenizer = model.get_tokenizer()
    prompts = [prompt for _ in missing_seeds for prompt, _ in problems]
    prompt_seeds = [seed for seed in missing_seeds for _ in problems]
    start_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, prompts, prompt_seeds, config.max_generation_len)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 3. outputs를 seed별로 나눠 채점하고 저장
    start = 0
    for seed in missing_seeds:
        end = start + len(problems)
        result = evaluate_outputs(outputs[start:end], [answer for _, answer in problems], tokenizer, config.model_name)
        start = end
        print(f"{hub_path} seed {seed} accuracy {result['accuracy']:.2%} reasoning len {mean(result['reasoning_lens']):.0f} truncated {mean(result['truncateds']):.2%}")
        save_result(experiment_name, hub_path, file_names[seed], result)


if __name__ == "__main__":
    main()
