from dataclasses import dataclass
from statistics import mean
import time
from typing import Literal

import math_verify
from simple_parsing import parse
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.dataset import load_math500
from utils.evaluation import generate_vllm, load_vllm
from utils.format import format_duration
from utils.hub import download_model, result_exists, save_result
from utils.model import GENERATION_CONFIGS
from utils.paths import make_hub_path, make_result_name


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["bf16", "gptq", "awq", "mxfp4"] = "bf16"
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128
    max_generation_len: int = 32768
    seeds: tuple[int, ...] = (0,)


def parse_math500_prediction(
    response: str | None,
) -> list | None:
    # math_verify로 response에서 답 수식 추출, response가 없으면 None
    # 예시) "... \boxed{\frac{1}{2}}" → [1/2, "\frac{1}{2}"]
    return None if response is None else math_verify.parse(response)


def split_outputs(
    outputs: list[CompletionOutput],
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])
    reasonings = []
    responses = []
    reasoning_lens = []
    for output in outputs:
        token_ids = list(output.token_ids)
        if reasoning_end_id in token_ids:
            reasoning_len = token_ids.index(reasoning_end_id)
            response = tokenizer.decode(token_ids[reasoning_len + 1 :], skip_special_tokens=True)
        else:
            # reasoning_end token이 없으면 reasoning이 안 끝난 것이라 response 없음
            # 대부분 max_generation_len에 닿은 경우
            reasoning_len = len(token_ids)
            response = None
        reasonings.append(tokenizer.decode(token_ids[:reasoning_len], skip_special_tokens=True))
        responses.append(response)
        reasoning_lens.append(reasoning_len)
    return {
        "reasoning_lens": reasoning_lens,
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "reasonings": reasonings,
        "responses": responses,
    }


def evaluate_outputs(
    outputs: list[CompletionOutput],
    answers: list[str],
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    split = split_outputs(outputs, tokenizer, model_name)
    predictions = [parse_math500_prediction(response) for response in split["responses"]]
    # 정답과 수식으로 비교
    # 예시) 답 1/2, 정답 "0.5" → 맞음
    corrects = [prediction is not None and math_verify.verify(math_verify.parse(f"${answer}$"), prediction) for prediction, answer in zip(predictions, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": [str(prediction[-1]) if prediction else None for prediction in predictions],
        **split,
    }


def main() -> None:
    config = parse(Config)
    experiment_name = "math500"
    hub_path = make_hub_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
    file_names = {seed: make_result_name(config.max_generation_len, seed) for seed in config.seeds}

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
