from statistics import mean
import time

import math_verify
from vllm import LLM, SamplingParams  # type: ignore
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.dataset import load_math500
from utils.format import format_duration
from utils.hub import result_exists, save_result
from utils.model import GENERATION_CONFIGS

# vllm이 한 sequence에 받는 prompt와 생성 token 수의 최대 합
MAX_MODEL_LEN = 40960


def load_vllm(
    model_path: str,
    model_name: str,
    hf_overrides: dict | None = None,
) -> LLM:
    return LLM(model_path, tokenizer=model_name, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, hf_overrides=hf_overrides)


def get_sampling_params(
    model_name: str,
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[SamplingParams]:
    return [SamplingParams(**GENERATION_CONFIGS[model_name]["sampling"], max_tokens=max_generation_len, seed=prompt_seed) for prompt_seed in prompt_seeds]


def generate_vllm(
    model: LLM,
    model_name: str,
    prompts: list[str],
    prompt_seeds: list[int],
    max_generation_len: int,
) -> list[CompletionOutput]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = model.chat(messages, get_sampling_params(model_name, prompt_seeds, max_generation_len), use_tqdm=True, chat_template_kwargs={"enable_thinking": True})
    return [output.outputs[0] for output in outputs]


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


def get_missing_seeds(
    experiment_name: str,
    hub_path: str,
    file_names: dict[int, str],
) -> list[int]:
    # seed마다 hub result에서 빠진 것 확인
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(experiment_name, hub_path, file_name)]
    if not missing_seeds:
        print(f"{hub_path} already evaluated")
    return missing_seeds


def evaluate_math500(
    model: LLM,
    model_name: str,
    experiment_name: str,
    hub_path: str,
    file_names: dict[int, str],
    missing_seeds: list[int],
    max_generation_len: int,
) -> None:
    problems = load_math500()
    tokenizer = model.get_tokenizer()

    # 1. vllm은 띄우는 데 오래 걸리기 때문에 빠진 seed를 모아 한 번에 생성
    prompts = [prompt for _ in missing_seeds for prompt, _ in problems]
    prompt_seeds = [seed for seed in missing_seeds for _ in problems]
    start_time = time.perf_counter()
    outputs = generate_vllm(model, model_name, prompts, prompt_seeds, max_generation_len)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 2. outputs를 seed별로 나눠 채점하고 저장
    start = 0
    for seed in missing_seeds:
        end = start + len(problems)
        result = evaluate_outputs(outputs[start:end], [answer for _, answer in problems], tokenizer, model_name)
        start = end
        print(f"{hub_path} {file_names[seed]} accuracy {result['accuracy']:.2%} reasoning len {mean(result['reasoning_lens']):.0f} truncated {mean(result['truncateds']):.2%}")
        save_result(experiment_name, hub_path, file_names[seed], result)
