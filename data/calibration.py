from dataclasses import dataclass
from statistics import mean
import time

from datasets import load_dataset
from simple_parsing import parse
import torch

from utils.dataset import MATH_PROMPT
from utils.evaluation import generate_vllm, load_vllm
from utils.format import format_duration
from utils.hub import save_tensors, tensors_exist
from utils.model import get_prompt_ids
from utils.paths import make_calibration_name, make_hub_path


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    num_calibration_problems: int = 128
    max_generation_len: int = 2048
    seed: int = 0


def load_numina(
    num_calibration_problems: int,
    seed: int,
) -> list[str]:
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train").shuffle(seed=seed)
    prompts = []
    for row in dataset:
        # 합성 문제, 서술형이 아닌 문제, 문제가 잘못된 것으로 표시된 문제 제외
        if row["synthetic"] or row["question_type"] != "math-word-problem" or row["problem_is_valid"] != "Yes":
            continue
        prompts.append(MATH_PROMPT.format(problem=row["problem"]))
        if len(prompts) == num_calibration_problems:
            break
    return prompts


def main() -> None:
    config = parse(Config)
    experiment_name = "calibration/numina"
    hub_path = make_hub_path("bf16", config.model_name)
    file_name = make_calibration_name(config.num_calibration_problems, config.max_generation_len, config.seed)
    if tensors_exist(experiment_name, hub_path, file_name):
        print(f"{hub_path} already generated")
        return
    prompts = load_numina(config.num_calibration_problems, config.seed)
    model = load_vllm(config.model_name, config.model_name)
    tokenizer = model.get_tokenizer()

    # 1. bf16으로 numina 문제의 reasoning 생성
    start_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, prompts, [config.seed] * len(prompts), config.max_generation_len)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)} output len {mean(len(output.token_ids) for output in outputs):.0f}")

    # 2. prompt 뒤에 생성한 token 전체를 이어 저장
    # 예시) prompt ids + "<think>\n...</think>\n\n... \\boxed{3}"
    calibration_ids = [torch.cat([get_prompt_ids(tokenizer, prompt), torch.tensor(output.token_ids)]) for prompt, output in zip(prompts, outputs)]
    save_tensors(experiment_name, hub_path, file_name, {"calibration_ids": torch.cat(calibration_ids), "calibration_lens": torch.tensor([len(ids) for ids in calibration_ids])})


if __name__ == "__main__":
    main()
