from dataclasses import dataclass
from statistics import mean, median
from typing import Literal

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, LENGTH_YMAX, RATIO_YMAX, get_figsize, get_title, plot_method_bars, save_figure

# 반복: 끝에서 LOOP_TAIL_LEN번째부터 LOOP_PATTERN_LEN 글자가 그 앞에 MIN_LOOP_COUNT번 이상 나옴
LOOP_TAIL_LEN = 200
LOOP_PATTERN_LEN = 50
MIN_LOOP_COUNT = 5


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    methods: tuple[Literal["gptq", "awq", "mxfp4"], ...] = ("gptq", "awq", "mxfp4")
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128
    max_generation_len: int = 32768
    seeds: tuple[int, ...] = (0,)


def get_accuracy(
    result: dict,
) -> float:
    return result["accuracy"]


def get_truncated_ratio(
    result: dict,
) -> float:
    return mean(truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def is_loop(
    text: str | None,
) -> bool:
    return text is not None and len(text) > LOOP_TAIL_LEN and text[:-LOOP_TAIL_LEN].count(text[-LOOP_TAIL_LEN : -LOOP_TAIL_LEN + LOOP_PATTERN_LEN]) >= MIN_LOOP_COUNT


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(reasoning) or is_loop(response) for reasoning, response in zip(result["reasonings"], result["responses"]))


def get_length_ratio(
    result: dict,
    bf16_result: dict,
) -> float:
    # bf16과 양자화 모델이 모두 맞힌 문제에서 bf16 대비 reasoning token 수 배의 중앙값
    return median(reasoning_len / bf16_reasoning_len for reasoning_len, bf16_reasoning_len, correct, bf16_correct in zip(result["reasoning_lens"], bf16_result["reasoning_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct)


def main() -> None:
    config = parse(Config)
    results = {"bf16": load_results("math500", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)}
    for method in config.methods:
        hub_path = make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
        results[method] = load_results("math500", hub_path, config.max_generation_len, config.seeds)
    methods = list(config.methods)

    # 정확도, 잘림, 반복은 bf16도 함께 표시
    fig, axes = plt.subplots(1, 4, figsize=get_figsize(len(results), 4), layout="constrained")
    plot_method_bars(axes[0], list(results), [mean(get_accuracy(result) for result in results[method]) for method in results], "%")
    plot_method_bars(axes[1], list(results), [mean(get_truncated_ratio(result) for result in results[method]) for method in results], "%")
    plot_method_bars(axes[2], list(results), [mean(get_loop_ratio(result) for result in results[method]) for method in results], "%")
    plot_method_bars(axes[3], methods, [mean(get_length_ratio(result, bf16_result) for result, bf16_result in zip(results[method], results["bf16"])) for method in methods], "배")
    for ax, ylabel in zip(axes, ["정확도", "잘림", "반복", "reasoning 길이"]):
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0, ACCURACY_YMAX)
    for ax in axes[1:3]:
        ax.set_ylim(0, RATIO_YMAX)
    axes[3].set_ylim(0, LENGTH_YMAX)

    fig.suptitle(get_title(config.model_name, "양자화 손실"))
    save_figure(fig, config.model_name, "math500_loss")


if __name__ == "__main__":
    main()
