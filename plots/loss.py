from dataclasses import dataclass
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.metrics import get_accuracy, get_length_ratio, get_loop_ratio, get_truncated_ratio
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, LENGTH_YMAX, METHOD_COLORS, RATIO_YMAX, get_figsize, get_panel_width, get_title, plot_bars, save_figure


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    methods: tuple[Literal["gptq", "awq", "mxfp4"], ...] = ("gptq", "awq", "mxfp4")
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128
    max_generation_len: int = 32768
    seeds: tuple[int, ...] = (0,)


def main() -> None:
    config = parse(Config)
    results = {"bf16": load_results("math500", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)}
    for method in config.methods:
        hub_path = make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
        results[method] = load_results("math500", hub_path, config.max_generation_len, config.seeds)
    methods = list(config.methods)
    colors = [METHOD_COLORS[method] for method in results]

    # 정확도, 잘림, 반복은 bf16도 함께 표시
    fig, axes = plt.subplots(1, 4, figsize=get_figsize([get_panel_width(len(results), 1)] * 4, 1), layout="constrained")
    plot_bars(axes[0], list(results), [([mean(get_accuracy(result) for result in results[method]) for method in results], colors)], "%")
    plot_bars(axes[1], list(results), [([mean(get_truncated_ratio(result) for result in results[method]) for method in results], colors)], "%")
    plot_bars(axes[2], list(results), [([mean(get_loop_ratio(result) for result in results[method]) for method in results], colors)], "%")
    plot_bars(axes[3], methods, [([mean(get_length_ratio(result, bf16_result) for result, bf16_result in zip(results[method], results["bf16"])) for method in methods], colors[1:])], "배")
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
