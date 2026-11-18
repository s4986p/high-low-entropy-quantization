from dataclasses import dataclass
import math
from statistics import mean
from typing import Literal

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results, result_exists
from utils.metrics import get_accuracy, get_length_ratio, get_loop_ratio, get_truncated_ratio
from utils.paths import make_hub_path, make_result_name
from utils.plot import ACCURACY_YMAX, ENTROPY_COLORS, ENTROPY_NAMES, LENGTH_YMAX, METHOD_COLORS, RATIO_YMAX, REFERENCE_LINESTYLE, get_figsize, get_panel_width, get_title, plot_bars, save_figure

BF16_POSITION_NAMES = {**ENTROPY_NAMES, "random": "무작위 위치"}
BF16_POSITION_COLORS = {**ENTROPY_COLORS, "random": "#9e9ac8"}
BF16_RATIOS = [0.1, 0.2, 0.8]
# 정확도, 잘림, 반복 panel마다 (지표 함수, y축 이름, unit, y축 상한)
PANELS = [(get_accuracy, "정확도", "%", ACCURACY_YMAX), (get_truncated_ratio, "잘림", "%", RATIO_YMAX), (get_loop_ratio, "반복", "%", RATIO_YMAX)]


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
    bf16_results = load_results("math500", make_hub_path("bf16", config.model_name), config.max_generation_len, config.seeds)
    quantized_results = {}
    sampling_results = {}
    for method in config.methods:
        hub_path = make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
        quantized_results[method] = load_results("math500", hub_path, config.max_generation_len, config.seeds)
        # 돌린 위치와 비율만 불러옴
        for bf16_position in BF16_POSITION_NAMES:
            for bf16_ratio in BF16_RATIOS:
                if all(result_exists("entropy_bf16_sampling/math500", hub_path, make_result_name(config.max_generation_len, seed, bf16_position, bf16_ratio)) for seed in config.seeds):
                    sampling_results[method, bf16_position, bf16_ratio] = load_results("entropy_bf16_sampling/math500", hub_path, config.max_generation_len, config.seeds, bf16_position, bf16_ratio)
    bf16_ratios = [bf16_ratio for bf16_ratio in BF16_RATIOS if any(case[2] == bf16_ratio for case in sampling_results)]
    cases = [f"{bf16_ratio:.0%}" for bf16_ratio in bf16_ratios]

    panel_widths = [get_panel_width(len(bf16_ratios), len(BF16_POSITION_NAMES))] * (len(PANELS) + 1)
    fig, axes = plt.subplots(len(config.methods), len(panel_widths), figsize=get_figsize(panel_widths, len(config.methods)), squeeze=False, layout="constrained")
    for method_axes, method in zip(axes, config.methods):
        # 1. 정확도, 잘림, 반복은 bf16과 bf16 샘플링 없는 양자화 모델을 점선으로 함께 표시
        for ax, (get_value, ylabel, unit, ymax) in zip(method_axes, PANELS):
            series = []
            for bf16_position, color in BF16_POSITION_COLORS.items():
                values = [mean(get_value(result) for result in sampling_results[method, bf16_position, bf16_ratio]) if (method, bf16_position, bf16_ratio) in sampling_results else math.nan for bf16_ratio in bf16_ratios]
                series.append((values, color))
            plot_bars(ax, cases, series, unit)
            ax.axhline(mean(get_value(result) for result in bf16_results), color=METHOD_COLORS["bf16"], linestyle=REFERENCE_LINESTYLE)
            ax.axhline(mean(get_value(result) for result in quantized_results[method]), color=METHOD_COLORS[method], linestyle=REFERENCE_LINESTYLE)
            ax.set_ylim(0, ymax)
            ax.set_ylabel(ylabel)

        # 2. reasoning 길이는 1배 기준선이 bf16
        series = []
        for bf16_position, color in BF16_POSITION_COLORS.items():
            values = [mean(get_length_ratio(result, bf16_result) for result, bf16_result in zip(sampling_results[method, bf16_position, bf16_ratio], bf16_results)) if (method, bf16_position, bf16_ratio) in sampling_results else math.nan for bf16_ratio in bf16_ratios]
            series.append((values, color))
        plot_bars(method_axes[-1], cases, series, "배")
        method_axes[-1].axhline(mean(get_length_ratio(result, bf16_result) for result, bf16_result in zip(quantized_results[method], bf16_results)), color=METHOD_COLORS[method], linestyle=REFERENCE_LINESTYLE)
        method_axes[-1].set_ylim(0, LENGTH_YMAX)
        method_axes[-1].set_ylabel("reasoning 길이")
        for ax in method_axes:
            ax.set_title(method)

    fig.suptitle(get_title(config.model_name, "bf16 entropy 위치별 bf16 샘플링"))
    handles = [Patch(color=color, label=BF16_POSITION_NAMES[bf16_position]) for bf16_position, color in BF16_POSITION_COLORS.items()]
    handles += [Line2D([], [], color=METHOD_COLORS[method], linestyle=REFERENCE_LINESTYLE, label=f"{method} bf16 샘플링 없음") for method in config.methods]
    handles.append(Line2D([], [], color=METHOD_COLORS["bf16"], linestyle=REFERENCE_LINESTYLE, label="bf16"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save_figure(fig, config.model_name, "math500_entropy_bf16_sampling")


if __name__ == "__main__":
    main()
