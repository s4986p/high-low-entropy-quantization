from dataclasses import dataclass
import math
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter
from simple_parsing import parse

from utils.hub import load_result, result_exists
from utils.metrics import get_length_ratio, get_loop_ratio, get_truncated_ratio
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, BF16_HATCH, ENTROPY_COLORS, ENTROPY_LABELS, fit_bar_width, format_accuracy, get_accuracy, get_figsize, get_panel_width, plot_bars, save_figure

BF16_POSITION_COLORS = {**ENTROPY_COLORS, "random": "#9e9ac8"}
BF16_POSITION_LABELS = {**ENTROPY_LABELS, "random": "무작위 위치"}
BF16_RATIOS = [0.1, 0.2, 0.8]


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
    file_names = [f"max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds]
    bf16_results = [load_result("math500", make_hub_path("bf16", config.model_name), file_name) for file_name in file_names]
    hub_paths = {method: make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems) for method in config.methods}
    quantized_results = {method: [load_result("math500", hub_paths[method], file_name) for file_name in file_names] for method in config.methods}
    case_results = {}
    for method in config.methods:
        for bf16_position in BF16_POSITION_LABELS:
            for bf16_ratio in BF16_RATIOS:
                case_file_names = [f"bf16_position={bf16_position},bf16_ratio={bf16_ratio},{file_name}" for file_name in file_names]
                if all(result_exists("entropy_bf16_sampling/math500", hub_paths[method], case_file_name) for case_file_name in case_file_names):
                    case_results[method, bf16_position, bf16_ratio] = [load_result("entropy_bf16_sampling/math500", hub_paths[method], case_file_name) for case_file_name in case_file_names]
    bf16_ratios = [bf16_ratio for bf16_ratio in BF16_RATIOS if any(case[2] == bf16_ratio for case in case_results)]

    panels = [
        (lambda result, bf16_result: get_accuracy(result), "정확도", "정확도", ACCURACY_YMAX, 0.2, "{x:.1%}", format_accuracy, PercentFormatter(1, decimals=0)),
        (lambda result, bf16_result: get_truncated_ratio(result), "잘림", "전체 문제 중 비율", 0.3, 0.05, "{x:.1%}", format_accuracy, PercentFormatter(1, decimals=0)),
        (lambda result, bf16_result: get_loop_ratio(result), "반복", "전체 문제 중 비율", 0.3, 0.05, "{x:.1%}", format_accuracy, PercentFormatter(1, decimals=0)),
        (get_length_ratio, "reasoning 길이", "bf16 대비 중앙값", 2, 0.5, "{x:.2f}배", lambda value, bf16_value: f"{value:.2f}배", StrMethodFormatter("{x:.1f}배")),
    ]
    panel_widths = [get_panel_width(len(bf16_ratios), len(BF16_POSITION_LABELS))] * len(panels)
    fig, axes = plt.subplots(len(config.methods), len(panels), figsize=get_figsize(panel_widths, len(config.methods)), squeeze=False, layout="constrained")
    for method_axes, method in zip(axes, config.methods):
        for ax, (get_value, title, ylabel, ymax, tick_step, value_format, format_text, formatter) in zip(method_axes, panels):
            bf16_value = mean(get_value(bf16_result, bf16_result) for bf16_result in bf16_results)
            quantized_value = mean(get_value(result, bf16_result) for result, bf16_result in zip(quantized_results[method], bf16_results))
            series = []
            for bf16_position, color in BF16_POSITION_COLORS.items():
                values = [mean(get_value(result, bf16_result) for result, bf16_result in zip(case_results[method, bf16_position, bf16_ratio], bf16_results)) if (method, bf16_position, bf16_ratio) in case_results else math.nan for bf16_ratio in bf16_ratios]
                series.append((values, color, ["" if math.isnan(value) else format_text(value, bf16_value) for value in values]))
            plot_bars(ax, [f"{bf16_ratio:.0%}" for bf16_ratio in bf16_ratios], series, [bf16_value] * len(series))
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_locator(MultipleLocator(tick_step))
            ax.yaxis.set_major_formatter(formatter)
            ax.set_xlabel("bf16으로 샘플링한 위치 비율")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{method} {title} ({method} {value_format.format(x=quantized_value)}, bf16 {value_format.format(x=bf16_value)})")

    fig.suptitle(f"{config.model_name} math500 bf16 entropy 위치별 bf16 샘플링\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    handles = [Patch(color=color, label=BF16_POSITION_LABELS[bf16_position]) for bf16_position, color in BF16_POSITION_COLORS.items()]
    handles.append(Patch(facecolor="white", edgecolor="#808080", hatch=BF16_HATCH, linewidth=0, label="bf16 대비 손실"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fit_bar_width(fig, list(axes[0]))
    save_figure(fig, config.model_name, "math500_entropy_bf16_sampling")


if __name__ == "__main__":
    main()
