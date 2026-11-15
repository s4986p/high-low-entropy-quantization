from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter
from simple_parsing import parse

from utils.hub import load_result
from utils.metrics import get_length_ratio, get_loop_ratio, get_truncated_ratio
from utils.paths import make_hub_path
from utils.plot import ACCURACY_YMAX, BF16_HATCH, METHOD_COLORS, fit_bar_width, format_accuracy, get_accuracy, get_figsize, get_panel_width, plot_bars, save_figure


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
    results = {}
    for method in config.methods:
        hub_path = make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
        results[method] = [load_result("math500", hub_path, file_name) for file_name in file_names]
    colors = [METHOD_COLORS[method] for method in results]

    panels = [
        (get_accuracy, "정확도", "정확도", ACCURACY_YMAX, 0.2),
        (get_truncated_ratio, "잘림", "전체 문제 중 비율", 0.3, 0.05),
        (get_loop_ratio, "반복", "전체 문제 중 비율", 0.3, 0.05),
    ]
    panel_widths = [get_panel_width(len(results), 1)] * (len(panels) + 1)
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=get_figsize(panel_widths, 1), layout="constrained")
    for ax, (get_value, title, ylabel, ymax, tick_step) in zip(axes, panels):
        bf16_value = mean(get_value(bf16_result) for bf16_result in bf16_results)
        values = [mean(get_value(result) for result in method_results) for method_results in results.values()]
        plot_bars(ax, list(results), [(values, colors, [format_accuracy(value, bf16_value) for value in values])], [bf16_value])
        ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (bf16 {bf16_value:.1%})")
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    length_ratios = [mean(get_length_ratio(result, bf16_result) for result, bf16_result in zip(method_results, bf16_results)) for method_results in results.values()]
    plot_bars(axes[-1], list(results), [(length_ratios, colors, [f"{length_ratio:.2f}배" for length_ratio in length_ratios])])
    axes[-1].set_ylim(0, 2)
    axes[-1].yaxis.set_major_locator(MultipleLocator(0.5))
    axes[-1].set_ylabel("bf16 대비 중앙값")
    axes[-1].set_title("reasoning 길이")
    axes[-1].yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}배"))

    fig.suptitle(f"{config.model_name} math500 양자화 손실\nmax_generation_len={config.max_generation_len},seeds={','.join(map(str, config.seeds))}")
    fig.legend(handles=[Patch(facecolor="white", edgecolor="#808080", hatch=BF16_HATCH, linewidth=0, label="bf16 대비 손실")], loc="outside lower center")
    fit_bar_width(fig, list(axes))
    save_figure(fig, config.model_name, "math500_loss")


if __name__ == "__main__":
    main()
