from dataclasses import dataclass
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer

from utils.hub import load_result, load_tensors
from utils.paths import make_hub_path
from utils.plot import ENTROPY_COLORS, ENTROPY_LABELS, METHOD_COLORS, fit_bar_width, get_figsize, get_panel_width, plot_bars, save_figure

TOKEN_CLASSES = ["digit", "symbol", "alpha", "other"]
TOKEN_CLASS_LABELS = {"digit": "숫자", "symbol": "기호", "alpha": "알파벳", "other": "기타"}


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    methods: tuple[Literal["gptq", "awq", "mxfp4"], ...] = ("gptq", "awq", "mxfp4")
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128
    max_generation_len: int = 32768
    seed: int = 0
    high_entropy_ratio: float = 0.2


def get_token_classes(
    tokens: list[str],
) -> Tensor:
    classes = []
    for token in tokens:
        token = token.strip()
        if token.isdigit():
            classes.append(TOKEN_CLASSES.index("digit"))
        elif token and not any(char.isalnum() for char in token):
            classes.append(TOKEN_CLASSES.index("symbol"))
        elif token.isalpha():
            classes.append(TOKEN_CLASSES.index("alpha"))
        else:
            classes.append(TOKEN_CLASSES.index("other"))
    return torch.tensor(classes)


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    results = {method: load_tensors("entropy_kl/math500", make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems), file_name) for method in config.methods}
    num_problems = sum(load_result("math500", make_hub_path("bf16", config.model_name), file_name)["corrects"])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    token_classes = get_token_classes([tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])
    class_labels = [TOKEN_CLASS_LABELS[token_class] for token_class in TOKEN_CLASSES]

    panel_widths = [get_panel_width(len(ENTROPY_LABELS), 1), get_panel_width(len(TOKEN_CLASSES), 2), get_panel_width(len(TOKEN_CLASSES), 2)]
    fig, axes = plt.subplots(len(results), len(panel_widths), figsize=get_figsize(panel_widths, len(results)), width_ratios=panel_widths, squeeze=False, layout="constrained")
    for (kl_ax, mismatch_ax, tail_mass_ax), (method, result) in zip(axes, results.items()):
        is_high_entropy = result["bf16_entropy"] > result["bf16_entropy"].quantile(1 - config.high_entropy_ratio)
        entropy_masks = {"low": ~is_high_entropy, "high": is_high_entropy}
        is_mismatch = (result["bf16_top1_ids"] != result["quantized_top1_ids"]).float()
        reasoning_classes = token_classes[result["reasoning_ids"]]

        kl_shares = [(result["kl"][entropy_mask].sum() / result["kl"].sum()).item() for entropy_mask in entropy_masks.values()]
        plot_bars(kl_ax, list(ENTROPY_LABELS.values()), [(kl_shares, list(ENTROPY_COLORS.values()), [f"{kl_share:.1%}" for kl_share in kl_shares])])
        kl_ax.set_ylabel("전체 kl 중 비율")
        kl_ax.set_title(f"{method} 위치별 kl 비중")

        series = []
        for entropy, entropy_mask in entropy_masks.items():
            mismatch_ratios = [is_mismatch[(reasoning_classes == class_index) & entropy_mask].mean().item() for class_index in range(len(TOKEN_CLASSES))]
            series.append((mismatch_ratios, ENTROPY_COLORS[entropy], [f"{mismatch_ratio:.1%}" for mismatch_ratio in mismatch_ratios]))
        plot_bars(mismatch_ax, class_labels, series)
        mismatch_ax.set_ylabel("bf16과 top-1이 다른 비율")
        mismatch_ax.set_title(f"{method} token 종류별 top-1 불일치")

        series = []
        for color, tail_mass in [(METHOD_COLORS["bf16"], result["bf16_tail_mass"]), (METHOD_COLORS[method], result["quantized_tail_mass"])]:
            class_tail_masses = [tail_mass[(reasoning_classes == class_index) & ~is_high_entropy].mean().item() for class_index in range(len(TOKEN_CLASSES))]
            series.append((class_tail_masses, color, [f"{class_tail_mass:.1%}" for class_tail_mass in class_tail_masses]))
        plot_bars(tail_mass_ax, class_labels, series)
        tail_mass_ax.set_ylabel("bf16 top-1 외 token 확률")
        tail_mass_ax.set_title(f"{method} 저엔트로피 위치의 tail mass")

        for ax in [kl_ax, mismatch_ax, tail_mass_ax]:
            ax.margins(y=0.2)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    fig.suptitle(f"{config.model_name} math500 bf16 entropy별 bf16과의 kl (사용한 문제 수 {num_problems})\n{file_name},high_entropy_ratio={config.high_entropy_ratio:g}")
    handles = [Patch(color=ENTROPY_COLORS[entropy], label=label) for entropy, label in ENTROPY_LABELS.items()]
    handles += [Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *results]]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fit_bar_width(fig, list(axes[0]))
    save_figure(fig, config.model_name, "math500_entropy_kl")


if __name__ == "__main__":
    main()
