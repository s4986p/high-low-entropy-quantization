from dataclasses import dataclass
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse
import torch
from torch import Tensor
from transformers import AutoTokenizer

from utils.hub import load_result, load_tensors
from utils.metrics import get_high_entropy_mask
from utils.paths import make_hub_path, make_result_name
from utils.plot import ENTROPY_COLORS, ENTROPY_NAMES, METHOD_COLORS, VALUE_MARGIN, get_figsize, get_panel_width, get_title, plot_bars, save_figure

TOKEN_TYPE_NAMES = {"digit": "숫자", "symbol": "기호", "alpha": "알파벳", "other": "기타"}
TOKEN_TYPES = list(TOKEN_TYPE_NAMES)


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    methods: tuple[Literal["gptq", "awq", "mxfp4"], ...] = ("gptq", "awq", "mxfp4")
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128
    high_entropy_ratio: float = 0.2
    max_generation_len: int = 32768
    seed: int = 0


def get_token_types(
    tokens: list[str],
) -> Tensor:
    # token마다 앞뒤 공백을 빼고 token 종류 index
    # 예시) [" 12", " +", " the", " x2"] → [숫자, 기호, 알파벳, 기타]
    token_types = []
    for token in tokens:
        token = token.strip()
        if token.isdigit():
            token_types.append(TOKEN_TYPES.index("digit"))
        elif token and not any(char.isalnum() for char in token):
            token_types.append(TOKEN_TYPES.index("symbol"))
        elif token.isalpha():
            token_types.append(TOKEN_TYPES.index("alpha"))
        else:
            token_types.append(TOKEN_TYPES.index("other"))
    return torch.tensor(token_types)


def main() -> None:
    config = parse(Config)
    file_name = make_result_name(config.max_generation_len, config.seed)
    results = {method: load_tensors("entropy_kl/math500", make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems), file_name) for method in config.methods}
    num_problems = sum(load_result("math500", make_hub_path("bf16", config.model_name), file_name)["corrects"])
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    vocab_types = get_token_types([tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])
    type_names = list(TOKEN_TYPE_NAMES.values())

    # kl 비중은 위치 2개, top-1 불일치와 tail mass는 token 종류마다 막대 2개
    panel_widths = [get_panel_width(len(ENTROPY_NAMES), 1), get_panel_width(len(TOKEN_TYPES), 2), get_panel_width(len(TOKEN_TYPES), 2)]
    fig, axes = plt.subplots(len(results), len(panel_widths), figsize=get_figsize(panel_widths, len(results)), width_ratios=panel_widths, squeeze=False, layout="constrained")
    for (kl_ax, mismatch_ax, tail_mass_ax), (method, result) in zip(axes, results.items()):
        is_high_entropy = get_high_entropy_mask(result["bf16_entropy"], config.high_entropy_ratio)
        entropy_masks = {"low": ~is_high_entropy, "high": is_high_entropy}
        is_mismatch = (result["bf16_top1_ids"] != result["quantized_top1_ids"]).float()
        reasoning_types = vocab_types[result["reasoning_ids"]]

        # 1. 위치마다 전체 kl 중 비율
        kl_shares = [(result["kl"][entropy_mask].sum() / result["kl"].sum()).item() for entropy_mask in entropy_masks.values()]
        plot_bars(kl_ax, list(ENTROPY_NAMES.values()), [(kl_shares, list(ENTROPY_COLORS.values()))], "%")
        kl_ax.set_ylabel("kl 비중")
        kl_ax.set_title(method)

        # 2. 위치와 token 종류마다 top-1 불일치 비율
        series = []
        for entropy, entropy_mask in entropy_masks.items():
            series.append(([is_mismatch[(reasoning_types == type_index) & entropy_mask].mean().item() for type_index in range(len(TOKEN_TYPES))], ENTROPY_COLORS[entropy]))
        plot_bars(mismatch_ax, type_names, series, "%")
        mismatch_ax.set_ylabel("top-1 불일치")
        mismatch_ax.set_title(method)

        # 3. 저엔트로피 위치에서 token 종류마다 bf16과 양자화 모델의 tail mass 평균
        series = []
        for color, tail_mass in [(METHOD_COLORS["bf16"], result["bf16_tail_mass"]), (METHOD_COLORS[method], result["quantized_tail_mass"])]:
            series.append(([tail_mass[(reasoning_types == type_index) & entropy_masks["low"]].mean().item() for type_index in range(len(TOKEN_TYPES))], color))
        plot_bars(tail_mass_ax, type_names, series, "%")
        tail_mass_ax.set_ylabel("tail mass")
        tail_mass_ax.set_title(f"{method} 저엔트로피 위치")

        for ax in [kl_ax, mismatch_ax, tail_mass_ax]:
            ax.margins(y=VALUE_MARGIN)
            ax.set_ylim(bottom=0)

    fig.suptitle(get_title(config.model_name, "bf16 entropy별 bf16과의 kl"))
    # 문제는 bf16이 맞힌 문제만
    handles = [Patch(color=ENTROPY_COLORS[entropy], label=name) for entropy, name in ENTROPY_NAMES.items()]
    handles.append(Patch(color=METHOD_COLORS["bf16"], label=f"bf16 ({num_problems}문제)"))
    handles += [Patch(color=METHOD_COLORS[method], label=method) for method in results]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save_figure(fig, config.model_name, "math500_entropy_kl")


if __name__ == "__main__":
    main()
