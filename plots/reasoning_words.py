from collections import Counter
from dataclasses import dataclass
import re
from statistics import mean
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_result
from utils.paths import make_hub_path
from utils.plot import BAR_GAP, METHOD_COLORS, save_figure

STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}
MARKER_WORDS = ["wait", "but", "however", "hmm", "wrong", "alternatively", "maybe", "perhaps", "verify", "confirm", "remember", "sure"]
WORD_BAR_INCHES = 0.12
WORD_PANEL_WIDTH = 4.5


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    methods: tuple[Literal["gptq", "awq", "mxfp4"], ...] = ("gptq", "awq", "mxfp4")
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128
    max_generation_len: int = 32768
    seed: int = 0
    num_words: int = 15


def get_words(
    text: str,
) -> list[str]:
    words = re.findall(r"(?<![\\a-z])[a-z]+(?:'[a-z]+)?(?:-[a-z]{2,})*", text.lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


def get_word_rates(
    reasoning: str,
    reasoning_len: int,
) -> dict[str, float]:
    return {word: count / reasoning_len * 1000 for word, count in Counter(get_words(reasoning)).items()}


def plot_words(
    axes: list[Axes],
    bf16_rates: dict[str, float],
    quantized_rates: dict[str, float],
    method: str,
    num_words: int,
) -> None:
    words = sorted(bf16_rates.keys() | quantized_rates.keys(), key=lambda word: quantized_rates.get(word, 0) - bf16_rates.get(word, 0), reverse=True)
    word_groups = [("늘어난 단어", words[:num_words]), ("줄어든 단어", words[::-1][:num_words]), ("의심, 분기, 확인 단어", MARKER_WORDS)]
    word_width = 2 + BAR_GAP
    num_slots = max(len(group_words) for _, group_words in word_groups)
    for ax, (title, group_words) in zip(axes, word_groups):
        for series_index, (color, rates) in enumerate([(METHOD_COLORS["bf16"], bf16_rates), (METHOD_COLORS[method], quantized_rates)]):
            ys = [word_index * word_width + series_index - 0.5 for word_index in range(len(group_words))]
            ax.barh(ys, [rates.get(word, 0) for word in group_words], height=1, color=color)
        for word_index, word in enumerate(group_words):
            bf16_rate, quantized_rate = bf16_rates.get(word, 0), quantized_rates.get(word, 0)
            ax.annotate(f"{quantized_rate / bf16_rate:.2f}배" if bf16_rate else "새 단어", (max(bf16_rate, quantized_rate), word_index * word_width), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks([word_index * word_width for word_index in range(len(group_words))], group_words)
        ax.set_ylim((num_slots - 0.5) * word_width, -word_width / 2)
        ax.margins(x=0.2)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("reasoning 1000 token당 평균 횟수")


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    bf16_result = load_result("math500", make_hub_path("bf16", config.model_name), file_name)

    row_height = WORD_BAR_INCHES * max(config.num_words, len(MARKER_WORDS)) * (2 + BAR_GAP) + 1.2
    fig = plt.figure(figsize=(WORD_PANEL_WIDTH * 3, row_height * len(config.methods) + 1), layout="constrained")
    for subfig, method in zip(fig.subfigures(len(config.methods), 1, squeeze=False)[:, 0], config.methods):
        result = load_result("math500", make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems), file_name)
        bf16_word_rates = []
        quantized_word_rates = []
        for reasoning, reasoning_len, truncated, bf16_reasoning, bf16_reasoning_len, bf16_correct in zip(result["reasonings"], result["reasoning_lens"], result["truncateds"], bf16_result["reasonings"], bf16_result["reasoning_lens"], bf16_result["corrects"]):
            if bf16_correct and not truncated:
                bf16_word_rates.append(get_word_rates(bf16_reasoning, bf16_reasoning_len))
                quantized_word_rates.append(get_word_rates(reasoning, reasoning_len))
        words = set().union(*bf16_word_rates, *quantized_word_rates)
        bf16_rates = {word: mean(rates.get(word, 0) for rates in bf16_word_rates) for word in words}
        quantized_rates = {word: mean(rates.get(word, 0) for rates in quantized_word_rates) for word in words}
        plot_words(subfig.subplots(1, 3), bf16_rates, quantized_rates, method, config.num_words)
        subfig.suptitle(f"{method} (사용한 문제 수 {len(quantized_word_rates)})")

    fig.suptitle(f"{config.model_name} math500 reasoning 단어\n{file_name}")
    fig.legend(handles=[Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *config.methods]], loc="outside lower center", ncol=len(config.methods) + 1)
    save_figure(fig, config.model_name, "math500_reasoning_words")


if __name__ == "__main__":
    main()
