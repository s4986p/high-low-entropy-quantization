from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_result
from utils.metrics import WORD_RATE_TOKENS, get_words
from utils.paths import make_hub_path, make_result_name
from utils.plot import METHOD_COLORS, get_title, get_words_figsize, plot_words, save_figure


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


def get_word_rates(
    reasoning: str,
    reasoning_len: int,
) -> dict[str, float]:
    return {word: count / reasoning_len * WORD_RATE_TOKENS for word, count in Counter(get_words(reasoning)).items()}


def main() -> None:
    config = parse(Config)
    file_name = make_result_name(config.max_generation_len, config.seed)
    bf16_result = load_result("math500", make_hub_path("bf16", config.model_name), file_name)

    fig = plt.figure(figsize=get_words_figsize(config.num_words, len(config.methods)), layout="constrained")
    for subfig, method in zip(fig.subfigures(len(config.methods), 1, squeeze=False)[:, 0], config.methods):
        result = load_result("math500", make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems), file_name)
        # bf16이 맞히고 양자화 모델이 잘리지 않은 문제만, 문제마다 단어 빈도를 구해 평균
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
        subfig.suptitle(f"{method} ({len(quantized_word_rates)}문제)")

    fig.suptitle(get_title(config.model_name, "overthinking 단어"))
    fig.legend(handles=[Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *config.methods]], loc="outside lower center", ncol=len(config.methods) + 1)
    save_figure(fig, config.model_name, "math500_overthinking_words")


if __name__ == "__main__":
    main()
