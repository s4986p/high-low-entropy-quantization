from collections import Counter
from dataclasses import dataclass
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse
from transformers import AutoTokenizer

from utils.hub import load_tensors
from utils.metrics import get_words
from utils.paths import make_hub_path
from utils.plot import METHOD_COLORS, get_words_figsize, plot_words, save_figure


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
    num_words: int = 15


def main() -> None:
    config = parse(Config)
    file_name = f"max_generation_len={config.max_generation_len},seed={config.seed}"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    token_words = [words[0] if len(words) == 1 else None for words in map(get_words, [tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])]

    fig = plt.figure(figsize=get_words_figsize(config.num_words, len(config.methods)), layout="constrained")
    for subfig, method in zip(fig.subfigures(len(config.methods), 1, squeeze=False)[:, 0], config.methods):
        result = load_tensors("entropy_kl/math500", make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems), file_name)
        is_high_entropy = result["bf16_entropy"] > result["bf16_entropy"].quantile(1 - config.high_entropy_ratio)
        num_high_entropy = is_high_entropy.sum().item()
        bf16_counts = Counter(token_words[token_id] for token_id in result["bf16_top1_ids"][is_high_entropy].tolist() if token_words[token_id])
        quantized_counts = Counter(token_words[token_id] for token_id in result["quantized_top1_ids"][is_high_entropy].tolist() if token_words[token_id])
        bf16_rates = {word: count / num_high_entropy * 1000 for word, count in bf16_counts.items()}
        quantized_rates = {word: count / num_high_entropy * 1000 for word, count in quantized_counts.items()}
        plot_words(subfig.subplots(1, 3), bf16_rates, quantized_rates, method, config.num_words, "고엔트로피 위치 1000개당 top-1 횟수")
        subfig.suptitle(f"{method} (고엔트로피 위치 수 {num_high_entropy})")

    fig.suptitle(f"{config.model_name} math500 고엔트로피 위치의 top-1 단어\n{file_name},high_entropy_ratio={config.high_entropy_ratio:g}")
    fig.legend(handles=[Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *config.methods]], loc="outside lower center", ncol=len(config.methods) + 1)
    save_figure(fig, config.model_name, "math500_entropy_words")


if __name__ == "__main__":
    main()
