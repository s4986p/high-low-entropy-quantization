from collections import Counter
from dataclasses import dataclass
from typing import Literal

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse
from transformers import AutoTokenizer

from utils.hub import load_tensors
from utils.metrics import WORD_RATE_TOKENS, get_high_entropy_mask, get_words
from utils.paths import make_hub_path, make_result_name
from utils.plot import METHOD_COLORS, get_title, get_words_figsize, plot_words, save_figure


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
    num_words: int = 15


def get_word_rates(
    top1_ids: list[int],
    token_words: list[str | None],
) -> dict[str, float]:
    # 고엔트로피 위치 WORD_RATE_TOKENS개당 top-1 token이 그 단어인 횟수
    counts = Counter(token_words[token_id] for token_id in top1_ids if token_words[token_id])
    return {word: count / len(top1_ids) * WORD_RATE_TOKENS for word, count in counts.items()}


def main() -> None:
    config = parse(Config)
    file_name = make_result_name(config.max_generation_len, config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # token 하나가 단어 하나이면 그 단어, 아니면 None
    # 예시) " wait" → "wait", " 12" → None
    token_words = [words[0] if len(words) == 1 else None for words in map(get_words, [tokenizer.decode([token_id]) for token_id in range(len(tokenizer))])]

    fig = plt.figure(figsize=get_words_figsize(config.num_words, len(config.methods)), layout="constrained")
    for subfig, method in zip(fig.subfigures(len(config.methods), 1, squeeze=False)[:, 0], config.methods):
        result = load_tensors("entropy_kl/math500", make_hub_path(method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems), file_name)
        is_high_entropy = get_high_entropy_mask(result["bf16_entropy"], config.high_entropy_ratio)
        bf16_rates = get_word_rates(result["bf16_top1_ids"][is_high_entropy].tolist(), token_words)
        quantized_rates = get_word_rates(result["quantized_top1_ids"][is_high_entropy].tolist(), token_words)
        plot_words(subfig.subplots(1, 3), bf16_rates, quantized_rates, method, config.num_words)
        subfig.suptitle(f"{method} (고엔트로피 위치 {is_high_entropy.sum().item()}개)")

    fig.suptitle(get_title(config.model_name, "bf16 entropy별 bf16과의 kl"))
    fig.legend(handles=[Patch(color=METHOD_COLORS[method], label=method) for method in ["bf16", *config.methods]], loc="outside lower center", ncol=len(config.methods) + 1)
    save_figure(fig, config.model_name, "math500_entropy_kl_words")


if __name__ == "__main__":
    main()
