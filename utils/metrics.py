import re
from statistics import mean, median

from torch import Tensor

# 단어 빈도: WORD_RATE_TOKENS token 또는 위치당 횟수
WORD_RATE_TOKENS = 1000
STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}
# 반복: 끝에서 LOOP_TAIL_LEN번째부터 LOOP_PATTERN_LEN 글자가 그 앞에 MIN_LOOP_COUNT번 이상 나옴
LOOP_TAIL_LEN = 200
LOOP_PATTERN_LEN = 50
MIN_LOOP_COUNT = 5


def get_words(
    text: str,
) -> list[str]:
    # 소문자 영어 단어만, latex 명령, 한 글자 단어, 불용어 제외
    # 예시) "Wait, \frac{1}{2} isn't right" → ["wait", "isn't", "right"]
    words = re.findall(r"(?<![\\a-z])[a-z]+(?:'[a-z]+)?(?:-[a-z]{2,})*", text.lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


def get_high_entropy_mask(
    bf16_entropy: Tensor,
    high_entropy_ratio: float,
) -> Tensor:
    # bf16 entropy 상위 high_entropy_ratio가 고엔트로피 위치, 나머지는 저엔트로피 위치
    return bf16_entropy > bf16_entropy.quantile(1 - high_entropy_ratio)


def get_accuracy(
    result: dict,
) -> float:
    return result["accuracy"]


def get_truncated_ratio(
    result: dict,
) -> float:
    return mean(truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def is_loop(
    text: str | None,
) -> bool:
    return text is not None and len(text) > LOOP_TAIL_LEN and text[:-LOOP_TAIL_LEN].count(text[-LOOP_TAIL_LEN : -LOOP_TAIL_LEN + LOOP_PATTERN_LEN]) >= MIN_LOOP_COUNT


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(reasoning) or is_loop(response) for reasoning, response in zip(result["reasonings"], result["responses"]))


def get_length_ratio(
    result: dict,
    bf16_result: dict,
) -> float:
    # bf16과 양자화 모델이 모두 맞힌 문제에서 bf16 대비 reasoning token 수 배의 중앙값
    return median(reasoning_len / bf16_reasoning_len for reasoning_len, bf16_reasoning_len, correct, bf16_correct in zip(result["reasoning_lens"], bf16_result["reasoning_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct)
