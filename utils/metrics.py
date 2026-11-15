import re
from statistics import mean, median

STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}


def get_words(
    text: str,
) -> list[str]:
    words = re.findall(r"(?<![\\a-z])[a-z]+(?:'[a-z]+)?(?:-[a-z]{2,})*", text.lower())
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


def get_truncated_ratio(
    result: dict,
) -> float:
    return mean(truncated and not correct for correct, truncated in zip(result["corrects"], result["truncateds"]))


def is_loop(
    text: str | None,
) -> bool:
    return text is not None and len(text) > 200 and text[:-200].count(text[-200:-150]) >= 5


def get_loop_ratio(
    result: dict,
) -> float:
    return mean(is_loop(reasoning) or is_loop(response) for reasoning, response in zip(result["reasonings"], result["responses"]))


def get_length_ratio(
    result: dict,
    bf16_result: dict,
) -> float:
    return median(reasoning_len / bf16_reasoning_len for reasoning_len, bf16_reasoning_len, correct, bf16_correct in zip(result["reasoning_lens"], bf16_result["reasoning_lens"], result["corrects"], bf16_result["corrects"]) if correct and bf16_correct)
