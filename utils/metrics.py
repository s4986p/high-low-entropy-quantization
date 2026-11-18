import re

from torch import Tensor

# 단어 빈도: WORD_RATE_TOKENS token 또는 위치당 횟수
WORD_RATE_TOKENS = 1000
STOPWORDS = {"the", "an", "is", "are", "was", "were", "be", "been", "being", "am", "it", "it's", "its", "this", "that", "that's", "these", "those", "of", "to", "in", "on", "at", "by", "with", "from", "as", "into", "for", "and", "or", "we", "me", "my", "our", "us", "you", "your", "he", "she", "they", "them", "their", "his", "her", "there", "there's", "here", "which", "what", "where", "who", "how", "has", "have", "had", "do", "does", "did", "can", "could", "would", "will"}


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
