import math_verify
from vllm.outputs import CompletionOutput  # type: ignore
from vllm.tokenizers import TokenizerLike  # type: ignore

from utils.model import GENERATION_CONFIGS


def split_outputs(
    outputs: list[CompletionOutput],
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    reasoning_end_id = tokenizer.convert_tokens_to_ids(GENERATION_CONFIGS[model_name]["reasoning_end"])
    reasonings = []
    responses = []
    reasoning_lens = []
    for output in outputs:
        token_ids = list(output.token_ids)
        if reasoning_end_id in token_ids:
            reasoning_len = token_ids.index(reasoning_end_id)
            response = tokenizer.decode(token_ids[reasoning_len + 1 :], skip_special_tokens=True)
        else:
            # reasoning_end token이 없으면 reasoning이 안 끝난 것이라 response 없음
            # 대부분 max_generation_len에 닿은 경우
            reasoning_len = len(token_ids)
            response = None
        reasonings.append(tokenizer.decode(token_ids[:reasoning_len], skip_special_tokens=True))
        responses.append(response)
        reasoning_lens.append(reasoning_len)
    return {
        "reasoning_lens": reasoning_lens,
        "truncateds": [output.finish_reason == "length" for output in outputs],
        "reasonings": reasonings,
        "responses": responses,
    }


def evaluate_outputs(
    outputs: list[CompletionOutput],
    answers: list[str],
    tokenizer: TokenizerLike,
    model_name: str,
) -> dict:
    split = split_outputs(outputs, tokenizer, model_name)
    # response의 답과 정답을 math_verify로 수식으로 비교
    # 예시) response "... \boxed{\frac{1}{2}}", 정답 "0.5" → 맞음
    parsed = [None if response is None else math_verify.parse(response) for response in split["responses"]]
    corrects = [prediction is not None and math_verify.verify(math_verify.parse(f"${answer}$"), prediction) for prediction, answer in zip(parsed, answers)]
    return {
        "accuracy": sum(corrects) / len(outputs),
        "corrects": corrects,
        "predictions": [str(prediction[-1]) if prediction else None for prediction in parsed],
        **split,
    }
