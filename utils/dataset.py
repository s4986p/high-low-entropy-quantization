from datasets import load_dataset
from torch import Tensor

from utils.hub import load_tensors
from utils.paths import make_calibration_name, make_hub_path

MATH_PROMPT = """{problem}
Please reason step by step, and put your final answer within \\boxed{{}}."""


def load_calibration(
    model_name: str,
    num_calibration_problems: int,
    max_generation_len: int,
    seed: int,
    max_calibration_len: int,
) -> list[Tensor]:
    tensors = load_tensors("calibration/numina", make_hub_path("bf16", model_name), make_calibration_name(num_calibration_problems, max_generation_len, seed))
    return [calibration_ids[:max_calibration_len] for calibration_ids in tensors["calibration_ids"].split(tensors["calibration_lens"].tolist())]


def load_math500() -> list[tuple[str, str]]:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [(MATH_PROMPT.format(problem=row["problem"]), row["answer"]) for row in dataset]
