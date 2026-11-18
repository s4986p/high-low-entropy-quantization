def make_hub_path(
    method: str,
    model_name: str,
    num_bits: int | None = None,
    group_size: int | None = None,
    num_calibration_problems: int | None = None,
) -> str:
    # bf16, mxfp4는 calibration이 없고 설정이 고정이라 경로에 설정 안 넣음
    if method in ("bf16", "mxfp4"):
        return f"{method}/{model_name}"
    return f"{method}/{model_name}/num_bits={num_bits},group_size={group_size},num_calibration_problems={num_calibration_problems}"


def make_result_name(
    max_generation_len: int,
    seed: int,
) -> str:
    return f"max_generation_len={max_generation_len},seed={seed}"


def make_calibration_name(
    num_calibration_problems: int,
    max_generation_len: int,
    seed: int,
) -> str:
    return f"num_calibration_problems={num_calibration_problems},max_generation_len={max_generation_len},seed={seed}"
