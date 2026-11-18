from dataclasses import dataclass
from typing import Literal

from simple_parsing import parse

from utils.evaluation import evaluate_math500, get_missing_seeds, load_vllm
from utils.hub import download_model
from utils.paths import make_hub_path, make_result_name


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["bf16", "gptq", "awq", "mxfp4"] = "bf16"
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128
    max_generation_len: int = 32768
    seeds: tuple[int, ...] = (0,)


def main() -> None:
    config = parse(Config)
    experiment_name = "math500"
    hub_path = make_hub_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
    file_names = {seed: make_result_name(config.max_generation_len, seed) for seed in config.seeds}
    missing_seeds = get_missing_seeds(experiment_name, hub_path, file_names)
    if not missing_seeds:
        return
    model_path = config.model_name if config.method == "bf16" else download_model(hub_path)
    evaluate_math500(load_vllm(model_path, config.model_name), config.model_name, experiment_name, hub_path, file_names, missing_seeds, config.max_generation_len)


if __name__ == "__main__":
    main()
