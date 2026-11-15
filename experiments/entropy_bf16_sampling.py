from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from statistics import mean
from tempfile import TemporaryDirectory
import time
from typing import Literal

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax
from vllm import ModelRegistry  # type: ignore
from vllm.compilation.decorators import support_torch_compile  # type: ignore
from vllm.config import VllmConfig  # type: ignore
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # type: ignore
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # type: ignore
from vllm.model_executor.models.qwen2 import Qwen2Model  # type: ignore
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer  # type: ignore
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix  # type: ignore

from utils.dataset import load_math500
from utils.evaluation import evaluate_outputs
from utils.format import format_duration
from utils.generation import generate_vllm, load_vllm
from utils.hub import download_model, load_tensors, result_exists, save_result
from utils.paths import make_hub_path


@dataclass
class Config:
    seeds: tuple[int, ...] = (0,)
    bf16_position: Literal["high", "low", "random"] = "high"
    bf16_ratio: float = 0.2
    max_generation_len: int = 32768
    # bf16과 섞을 양자화 모델
    model_name: str = "Qwen/Qwen3-0.6B"
    method: Literal["gptq", "awq", "mxfp4"] = "gptq"
    num_bits: int = 4
    group_size: int = 128
    num_calibration_problems: int = 128


def merge_models(
    bf16_model_path: str,
    quantized_model_path: str,
    model_path: str,
) -> None:
    # vllm은 모델 하나만 띄울 수 있기 때문에 bf16과 양자화 모델의 layer를 이어 한 모델로 합침
    # 예시) layer 28개 → 앞 28개는 bf16 layer, 뒤 28개는 양자화 layer
    model_config = json.loads((Path(quantized_model_path) / "config.json").read_text())
    num_layers = model_config["num_hidden_layers"]
    model_config["architectures"] = [MergedQwen3ForCausalLM.__name__]
    model_config["num_hidden_layers"] = 2 * num_layers
    model_config["max_window_layers"] = 2 * num_layers
    model_config["layer_types"] = model_config["layer_types"] * 2

    weights = {}
    for file in sorted(Path(bf16_model_path).glob("model*.safetensors")):
        weights.update(load_file(file))
    for file in sorted(Path(quantized_model_path).glob("model*.safetensors")):
        for name, tensor in load_file(file).items():
            match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
            if match:
                weights[f"model.layers.{int(match[1]) + num_layers}.{match[2]}"] = tensor
            else:
                # layer 밖 weight는 양자화하지 않아 bf16과 같아야 함
                assert torch.equal(tensor, weights[name]), name

    (Path(model_path) / "config.json").write_text(json.dumps(model_config, indent=2))
    shutil.copy(Path(bf16_model_path) / "generation_config.json", Path(model_path) / "generation_config.json")
    save_file(weights, Path(model_path) / "model.safetensors")
    print(f"merged {len(weights)} tensors num_hidden_layers {model_config['num_hidden_layers']}")


# torch.compile은 token 수가 바뀌면 다시 compile함
# vllm은 여러 request를 한 입력으로 묶어 step마다 token 수가 바뀌기 때문에 dynamic_arg_dims로 바뀌는 차원 표시
@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "intermediate_tensors": 0, "inputs_embeds": 0})
class MergedQwen3Model(Qwen2Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        # 같은 입력을 bf16 layer와 양자화 layer에 각각 통과시킴
        # vllm은 tensor 하나를 기대하기 때문에 두 hidden을 이어 붙이고 compute_logits에서 나눔
        num_layers = len(self.layers) // 2
        hiddens = []
        for layers in [self.layers[:num_layers], self.layers[num_layers:]]:
            hidden_states = self.embed_input_ids(input_ids)
            residual = None
            for layer in layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            hiddens.append(hidden_states)
        return torch.cat(hiddens, dim=-1)


class MergedQwen3ForCausalLM(Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = MergedQwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        bf16_hiddens, quantized_hiddens = hidden_states.chunk(2, dim=-1)
        bf16_logits = self.logits_processor(self.lm_head, bf16_hiddens.contiguous())
        quantized_logits = self.logits_processor(self.lm_head, quantized_hiddens.contiguous())

        # bf16_position에 해당하는 위치만 bf16 logit으로 샘플링
        if self.config.bf16_position == "random":
            is_bf16 = torch.rand(len(hidden_states), device=hidden_states.device) < self.config.bf16_ratio
        else:
            bf16_logprobs = log_softmax(bf16_logits.float(), dim=-1)
            bf16_entropy = -(bf16_logprobs.exp() * bf16_logprobs).sum(dim=-1)
            if self.config.bf16_position == "high":
                is_bf16 = bf16_entropy > self.config.entropy_threshold
            else:
                is_bf16 = bf16_entropy < self.config.entropy_threshold
        return torch.where(is_bf16[:, None], bf16_logits, quantized_logits)

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
    ) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


def main() -> None:
    config = parse(Config)
    experiment_name = "entropy_bf16_sampling/math500"
    hub_path = make_hub_path(config.method, config.model_name, config.num_bits, config.group_size, config.num_calibration_problems)
    file_names = {seed: f"bf16_position={config.bf16_position},bf16_ratio={config.bf16_ratio},max_generation_len={config.max_generation_len},seed={seed}" for seed in config.seeds}

    # 1. seed마다 hub result에서 빠진 것 확인, 빠진 것이 없으면 종료
    missing_seeds = [seed for seed, file_name in file_names.items() if not result_exists(experiment_name, hub_path, file_name)]
    if not missing_seeds:
        print(f"{hub_path} already evaluated")
        return
    problems = load_math500()

    # 2. seed 0 bf16 reasoning의 bf16 entropy 분위수로 경계 설정, random은 경계 없음
    # 예시) high, bf16_ratio 0.2 → bf16 entropy 상위 20% 경계
    bf16_entropy = load_tensors("entropy_kl/math500", hub_path, f"max_generation_len={config.max_generation_len},seed=0")["bf16_entropy"]
    entropy_threshold = None
    if config.bf16_position == "high":
        entropy_threshold = bf16_entropy.quantile(1 - config.bf16_ratio).item()
    elif config.bf16_position == "low":
        entropy_threshold = bf16_entropy.quantile(config.bf16_ratio).item()
    print(f"{hub_path} entropy threshold {entropy_threshold}")

    # 3. bf16과 양자화 모델을 합쳐 vllm으로 띄움
    # register_model은 이 process에만 적용되기 때문에 vllm engine을 같은 process에서 실행
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    ModelRegistry.register_model(MergedQwen3ForCausalLM.__name__, MergedQwen3ForCausalLM)
    hf_overrides = {"bf16_position": config.bf16_position, "bf16_ratio": config.bf16_ratio, "entropy_threshold": entropy_threshold}
    with TemporaryDirectory() as model_path:
        merge_models(snapshot_download(config.model_name, allow_patterns=["*.json", "*.safetensors"]), download_model(hub_path), model_path)
        model = load_vllm(model_path, config.model_name, hf_overrides)
    tokenizer = model.get_tokenizer()

    # 4. vllm은 띄우는 데 오래 걸리기 때문에 빠진 seed를 모아 한 번에 생성
    prompts = [prompt for _ in missing_seeds for prompt, _ in problems]
    prompt_seeds = [seed for seed in missing_seeds for _ in problems]
    start_time = time.perf_counter()
    outputs = generate_vllm(model, config.model_name, prompts, prompt_seeds, config.max_generation_len)
    print(f"{hub_path} time {format_duration(time.perf_counter() - start_time)}")

    # 5. outputs를 seed별로 나눠 채점하고 저장
    start = 0
    for seed in missing_seeds:
        end = start + len(problems)
        result = evaluate_outputs(outputs[start:end], [answer for _, answer in problems], tokenizer, config.model_name)
        start = end
        print(f"{hub_path} seed {seed} {config.bf16_position} {config.bf16_ratio} accuracy {result['accuracy']:.2%} reasoning len {mean(result['reasoning_lens']):.0f} truncated {mean(result['truncateds']):.2%}")
        save_result(experiment_name, hub_path, file_names[seed], result)


if __name__ == "__main__":
    main()
