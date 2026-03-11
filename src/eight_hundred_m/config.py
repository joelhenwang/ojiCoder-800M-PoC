from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


DEFAULT_SPECIAL_TOKENS = (
    "<|repo|>",
    "<|path|>",
    "<|file_sep|>",
    "<|diff|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|memory|>",
    "<|plan|>",
    "<|patch|>",
)

FORBIDDEN_BASE_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
)


def _as_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _as_int(value: object, field_name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _as_float(value: object, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _as_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _as_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _as_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be a list of strings")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings")
        result.append(item)
    return tuple(result)


def _get_with_default(payload: Mapping[str, object], key: str, default: object) -> object:
    return payload[key] if key in payload else default


@dataclass(slots=True)
class TokenizerConfig:
    vocab_size: int = 64_000
    tokenizer_type: str = "bpe"
    byte_fallback: bool = True
    normalization: str = "code_preserving_v1"
    artifact_name: str = "base-bpe"
    artifact_version: str = "v1"
    special_tokens: Sequence[str] = field(default_factory=lambda: DEFAULT_SPECIAL_TOKENS)
    validation_targets: Sequence[str] = field(
        default_factory=lambda: (
            "whitespace_fidelity",
            "path_preservation",
            "diff_markup",
            "tool_transcript_markup",
        )
    )

    def __post_init__(self) -> None:
        if self.tokenizer_type != "bpe":
            raise ValueError("v1 tokenizer_type must be 'bpe'")
        overlap = set(self.special_tokens).intersection(FORBIDDEN_BASE_SPECIAL_TOKENS)
        if overlap:
            raise ValueError(
                "base tokenizer must not reserve chat/thinking wrapper tokens: "
                + ", ".join(sorted(overlap))
            )

    def to_dict(self) -> dict[str, str | int | bool | list[str]]:
        return {
            "vocab_size": self.vocab_size,
            "tokenizer_type": self.tokenizer_type,
            "byte_fallback": self.byte_fallback,
            "normalization": self.normalization,
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "special_tokens": list(self.special_tokens),
            "validation_targets": list(self.validation_targets),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TokenizerConfig:
        normalized_payload = _as_mapping(payload, "tokenizer_config")
        return cls(
            vocab_size=_as_int(
                _get_with_default(normalized_payload, "vocab_size", 64_000),
                "vocab_size",
            ),
            tokenizer_type=_as_str(
                _get_with_default(normalized_payload, "tokenizer_type", "bpe"),
                "tokenizer_type",
            ),
            byte_fallback=_as_bool(
                _get_with_default(normalized_payload, "byte_fallback", True),
                "byte_fallback",
            ),
            normalization=_as_str(
                _get_with_default(normalized_payload, "normalization", "code_preserving_v1"),
                "normalization",
            ),
            artifact_name=_as_str(
                _get_with_default(normalized_payload, "artifact_name", "base-bpe"),
                "artifact_name",
            ),
            artifact_version=_as_str(
                _get_with_default(normalized_payload, "artifact_version", "v1"),
                "artifact_version",
            ),
            special_tokens=_as_str_tuple(
                _get_with_default(normalized_payload, "special_tokens", DEFAULT_SPECIAL_TOKENS),
                "special_tokens",
            ),
            validation_targets=_as_str_tuple(
                _get_with_default(
                    normalized_payload,
                    "validation_targets",
                    (
                        "whitespace_fidelity",
                        "path_preservation",
                        "diff_markup",
                        "tool_transcript_markup",
                    ),
                ),
                "validation_targets",
            ),
        )


@dataclass(slots=True)
class TrainingStageConfig:
    name: str
    sequence_length: int
    objective: str = "next_token"
    target_tokens: int = 0
    sequence_length_mix: Sequence[tuple[int, float]] = field(default_factory=tuple)
    code_ratio: float = 0.0
    docs_ratio: float = 0.0
    synthetic_ratio: float = 0.0
    general_ratio: float = 0.0
    tool_ratio: float = 0.0
    fim_ratio: float = 0.0
    diff_ratio: float = 0.0
    repair_ratio: float = 0.0
    trajectory_summary_ratio: float = 0.0
    use_repo_formatting: bool = False
    use_tool_formatting: bool = False

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.sequence_length_mix:
            total_mix = sum(weight for _, weight in self.sequence_length_mix)
            if abs(total_mix - 1.0) > 1e-6:
                raise ValueError("sequence_length_mix must sum to 1.0")
        total_ratio = self.total_ratio()
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError("data mixture ratios must sum to 1.0")

    def total_ratio(self) -> float:
        return (
            self.code_ratio
            + self.docs_ratio
            + self.synthetic_ratio
            + self.general_ratio
            + self.tool_ratio
        )

    def to_dict(self) -> dict[str, str | int | float | bool | list[dict[str, float | int]]]:
        return {
            "name": self.name,
            "sequence_length": self.sequence_length,
            "objective": self.objective,
            "target_tokens": self.target_tokens,
            "sequence_length_mix": [
                {"sequence_length": seq_len, "weight": weight}
                for seq_len, weight in self.sequence_length_mix
            ],
            "code_ratio": self.code_ratio,
            "docs_ratio": self.docs_ratio,
            "synthetic_ratio": self.synthetic_ratio,
            "general_ratio": self.general_ratio,
            "tool_ratio": self.tool_ratio,
            "fim_ratio": self.fim_ratio,
            "diff_ratio": self.diff_ratio,
            "repair_ratio": self.repair_ratio,
            "trajectory_summary_ratio": self.trajectory_summary_ratio,
            "use_repo_formatting": self.use_repo_formatting,
            "use_tool_formatting": self.use_tool_formatting,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrainingStageConfig:
        normalized_payload = _as_mapping(payload, "training_stage")
        data_mix = _as_mapping(_get_with_default(normalized_payload, "data_mix", {}), "data_mix")
        objective_mix = _as_mapping(
            _get_with_default(normalized_payload, "objective_mix", {}),
            "objective_mix",
        )
        sequence_length_mix_payload = _get_with_default(
            normalized_payload,
            "sequence_length_mix",
            (),
        )
        if not isinstance(sequence_length_mix_payload, list | tuple):
            raise ValueError("sequence_length_mix must be a list")

        sequence_length_mix = tuple(
            (
                _as_int(_as_mapping(item, "sequence_length_mix_item")["sequence_length"], "sequence_length"),
                _as_float(_as_mapping(item, "sequence_length_mix_item")["weight"], "weight"),
            )
            for item in sequence_length_mix_payload
        )

        return cls(
            name=_as_str(normalized_payload["name"], "name"),
            sequence_length=_as_int(normalized_payload["sequence_length"], "sequence_length"),
            objective=_as_str(
                _get_with_default(normalized_payload, "objective", "next_token"),
                "objective",
            ),
            target_tokens=_as_int(normalized_payload["target_tokens"], "target_tokens"),
            sequence_length_mix=sequence_length_mix,
            code_ratio=_as_float(
                _get_with_default(data_mix, "code_ratio", _get_with_default(normalized_payload, "code_ratio", 0.0)),
                "code_ratio",
            ),
            docs_ratio=_as_float(
                _get_with_default(data_mix, "docs_ratio", _get_with_default(normalized_payload, "docs_ratio", 0.0)),
                "docs_ratio",
            ),
            synthetic_ratio=_as_float(
                _get_with_default(
                    data_mix,
                    "synthetic_ratio",
                    _get_with_default(normalized_payload, "synthetic_ratio", 0.0),
                ),
                "synthetic_ratio",
            ),
            general_ratio=_as_float(
                _get_with_default(
                    data_mix,
                    "general_ratio",
                    _get_with_default(normalized_payload, "general_ratio", 0.0),
                ),
                "general_ratio",
            ),
            tool_ratio=_as_float(
                _get_with_default(data_mix, "tool_ratio", _get_with_default(normalized_payload, "tool_ratio", 0.0)),
                "tool_ratio",
            ),
            fim_ratio=_as_float(
                _get_with_default(
                    objective_mix,
                    "fim_ratio",
                    _get_with_default(normalized_payload, "fim_ratio", 0.0),
                ),
                "fim_ratio",
            ),
            diff_ratio=_as_float(
                _get_with_default(
                    objective_mix,
                    "diff_ratio",
                    _get_with_default(normalized_payload, "diff_ratio", 0.0),
                ),
                "diff_ratio",
            ),
            repair_ratio=_as_float(
                _get_with_default(
                    objective_mix,
                    "repair_ratio",
                    _get_with_default(normalized_payload, "repair_ratio", 0.0),
                ),
                "repair_ratio",
            ),
            trajectory_summary_ratio=_as_float(
                _get_with_default(
                    objective_mix,
                    "trajectory_summary_ratio",
                    _get_with_default(normalized_payload, "trajectory_summary_ratio", 0.0),
                ),
                "trajectory_summary_ratio",
            ),
            use_repo_formatting=_as_bool(
                _get_with_default(normalized_payload, "use_repo_formatting", False),
                "use_repo_formatting",
            ),
            use_tool_formatting=_as_bool(
                _get_with_default(normalized_payload, "use_tool_formatting", False),
                "use_tool_formatting",
            ),
        )


def build_default_training_stages() -> tuple[TrainingStageConfig, ...]:
    return (
        TrainingStageConfig(
            name="prototype",
            sequence_length=4096,
            objective="next_token",
            target_tokens=1_500_000_000,
            sequence_length_mix=((4096, 0.9), (8192, 0.1)),
            code_ratio=0.45,
            docs_ratio=0.25,
            synthetic_ratio=0.10,
            general_ratio=0.20,
            tool_ratio=0.0,
            fim_ratio=0.05,
        ),
        TrainingStageConfig(
            name="stage_1_base",
            sequence_length=4096,
            objective="next_token",
            target_tokens=25_000_000_000,
            sequence_length_mix=((4096, 0.85), (8192, 0.15)),
            code_ratio=0.45,
            docs_ratio=0.25,
            synthetic_ratio=0.10,
            general_ratio=0.20,
            tool_ratio=0.0,
            fim_ratio=0.05,
        ),
        TrainingStageConfig(
            name="stage_2_code_cpt",
            sequence_length=8192,
            objective="next_token+fim",
            target_tokens=20_000_000_000,
            sequence_length_mix=((4096, 0.35), (8192, 0.65)),
            code_ratio=0.70,
            docs_ratio=0.15,
            synthetic_ratio=0.10,
            general_ratio=0.05,
            tool_ratio=0.0,
            fim_ratio=0.15,
            use_repo_formatting=True,
        ),
        TrainingStageConfig(
            name="stage_3_repair",
            sequence_length=8192,
            objective="repair+diff",
            target_tokens=5_000_000_000,
            sequence_length_mix=((8192, 1.0),),
            code_ratio=0.30,
            docs_ratio=0.10,
            synthetic_ratio=0.45,
            general_ratio=0.0,
            tool_ratio=0.15,
            diff_ratio=0.25,
            repair_ratio=0.45,
            use_repo_formatting=True,
            use_tool_formatting=True,
        ),
        TrainingStageConfig(
            name="stage_4_tool_policy",
            sequence_length=8192,
            objective="tool_trajectory",
            target_tokens=500_000_000,
            sequence_length_mix=((8192, 1.0),),
            code_ratio=0.0,
            docs_ratio=0.0,
            synthetic_ratio=0.25,
            general_ratio=0.0,
            tool_ratio=0.75,
            trajectory_summary_ratio=0.30,
            repair_ratio=0.30,
            use_repo_formatting=True,
            use_tool_formatting=True,
        ),
    )


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 64_000
    max_position_embeddings: int = 8_192
    num_hidden_layers: int = 24
    hidden_size: int = 1_536
    intermediate_size: int = 4_096
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 96
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    layer_scale_init: float = 0.1
    tie_word_embeddings: bool = True
    use_qk_norm: bool = True
    use_value_residual: bool = True
    use_per_head_gating: bool = True
    special_tokens: Sequence[str] = field(default_factory=lambda: DEFAULT_SPECIAL_TOKENS)

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.layer_scale_init <= 0:
            raise ValueError("layer_scale_init must be positive")

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def approx_parameter_count(self) -> int:
        embedding = self.vocab_size * self.hidden_size
        attention = (
            self.hidden_size * self.hidden_size
            + self.hidden_size * (self.num_key_value_heads * self.head_dim)
            + self.hidden_size * (self.num_key_value_heads * self.head_dim)
            + self.hidden_size * self.hidden_size
        )
        mlp = (
            self.hidden_size * self.intermediate_size
            + self.hidden_size * self.intermediate_size
            + self.intermediate_size * self.hidden_size
        )
        norms = 4 * self.hidden_size
        head_gates = self.num_attention_heads
        per_layer = attention + mlp + norms + head_gates
        total = embedding + self.num_hidden_layers * per_layer
        if not self.tie_word_embeddings:
            total += embedding
        return total

    def to_dict(self) -> dict[str, int | float | bool | Sequence[str]]:
        return {
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "initializer_range": self.initializer_range,
            "layer_scale_init": self.layer_scale_init,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_qk_norm": self.use_qk_norm,
            "use_value_residual": self.use_value_residual,
            "use_per_head_gating": self.use_per_head_gating,
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ModelConfig:
        normalized_payload = _as_mapping(payload, "model_config")
        return cls(
            vocab_size=_as_int(_get_with_default(normalized_payload, "vocab_size", 64_000), "vocab_size"),
            max_position_embeddings=_as_int(
                _get_with_default(normalized_payload, "max_position_embeddings", 8_192),
                "max_position_embeddings",
            ),
            num_hidden_layers=_as_int(
                _get_with_default(normalized_payload, "num_hidden_layers", 24),
                "num_hidden_layers",
            ),
            hidden_size=_as_int(_get_with_default(normalized_payload, "hidden_size", 1_536), "hidden_size"),
            intermediate_size=_as_int(
                _get_with_default(normalized_payload, "intermediate_size", 4_096),
                "intermediate_size",
            ),
            num_attention_heads=_as_int(
                _get_with_default(normalized_payload, "num_attention_heads", 16),
                "num_attention_heads",
            ),
            num_key_value_heads=_as_int(
                _get_with_default(normalized_payload, "num_key_value_heads", 4),
                "num_key_value_heads",
            ),
            head_dim=_as_int(_get_with_default(normalized_payload, "head_dim", 96), "head_dim"),
            rope_theta=_as_float(_get_with_default(normalized_payload, "rope_theta", 10_000.0), "rope_theta"),
            rms_norm_eps=_as_float(
                _get_with_default(normalized_payload, "rms_norm_eps", 1e-6),
                "rms_norm_eps",
            ),
            hidden_dropout=_as_float(
                _get_with_default(normalized_payload, "hidden_dropout", 0.0),
                "hidden_dropout",
            ),
            attention_dropout=_as_float(
                _get_with_default(normalized_payload, "attention_dropout", 0.0),
                "attention_dropout",
            ),
            initializer_range=_as_float(
                _get_with_default(normalized_payload, "initializer_range", 0.02),
                "initializer_range",
            ),
            layer_scale_init=_as_float(
                _get_with_default(normalized_payload, "layer_scale_init", 0.1),
                "layer_scale_init",
            ),
            tie_word_embeddings=_as_bool(
                _get_with_default(normalized_payload, "tie_word_embeddings", True),
                "tie_word_embeddings",
            ),
            use_qk_norm=_as_bool(_get_with_default(normalized_payload, "use_qk_norm", True), "use_qk_norm"),
            use_value_residual=_as_bool(
                _get_with_default(normalized_payload, "use_value_residual", True),
                "use_value_residual",
            ),
            use_per_head_gating=_as_bool(
                _get_with_default(normalized_payload, "use_per_head_gating", True),
                "use_per_head_gating",
            ),
            special_tokens=_as_str_tuple(
                _get_with_default(normalized_payload, "special_tokens", DEFAULT_SPECIAL_TOKENS),
                "special_tokens",
            ),
        )


def load_json_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("config payload must be a JSON object")
    return payload


def load_tokenizer_config(path: str | Path) -> TokenizerConfig:
    return TokenizerConfig.from_dict(load_json_config(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.from_dict(load_json_config(path))


def load_training_stages(path: str | Path) -> tuple[TrainingStageConfig, ...]:
    payload = load_json_config(path)
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError("training stage config must contain a 'stages' list")
    return tuple(TrainingStageConfig.from_dict(stage) for stage in stages)
