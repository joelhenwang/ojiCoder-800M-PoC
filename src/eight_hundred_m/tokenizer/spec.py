from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence, cast


NormalizationMode = Literal["code_preserving_v1"]
TokenizerContentType = Literal["code", "docs", "synthetic", "tool"]


def _as_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _as_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _as_float(value: object, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _as_content_type(value: object, field_name: str) -> TokenizerContentType:
    text = _as_str(value, field_name)
    if text not in {"code", "docs", "synthetic", "tool"}:
        raise ValueError(f"{field_name} has unsupported content type: {text}")
    return cast(TokenizerContentType, text)


def _as_normalization_mode(value: object, field_name: str) -> NormalizationMode:
    text = _as_str(value, field_name)
    if text != "code_preserving_v1":
        raise ValueError(f"{field_name} has unsupported normalization mode: {text}")
    return cast(NormalizationMode, text)


@dataclass(slots=True)
class TokenizerNormalizationConfig:
    mode: NormalizationMode = "code_preserving_v1"
    normalize_newlines: bool = True
    ensure_trailing_newline: bool = True

    def normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n") if self.normalize_newlines else text
        if self.ensure_trailing_newline and normalized and not normalized.endswith("\n"):
            normalized += "\n"
        return normalized


@dataclass(slots=True)
class TokenizerCorpusEntry:
    name: str
    path_glob: str
    weight: float
    content_type: TokenizerContentType
    license_required: bool = True

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("weight must be positive")

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "name": self.name,
            "path_glob": self.path_glob,
            "weight": self.weight,
            "content_type": self.content_type,
            "license_required": self.license_required,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TokenizerCorpusEntry:
        return cls(
            name=_as_str(payload["name"], "name"),
            path_glob=_as_str(payload["path_glob"], "path_glob"),
            weight=_as_float(payload["weight"], "weight"),
            content_type=_as_content_type(payload["content_type"], "content_type"),
            license_required=_as_bool(payload.get("license_required", True), "license_required"),
        )


@dataclass(slots=True)
class TokenizerCorpusManifest:
    artifact_name: str = "tokenizer-corpus-manifest"
    artifact_version: str = "v1"
    authoritative_tokenizer_config: str = "configs/tokenizer/base_bpe.json"
    normalization: TokenizerNormalizationConfig = field(default_factory=TokenizerNormalizationConfig)
    entries: Sequence[TokenizerCorpusEntry] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("entries must not be empty")
        total_weight = sum(entry.weight for entry in self.entries)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("tokenizer corpus entry weights must sum to 1.0")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "authoritative_tokenizer_config": self.authoritative_tokenizer_config,
            "normalization": {
                "mode": self.normalization.mode,
                "normalize_newlines": self.normalization.normalize_newlines,
                "ensure_trailing_newline": self.normalization.ensure_trailing_newline,
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TokenizerCorpusManifest:
        normalization_payload = payload.get("normalization", {})
        if not isinstance(normalization_payload, Mapping):
            raise ValueError("normalization must be a JSON object")
        entries_payload = payload.get("entries", [])
        if not isinstance(entries_payload, list):
            raise ValueError("entries must be a list")
        normalization = TokenizerNormalizationConfig(
            mode=_as_normalization_mode(
                normalization_payload.get("mode", "code_preserving_v1"),
                "mode",
            ),
            normalize_newlines=_as_bool(
                normalization_payload.get("normalize_newlines", True),
                "normalize_newlines",
            ),
            ensure_trailing_newline=_as_bool(
                normalization_payload.get("ensure_trailing_newline", True),
                "ensure_trailing_newline",
            ),
        )
        return cls(
            artifact_name=_as_str(payload.get("artifact_name", "tokenizer-corpus-manifest"), "artifact_name"),
            artifact_version=_as_str(payload.get("artifact_version", "v1"), "artifact_version"),
            authoritative_tokenizer_config=_as_str(
                payload.get("authoritative_tokenizer_config", "configs/tokenizer/base_bpe.json")
                ,
                "authoritative_tokenizer_config",
            ),
            normalization=normalization,
            entries=tuple(TokenizerCorpusEntry.from_dict(entry) for entry in entries_payload),
        )


def load_tokenizer_corpus_manifest(path: str | Path) -> TokenizerCorpusManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("tokenizer corpus manifest must be a JSON object")
    return TokenizerCorpusManifest.from_dict(payload)
