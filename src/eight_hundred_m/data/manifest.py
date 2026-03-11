from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence, cast


DEFAULT_PERMISSIVE_LICENSES = (
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
    "MIT",
    "Python-2.0",
    "Unicode-3.0",
    "Unlicense",
    "Zlib",
)

DataContentType = Literal["code", "docs", "synthetic", "general", "tool"]
DataSourceKind = Literal["local-files", "jsonl-records"]


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


def _as_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be a list of strings")
    items: list[str] = []
    for item in value:
        items.append(_as_str(item, field_name))
    return tuple(items)


def _as_content_type(value: object, field_name: str) -> DataContentType:
    text = _as_str(value, field_name)
    if text not in {"code", "docs", "synthetic", "general", "tool"}:
        raise ValueError(f"{field_name} has unsupported content type: {text}")
    return cast(DataContentType, text)


def _as_source_kind(value: object, field_name: str) -> DataSourceKind:
    text = _as_str(value, field_name)
    if text not in {"local-files", "jsonl-records"}:
        raise ValueError(f"{field_name} has unsupported source kind: {text}")
    return cast(DataSourceKind, text)


@dataclass(slots=True)
class DataFilterPolicy:
    allow_licenses: Sequence[str] = field(default_factory=lambda: DEFAULT_PERMISSIVE_LICENSES)
    require_parser_valid_when_available: bool = True
    exclude_generated: bool = True
    exclude_minified: bool = True
    exclude_vendor: bool = True
    file_level_dedup: bool = True
    near_duplicate_dedup: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_licenses": list(self.allow_licenses),
            "require_parser_valid_when_available": self.require_parser_valid_when_available,
            "exclude_generated": self.exclude_generated,
            "exclude_minified": self.exclude_minified,
            "exclude_vendor": self.exclude_vendor,
            "file_level_dedup": self.file_level_dedup,
            "near_duplicate_dedup": self.near_duplicate_dedup,
        }


@dataclass(slots=True)
class DataSourceManifestEntry:
    name: str
    content_type: DataContentType
    languages: Sequence[str]
    path_glob: str
    weight: float
    license: str | None = None
    source_kind: DataSourceKind = "local-files"
    dataset_name: str | None = None
    record_format: str | None = None
    text_mode: str | None = None
    provenance_required: bool = True

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if not self.languages:
            raise ValueError("languages must not be empty")
        if self.source_kind == "jsonl-records" and not self.record_format:
            raise ValueError("record_format is required for jsonl-records sources")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "content_type": self.content_type,
            "languages": list(self.languages),
            "path_glob": self.path_glob,
            "weight": self.weight,
            "license": self.license,
            "source_kind": self.source_kind,
            "dataset_name": self.dataset_name,
            "record_format": self.record_format,
            "text_mode": self.text_mode,
            "provenance_required": self.provenance_required,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DataSourceManifestEntry:
        languages = payload.get("languages", [])
        if not isinstance(languages, list):
            raise ValueError("languages must be a list")
        return cls(
            name=_as_str(payload["name"], "name"),
            content_type=_as_content_type(payload["content_type"], "content_type"),
            languages=_as_str_tuple(languages, "languages"),
            path_glob=_as_str(payload["path_glob"], "path_glob"),
            weight=_as_float(payload["weight"], "weight"),
            license=_as_str(payload["license"], "license") if payload.get("license") is not None else None,
            source_kind=_as_source_kind(payload.get("source_kind", "local-files"), "source_kind"),
            dataset_name=_as_str(payload["dataset_name"], "dataset_name") if payload.get("dataset_name") is not None else None,
            record_format=_as_str(payload["record_format"], "record_format") if payload.get("record_format") is not None else None,
            text_mode=_as_str(payload["text_mode"], "text_mode") if payload.get("text_mode") is not None else None,
            provenance_required=_as_bool(payload.get("provenance_required", True), "provenance_required"),
        )


@dataclass(slots=True)
class DataManifest:
    artifact_name: str = "base-data-manifest"
    artifact_version: str = "v1"
    filter_policy: DataFilterPolicy = field(default_factory=DataFilterPolicy)
    entries: Sequence[DataSourceManifestEntry] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("entries must not be empty")
        total_weight = sum(entry.weight for entry in self.entries)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("data manifest entry weights must sum to 1.0")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_name": self.artifact_name,
            "artifact_version": self.artifact_version,
            "filter_policy": self.filter_policy.to_dict(),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DataManifest:
        filter_payload = payload.get("filter_policy", {})
        if not isinstance(filter_payload, Mapping):
            raise ValueError("filter_policy must be a JSON object")
        entries_payload = payload.get("entries", [])
        if not isinstance(entries_payload, list):
            raise ValueError("entries must be a list")
        filter_policy = DataFilterPolicy(
            allow_licenses=_as_str_tuple(
                filter_payload.get("allow_licenses", DEFAULT_PERMISSIVE_LICENSES),
                "allow_licenses",
            ),
            require_parser_valid_when_available=_as_bool(
                filter_payload.get("require_parser_valid_when_available", True)
                ,
                "require_parser_valid_when_available",
            ),
            exclude_generated=_as_bool(filter_payload.get("exclude_generated", True), "exclude_generated"),
            exclude_minified=_as_bool(filter_payload.get("exclude_minified", True), "exclude_minified"),
            exclude_vendor=_as_bool(filter_payload.get("exclude_vendor", True), "exclude_vendor"),
            file_level_dedup=_as_bool(filter_payload.get("file_level_dedup", True), "file_level_dedup"),
            near_duplicate_dedup=_as_bool(
                filter_payload.get("near_duplicate_dedup", False),
                "near_duplicate_dedup",
            ),
        )
        return cls(
            artifact_name=_as_str(payload.get("artifact_name", "base-data-manifest"), "artifact_name"),
            artifact_version=_as_str(payload.get("artifact_version", "v1"), "artifact_version"),
            filter_policy=filter_policy,
            entries=tuple(DataSourceManifestEntry.from_dict(entry) for entry in entries_payload),
        )


def load_data_manifest(path: str | Path) -> DataManifest:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("data manifest must be a JSON object")
    return DataManifest.from_dict(payload)
