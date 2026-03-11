from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .manifest import DataFilterPolicy, DataManifest


@dataclass(slots=True)
class PlannedDataFile:
    entry_name: str
    path: Path
    content_type: str
    languages: tuple[str, ...]
    weight: float
    license: str | None


@dataclass(slots=True)
class DataShardDescriptor:
    shard_id: int
    file_count: int
    content_types: tuple[str, ...]
    paths: tuple[Path, ...]


@dataclass(slots=True)
class DataPlanningSummary:
    total_files: int
    by_entry: dict[str, int]
    by_content_type: dict[str, int]


def is_vendor_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    return "vendor" in lowered_parts or "node_modules" in lowered_parts or "third_party" in lowered_parts


def is_probably_generated_path(path: Path) -> bool:
    lowered_name = path.name.lower()
    generated_markers = (
        ".min.",
        ".generated.",
        "generated_",
        "_generated",
        ".pb.",
    )
    return any(marker in lowered_name for marker in generated_markers)


def license_allowed(policy: DataFilterPolicy, license_name: str | None) -> bool:
    if license_name is None:
        return True
    return license_name in set(policy.allow_licenses)


def discover_data_files(
    manifest: DataManifest,
    root: str | Path,
) -> tuple[PlannedDataFile, ...]:
    root_path = Path(root)
    discovered: list[PlannedDataFile] = []
    for entry in manifest.entries:
        for path in root_path.glob(entry.path_glob):
            if path.is_file():
                discovered.append(
                    PlannedDataFile(
                        entry_name=entry.name,
                        path=path,
                        content_type=entry.content_type,
                        languages=tuple(entry.languages),
                        weight=entry.weight,
                        license=entry.license,
                    )
                )
    return tuple(sorted(discovered, key=lambda item: (item.entry_name, str(item.path))))


def filter_planned_data_files(
    files: Sequence[PlannedDataFile],
    policy: DataFilterPolicy,
) -> tuple[PlannedDataFile, ...]:
    filtered: list[PlannedDataFile] = []
    seen_paths: set[Path] = set()
    for item in files:
        if policy.exclude_vendor and is_vendor_path(item.path):
            continue
        if (policy.exclude_generated or policy.exclude_minified) and is_probably_generated_path(item.path):
            continue
        if not license_allowed(policy, item.license):
            continue
        if policy.file_level_dedup and item.path in seen_paths:
            continue
        seen_paths.add(item.path)
        filtered.append(item)
    return tuple(filtered)


def plan_data_shards(
    files: Sequence[PlannedDataFile],
    max_files_per_shard: int = 128,
) -> tuple[tuple[PlannedDataFile, ...], ...]:
    if max_files_per_shard <= 0:
        raise ValueError("max_files_per_shard must be positive")
    shards: list[tuple[PlannedDataFile, ...]] = []
    current: list[PlannedDataFile] = []
    for item in files:
        current.append(item)
        if len(current) >= max_files_per_shard:
            shards.append(tuple(current))
            current = []
    if current:
        shards.append(tuple(current))
    return tuple(shards)


def summarize_planned_data_files(files: Sequence[PlannedDataFile]) -> DataPlanningSummary:
    by_entry: dict[str, int] = {}
    by_content_type: dict[str, int] = {}
    for item in files:
        by_entry[item.entry_name] = by_entry.get(item.entry_name, 0) + 1
        by_content_type[item.content_type] = by_content_type.get(item.content_type, 0) + 1
    return DataPlanningSummary(
        total_files=len(files),
        by_entry=by_entry,
        by_content_type=by_content_type,
    )


def build_data_shard_descriptors(
    files: Sequence[PlannedDataFile],
    max_files_per_shard: int = 128,
) -> tuple[DataShardDescriptor, ...]:
    shards = plan_data_shards(files, max_files_per_shard=max_files_per_shard)
    descriptors: list[DataShardDescriptor] = []
    for index, shard in enumerate(shards):
        descriptors.append(
            DataShardDescriptor(
                shard_id=index,
                file_count=len(shard),
                content_types=tuple(sorted({item.content_type for item in shard})),
                paths=tuple(item.path for item in shard),
            )
        )
    return tuple(descriptors)


def write_data_planning_artifacts(
    files: Sequence[PlannedDataFile],
    output_dir: str | Path,
    max_files_per_shard: int = 128,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = summarize_planned_data_files(files)
    descriptors = build_data_shard_descriptors(files, max_files_per_shard=max_files_per_shard)

    summary_path = output_path / "summary.json"
    shards_path = output_path / "shards.json"

    summary_payload = {
        "total_files": summary.total_files,
        "by_entry": summary.by_entry,
        "by_content_type": summary.by_content_type,
    }
    shard_payload = {
        "shards": [
            {
                "shard_id": descriptor.shard_id,
                "file_count": descriptor.file_count,
                "content_types": list(descriptor.content_types),
                "paths": [str(path) for path in descriptor.paths],
            }
            for descriptor in descriptors
        ]
    }

    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    shards_path.write_text(json.dumps(shard_payload, indent=2) + "\n", encoding="utf-8")
    return summary_path, shards_path
