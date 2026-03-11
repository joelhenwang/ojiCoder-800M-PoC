from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from .languages import STAGE1_EXTERNAL_LANGUAGES, canonicalize_language, language_is_selected
from .manifest import DataFilterPolicy, DataManifest, DataSourceManifestEntry
from .planning import license_allowed

ExternalRecordFormat = Literal["the-stack-v2-dedup", "commitpackft"]


@dataclass(slots=True)
class ExternalDataSample:
    source_entry_name: str
    dataset_name: str
    record_format: ExternalRecordFormat
    canonical_language: str
    source_language: str
    content_type: str
    text: str
    weight: float
    repo_name: str | None = None
    path: str | None = None
    revision: str | None = None
    commit: str | None = None
    license_name: str | None = None
    license_evidence: tuple[str, ...] = ()
    encoding: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_entry_name": self.source_entry_name,
            "dataset_name": self.dataset_name,
            "record_format": self.record_format,
            "canonical_language": self.canonical_language,
            "source_language": self.source_language,
            "content_type": self.content_type,
            "text": self.text,
            "weight": self.weight,
            "repo_name": self.repo_name,
            "path": self.path,
            "revision": self.revision,
            "commit": self.commit,
            "license_name": self.license_name,
            "license_evidence": list(self.license_evidence),
            "encoding": self.encoding,
        }


@dataclass(slots=True)
class ExternalSampleSummary:
    total_samples: int
    by_dataset: dict[str, int]
    by_language: dict[str, int]
    by_content_type: dict[str, int]


@dataclass(slots=True)
class ExternalSampleShard:
    shard_id: int
    sample_count: int
    languages: tuple[str, ...]
    datasets: tuple[str, ...]
    output_path: Path


def _as_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def _choose_stack_text(record: dict[str, Any]) -> str | None:
    for field in ("content", "text", "raw_text", "code"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _choose_commitpack_text(record: dict[str, Any], text_mode: str | None) -> str | None:
    if text_mode == "new_contents" or text_mode is None:
        value = record.get("new_contents")
        if isinstance(value, str) and value:
            return value
    return None


def adapt_the_stack_v2_record(
    entry: DataSourceManifestEntry,
    record: dict[str, Any],
    selected_languages: tuple[str, ...] = STAGE1_EXTERNAL_LANGUAGES,
) -> ExternalDataSample | None:
    source_language = _as_str(record.get("language"), "") or ""
    canonical_language = canonicalize_language(source_language)
    if not language_is_selected(canonical_language, selected_languages):
        return None
    text = _choose_stack_text(record)
    if not text:
        return None
    license_evidence = _as_str_tuple(record.get("detected_licenses"))
    license_name = _as_str(record.get("license_type")) or _as_str(record.get("license"))
    return ExternalDataSample(
        source_entry_name=entry.name,
        dataset_name=entry.dataset_name or "bigcode/the-stack-v2-dedup",
        record_format="the-stack-v2-dedup",
        canonical_language=canonical_language,
        source_language=source_language,
        content_type=entry.content_type,
        text=text,
        weight=entry.weight,
        repo_name=_as_str(record.get("repo_name")),
        path=_as_str(record.get("path")),
        revision=_as_str(record.get("revision")) or _as_str(record.get("snapshot_id")),
        license_name=license_name,
        license_evidence=license_evidence,
        encoding=_as_str(record.get("src_encoding")),
    )


def adapt_commitpackft_record(
    entry: DataSourceManifestEntry,
    record: dict[str, Any],
    selected_languages: tuple[str, ...] = STAGE1_EXTERNAL_LANGUAGES,
) -> ExternalDataSample | None:
    source_language = _as_str(record.get("lang"), "") or _as_str(record.get("language"), "") or ""
    canonical_language = canonicalize_language(source_language)
    if not language_is_selected(canonical_language, selected_languages):
        return None
    text = _choose_commitpack_text(record, entry.text_mode)
    if not text:
        return None
    repo_name = _as_str(record.get("repo_name"))
    if repo_name is None:
        repos = record.get("repos")
        if isinstance(repos, list) and repos:
            repo_name = _as_str(repos[0])
    return ExternalDataSample(
        source_entry_name=entry.name,
        dataset_name=entry.dataset_name or "bigcode/commitpackft",
        record_format="commitpackft",
        canonical_language=canonical_language,
        source_language=source_language,
        content_type=entry.content_type,
        text=text,
        weight=entry.weight,
        repo_name=repo_name,
        path=_as_str(record.get("new_file")) or _as_str(record.get("path")),
        commit=_as_str(record.get("commit")),
        license_name=_as_str(record.get("license")),
        license_evidence=(),
        encoding=None,
    )


def adapt_external_record(
    entry: DataSourceManifestEntry,
    record: dict[str, Any],
    selected_languages: tuple[str, ...] = STAGE1_EXTERNAL_LANGUAGES,
) -> ExternalDataSample | None:
    if entry.record_format == "the-stack-v2-dedup":
        return adapt_the_stack_v2_record(entry, record, selected_languages=selected_languages)
    if entry.record_format == "commitpackft":
        return adapt_commitpackft_record(entry, record, selected_languages=selected_languages)
    raise ValueError(f"Unsupported external record format: {entry.record_format}")


def discover_external_record_files(
    manifest: DataManifest,
    root: str | Path,
) -> tuple[tuple[DataSourceManifestEntry, Path], ...]:
    root_path = Path(root)
    discovered: list[tuple[DataSourceManifestEntry, Path]] = []
    for entry in manifest.entries:
        if entry.source_kind != "jsonl-records":
            continue
        for path in root_path.glob(entry.path_glob):
            if path.is_file():
                discovered.append((entry, path))
    return tuple(sorted(discovered, key=lambda item: (item[0].name, str(item[1]))))


def load_jsonl_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("JSONL record must be a JSON object")
            records.append(payload)
    return tuple(records)


def collect_external_samples(
    manifest: DataManifest,
    root: str | Path,
    selected_languages: tuple[str, ...] = STAGE1_EXTERNAL_LANGUAGES,
) -> tuple[ExternalDataSample, ...]:
    samples: list[ExternalDataSample] = []
    for entry, path in discover_external_record_files(manifest, root):
        for record in load_jsonl_records(path):
            sample = adapt_external_record(entry, record, selected_languages=selected_languages)
            if sample is not None:
                samples.append(sample)
    return tuple(samples)


def filter_external_samples(
    samples: Sequence[ExternalDataSample],
    policy: DataFilterPolicy,
) -> tuple[ExternalDataSample, ...]:
    filtered: list[ExternalDataSample] = []
    seen_keys: set[tuple[str | None, str | None, str]] = set()
    for sample in samples:
        if not license_allowed(policy, sample.license_name):
            continue
        dedup_key = (sample.repo_name, sample.path, sample.text)
        if policy.file_level_dedup and dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        filtered.append(sample)
    return tuple(filtered)


def summarize_external_samples(samples: Sequence[ExternalDataSample]) -> ExternalSampleSummary:
    by_dataset: dict[str, int] = {}
    by_language: dict[str, int] = {}
    by_content_type: dict[str, int] = {}
    for sample in samples:
        by_dataset[sample.dataset_name] = by_dataset.get(sample.dataset_name, 0) + 1
        by_language[sample.canonical_language] = by_language.get(sample.canonical_language, 0) + 1
        by_content_type[sample.content_type] = by_content_type.get(sample.content_type, 0) + 1
    return ExternalSampleSummary(
        total_samples=len(samples),
        by_dataset=by_dataset,
        by_language=by_language,
        by_content_type=by_content_type,
    )


def write_external_sample_shards(
    samples: Sequence[ExternalDataSample],
    output_dir: str | Path,
    max_samples_per_shard: int = 128,
) -> tuple[ExternalSampleShard, ...]:
    if max_samples_per_shard <= 0:
        raise ValueError("max_samples_per_shard must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shards: list[ExternalSampleShard] = []
    for shard_id, start in enumerate(range(0, len(samples), max_samples_per_shard)):
        shard_samples = samples[start : start + max_samples_per_shard]
        shard_path = output_path / f"samples-{shard_id:05d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as handle:
            for sample in shard_samples:
                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        shards.append(
            ExternalSampleShard(
                shard_id=shard_id,
                sample_count=len(shard_samples),
                languages=tuple(sorted({sample.canonical_language for sample in shard_samples})),
                datasets=tuple(sorted({sample.dataset_name for sample in shard_samples})),
                output_path=shard_path,
            )
        )
    return tuple(shards)
