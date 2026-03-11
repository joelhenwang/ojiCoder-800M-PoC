from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Callable, Sequence

from .external import ExternalDataSample


TokenCounter = Callable[[str], int]


def estimate_text_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, ceil(len(stripped) / 4))


@dataclass(slots=True)
class PackedTrainingSample:
    pack_id: int
    text: str
    approx_tokens: int
    sample_count: int
    datasets: tuple[str, ...]
    languages: tuple[str, ...]
    token_count_method: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return {
            "pack_id": self.pack_id,
            "text": self.text,
            "approx_tokens": self.approx_tokens,
            "sample_count": self.sample_count,
            "datasets": list(self.datasets),
            "languages": list(self.languages),
            "token_count_method": self.token_count_method,
        }


@dataclass(slots=True)
class PackedSampleSummary:
    pack_count: int
    total_samples: int
    total_approx_tokens: int


def normalize_stage1_sample_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if normalized.endswith("\n") or not normalized else normalized + "\n"


def pack_external_samples(
    samples: Sequence[ExternalDataSample],
    target_tokens: int,
    max_samples_per_pack: int = 16,
    token_counter: TokenCounter | None = None,
    token_count_method: str = "heuristic",
) -> tuple[PackedTrainingSample, ...]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if max_samples_per_pack <= 0:
        raise ValueError("max_samples_per_pack must be positive")

    packs: list[PackedTrainingSample] = []
    current_texts: list[str] = []
    current_datasets: set[str] = set()
    current_languages: set[str] = set()
    current_tokens = 0
    current_sample_count = 0
    pack_id = 0

    def flush() -> None:
        nonlocal current_texts, current_datasets, current_languages, current_tokens, current_sample_count, pack_id
        if not current_texts:
            return
        packs.append(
            PackedTrainingSample(
                pack_id=pack_id,
                text="".join(current_texts),
                approx_tokens=current_tokens,
                sample_count=current_sample_count,
                datasets=tuple(sorted(current_datasets)),
                languages=tuple(sorted(current_languages)),
                token_count_method=token_count_method,
            )
        )
        pack_id += 1
        current_texts = []
        current_datasets = set()
        current_languages = set()
        current_tokens = 0
        current_sample_count = 0

    for sample in samples:
        text = normalize_stage1_sample_text(sample.text)
        sample_tokens = token_counter(text) if token_counter is not None else estimate_text_tokens(text)
        would_exceed_tokens = current_tokens + sample_tokens > target_tokens
        would_exceed_count = current_sample_count >= max_samples_per_pack
        if current_texts and (would_exceed_tokens or would_exceed_count):
            flush()
        current_texts.append(text)
        current_datasets.add(sample.dataset_name)
        current_languages.add(sample.canonical_language)
        current_tokens += sample_tokens
        current_sample_count += 1
    flush()
    return tuple(packs)


def summarize_packed_samples(packs: Sequence[PackedTrainingSample]) -> PackedSampleSummary:
    return PackedSampleSummary(
        pack_count=len(packs),
        total_samples=sum(pack.sample_count for pack in packs),
        total_approx_tokens=sum(pack.approx_tokens for pack in packs),
    )


def write_packed_training_samples(
    packs: Sequence[PackedTrainingSample],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    packs_path = output_path / "packed-samples.jsonl"
    summary_path = output_path / "packed-summary.json"

    with packs_path.open("w", encoding="utf-8") as handle:
        for pack in packs:
            handle.write(json.dumps(pack.to_dict(), ensure_ascii=False) + "\n")

    summary = summarize_packed_samples(packs)
    summary_path.write_text(
        json.dumps(
            {
                "pack_count": summary.pack_count,
                "total_samples": summary.total_samples,
                "total_approx_tokens": summary.total_approx_tokens,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return packs_path, summary_path


def load_packed_training_samples(path: str | Path) -> tuple[PackedTrainingSample, ...]:
    packs: list[PackedTrainingSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("packed training sample record must be a JSON object")
            packs.append(
                PackedTrainingSample(
                    pack_id=int(payload["pack_id"]),
                    text=str(payload["text"]),
                    approx_tokens=int(payload["approx_tokens"]),
                    sample_count=int(payload["sample_count"]),
                    datasets=tuple(str(item) for item in payload.get("datasets", [])),
                    languages=tuple(str(item) for item in payload.get("languages", [])),
                    token_count_method=str(payload.get("token_count_method", "heuristic")),
                )
            )
    return tuple(packs)