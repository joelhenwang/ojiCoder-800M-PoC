from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data import load_packed_training_samples
from ..tokenizer import load_tokenizer_runtime


@dataclass(slots=True)
class SmokeBatchSummary:
    batch_size: int
    sequence_length: int
    example_token_lengths: tuple[int, ...]
    tokenizer_path: Path
    source_path: Path


def _pad_or_truncate(token_ids: list[int], sequence_length: int, pad_token_id: int = 0) -> list[int]:
    if len(token_ids) >= sequence_length:
        return token_ids[:sequence_length]
    return token_ids + [pad_token_id] * (sequence_length - len(token_ids))


def build_smoke_batch_summary(
    packed_samples_path: str | Path,
    tokenizer_path: str | Path,
    sequence_length: int,
    batch_size: int = 2,
) -> SmokeBatchSummary:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    runtime = load_tokenizer_runtime(tokenizer_path)
    packed_samples = load_packed_training_samples(packed_samples_path)
    selected = packed_samples[:batch_size]
    lengths: list[int] = []
    for sample in selected:
        token_ids = runtime.encode(sample.text)
        padded = _pad_or_truncate(token_ids, sequence_length)
        lengths.append(min(len(token_ids), len(padded)))
    return SmokeBatchSummary(
        batch_size=len(selected),
        sequence_length=sequence_length,
        example_token_lengths=tuple(lengths),
        tokenizer_path=Path(tokenizer_path),
        source_path=Path(packed_samples_path),
    )