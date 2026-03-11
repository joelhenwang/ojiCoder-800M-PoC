from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .corpus import discover_tokenizer_corpus_files, load_normalized_corpus_texts
from .spec import TokenizerCorpusManifest, load_tokenizer_corpus_manifest
from .validation import summarize_validation_results


@dataclass(slots=True)
class TokenizerTrainingPlan:
    manifest: TokenizerCorpusManifest
    corpus_root: Path
    output_dir: Path
    vocab_size: int
    byte_fallback: bool
    special_tokens: tuple[str, ...]

    @property
    def tokenizer_output_path(self) -> Path:
        return self.output_dir / "tokenizer.json"

    @property
    def metadata_output_path(self) -> Path:
        return self.output_dir / "training_summary.json"


def build_tokenizer_training_plan(
    manifest_path: str | Path,
    corpus_root: str | Path,
    output_dir: str | Path,
    vocab_size: int = 64_000,
    byte_fallback: bool = True,
    special_tokens: Sequence[str] | None = None,
) -> TokenizerTrainingPlan:
    manifest = load_tokenizer_corpus_manifest(manifest_path)
    return TokenizerTrainingPlan(
        manifest=manifest,
        corpus_root=Path(corpus_root),
        output_dir=Path(output_dir),
        vocab_size=vocab_size,
        byte_fallback=byte_fallback,
        special_tokens=tuple(special_tokens or ()),
    )


def collect_tokenizer_training_texts(
    plan: TokenizerTrainingPlan,
    limit: int | None = None,
) -> tuple[str, ...]:
    files = discover_tokenizer_corpus_files(plan.manifest, plan.corpus_root)
    if limit is not None:
        files = files[:limit]
    return load_normalized_corpus_texts(files, plan.manifest)


def estimate_corpus_bytes(texts: Sequence[str]) -> int:
    return sum(len(text.encode("utf-8")) for text in texts)


def write_tokenizer_training_metadata(
    plan: TokenizerTrainingPlan,
    texts: Sequence[str],
) -> Path:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    validation_summary = summarize_validation_results(texts, plan.manifest.normalization)
    payload = {
        "artifact_name": plan.manifest.artifact_name,
        "artifact_version": plan.manifest.artifact_version,
        "vocab_size": plan.vocab_size,
        "byte_fallback": plan.byte_fallback,
        "special_tokens": list(plan.special_tokens),
        "text_count": len(texts),
        "corpus_bytes": estimate_corpus_bytes(texts),
        "corpus_root": str(plan.corpus_root),
        "validation_summary": {
            "sample_count": validation_summary.sample_count,
            "passed_counts": validation_summary.passed_counts,
        },
    }
    plan.metadata_output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return plan.metadata_output_path


def train_bpe_tokenizer(plan: TokenizerTrainingPlan, texts: Sequence[str]) -> Path:
    tokenizers = importlib.import_module("tokenizers")

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(byte_fallback=plan.byte_fallback))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=plan.vocab_size,
        special_tokens=list(plan.special_tokens),
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(plan.tokenizer_output_path))
    write_tokenizer_training_metadata(plan, texts)
    return plan.tokenizer_output_path
