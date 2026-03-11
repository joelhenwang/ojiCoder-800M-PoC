from pathlib import Path

from eight_hundred_m.data import load_data_manifest
from eight_hundred_m.tokenizer import load_tokenizer_corpus_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_load_tokenizer_corpus_manifest() -> None:
    manifest = load_tokenizer_corpus_manifest(
        ROOT / "configs" / "tokenizer" / "corpus_manifest.json"
    )

    assert manifest.authoritative_tokenizer_config == "configs/tokenizer/base_bpe.json"
    assert len(manifest.entries) == 3
    assert abs(sum(entry.weight for entry in manifest.entries) - 1.0) < 1e-6


def test_code_preserving_normalization() -> None:
    manifest = load_tokenizer_corpus_manifest(
        ROOT / "configs" / "tokenizer" / "corpus_manifest.json"
    )

    normalized = manifest.normalization.normalize_text("def f():\r\n    return 1")

    assert normalized == "def f():\n    return 1\n"


def test_load_data_manifest() -> None:
    manifest = load_data_manifest(ROOT / "configs" / "data" / "base_manifest.json")

    assert manifest.filter_policy.exclude_generated is True
    assert manifest.filter_policy.file_level_dedup is True
    assert len(manifest.entries) == 4
    assert abs(sum(entry.weight for entry in manifest.entries) - 1.0) < 1e-6