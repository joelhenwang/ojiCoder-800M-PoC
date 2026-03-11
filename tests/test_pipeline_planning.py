import json
from pathlib import Path

from eight_hundred_m.config import TrainingStageConfig
from eight_hundred_m.data import (
    DataFilterPolicy,
    build_data_shard_descriptors,
    discover_data_files,
    filter_planned_data_files,
    plan_data_shards,
    summarize_planned_data_files,
    write_data_planning_artifacts,
)
from eight_hundred_m.tokenizer import (
    discover_tokenizer_corpus_files,
    iter_manifest_glob_matches,
    load_normalized_corpus_texts,
    load_tokenizer_corpus_manifest,
    summarize_tokenizer_corpus,
    summarize_validation_results,
    validate_normalized_text,
)
from eight_hundred_m.training import (
    build_stage_runtime_plans,
    estimate_stage_sequences,
    get_stage_by_name,
)


ROOT = Path(__file__).resolve().parents[1]


def test_iter_manifest_glob_matches_expands_braces(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "code"
    dataset_root.mkdir(parents=True)
    (dataset_root / "one.py").write_text("print('x')\n", encoding="utf-8")
    (dataset_root / "two.ts").write_text("export const x = 1;\n", encoding="utf-8")

    matches = iter_manifest_glob_matches(tmp_path, "datasets/code/**/*.{py,ts}")

    assert len(matches) == 2


def test_discover_tokenizer_corpus_files_and_summarize(tmp_path: Path) -> None:
    code_dir = tmp_path / "datasets" / "code"
    docs_dir = tmp_path / "datasets" / "docs"
    synthetic_dir = tmp_path / "datasets" / "synthetic"
    code_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)
    (code_dir / "sample.py").write_text("print('x')\r\n", encoding="utf-8")
    (docs_dir / "README.md").write_text("docs\n", encoding="utf-8")
    (synthetic_dir / "task.jsonl").write_text('{"task": 1}\n', encoding="utf-8")

    manifest = load_tokenizer_corpus_manifest(ROOT / "configs" / "tokenizer" / "corpus_manifest.json")
    files = discover_tokenizer_corpus_files(manifest, tmp_path)
    summary = summarize_tokenizer_corpus(manifest, tmp_path)
    texts = load_normalized_corpus_texts(files, manifest)

    assert len(files) == 3
    assert summary["code-primary"] == 1
    assert summary["docs-adjacent"] == 1
    assert summary["synthetic-bootstrap"] == 1
    assert any(text.endswith("\n") for text in texts)


def test_validate_normalized_text_and_summary() -> None:
    manifest = load_tokenizer_corpus_manifest(ROOT / "configs" / "tokenizer" / "corpus_manifest.json")
    sample = "@@ -1,2 +1,2 @@\r\n<|tool_call|>{}\r\npath/to/file.py\r\n    indented\r\n"

    result = validate_normalized_text(sample, manifest.normalization)
    summary = summarize_validation_results((sample,), manifest.normalization)

    assert result.whitespace_fidelity is True
    assert result.path_preservation is True
    assert result.diff_markup is True
    assert result.tool_transcript_markup is True
    assert summary.passed_counts["whitespace_fidelity"] == 1


def test_discover_filter_and_shard_data_files(tmp_path: Path) -> None:
    data_root = tmp_path / "datasets"
    (data_root / "code" / "src").mkdir(parents=True)
    (data_root / "code" / "vendor").mkdir(parents=True)
    (data_root / "docs").mkdir(parents=True)
    (data_root / "synthetic").mkdir(parents=True)
    (data_root / "tool").mkdir(parents=True)
    (data_root / "code" / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (data_root / "code" / "vendor" / "vendored.py").write_text("print('vendor')\n", encoding="utf-8")
    (data_root / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    (data_root / "synthetic" / "generated.min.jsonl").write_text("{}\n", encoding="utf-8")
    (data_root / "tool" / "trace.jsonl").write_text("{}\n", encoding="utf-8")

    from eight_hundred_m.data import load_data_manifest

    manifest = load_data_manifest(ROOT / "configs" / "data" / "base_manifest.json")
    discovered = discover_data_files(manifest, tmp_path)
    filtered = filter_planned_data_files(discovered, manifest.filter_policy)
    shards = plan_data_shards(filtered, max_files_per_shard=2)
    descriptors = build_data_shard_descriptors(filtered, max_files_per_shard=2)
    summary = summarize_planned_data_files(filtered)
    summary_path, shards_path = write_data_planning_artifacts(
        filtered,
        tmp_path / "artifacts" / "data-plan",
        max_files_per_shard=2,
    )
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    written_shards = json.loads(shards_path.read_text(encoding="utf-8"))

    assert len(discovered) == 5
    assert len(filtered) == 3
    assert len(shards) == 2
    assert len(descriptors) == 2
    assert summary.total_files == 3
    assert written_summary["total_files"] == 3
    assert len(written_shards["shards"]) == 2


def test_estimate_and_lookup_stage_runtime_plans() -> None:
    plans = build_stage_runtime_plans(ROOT / "configs" / "train" / "stages.json")
    stages = tuple(plan.stage for plan in plans)
    stage = get_stage_by_name(stages, "stage_1_base")

    assert isinstance(stage, TrainingStageConfig)
    assert estimate_stage_sequences(stage) == stage.target_tokens // stage.sequence_length
    assert plans[0].estimated_sequences > 0


def test_license_filter_helper() -> None:
    policy = DataFilterPolicy(allow_licenses=("MIT",))

    assert filter_planned_data_files((), policy) == ()