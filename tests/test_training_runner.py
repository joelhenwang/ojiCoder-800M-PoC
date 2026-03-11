import json
from pathlib import Path

from eight_hundred_m.tokenizer import (
    build_tokenizer_training_plan,
    collect_tokenizer_training_texts,
    estimate_corpus_bytes,
    write_tokenizer_training_metadata,
)
from eight_hundred_m.training import (
    TrainingRunConfig,
    build_stage_execution_summary,
    build_training_run_plan,
    estimate_optimizer_steps,
)


ROOT = Path(__file__).resolve().parents[1]


def test_tokenizer_training_plan_collects_texts_and_writes_metadata(tmp_path: Path) -> None:
    code_dir = tmp_path / "datasets" / "code"
    docs_dir = tmp_path / "datasets" / "docs"
    synthetic_dir = tmp_path / "datasets" / "synthetic"
    code_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)
    (code_dir / "a.py").write_text("print('a')\r\n", encoding="utf-8")
    (docs_dir / "README.md").write_text("hello", encoding="utf-8")
    (synthetic_dir / "task.jsonl").write_text('{"task": 1}', encoding="utf-8")

    output_dir = tmp_path / "artifacts" / "tokenizer"
    plan = build_tokenizer_training_plan(
        ROOT / "configs" / "tokenizer" / "corpus_manifest.json",
        tmp_path,
        output_dir,
        special_tokens=("<|repo|>", "<|path|>"),
    )
    texts = collect_tokenizer_training_texts(plan)
    metadata_path = write_tokenizer_training_metadata(plan, texts)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(texts) == 3
    assert estimate_corpus_bytes(texts) > 0
    assert payload["text_count"] == 3
    assert payload["special_tokens"] == ["<|repo|>", "<|path|>"]


def test_stage_execution_summary_and_run_plan() -> None:
    run_config = TrainingRunConfig(
        run_name="smoke",
        model_config_path=ROOT / "configs" / "model" / "800m.json",
        tokenizer_config_path=ROOT / "configs" / "tokenizer" / "base_bpe.json",
        stage_config_path=ROOT / "configs" / "train" / "stages.json",
        output_dir=ROOT / "artifacts" / "runs" / "smoke",
        global_batch_size_tokens=8_192,
    )
    run_plan = build_training_run_plan(run_config)
    summary = build_stage_execution_summary(run_plan.stages[0], run_config.global_batch_size_tokens)

    assert run_plan.model_config.hidden_size == 1536
    assert run_plan.tokenizer_config.tokenizer_type == "bpe"
    assert run_plan.total_target_tokens > 0
    assert summary.estimated_optimizer_steps == estimate_optimizer_steps(
        run_plan.stages[0],
        run_config.global_batch_size_tokens,
    )
    assert any(stage.stage_name == "stage_4_tool_policy" for stage in run_plan.stage_summaries)