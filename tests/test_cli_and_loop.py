import json
import importlib
from pathlib import Path
from types import SimpleNamespace

from eight_hundred_m.cli import main
from eight_hundred_m.training import TrainingRunConfig, build_training_run_plan
from eight_hundred_m.training.loop import execute_training_run_skeleton


ROOT = Path(__file__).resolve().parents[1]


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizerBackend:
    def encode(self, text: str) -> _FakeEncoding:
        token_count = max(1, len(text.strip().split()))
        return _FakeEncoding(list(range(token_count)))

    @classmethod
    def from_file(cls, _path: str):
        return cls()


def _fake_import_module(name: str) -> object:
    if name == "tokenizers":
        return SimpleNamespace(Tokenizer=_FakeTokenizerBackend)
    return importlib.import_module(name)


def test_cli_tokenizer_summary(tmp_path: Path, capsys) -> None:
    code_dir = tmp_path / "datasets" / "code"
    docs_dir = tmp_path / "datasets" / "docs"
    synthetic_dir = tmp_path / "datasets" / "synthetic"
    code_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    synthetic_dir.mkdir(parents=True)
    (code_dir / "sample.py").write_text("print('x')\n", encoding="utf-8")
    (docs_dir / "README.md").write_text("docs\n", encoding="utf-8")
    (synthetic_dir / "task.jsonl").write_text('{"task":1}\n', encoding="utf-8")

    exit_code = main([
        "tokenizer-summary",
        "--manifest",
        str(ROOT / "configs" / "tokenizer" / "corpus_manifest.json"),
        "--root",
        str(tmp_path),
    ])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["file_count"] == 3
    assert output["summary"]["code-primary"] == 1


def test_cli_data_plan_and_init_run(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "datasets"
    (data_root / "code").mkdir(parents=True)
    (data_root / "docs").mkdir(parents=True)
    (data_root / "synthetic").mkdir(parents=True)
    (data_root / "tool").mkdir(parents=True)
    (data_root / "code" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (data_root / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    (data_root / "synthetic" / "task.jsonl").write_text("{}\n", encoding="utf-8")
    (data_root / "tool" / "trace.jsonl").write_text("{}\n", encoding="utf-8")

    exit_code = main([
        "data-plan",
        "--manifest",
        str(ROOT / "configs" / "data" / "base_manifest.json"),
        "--root",
        str(tmp_path),
        "--max-files-per-shard",
        "2",
    ])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["total_files"] == 4
    assert output["shard_count"] == 2

    run_exit = main([
        "init-run",
        "--model-config",
        str(ROOT / "configs" / "model" / "800m.json"),
        "--tokenizer-config",
        str(ROOT / "configs" / "tokenizer" / "base_bpe.json"),
        "--stage-config",
        str(ROOT / "configs" / "train" / "stages.json"),
        "--output-dir",
        str(tmp_path / "artifacts" / "runs"),
        "--run-name",
        "smoke",
        "--global-batch-size-tokens",
        "8192",
    ])
    run_output = json.loads(capsys.readouterr().out)

    assert run_exit == 0
    assert run_output["checkpoint_count"] == 5


def test_cli_external_pack_and_init_run_with_external_data(tmp_path: Path, capsys) -> None:
    stack_dir = tmp_path / "datasets" / "the-stack-v2-dedup" / "python"
    commit_dir = tmp_path / "datasets" / "commitpackft" / "python"
    stack_dir.mkdir(parents=True)
    commit_dir.mkdir(parents=True)
    (stack_dir / "sample.jsonl").write_text(
        json.dumps(
            {
                "language": "Python",
                "content": "print('stack')\n",
                "repo_name": "org/repo",
                "path": "src/main.py",
                "revision": "rev-1",
                "license_type": "MIT",
                "detected_licenses": ["MIT"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (commit_dir / "sample.jsonl").write_text(
        json.dumps(
            {
                "lang": "python",
                "new_contents": "print('commit')\n",
                "repos": ["org/commit-repo"],
                "new_file": "app.py",
                "commit": "abc123",
                "license": "Apache-2.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    external_exit = main([
        "external-pack",
        "--manifest",
        str(ROOT / "configs" / "data" / "stage1_external_manifest.json"),
        "--root",
        str(tmp_path),
        "--target-tokens",
        "8",
        "--max-samples-per-pack",
        "1",
    ])
    external_output = json.loads(capsys.readouterr().out)

    assert external_exit == 0
    assert external_output["sample_count"] == 2
    assert external_output["pack_count"] == 2

    run_exit = main([
        "init-run",
        "--model-config",
        str(ROOT / "configs" / "model" / "800m.json"),
        "--tokenizer-config",
        str(ROOT / "configs" / "tokenizer" / "base_bpe.json"),
        "--stage-config",
        str(ROOT / "configs" / "train" / "stages.json"),
        "--output-dir",
        str(tmp_path / "artifacts" / "runs"),
        "--run-name",
        "external-smoke",
        "--global-batch-size-tokens",
        "8192",
        "--external-data-manifest",
        str(ROOT / "configs" / "data" / "stage1_external_manifest.json"),
        "--external-data-root",
        str(tmp_path),
        "--pack-target-tokens",
        "8",
        "--max-samples-per-pack",
        "1",
    ])
    run_output = json.loads(capsys.readouterr().out)

    assert run_exit == 0
    assert run_output["stage1_data_dir"] is not None
    assert Path(run_output["stage1_data_dir"]).exists()


def test_cli_external_pack_with_tokenizer_artifact(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("eight_hundred_m.tokenizer.runtime.importlib.import_module", _fake_import_module)
    stack_dir = tmp_path / "datasets" / "the-stack-v2-dedup" / "python"
    stack_dir.mkdir(parents=True)
    (stack_dir / "sample.jsonl").write_text(
        json.dumps(
            {
                "language": "Python",
                "content": "print('stack data')\n",
                "repo_name": "org/repo",
                "path": "src/main.py",
                "revision": "rev-1",
                "license_type": "MIT",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}\n", encoding="utf-8")

    exit_code = main([
        "external-pack",
        "--manifest",
        str(ROOT / "configs" / "data" / "stage1_external_manifest.json"),
        "--root",
        str(tmp_path),
        "--target-tokens",
        "8",
        "--max-samples-per-pack",
        "1",
        "--tokenizer-artifact",
        str(tokenizer_path),
    ])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["token_count_method"] == "tokenizer"


def test_training_loop_writes_run_plan_and_checkpoints(tmp_path: Path) -> None:
    run_config = TrainingRunConfig(
        run_name="loop-smoke",
        model_config_path=ROOT / "configs" / "model" / "800m.json",
        tokenizer_config_path=ROOT / "configs" / "tokenizer" / "base_bpe.json",
        stage_config_path=ROOT / "configs" / "train" / "stages.json",
        output_dir=tmp_path / "artifacts" / "runs",
        global_batch_size_tokens=8192,
    )
    run_plan = build_training_run_plan(run_config)
    result = execute_training_run_skeleton(run_plan)

    assert result.plan_path.exists()
    assert len(result.stage_checkpoints) == len(run_plan.stages)
    assert all(checkpoint.metadata_path.exists() for checkpoint in result.stage_checkpoints)
