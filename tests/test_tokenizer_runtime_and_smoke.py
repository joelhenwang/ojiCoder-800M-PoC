from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from eight_hundred_m.cli import main
from eight_hundred_m.data import PackedTrainingSample, write_packed_training_samples
from eight_hundred_m.tokenizer import load_tokenizer_runtime
from eight_hundred_m.training import build_smoke_batch_summary


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


def test_load_tokenizer_runtime_and_smoke_batch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("eight_hundred_m.tokenizer.runtime.importlib.import_module", _fake_import_module)

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}\n", encoding="utf-8")
    runtime = load_tokenizer_runtime(tokenizer_path)

    assert runtime.count_tokens("hello world") == 2

    packs_path, _ = write_packed_training_samples(
        (
            PackedTrainingSample(
                pack_id=0,
                text="hello world\n",
                approx_tokens=2,
                sample_count=1,
                datasets=("dataset",),
                languages=("Python",),
            ),
        ),
        tmp_path / "packed",
    )
    summary = build_smoke_batch_summary(
        packed_samples_path=packs_path,
        tokenizer_path=tokenizer_path,
        sequence_length=8,
        batch_size=1,
    )

    assert summary.batch_size == 1
    assert summary.example_token_lengths == (2,)


def test_cli_smoke_batch_with_fake_tokenizer(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr("eight_hundred_m.tokenizer.runtime.importlib.import_module", _fake_import_module)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}\n", encoding="utf-8")

    packs_path, _ = write_packed_training_samples(
        (
            PackedTrainingSample(
                pack_id=0,
                text="one two three\n",
                approx_tokens=3,
                sample_count=1,
                datasets=("dataset",),
                languages=("Python",),
            ),
        ),
        tmp_path / "packed",
    )

    exit_code = main(
        [
            "smoke-batch",
            "--packed-samples",
            str(packs_path),
            "--tokenizer-artifact",
            str(tokenizer_path),
            "--sequence-length",
            "8",
            "--batch-size",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["example_token_lengths"] == [3]
