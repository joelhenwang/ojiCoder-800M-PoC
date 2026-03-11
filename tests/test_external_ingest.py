import json
from pathlib import Path

from eight_hundred_m.data import (
    STAGE1_EXTERNAL_LANGUAGES,
    canonicalize_language,
    collect_external_samples,
    estimate_text_tokens,
    filter_external_samples,
    load_data_manifest,
    pack_external_samples,
    summarize_external_samples,
    summarize_packed_samples,
    write_packed_training_samples,
    write_external_sample_shards,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonicalize_language_aliases() -> None:
    assert canonicalize_language("python") == "Python"
    assert canonicalize_language("bash") == "Shell"
    assert canonicalize_language("typescript") == "TypeScript"
    assert "Python" in STAGE1_EXTERNAL_LANGUAGES


def test_collect_and_filter_external_samples(tmp_path: Path) -> None:
    stack_dir = tmp_path / "datasets" / "the-stack-v2-dedup" / "python"
    commit_dir = tmp_path / "datasets" / "commitpackft" / "python"
    stack_dir.mkdir(parents=True)
    commit_dir.mkdir(parents=True)

    stack_records = [
        {
            "language": "Python",
            "content": "print('stack')\n",
            "repo_name": "org/repo",
            "path": "src/main.py",
            "revision": "rev-1",
            "license_type": "MIT",
            "detected_licenses": ["MIT"],
            "src_encoding": "utf-8"
        },
        {
            "language": "UnknownLang",
            "content": "ignore me\n",
            "repo_name": "org/repo",
            "path": "src/ignore.txt",
            "license_type": "MIT"
        }
    ]
    commit_records = [
        {
            "lang": "python",
            "new_contents": "print('commit')\n",
            "repos": ["org/commit-repo"],
            "new_file": "app.py",
            "commit": "abc123",
            "license": "Apache-2.0"
        },
        {
            "lang": "python",
            "new_contents": "print('bad-license')\n",
            "repos": ["org/commit-repo"],
            "new_file": "blocked.py",
            "commit": "abc124",
            "license": "Proprietary"
        }
    ]

    (stack_dir / "sample.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in stack_records),
        encoding="utf-8",
    )
    (commit_dir / "sample.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in commit_records),
        encoding="utf-8",
    )

    manifest = load_data_manifest(ROOT / "configs" / "data" / "stage1_external_manifest.json")
    samples = collect_external_samples(manifest, tmp_path)
    filtered = filter_external_samples(samples, manifest.filter_policy)
    summary = summarize_external_samples(filtered)
    shards = write_external_sample_shards(filtered, tmp_path / "artifacts" / "external-shards", max_samples_per_shard=1)

    assert len(samples) == 3
    assert len(filtered) == 2
    assert summary.total_samples == 2
    assert summary.by_dataset["bigcode/the-stack-v2-dedup"] == 1
    assert summary.by_dataset["bigcode/commitpackft"] == 1
    assert len(shards) == 2
    assert all(shard.output_path.exists() for shard in shards)

    packs = pack_external_samples(filtered, target_tokens=8, max_samples_per_pack=1)
    pack_summary = summarize_packed_samples(packs)
    packs_path, summary_path = write_packed_training_samples(
        packs,
        tmp_path / "artifacts" / "packed",
    )

    assert len(packs) == 2
    assert pack_summary.total_samples == 2
    assert packs_path.exists()
    assert summary_path.exists()
    assert estimate_text_tokens("print('x')\n") > 0

    tokenizer_packs = pack_external_samples(
        filtered,
        target_tokens=8,
        max_samples_per_pack=1,
        token_counter=lambda text: max(1, len(text.split())),
        token_count_method="tokenizer",
    )

    assert all(pack.token_count_method == "tokenizer" for pack in tokenizer_packs)


def test_load_stage1_external_manifest() -> None:
    manifest = load_data_manifest(ROOT / "configs" / "data" / "stage1_external_manifest.json")

    assert manifest.artifact_name == "stage1-external-manifest"
    assert len(manifest.entries) == 2
    assert manifest.entries[0].source_kind == "jsonl-records"
    assert manifest.entries[1].text_mode == "new_contents"
