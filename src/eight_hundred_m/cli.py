from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_tokenizer_config
from .data import (
    build_data_shard_descriptors,
    collect_external_samples,
    discover_data_files,
    filter_external_samples,
    filter_planned_data_files,
    load_data_manifest,
    pack_external_samples,
    summarize_planned_data_files,
    summarize_external_samples,
)
from .tokenizer import (
    build_tokenizer_training_plan,
    collect_tokenizer_training_texts,
    discover_tokenizer_corpus_files,
    load_tokenizer_corpus_manifest,
    load_tokenizer_runtime,
    summarize_tokenizer_corpus,
)
from .training import TrainingRunConfig, build_smoke_batch_summary, build_training_run_plan
from .training.loop import execute_training_run_skeleton


def _json_dump(payload: object) -> str:
    return json.dumps(payload, indent=2) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eight-hundred-m")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenizer_summary = subparsers.add_parser("tokenizer-summary")
    tokenizer_summary.add_argument("--manifest", required=True)
    tokenizer_summary.add_argument("--root", required=True)

    data_plan = subparsers.add_parser("data-plan")
    data_plan.add_argument("--manifest", required=True)
    data_plan.add_argument("--root", required=True)
    data_plan.add_argument("--max-files-per-shard", type=int, default=128)

    external_pack = subparsers.add_parser("external-pack")
    external_pack.add_argument("--manifest", required=True)
    external_pack.add_argument("--root", required=True)
    external_pack.add_argument("--target-tokens", type=int, default=4096)
    external_pack.add_argument("--max-samples-per-pack", type=int, default=16)
    external_pack.add_argument("--tokenizer-artifact")

    smoke_batch = subparsers.add_parser("smoke-batch")
    smoke_batch.add_argument("--packed-samples", required=True)
    smoke_batch.add_argument("--tokenizer-artifact", required=True)
    smoke_batch.add_argument("--sequence-length", type=int, required=True)
    smoke_batch.add_argument("--batch-size", type=int, default=2)

    run_summary = subparsers.add_parser("run-summary")
    run_summary.add_argument("--model-config", required=True)
    run_summary.add_argument("--tokenizer-config", required=True)
    run_summary.add_argument("--stage-config", required=True)
    run_summary.add_argument("--output-dir", required=True)
    run_summary.add_argument("--run-name", required=True)
    run_summary.add_argument("--global-batch-size-tokens", type=int, default=4_194_304)

    init_run = subparsers.add_parser("init-run")
    init_run.add_argument("--model-config", required=True)
    init_run.add_argument("--tokenizer-config", required=True)
    init_run.add_argument("--stage-config", required=True)
    init_run.add_argument("--output-dir", required=True)
    init_run.add_argument("--run-name", required=True)
    init_run.add_argument("--global-batch-size-tokens", type=int, default=4_194_304)
    init_run.add_argument("--external-data-manifest")
    init_run.add_argument("--external-data-root")
    init_run.add_argument("--pack-target-tokens", type=int)
    init_run.add_argument("--max-samples-per-pack", type=int, default=16)
    init_run.add_argument("--tokenizer-artifact")

    return parser


def command_tokenizer_summary(manifest: str, root: str) -> dict[str, object]:
    tokenizer_manifest = load_tokenizer_corpus_manifest(manifest)
    files = discover_tokenizer_corpus_files(tokenizer_manifest, root)
    summary = summarize_tokenizer_corpus(tokenizer_manifest, root)
    return {
        "artifact_name": tokenizer_manifest.artifact_name,
        "artifact_version": tokenizer_manifest.artifact_version,
        "file_count": len(files),
        "summary": summary,
    }


def command_data_plan(manifest: str, root: str, max_files_per_shard: int) -> dict[str, object]:
    data_manifest = load_data_manifest(manifest)
    discovered = discover_data_files(data_manifest, root)
    filtered = filter_planned_data_files(discovered, data_manifest.filter_policy)
    summary = summarize_planned_data_files(filtered)
    shard_descriptors = build_data_shard_descriptors(filtered, max_files_per_shard=max_files_per_shard)
    return {
        "artifact_name": data_manifest.artifact_name,
        "total_files": summary.total_files,
        "by_entry": summary.by_entry,
        "by_content_type": summary.by_content_type,
        "shard_count": len(shard_descriptors),
    }


def command_external_pack(
    manifest: str,
    root: str,
    target_tokens: int,
    max_samples_per_pack: int,
    tokenizer_artifact: str | None = None,
) -> dict[str, object]:
    data_manifest = load_data_manifest(manifest)
    samples = collect_external_samples(data_manifest, root)
    filtered = filter_external_samples(samples, data_manifest.filter_policy)
    token_counter = None
    token_count_method = "heuristic"
    if tokenizer_artifact is not None:
        runtime = load_tokenizer_runtime(tokenizer_artifact)
        token_counter = runtime.count_tokens
        token_count_method = "tokenizer"
    packs = pack_external_samples(
        filtered,
        target_tokens=target_tokens,
        max_samples_per_pack=max_samples_per_pack,
        token_counter=token_counter,
        token_count_method=token_count_method,
    )
    summary = summarize_external_samples(filtered)
    return {
        "artifact_name": data_manifest.artifact_name,
        "sample_count": summary.total_samples,
        "pack_count": len(packs),
        "by_dataset": summary.by_dataset,
        "by_language": summary.by_language,
        "token_count_method": token_count_method,
    }


def command_smoke_batch(
    packed_samples: str,
    tokenizer_artifact: str,
    sequence_length: int,
    batch_size: int,
) -> dict[str, object]:
    summary = build_smoke_batch_summary(
        packed_samples_path=packed_samples,
        tokenizer_path=tokenizer_artifact,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    return {
        "batch_size": summary.batch_size,
        "sequence_length": summary.sequence_length,
        "example_token_lengths": list(summary.example_token_lengths),
        "tokenizer_path": str(summary.tokenizer_path),
        "source_path": str(summary.source_path),
    }


def _build_run_config(args: argparse.Namespace) -> TrainingRunConfig:
    return TrainingRunConfig(
        run_name=args.run_name,
        model_config_path=Path(args.model_config),
        tokenizer_config_path=Path(args.tokenizer_config),
        stage_config_path=Path(args.stage_config),
        output_dir=Path(args.output_dir),
        global_batch_size_tokens=args.global_batch_size_tokens,
    )


def command_run_summary(args: argparse.Namespace) -> dict[str, object]:
    run_plan = build_training_run_plan(_build_run_config(args))
    tokenizer_config = load_tokenizer_config(args.tokenizer_config)
    return {
        "run_name": run_plan.run_config.run_name,
        "tokenizer_type": tokenizer_config.tokenizer_type,
        "total_target_tokens": run_plan.total_target_tokens,
        "stage_count": len(run_plan.stages),
        "stages": [
            {
                "stage_name": summary.stage_name,
                "estimated_optimizer_steps": summary.estimated_optimizer_steps,
            }
            for summary in run_plan.stage_summaries
        ],
    }


def command_init_run(args: argparse.Namespace) -> dict[str, object]:
    run_plan = build_training_run_plan(_build_run_config(args))
    token_counter = None
    token_count_method = "heuristic"
    if args.tokenizer_artifact is not None:
        runtime = load_tokenizer_runtime(args.tokenizer_artifact)
        token_counter = runtime.count_tokens
        token_count_method = "tokenizer"
    result = execute_training_run_skeleton(
        run_plan,
        stage1_external_manifest_path=args.external_data_manifest,
        stage1_external_root=args.external_data_root,
        pack_target_tokens=args.pack_target_tokens,
        max_samples_per_pack=args.max_samples_per_pack,
        token_counter=token_counter,
        token_count_method=token_count_method,
    )
    return {
        "run_dir": str(result.run_dir),
        "plan_path": str(result.plan_path),
        "checkpoint_count": len(result.stage_checkpoints),
        "stage1_data_dir": str(result.stage1_data_dir) if result.stage1_data_dir is not None else None,
        "token_count_method": token_count_method,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "tokenizer-summary":
        payload = command_tokenizer_summary(args.manifest, args.root)
    elif args.command == "data-plan":
        payload = command_data_plan(args.manifest, args.root, args.max_files_per_shard)
    elif args.command == "external-pack":
        payload = command_external_pack(
            args.manifest,
            args.root,
            args.target_tokens,
            args.max_samples_per_pack,
            args.tokenizer_artifact,
        )
    elif args.command == "smoke-batch":
        payload = command_smoke_batch(
            args.packed_samples,
            args.tokenizer_artifact,
            args.sequence_length,
            args.batch_size,
        )
    elif args.command == "run-summary":
        payload = command_run_summary(args)
    elif args.command == "init-run":
        payload = command_init_run(args)
    else:
        parser.error(f"unknown command: {args.command}")
        return 2

    print(_json_dump(payload), end="")
    return 0
