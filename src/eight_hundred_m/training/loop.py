from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..data import (
    collect_external_samples,
    filter_external_samples,
    load_data_manifest,
    pack_external_samples,
    summarize_external_samples,
    write_external_sample_shards,
    write_packed_training_samples,
)
from .runner import TrainingRunPlan, build_training_run_plan


@dataclass(slots=True)
class StageCheckpointPlaceholder:
    stage_name: str
    checkpoint_path: Path
    metadata_path: Path


@dataclass(slots=True)
class TrainingSkeletonResult:
    run_dir: Path
    plan_path: Path
    stage_checkpoints: tuple[StageCheckpointPlaceholder, ...]
    stage1_data_dir: Path | None = None


def initialize_run_directory(run_plan: TrainingRunPlan) -> Path:
    run_dir = run_plan.run_config.output_dir / run_plan.run_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_plan_metadata(run_plan: TrainingRunPlan, run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_plan.run_config.run_name,
        "model_config_path": str(run_plan.run_config.model_config_path),
        "tokenizer_config_path": str(run_plan.run_config.tokenizer_config_path),
        "stage_config_path": str(run_plan.run_config.stage_config_path),
        "global_batch_size_tokens": run_plan.run_config.global_batch_size_tokens,
        "total_target_tokens": run_plan.total_target_tokens,
        "stages": [
            {
                "stage_name": summary.stage_name,
                "objective": summary.objective,
                "sequence_length": summary.sequence_length,
                "target_tokens": summary.target_tokens,
                "estimated_sequences": summary.estimated_sequences,
                "estimated_optimizer_steps": summary.estimated_optimizer_steps,
                "use_repo_formatting": summary.use_repo_formatting,
                "use_tool_formatting": summary.use_tool_formatting,
            }
            for summary in run_plan.stage_summaries
        ],
    }
    plan_path = run_path / "run_plan.json"
    plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return plan_path


def create_checkpoint_placeholder(
    run_dir: str | Path,
    stage_name: str,
    step: int = 0,
) -> StageCheckpointPlaceholder:
    stage_dir = Path(run_dir) / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = stage_dir / f"checkpoint-step-{step:08d}.bin"
    metadata_path = stage_dir / "checkpoint.json"
    checkpoint_path.write_bytes(b"")
    metadata_path.write_text(
        json.dumps(
            {
                "stage_name": stage_name,
                "step": step,
                "placeholder": True,
                "checkpoint_file": checkpoint_path.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return StageCheckpointPlaceholder(
        stage_name=stage_name,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
    )


def prepare_stage1_external_data(
    manifest_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    pack_target_tokens: int,
    max_samples_per_pack: int = 16,
    token_counter: Callable[[str], int] | None = None,
    token_count_method: str = "heuristic",
) -> Path:
    manifest = load_data_manifest(manifest_path)
    samples = collect_external_samples(manifest, data_root)
    filtered = filter_external_samples(samples, manifest.filter_policy)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    external_summary = summarize_external_samples(filtered)
    write_external_sample_shards(filtered, output_path / "external-shards", max_samples_per_shard=max_samples_per_pack)
    packs = pack_external_samples(
        filtered,
        target_tokens=pack_target_tokens,
        max_samples_per_pack=max_samples_per_pack,
        token_counter=token_counter,
        token_count_method=token_count_method,
    )
    packed_path, packed_summary_path = write_packed_training_samples(packs, output_path / "packed")
    summary_path = output_path / "external-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "total_samples": external_summary.total_samples,
                "by_dataset": external_summary.by_dataset,
                "by_language": external_summary.by_language,
                "by_content_type": external_summary.by_content_type,
                "packed_samples_path": str(packed_path),
                "packed_summary_path": str(packed_summary_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def execute_training_run_skeleton(
    run_plan: TrainingRunPlan,
    stage1_external_manifest_path: str | Path | None = None,
    stage1_external_root: str | Path | None = None,
    pack_target_tokens: int | None = None,
    max_samples_per_pack: int = 16,
    token_counter: Callable[[str], int] | None = None,
    token_count_method: str = "heuristic",
) -> TrainingSkeletonResult:
    run_dir = initialize_run_directory(run_plan)
    plan_path = write_run_plan_metadata(run_plan, run_dir)
    stage_checkpoints = tuple(
        create_checkpoint_placeholder(run_dir, summary.stage_name)
        for summary in run_plan.stage_summaries
    )
    stage1_data_dir: Path | None = None
    if stage1_external_manifest_path is not None and stage1_external_root is not None:
        stage1_data_dir = prepare_stage1_external_data(
            manifest_path=stage1_external_manifest_path,
            data_root=stage1_external_root,
            output_dir=run_dir / "stage1-data",
            pack_target_tokens=pack_target_tokens or run_plan.stages[0].sequence_length,
            max_samples_per_pack=max_samples_per_pack,
            token_counter=token_counter,
            token_count_method=token_count_method,
        )
    return TrainingSkeletonResult(
        run_dir=run_dir,
        plan_path=plan_path,
        stage_checkpoints=stage_checkpoints,
        stage1_data_dir=stage1_data_dir,
    )


def build_and_execute_training_run(run_config_path: TrainingRunPlan | None = None) -> None:
    raise NotImplementedError("Use build_training_run_plan() and execute_training_run_skeleton() explicitly")
