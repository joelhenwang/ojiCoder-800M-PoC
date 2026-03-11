from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import TrainingStageConfig, load_training_stages


@dataclass(slots=True)
class StageRuntimePlan:
    stage: TrainingStageConfig
    estimated_sequences: int


def estimate_stage_sequences(stage: TrainingStageConfig) -> int:
    return stage.target_tokens // stage.sequence_length


def build_stage_runtime_plans(
    config_path: str | Path,
) -> tuple[StageRuntimePlan, ...]:
    stages = load_training_stages(config_path)
    return tuple(
        StageRuntimePlan(stage=stage, estimated_sequences=estimate_stage_sequences(stage))
        for stage in stages
    )


def get_stage_by_name(
    stages: tuple[TrainingStageConfig, ...],
    stage_name: str,
) -> TrainingStageConfig:
    for stage in stages:
        if stage.name == stage_name:
            return stage
    raise KeyError(f"Unknown stage: {stage_name}")
