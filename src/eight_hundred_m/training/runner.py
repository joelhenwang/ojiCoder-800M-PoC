from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

from ..config import (
    ModelConfig,
    TokenizerConfig,
    TrainingStageConfig,
    load_model_config,
    load_tokenizer_config,
)
from .stages import build_stage_runtime_plans


@dataclass(slots=True)
class TrainingRunConfig:
    run_name: str
    model_config_path: Path
    tokenizer_config_path: Path
    stage_config_path: Path
    output_dir: Path
    global_batch_size_tokens: int = 4_194_304


@dataclass(slots=True)
class StageExecutionSummary:
    stage_name: str
    objective: str
    sequence_length: int
    target_tokens: int
    estimated_sequences: int
    estimated_optimizer_steps: int
    use_repo_formatting: bool
    use_tool_formatting: bool


@dataclass(slots=True)
class TrainingRunPlan:
    run_config: TrainingRunConfig
    model_config: ModelConfig
    tokenizer_config: TokenizerConfig
    stages: tuple[TrainingStageConfig, ...]
    stage_summaries: tuple[StageExecutionSummary, ...]

    @property
    def total_target_tokens(self) -> int:
        return sum(stage.target_tokens for stage in self.stages)


def estimate_optimizer_steps(
    stage: TrainingStageConfig,
    global_batch_size_tokens: int,
) -> int:
    if global_batch_size_tokens <= 0:
        raise ValueError("global_batch_size_tokens must be positive")
    return ceil(stage.target_tokens / global_batch_size_tokens)


def build_stage_execution_summary(
    stage: TrainingStageConfig,
    global_batch_size_tokens: int,
) -> StageExecutionSummary:
    estimated_sequences = stage.target_tokens // stage.sequence_length
    return StageExecutionSummary(
        stage_name=stage.name,
        objective=stage.objective,
        sequence_length=stage.sequence_length,
        target_tokens=stage.target_tokens,
        estimated_sequences=estimated_sequences,
        estimated_optimizer_steps=estimate_optimizer_steps(stage, global_batch_size_tokens),
        use_repo_formatting=stage.use_repo_formatting,
        use_tool_formatting=stage.use_tool_formatting,
    )


def build_training_run_plan(config: TrainingRunConfig) -> TrainingRunPlan:
    model_config = load_model_config(config.model_config_path)
    tokenizer_config = load_tokenizer_config(config.tokenizer_config_path)
    stage_runtime_plans = build_stage_runtime_plans(config.stage_config_path)
    stages = tuple(runtime_plan.stage for runtime_plan in stage_runtime_plans)
    summaries = tuple(
        build_stage_execution_summary(stage, config.global_batch_size_tokens)
        for stage in stages
    )
    return TrainingRunPlan(
        run_config=config,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        stages=stages,
        stage_summaries=summaries,
    )
