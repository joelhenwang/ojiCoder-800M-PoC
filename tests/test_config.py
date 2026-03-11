import pytest
from pathlib import Path

from eight_hundred_m import (
    ModelConfig,
    TokenizerConfig,
    build_default_training_stages,
    load_model_config,
    load_tokenizer_config,
    load_training_stages,
)


ROOT = Path(__file__).resolve().parents[1]


def test_model_config_validates_gqa_shapes() -> None:
    config = ModelConfig()

    assert config.hidden_size == config.num_attention_heads * config.head_dim
    assert config.num_attention_heads % config.num_key_value_heads == 0


def test_parameter_estimate_is_reasonable() -> None:
    config = ModelConfig()
    estimate = config.approx_parameter_count()

    assert estimate > 600_000_000
    assert estimate < 900_000_000


def test_tokenizer_config_rejects_chat_wrappers() -> None:
    with pytest.raises(ValueError):
        TokenizerConfig(special_tokens=("<|repo|>", "<|im_start|>"))


def test_default_training_stages_match_planned_budgets() -> None:
    stages = {stage.name: stage for stage in build_default_training_stages()}

    assert stages["prototype"].target_tokens == 1_500_000_000
    assert stages["stage_1_base"].target_tokens == 25_000_000_000
    assert stages["stage_2_code_cpt"].target_tokens == 20_000_000_000
    assert stages["stage_3_repair"].target_tokens == 5_000_000_000
    assert stages["stage_4_tool_policy"].target_tokens == 500_000_000
    assert stages["stage_2_code_cpt"].use_repo_formatting is True
    assert stages["stage_4_tool_policy"].use_tool_formatting is True


def test_load_tokenizer_config_from_artifact() -> None:
    config = load_tokenizer_config(ROOT / "configs" / "tokenizer" / "base_bpe.json")

    assert config.tokenizer_type == "bpe"
    assert config.byte_fallback is True
    assert "<|im_start|>" not in config.special_tokens


def test_load_training_stages_from_artifact() -> None:
    stages = load_training_stages(ROOT / "configs" / "train" / "stages.json")
    stages_by_name = {stage.name: stage for stage in stages}

    assert len(stages) == 5
    assert stages_by_name["stage_2_code_cpt"].use_repo_formatting is True
    assert stages_by_name["stage_4_tool_policy"].tool_ratio == 0.75


def test_load_model_config_from_artifact() -> None:
    config = load_model_config(ROOT / "configs" / "model" / "800m.json")

    assert isinstance(config, ModelConfig)
    assert config.num_hidden_layers == 24
    assert config.hidden_size == 1536
