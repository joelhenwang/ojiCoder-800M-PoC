import torch

from eight_hundred_m import EightHundredMForCausalLM, ModelConfig


def test_model_forward_shape() -> None:
    config = ModelConfig(
        vocab_size=512,
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    model = EightHundredMForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids)

    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.hidden_states.shape == (2, 8, config.hidden_size)
