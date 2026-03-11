from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class LayerScale(nn.Module):
    def __init__(self, hidden_size: int, init_value: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((hidden_size,), init_value))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: Tensor, num_groups: int) -> Tensor:
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, num_groups, seq_len, head_dim)
    return hidden_states.reshape(batch, kv_heads * num_groups, seq_len, head_dim)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.use_qk_norm = config.use_qk_norm
        self.use_per_head_gating = config.use_per_head_gating
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if self.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if self.use_qk_norm else nn.Identity()
        self.head_gate = (
            nn.Parameter(torch.ones(self.num_heads)) if self.use_per_head_gating else None
        )
        self.dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
        value_residual: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        if value_residual is not None:
            v = v + value_residual

        k = repeat_kv(k, self.num_groups)
        v = repeat_kv(v, self.num_groups)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None,
            scale=self.scaling,
        )

        if self.head_gate is not None:
            attn_output = attn_output * self.head_gate.view(1, self.num_heads, 1, 1)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.o_proj(attn_output), v[:, :: self.num_groups, :, :].contiguous()


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.attention_scale = LayerScale(config.hidden_size, config.layer_scale_init)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)
        self.mlp_scale = LayerScale(config.hidden_size, config.layer_scale_init)
        self.use_value_residual = config.use_value_residual

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
        value_residual: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        residual = hidden_states
        attn_input = self.input_norm(hidden_states)
        attn_output, next_value_residual = self.attention(
            attn_input,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            value_residual=value_residual if self.use_value_residual else None,
        )
        hidden_states = residual + self.attention_scale(attn_output)

        residual = hidden_states
        mlp_input = self.post_attn_norm(hidden_states)
        mlp_output = self.mlp(mlp_input)
        hidden_states = residual + self.mlp_scale(mlp_output)
        return hidden_states, next_value_residual if self.use_value_residual else None


@dataclass(slots=True)
class CausalLMOutput:
    logits: Tensor
    hidden_states: Tensor


class EightHundredMForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=self.config.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        if attention_mask is not None:
            if attention_mask.ndim != 4:
                raise ValueError("attention_mask must have shape [batch, heads|1, query, key]")
            causal_mask = attention_mask
        else:
            causal_mask = None

        value_residual = None
        for layer in self.layers:
            hidden_states, value_residual = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=causal_mask,
                value_residual=value_residual,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits, hidden_states=hidden_states)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
