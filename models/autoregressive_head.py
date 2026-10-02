# Autoregressive head (draft model): FC layer + LLaMA-style decoder layer(s)

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor):
        # (B, S) -> cos, sin of shape (B, 1, S, head_dim)
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None], emb.sin()[:, None]


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_mask: Optional[torch.Tensor], cache=None, layer_idx=0):
        B, S, _ = x.shape

        def split(t):  # (B, S, D) -> (B, heads, S, head_dim)
            return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)  # (B, heads, T, head_dim), T >= S
            if attn_mask is None and S > 1:
                # new queries sit at the end of the T cached keys
                attn_mask = torch.ones(S, k.shape[2], dtype=torch.bool, device=x.device).tril(k.shape[2] - S)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None and cache is None
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, eps):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps)
        self.attn = Attention(hidden_size, num_heads)
        self.mlp_norm = RMSNorm(hidden_size, eps)
        self.mlp = MLP(hidden_size, intermediate_size)

    def forward(self, x, cos, sin, attn_mask, cache=None, layer_idx=0):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask, cache, layer_idx)
        return x + self.mlp(self.mlp_norm(x))


class AutoregressiveHead(nn.Module):
    """
    Predicts the next feature from [feature_i ; embed(token_{i+1})].

        concat (B, S, F+E) --FC--> (B, S, H) --decoder layer(s)--> (B, S, H)

    The output lives in the target LLM's feature space (H == feature_dim), so
    `TargetLLM.compute_logits` turns it into a token distribution. Embeddings
    come from the target LLM and are not owned by this module.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int = 1,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size != feature_dim:
            raise ValueError(
                "decoder_hidden_size must equal feature_dim: the head predicts "
                "features that are fed to the target LM head"
            )

        self.fc = nn.Linear(feature_dim + embedding_dim, hidden_size)
        self.rotary = RotaryEmbedding(hidden_size // num_heads, rope_theta)
        self.layers = nn.ModuleList(
            DecoderLayer(hidden_size, num_heads, intermediate_size, rms_norm_eps)
            for _ in range(num_layers)
        )

    @classmethod
    def from_config(cls, cfg: dict):
        """Build from the `eagle` section of model_config.yaml."""
        e = cfg["eagle"]
        if e["concat_dim"] != e["feature_dim"] + e["embedding_dim"]:
            raise ValueError("concat_dim must equal feature_dim + embedding_dim")
        return cls(
            feature_dim=e["feature_dim"],
            embedding_dim=e["embedding_dim"],
            hidden_size=e["decoder_hidden_size"],
            num_heads=e["num_heads"],
            intermediate_size=e["intermediate_size"],
            num_layers=e["num_layers"],
            rope_theta=e["rope_theta"],
            rms_norm_eps=e["rms_norm_eps"],
        )

    def forward(
        self,
        features: torch.Tensor,
        shifted_embeddings: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cache=None,
    ) -> torch.Tensor:
        """
        Args:
            features:           (B, S, F)
            shifted_embeddings: (B, S, E) embeddings of tokens advanced by one step
            position_ids:       (B, S); defaults to cache.seq_len + 0..S-1
            attn_mask:          optional boolean/additive mask broadcastable to
                                (B, heads, S, T), e.g. a tree mask at inference,
                                where T = cache.seq_len + S with a cache, else S;
                                defaults to a causal mask
            cache:              optional KV cache (see inference/draft_tree.py);
                                exposes `seq_len` and `update(layer_idx, k, v)`.
                                New keys/values are appended to it.

        Returns:
            predicted next features (B, S, H)
        """
        x = torch.cat((features, shifted_embeddings), dim=-1)
        x = self.fc(x.to(self.fc.weight.dtype))

        if position_ids is None:
            start = cache.seq_len if cache is not None else 0
            position_ids = torch.arange(start, start + x.shape[1], device=x.device).expand(x.shape[0], -1)
        cos, sin = self.rotary(position_ids)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, attn_mask, cache, i)
        return x
