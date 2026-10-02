# Draft-side KV cache for the autoregressive head (tree drafting builds on this)

import torch

from models.autoregressive_head import AutoregressiveHead


class DraftKVCache:
    """
    Preallocated key/value cache for the draft head, one buffer pair per layer.

    Keys are stored after RoPE, so entries are tied to the position_ids they
    were written with. Drafting appends speculative entries past the committed
    prefix; after verification call `crop(committed_len)` to drop them.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (num_layers, batch_size, num_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self._lengths = [0] * num_layers

    @classmethod
    def for_head(
        cls, head: AutoregressiveHead, batch_size: int, max_len: int
    ) -> "DraftKVCache":
        attn = head.layers[0].attn
        weight = head.fc.weight
        return cls(
            len(head.layers), batch_size, attn.num_heads, attn.head_dim,
            max_len, weight.device, weight.dtype,
        )

    @property
    def seq_len(self) -> int:
        return self._lengths[0]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append (B, heads, S, head_dim) and return all cached keys/values."""
        start = self._lengths[layer_idx]
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > max_len={self.max_len}")
        self.k[layer_idx, :, :, start:end] = k
        self.v[layer_idx, :, :, start:end] = v
        self._lengths[layer_idx] = end
        return self.k[layer_idx, :, :, :end], self.v[layer_idx, :, :, :end]

    def crop(self, length: int) -> None:
        """Discard entries at positions >= length (e.g. rejected draft tokens)."""
        if not 0 <= length <= self.seq_len:
            raise ValueError(f"cannot crop to {length}, seq_len={self.seq_len}")
        self._lengths = [length] * len(self._lengths)

    def reset(self) -> None:
        self.crop(0)
