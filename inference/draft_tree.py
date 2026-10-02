# Draft tree: static tree structure, draft-side KV cache and tree drafting

import torch

from models.autoregressive_head import AutoregressiveHead
from models.target_llm import TargetLLM


class DraftTreeStructure:
    """
    Static draft-tree shape, given as paths of top-k ranks from the root.

    `[0]` is the best first draft token, `[0, 1]` the second-best token
    following it. The root (the token the target LLM has just produced) is
    implicit. Nodes are ordered breadth-first, so every depth is a contiguous
    block `levels[d - 1] = (start, end)`.
    """

    def __init__(self, choices):
        paths = sorted({tuple(c) for c in choices}, key=lambda p: (len(p), p))
        index = {p: i for i, p in enumerate(paths)}

        self.parents = []  # parent node index, -1 for children of the root
        for path in paths:
            parent = path[:-1]
            if parent and parent not in index:
                raise ValueError(f"tree path {list(path)} has no parent {list(parent)}")
            self.parents.append(index[parent] if parent else -1)

        self.num_nodes = len(paths)
        self.ranks = [path[-1] for path in paths]
        self.depths = [len(path) for path in paths]
        self.max_depth = max(self.depths)

        self.levels = []
        for d in range(1, self.max_depth + 1):
            ids = [i for i, depth in enumerate(self.depths) if depth == d]
            self.levels.append((ids[0], ids[-1] + 1))

        # children[i + 1] are the children of node i; children[0] those of the root.
        # Ordered by rank, i.e. by decreasing draft probability.
        self.children = [[] for _ in range(self.num_nodes + 1)]
        for i, parent in enumerate(self.parents):
            self.children[parent + 1].append(i)

        # ancestor_mask[i, j]: node j is node i or one of its ancestors
        self.ancestor_mask = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.bool)
        for i in range(self.num_nodes):
            j = i
            while j != -1:
                self.ancestor_mask[i, j] = True
                j = self.parents[j]

    @classmethod
    def from_config(cls, cfg: dict) -> "DraftTreeStructure":
        """Build from the `draft_tree` section of model_config.yaml."""
        return cls(cfg["draft_tree"]["choices"])


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


@torch.no_grad()
def draft_tree(
    head: AutoregressiveHead,
    target: TargetLLM,
    cache: DraftKVCache,
    features: torch.Tensor,
    next_tokens: torch.Tensor,
    tree: DraftTreeStructure,
) -> torch.Tensor:
    """
    Build the draft tree of candidate tokens, level by level.

    Args:
        features:    (1, M, H) target-LLM features of the M newly committed
                     positions (including the root token's position).
        next_tokens: (1, M) the token that follows each of those positions; the
                     last one is the root token produced by the target LLM.
        cache:       draft KV cache holding all earlier committed positions.

    The committed positions are written to `cache` and kept. Speculative tree
    entries are written and then removed again.

    Returns:
        (num_nodes,) candidate tokens in the tree's node order.
    """
    # 1. Commit the new positions. The last output estimates the root's feature.
    out = head(features, target.embed_tokens(next_tokens), cache=cache)
    committed = cache.seq_len
    parent_features = out[0, -1:]  # (P, H): features of the previous level's nodes

    tokens = torch.empty(tree.num_nodes, dtype=torch.long, device=features.device)
    ancestor_mask = tree.ancestor_mask.to(features.device)
    parents = torch.tensor(tree.parents, device=features.device)
    ranks = torch.tensor(tree.ranks, device=features.device)

    prev_start = 0
    for depth, (start, end) in enumerate(tree.levels, start=1):
        # 2. Children of each parent = its top-k tokens under the target LM head.
        k = int(ranks[start:end].max()) + 1
        topk = target.compute_logits(parent_features).topk(k, dim=-1).indices  # (P, k)
        parent_idx = (parents[start:end] - prev_start).clamp(min=0)  # root -> row 0
        tokens[start:end] = topk[parent_idx, ranks[start:end]]

        if depth == tree.max_depth:
            break

        # 3. Run the head on this level; each node attends to the committed
        #    prefix and to its own ancestors only.
        mask = torch.ones(1, 1, end - start, committed + end, dtype=torch.bool, device=features.device)
        mask[0, 0, :, committed:] = ancestor_mask[start:end, :end]
        positions = torch.full((1, end - start), committed + depth - 1, device=features.device)

        out = head(
            parent_features[parent_idx][None],
            target.embed_tokens(tokens[start:end])[None],
            position_ids=positions,
            attn_mask=mask,
            cache=cache,
        )
        parent_features, prev_start = out[0], start

    cache.crop(committed)
    return tokens
