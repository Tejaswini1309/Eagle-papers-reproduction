# Tree verification: score every draft-tree candidate in one target-LLM pass

import torch
from transformers import DynamicCache

from inference.draft_tree import DraftTreeStructure
from models.target_llm import TargetLLM


def build_tree_attention_mask(
    tree: DraftTreeStructure, prefix_len: int, device: torch.device
) -> torch.Tensor:
    """
    Boolean mask of shape (1, 1, N + 1, prefix_len + N + 1) for the sequence
    [root, node_0, ..., node_{N-1}] appended after the cached prefix.

    Every row sees the whole prefix and the root, plus its own ancestors and
    itself, so each candidate is scored as if it were generated alone.
    """
    n = tree.num_nodes + 1
    mask = torch.zeros(1, 1, n, prefix_len + n, dtype=torch.bool, device=device)
    mask[..., :prefix_len] = True
    mask[..., prefix_len] = True  # root
    mask[0, 0, torch.arange(n, device=device), prefix_len + torch.arange(n, device=device)] = True
    mask[0, 0, 1:, prefix_len + 1:] = tree.ancestor_mask.to(device)
    return mask


@torch.no_grad()
def verify_tree(
    target: TargetLLM,
    cache: DynamicCache,
    root_token: int,
    node_tokens: torch.Tensor,
    tree: DraftTreeStructure,
):
    """
    Run the target LLM once over [root, tree nodes] with the tree mask.

    `cache` holds the committed prefix and is extended in place by N + 1
    entries; call `keep_accepted` afterwards to drop the rejected ones.

    Returns:
        features: (N + 1, H) target features; row 0 is the root, row i + 1 node i
        logits:   (N + 1, V) target logits; row r gives the distribution over
                  the token that follows row r's token
    """
    prefix_len = cache.get_seq_length()
    device = node_tokens.device

    input_ids = torch.cat([torch.tensor([root_token], device=device), node_tokens])[None]
    depths = torch.tensor([0] + tree.depths, device=device)
    position_ids = (prefix_len + depths)[None]

    features = target.get_features(
        input_ids,
        attention_mask=build_tree_attention_mask(tree, prefix_len, device),
        position_ids=position_ids,
        past_key_values=cache,
    )[0]
    return features, target.compute_logits(features)


def keep_accepted(cache: DynamicCache, prefix_len: int, accepted_nodes: list) -> None:
    """
    Keep the prefix, the root and the accepted nodes in the target KV cache.

    Accepted nodes form a root-to-node path, so each already sits at the
    position it will have in the final sequence and its keys can be reused.
    """
    keep = list(range(prefix_len + 1)) + [prefix_len + 1 + i for i in accepted_nodes]
    for layer in cache.layers:
        index = torch.tensor(keep, device=layer.keys.device)
        layer.keys = layer.keys[:, :, index]
        layer.values = layer.values[:, :, index]
