# EAGLE generation loop: draft a tree, verify it in one target pass, accept a prefix

from typing import List, NamedTuple, Optional

import torch
from transformers import DynamicCache

from inference.acceptance import accept_tree, sample_from_logits
from inference.draft_tree import DraftKVCache, DraftTreeStructure, draft_tree
from inference.tree_verify import keep_accepted, verify_tree
from models.autoregressive_head import AutoregressiveHead
from models.target_llm import TargetLLM


class GenerationResult(NamedTuple):
    tokens: torch.Tensor  # (1, T) generated tokens, prompt excluded
    num_rounds: int       # target-LLM verification passes used

    @property
    def mean_accepted_length(self) -> float:
        """Average number of tokens produced per target-LLM pass."""
        return self.tokens.shape[1] / max(self.num_rounds, 1)


def _is_done(tokens: List[int], max_new_tokens: int, eos_token_id: Optional[int]) -> bool:
    return len(tokens) >= max_new_tokens or (eos_token_id is not None and eos_token_id in tokens)


def _finish(tokens: List[int], max_new_tokens: int, eos_token_id: Optional[int]):
    """Truncate to the token budget and to the first EOS (kept)."""
    tokens = tokens[:max_new_tokens]
    if eos_token_id is not None and eos_token_id in tokens:
        tokens = tokens[: tokens.index(eos_token_id) + 1]
    return tokens


class EagleGenerator:
    """Batch-size-1 speculative decoding with a draft tree (EAGLE-1)."""

    def __init__(self, target: TargetLLM, head: AutoregressiveHead, tree: DraftTreeStructure):
        self.target, self.head, self.tree = target, head, tree

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
    ) -> GenerationResult:
        """
        Args:
            input_ids: (1, S) prompt tokens. The output follows the target LLM's
                       own distribution (greedy if temperature is 0).
        """
        target, head, tree = self.target, self.head, self.tree
        input_ids = input_ids.to(target.device)

        target_cache = DynamicCache()
        draft_cache = DraftKVCache.for_head(
            head, 1, input_ids.shape[1] + max_new_tokens + tree.num_nodes + tree.max_depth + 2
        )

        # Prefill: the target LLM reads the prompt and produces the first root token.
        features = target.get_features(input_ids, past_key_values=target_cache)
        root = sample_from_logits(target.compute_logits(features[0, -1]), temperature)
        next_tokens = torch.cat([input_ids[:, 1:], torch.tensor([[root]], device=input_ids.device)], dim=1)

        generated, rounds = [root], 0
        while not _is_done(generated, max_new_tokens, eos_token_id):
            node_tokens = draft_tree(head, target, draft_cache, features, next_tokens, tree)

            prefix_len = target_cache.get_seq_length()
            node_features, logits = verify_tree(target, target_cache, root, node_tokens, tree)
            accepted, root = accept_tree(logits, node_tokens, tree, temperature)
            keep_accepted(target_cache, prefix_len, accepted)

            new_tokens = [int(node_tokens[i]) for i in accepted] + [root]
            generated += new_tokens
            rounds += 1

            # The next round commits the root and the accepted nodes to the draft
            # cache, each paired with the token that followed it.
            rows = torch.tensor([0] + [i + 1 for i in accepted], device=node_features.device)
            features = node_features[rows][None]
            next_tokens = torch.tensor([new_tokens], device=features.device)

        tokens = torch.tensor([_finish(generated, max_new_tokens, eos_token_id)], device=input_ids.device)
        return GenerationResult(tokens, rounds)


@torch.no_grad()
def vanilla_generate(
    target: TargetLLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Standard one-token-per-pass decoding with a KV cache (speed baseline)."""
    input_ids = input_ids.to(target.device)
    cache = DynamicCache()
    features = target.get_features(input_ids, past_key_values=cache)

    generated = []
    while len(generated) < max_new_tokens:
        token = sample_from_logits(target.compute_logits(features[0, -1]), temperature)
        generated.append(token)
        if token == eos_token_id:
            break
        features = target.get_features(
            torch.tensor([[token]], device=input_ids.device), past_key_values=cache
        )
    return torch.tensor([generated], device=input_ids.device)
