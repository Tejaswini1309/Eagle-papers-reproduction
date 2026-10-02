# Tests for inference/draft_tree.py

import pytest
import torch

from tests.conftest import HIDDEN, VOCAB
from inference.draft_tree import DraftKVCache, DraftTreeStructure, draft_tree
from models import AutoregressiveHead


class TestDraftTreeStructure:
    def test_breadth_first_order_and_levels(self):
        tree = DraftTreeStructure([[0, 0], [1], [0], [0, 1]])
        assert tree.num_nodes == 4 and tree.max_depth == 2
        assert tree.depths == [1, 1, 2, 2]
        assert tree.levels == [(0, 2), (2, 4)]

    def test_parents_ranks_and_children(self):
        tree = DraftTreeStructure([[0], [1], [0, 0], [0, 1]])
        assert tree.parents == [-1, -1, 0, 0]
        assert tree.ranks == [0, 1, 0, 1]
        assert tree.children[0] == [0, 1]  # children of the root, by rank
        assert tree.children[1] == [2, 3]  # children of node 0
        assert tree.children[2] == []

    def test_ancestor_mask_is_ancestors_and_self_only(self):
        tree = DraftTreeStructure([[0], [1], [0, 0]])
        expected = torch.tensor(
            [[1, 0, 0],
             [0, 1, 0],
             [1, 0, 1]], dtype=torch.bool
        )
        assert torch.equal(tree.ancestor_mask, expected)

    def test_missing_parent_is_rejected(self):
        with pytest.raises(ValueError):
            DraftTreeStructure([[0], [1, 0]])

    def test_from_config(self):
        tree = DraftTreeStructure.from_config({"draft_tree": {"choices": [[0], [0, 0]]}})
        assert tree.num_nodes == 2


class TestDraftKVCache:
    @staticmethod
    def make_cache(max_len=8):
        return DraftKVCache(2, 1, 4, 16, max_len, torch.device("cpu"), torch.float32)

    def test_update_appends_and_returns_everything(self):
        cache = self.make_cache()
        k = torch.randn(1, 4, 3, 16)
        keys, values = cache.update(0, k, k)
        assert keys.shape == (1, 4, 3, 16) and torch.equal(keys, k)
        keys, _ = cache.update(0, k[:, :, :2], k[:, :, :2])
        assert keys.shape[2] == 5

    def test_crop_discards_later_entries(self):
        cache = self.make_cache()
        for layer in range(2):
            cache.update(layer, torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16))
        cache.crop(2)
        assert cache.seq_len == 2
        keys, _ = cache.update(0, torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        assert keys.shape[2] == 3

    def test_overflow_and_bad_crop_raise(self):
        cache = self.make_cache(max_len=4)
        with pytest.raises(ValueError):
            cache.update(0, torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16))
        with pytest.raises(ValueError):
            cache.crop(1)

    def test_reset(self):
        cache = self.make_cache()
        cache.update(0, torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16))
        cache.reset()
        assert cache.seq_len == 0


class TestDraftTree:
    @staticmethod
    def committed_inputs(target, length=6):
        ids = torch.randint(0, VOCAB, (1, length))
        features = target.get_features(ids)
        next_tokens = torch.randint(0, VOCAB, (1, length))  # last entry is the root token
        return features, next_tokens

    def test_cache_keeps_only_committed_positions(self, target, head, tree):
        features, next_tokens = self.committed_inputs(target)
        cache = DraftKVCache.for_head(head, 1, 64)
        tokens = draft_tree(head, target, cache, features, next_tokens, tree)
        assert tokens.shape == (tree.num_nodes,)
        assert cache.seq_len == features.shape[1]

    def test_second_round_extends_committed_prefix(self, target, head, tree):
        features, next_tokens = self.committed_inputs(target)
        cache = DraftKVCache.for_head(head, 1, 64)
        draft_tree(head, target, cache, features, next_tokens, tree)
        draft_tree(head, target, cache, features[:, :2], next_tokens[:, :2], tree)
        assert cache.seq_len == features.shape[1] + 2

    @pytest.mark.parametrize("seed", range(10))
    def test_every_node_is_the_topk_token_of_its_own_path(self, target, tree, seed):
        """
        Reference: run the head from scratch (no cache, plain causal mask) on the
        committed prefix followed by the node's ancestors. Each node must be the
        top-k token, at the node's rank, of the LM head on that path's last feature.
        Tokens are discrete, so several random heads and prompts are used to make
        a wrong mask or position change at least one of them.
        """
        torch.manual_seed(seed)
        head = AutoregressiveHead(HIDDEN, HIDDEN, HIDDEN, num_heads=4, intermediate_size=128).eval()
        features, next_tokens = self.committed_inputs(target)
        cache = DraftKVCache.for_head(head, 1, 64)
        tokens = draft_tree(head, target, cache, features, next_tokens, tree)

        def path_to(node):
            path = []
            while node != -1:
                path.append(node)
                node = tree.parents[node]
            return path[::-1]

        with torch.no_grad():
            for node in range(tree.num_nodes):
                feats, embeds = features, target.embed_tokens(next_tokens)
                out = head(feats, embeds)[:, -1:]  # estimated feature of the root
                for ancestor in path_to(node)[:-1]:
                    feats = torch.cat([feats, out], dim=1)
                    embeds = torch.cat([embeds, target.embed_tokens(tokens[ancestor].view(1, 1))], dim=1)
                    out = head(feats, embeds)[:, -1:]
                k = tree.ranks[node] + 1
                expected = target.compute_logits(out[0]).topk(k).indices[0, tree.ranks[node]]
                assert tokens[node] == expected, f"node {node}"
