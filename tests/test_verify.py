# Tests for inference/tree_verify.py

import torch
from transformers import DynamicCache

from tests.conftest import VOCAB
from inference.draft_tree import DraftTreeStructure
from inference.tree_verify import build_tree_attention_mask, keep_accepted, verify_tree

PREFIX_LEN = 6
ROOT = 7


def path_to(tree, node):
    path = []
    while node != -1:
        path.append(node)
        node = tree.parents[node]
    return path[::-1]


def prefilled_cache(target):
    prompt = torch.randint(0, VOCAB, (1, PREFIX_LEN))
    cache = DynamicCache()
    target.get_features(prompt, past_key_values=cache)
    return prompt, cache


def test_tree_attention_mask_layout():
    tree = DraftTreeStructure([[0], [1], [0, 0]])
    mask = build_tree_attention_mask(tree, prefix_len=2, device=torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 6)
    expected_tree_part = torch.tensor(
        [[1, 0, 0, 0],   # root sees only itself
         [1, 1, 0, 0],   # node 0 sees root + itself
         [1, 0, 1, 0],   # node 1 does not see node 0
         [1, 1, 0, 1]],  # node 2 (child of node 0) sees root, node 0, itself
        dtype=torch.bool,
    )
    assert mask[0, 0, :, :2].all()  # whole prefix visible
    assert torch.equal(mask[0, 0, :, 2:], expected_tree_part)


def test_logits_match_running_each_path_alone(target, tree):
    prompt, cache = prefilled_cache(target)
    nodes = torch.randint(0, VOCAB, (tree.num_nodes,))
    _, logits = verify_tree(target, cache, ROOT, nodes, tree)

    assert logits.shape == (tree.num_nodes + 1, VOCAB)
    with torch.no_grad():
        root_only = torch.cat([prompt[0], torch.tensor([ROOT])])[None]
        assert torch.allclose(target.model(root_only).logits[0, -1], logits[0], atol=1e-4)
        for node in range(tree.num_nodes):
            path_tokens = [int(nodes[n]) for n in path_to(tree, node)]
            sequence = torch.cat([prompt[0], torch.tensor([ROOT] + path_tokens)])[None]
            assert torch.allclose(target.model(sequence).logits[0, -1], logits[node + 1], atol=1e-4)


def test_verify_extends_cache_by_root_plus_nodes(target, tree):
    _, cache = prefilled_cache(target)
    verify_tree(target, cache, ROOT, torch.randint(0, VOCAB, (tree.num_nodes,)), tree)
    assert cache.get_seq_length() == PREFIX_LEN + 1 + tree.num_nodes


def test_keep_accepted_leaves_the_cache_of_a_sequential_run(target, tree):
    prompt, cache = prefilled_cache(target)
    nodes = torch.randint(0, VOCAB, (tree.num_nodes,))
    verify_tree(target, cache, ROOT, nodes, tree)

    accepted = path_to(tree, tree.num_nodes - 1)  # deepest path, so not a contiguous prefix of the nodes
    keep_accepted(cache, PREFIX_LEN, accepted)
    assert cache.get_seq_length() == PREFIX_LEN + 1 + len(accepted)

    # Reference cache: the same tokens fed sequentially.
    tokens = [ROOT] + [int(nodes[n]) for n in accepted]
    reference = DynamicCache()
    target.get_features(prompt, past_key_values=reference)
    target.get_features(torch.tensor([tokens]), past_key_values=reference)

    for kept, ref in zip(cache.layers, reference.layers):
        assert torch.allclose(kept.keys, ref.keys, atol=1e-5)
        assert torch.allclose(kept.values, ref.values, atol=1e-5)

    # Decoding on top of the pruned cache continues exactly as the reference does.
    next_token = torch.tensor([[3]])
    kept_out = target.get_features(next_token, past_key_values=cache)
    ref_out = target.get_features(next_token, past_key_values=reference)
    assert torch.allclose(kept_out, ref_out, atol=1e-5)
