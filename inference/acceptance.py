# Acceptance algorithm: pick the longest valid candidate path, then one new token

import torch

from inference.draft_tree import DraftTreeStructure


def sample_from_logits(logits: torch.Tensor, temperature: float) -> int:
    """Greedy for temperature 0, otherwise sample from softmax(logits / T)."""
    if temperature == 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1))


def _accept_child(row_logits, children, node_tokens, temperature):
    """
    Try the children of one node. Returns (child, None) if a child is accepted,
    else (None, token) with the token to emit instead.
    """
    if temperature == 0:
        token = int(row_logits.argmax())
        match = next((c for c in children if int(node_tokens[c]) == token), None)
        return match, token

    probs = torch.softmax(row_logits.float() / temperature, dim=-1)
    for c in children:
        token = int(node_tokens[c])
        if torch.rand(()).item() < probs[token].item():
            return c, None
        probs[token] = 0
        probs = probs / probs.sum()
    return None, int(torch.multinomial(probs, 1))


def accept_tree(
    logits: torch.Tensor,
    node_tokens: torch.Tensor,
    tree: DraftTreeStructure,
    temperature: float = 0.0,
):
    """
    Walk down the draft tree from the root, accepting at most one child per level.

    Args:
        logits:      (N + 1, V) target logits from `verify_tree`
        node_tokens: (N,) draft token of each node

    Greedy (temperature 0): a child is accepted if it equals the target's argmax.

    Sampling: children are tried in rank order, each treated as a point-mass
    proposal x. Accept with probability p(x); on rejection set p(x) = 0 and
    renormalise. If every child is rejected, sample from the remaining p. Each
    step leaves the output distribution equal to p, so the result is distributed
    exactly as sampling from the target LLM, for any choice of candidates.

    Returns:
        accepted_nodes: node indices along the accepted path, root first
        new_token:      the next token, drawn from the target LLM at the last
                        accepted node
    """
    accepted = []
    row = 0  # logits row of the current node (0 is the root)

    while True:
        match, new_token = _accept_child(
            logits[row], tree.children[row], node_tokens, temperature
        )
        if match is None:
            return accepted, new_token
        accepted.append(match)
        row = match + 1
