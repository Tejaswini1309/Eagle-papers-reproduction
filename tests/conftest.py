# Shared fixtures: a tiny random Llama target LLM and a matching draft head

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from inference.draft_tree import DraftTreeStructure
from models import AutoregressiveHead, TargetLLM

HIDDEN, VOCAB = 64, 100


@pytest.fixture(scope="session")
def target():
    """Tiny random target LLM (fp32); bypasses the pretrained-weights loader."""
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
    )
    llm = TargetLLM.__new__(TargetLLM)
    nn.Module.__init__(llm)
    llm.model = LlamaForCausalLM(config).eval()
    # Default init gives tiny embeddings; scale them up so tokens influence the draft head.
    with torch.no_grad():
        llm.model.get_input_embeddings().weight.mul_(30)
    llm.model.requires_grad_(False)
    return llm


@pytest.fixture()
def head():
    torch.manual_seed(1)
    return AutoregressiveHead(HIDDEN, HIDDEN, HIDDEN, num_heads=4, intermediate_size=128).eval()


@pytest.fixture(scope="session")
def tree():
    # Branches below non-first parents at every depth, so parent bookkeeping is exercised.
    return DraftTreeStructure(
        [[0], [1], [2],
         [0, 0], [0, 1], [1, 0], [1, 1], [2, 0],
         [0, 0, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0],
         [1, 0, 0, 0], [2, 0, 0, 0]]
    )
