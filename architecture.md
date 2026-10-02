# EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) — Architecture

Reference: Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", arXiv:2401.15077.

## Folder structure

```
Eagle/
├── configs/
├── data/
├── evaluation/
├── inference/
│   ├── draft_tree.py
│   ├── tree_verify.py
│   └── acceptance.py
├── models/
│   ├── target_llm.py
│   └── autoregressive_head.py
├── scripts/
│   ├── launch_training.sh
│   └── run_generation.py
├── tests/
├── training/
│   ├── loss.py
│   └── train.py
├── architecture.md
├── main.py
└── requirements.txt
```

## models/

### target_llm.py
- Imports (loads) the original target Large Language Model (LLM).
- Provides a function that runs the target LLM and returns:
  - the features (the decoder output after the final norm, i.e. the input of the LM head; the paper's "second-to-top layer" counts the LM head as the top layer, so this is not HF `hidden_states[-2]`),
  - the token embeddings (from the target LLM's embedding layer).
- These features and embeddings are consumed by `autoregressive_head.py`.

### autoregressive_head.py
- Contains the autoregressive head (the draft model).
- Takes the features and the embeddings of the shifted tokens (tokens advanced by one time step) from `target_llm.py`.
- Predicts the next feature vector, from which the output token distribution is obtained.
- Optionally takes a KV cache (`DraftKVCache` from `inference/draft_tree.py`) so drafting only processes new tokens; without one it behaves as before (training).

## Workflow (shared by training and inference)

1. The target LLM is run to obtain the features and the shifted tokens.
2. Control passes to the draft model (`autoregressive_head.py`), which predicts the next feature vector and then the output.

## training/

### loss.py
- Calculates the training loss from the draft model's predictions:
  - Regression loss: Smooth L1 loss between the predicted feature and the target LLM's feature.
  - Classification loss: cross-entropy of the draft model's predictions.
  - Objective: L = L_reg + w · L_cls

### train.py
- Performs the training updates (parameter updates of the autoregressive head) using the loss from `loss.py`.
- Optimizer: Adam.

## inference/

### draft_tree.py
- Takes the tokens predicted by the draft model and builds the draft tree of candidate tokens.
- Holds `DraftKVCache`, the draft head's key/value cache (preallocated per layer, keys stored post-RoPE):
  - `update(layer, k, v)` appends new entries and returns all cached keys/values.
  - `crop(n)` drops speculative entries beyond the committed prefix after verification.

### tree_verify.py
- Passes the draft tree back to the target LLM.
- Implements the tree mask so the target LLM scores all tree candidates.

### acceptance.py
- Verifies the candidates using the acceptance algorithm.
- Accepts the prefix tokens and returns the output.

## scripts/

### launch_training.sh
- Launches training across multiple GPUs.

### run_generation.py
- Runs text generation.
- Runs speedup measurements.