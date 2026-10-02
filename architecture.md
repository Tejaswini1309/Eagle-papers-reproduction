# EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) — Architecture

Reference: Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", arXiv:2401.15077.

## Folder structure

```
Eagle/
├── configs/
├── data/
│   └── dataset.py
├── evaluation/
│   ├── __init__.py
│   └── evaluate_benchmarks.py
├── inference/
│   ├── draft_tree.py
│   ├── tree_verify.py
│   ├── acceptance.py
│   └── generate.py
├── models/
│   ├── target_llm.py
│   └── autoregressive_head.py
├── scripts/
│   ├── launch_training.sh
│   └── run_generation.py
├── tests/
│   ├──conftest.py
│   ├──test_draft.py
│   ├──test_evaluate.py
│   └──test_verify.py
├── training/
│   ├── loss.py
│   └── train.py
├── architecture.md
├── main.py
└── requirements.txt
```

## data/

### dataset.py
- Loads ShareGPT-style conversations (`.json` / `.jsonl`) and tokenizes them once with the target LLM's tokenizer.
- Formats each conversation in the Vicuna v1.1 template (`SYSTEM USER: ... ASSISTANT: ...</s>`).
- Drops malformed conversations (unknown role, wrong turn order, empty turn) and conversations with no assistant token left after truncation to `max_length`.
- Each sample holds:
  - `input_ids`,
  - `loss_mask` (True on assistant tokens, including the closing end-of-sequence token).
- The collate function right-pads each batch to its longest sequence and adds `attention_mask`.
- Contains no target-LLM features: these are computed online by `models/target_llm.py` during training.

## models/

### target_llm.py
- Imports (loads) the original target Large Language Model (LLM).
- Provides a function that runs the target LLM (optionally with a KV cache and a tree mask for inference) and returns:
  - the features (second-to-top-layer hidden states, before the LM head),
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
- Restricts the loss to assistant tokens via `loss_mask` (configurable).
- Optimizer: Adam.
- Runs on one device, or data-parallel across GPUs when started with `torchrun`.

## inference/

### draft_tree.py
- `DraftTreeStructure`: the static tree shape, read from `draft_tree.choices` in `configs/model_config.yaml` (paths of top-k ranks from the root).
- `draft_tree(...)`: builds the draft tree of candidate tokens level by level.
  - Commits the newly accepted positions to the draft cache; the last output is the root's estimated feature.
  - Each node's children are the top-k tokens of the target LM head applied to the node's predicted feature.
  - Each level runs the head once, with a mask so a node sees only the committed prefix and its ancestors.
- Holds `DraftKVCache`, the draft head's key/value cache (preallocated per layer, keys stored post-RoPE):
  - `update(layer, k, v)` appends new entries and returns all cached keys/values.
  - `crop(n)` drops speculative entries beyond the committed prefix after verification.

### tree_verify.py
- Passes the draft tree back to the target LLM.
- Implements the tree mask so the target LLM scores all tree candidates in a single pass: each node sees the prefix, the root and its own ancestors.
- After acceptance, `keep_accepted` keeps only the prefix, the root and the accepted nodes in the target KV cache.

### acceptance.py
- Verifies the candidates using the acceptance algorithm.
  - Temperature 0: a child is accepted if it equals the target's argmax.
  - Temperature > 0: multi-round speculative sampling; each child is a point-mass proposal, so the output follows the target distribution exactly.
- Accepts the prefix tokens and returns the output, together with one new token drawn from the target LLM.

### generate.py
- `EagleGenerator`: the generation loop (batch size 1). Each round drafts a tree, verifies it in one target pass, accepts a prefix, and updates both KV caches.
- `vanilla_generate`: one-token-per-pass baseline, used for speedup measurement and for checking that the output is unchanged.

## evaluation/

### __init__.py
- Marks `evaluation/` as a Python package.

### evaluate_benchmarks.py
- Evaluation scripts for MT-bench, HumanEval, GSM8K, and Alpaca (loaded from the Hugging Face Hub with `datasets`).
- Each conversation is answered by EAGLE and by vanilla decoding on identical prompts; MT-bench's second turn reuses EAGLE's first answer as history for both.
- Reports per benchmark:
  - mean acceptance length (tokens produced per target-LLM pass),
  - EAGLE and vanilla throughput in tokens per second,
  - speedup ratio (EAGLE throughput / vanilla throughput),
  - at temperature 0, how many answers are identical to vanilla decoding.
- Run with `python main.py evaluate --checkpoint ...` or `python -m evaluation.evaluate_benchmarks`; `--output` saves per-turn results as JSON.

## scripts/

### launch_training.sh
- Launches training across multiple GPUs with `torchrun` (one process per GPU, data parallel).
- Each process keeps its own frozen target LLM and a replica of the head; gradients are synchronised with `DistributedDataParallel`, and only rank 0 logs and saves checkpoints.
- `NUM_GPUS` selects how many GPUs to use; extra arguments are passed to `training/train.py`.

### run_generation.py
- Runs text generation.
- Runs speedup measurements.

## main.py
- Command-line interface (CLI) entry point for:
  - training (`python main.py train`, single device; use `scripts/launch_training.sh` for multiple GPUs),
  - text generation and speedup measurement (`python main.py generate`, same options as `scripts/run_generation.py`),
  - benchmark evaluation (`python main.py evaluate`).

## tests/

Run with `python -m pytest tests`.

### conftest.py
- Shared fixtures: a tiny random target LLM (fp32), a matching draft head and a small draft tree.

### test_draft.py
- Tests `inference/draft_tree.py`.
  - Tree structure: node order, parents, children and the ancestor mask.
  - `DraftKVCache`: append, crop, overflow, reset.
  - `draft_tree`: every node equals the top-k token obtained by running the head from scratch on that node's path; the cache keeps only committed positions.

### test_verify.py
- Tests `inference/tree_verify.py`.
  - The tree mask layout.
  - The target logits of every node equal those of running its root-to-node path alone.
  - After `keep_accepted`, the target KV cache equals that of a sequential run and decoding continues identically.

### test_evaluate.py
- Tests `evaluation/evaluate_benchmarks.py` without network access or real models.
  - `summarize`: acceptance length, throughputs, speedup and the identical-output count.
  - `print_table`: output with and without a baseline.
  - Benchmark loaders: turn handling and the Alpaca instruction/input join, with the dataset download replaced by fake rows.
  - `run_conversation`: greedy output equals vanilla decoding, and later turns see earlier answers (stub tokenizer, tiny target).