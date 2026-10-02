# Text generation with EAGLE, plus speedup measurement against vanilla decoding

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python scripts/run_generation.py`

from data.dataset import build_text
from inference.draft_tree import DraftTreeStructure
from inference.generate import EagleGenerator, vanilla_generate
from models import AutoregressiveHead, TargetLLM
from training.train import load_yaml, resolve_device

DEFAULT_PROMPTS = [
    "Explain speculative decoding in two sentences.",
    "Write a Python function that checks whether a number is prime.",
    "What are the main causes of the French Revolution?",
]


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed(fn, device: str):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, time.perf_counter() - start


def load_head(model_cfg: dict, checkpoint: str, device: str) -> AutoregressiveHead:
    head = AutoregressiveHead.from_config(model_cfg).to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["head"])
    return head.eval()


def read_prompts(args) -> list:
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]
    return args.prompt or DEFAULT_PROMPTS


def add_arguments(parser):
    parser.add_argument("--checkpoint", required=True, help="draft-head checkpoint from training")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--prompt", action="append", help="prompt text (repeatable)")
    parser.add_argument("--prompts-file", help="text file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", help="target LLM precision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-baseline", action="store_true", help="skip vanilla decoding (no speedup)")
    parser.add_argument("--show-output", action="store_true", help="print the generated text")


def run(args):

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model_cfg = load_yaml(args.model_config)

    target = TargetLLM.from_config(model_cfg, device=device, dtype=getattr(torch, args.dtype))
    head = load_head(model_cfg, args.checkpoint, device)
    generator = EagleGenerator(target, head, DraftTreeStructure.from_config(model_cfg))

    tokenizer = target.tokenizer
    eos = tokenizer.eos_token_id
    prompts = [
        tokenizer(build_text([("user", p)], tokenizer.eos_token)[0], return_tensors="pt").input_ids
        for p in read_prompts(args)
    ]

    def run_eagle(ids, n):
        return generator.generate(ids, n, args.temperature, eos)

    def run_baseline(ids, n):
        return vanilla_generate(target, ids, n, args.temperature, eos)

    # Warm-up so kernel compilation and allocator growth are not timed.
    run_eagle(prompts[0], 8)
    if not args.no_baseline:
        run_baseline(prompts[0], 8)

    totals = {"tokens": 0, "rounds": 0, "eagle_s": 0.0, "base_tokens": 0, "base_s": 0.0, "match": 0}
    for i, ids in enumerate(prompts):
        result, eagle_s = timed(lambda: run_eagle(ids, args.max_new_tokens), device)
        totals["tokens"] += result.tokens.shape[1]
        totals["rounds"] += result.num_rounds
        totals["eagle_s"] += eagle_s
        line = (
            f"[{i}] {result.tokens.shape[1]} tokens, "
            f"{result.mean_accepted_length:.2f} tokens/pass, "
            f"{result.tokens.shape[1] / eagle_s:.1f} tok/s (EAGLE)"
        )

        if not args.no_baseline:
            baseline, base_s = timed(lambda: run_baseline(ids, args.max_new_tokens), device)
            totals["base_tokens"] += baseline.shape[1]
            totals["base_s"] += base_s
            totals["match"] += torch.equal(baseline.cpu(), result.tokens.cpu())
            line += f", {baseline.shape[1] / base_s:.1f} tok/s (vanilla)"
        print(line)

        if args.show_output:
            print(tokenizer.decode(result.tokens[0], skip_special_tokens=True), "\n")

    print("\n=== Summary ===")
    print(f"mean accepted length : {totals['tokens'] / totals['rounds']:.2f} tokens per target pass")
    print(f"EAGLE throughput     : {totals['tokens'] / totals['eagle_s']:.1f} tok/s")
    if not args.no_baseline:
        eagle_rate = totals["tokens"] / totals["eagle_s"]
        base_rate = totals["base_tokens"] / totals["base_s"]
        print(f"vanilla throughput   : {base_rate:.1f} tok/s")
        print(f"speedup              : {eagle_rate / base_rate:.2f}x")
        if args.temperature == 0:
            print(f"identical to vanilla : {totals['match']}/{len(prompts)} prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())
