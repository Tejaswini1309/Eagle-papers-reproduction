# Benchmark evaluation on MT-bench, HumanEval, GSM8K and Alpaca:
# speedup over vanilla decoding and average acceptance length

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch

from data.dataset import build_text
from inference.draft_tree import DraftTreeStructure
from inference.generate import EagleGenerator, vanilla_generate
from models import TargetLLM
from scripts.run_generation import load_head, timed
from training.train import load_yaml, resolve_device

# Benchmarks: each loader returns a list of conversations, each a list of user turns

def _load(name: str, config: Optional[str], split: str):
    from datasets import load_dataset

    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def load_mt_bench() -> List[List[str]]:
    return [list(row["prompt"]) for row in _load("HuggingFaceH4/mt_bench_prompts", None, "train")]


def load_humaneval() -> List[List[str]]:
    return [[row["prompt"]] for row in _load("openai/openai_humaneval", None, "test")]


def load_gsm8k() -> List[List[str]]:
    return [[row["question"]] for row in _load("openai/gsm8k", "main", "test")]


def load_alpaca() -> List[List[str]]:
    rows = _load("tatsu-lab/alpaca", None, "train")
    return [
        [row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")] for row in rows
    ]


BENCHMARKS = {
    "mt_bench": load_mt_bench,
    "humaneval": load_humaneval,
    "gsm8k": load_gsm8k,
    "alpaca": load_alpaca,
}

# Measurement
@dataclass
class TurnResult:
    tokens: int
    rounds: int
    eagle_seconds: float
    baseline_tokens: Optional[int] = None
    baseline_seconds: Optional[float] = None
    identical: Optional[bool] = None


def run_conversation(
    user_turns, generator, device, max_new_tokens, temperature, baseline=True
) -> List[TurnResult]:
    """
    Answer each user turn with EAGLE and (optionally) vanilla decoding.
    Later turns use EAGLE's earlier answers as history for both methods, so
    the two always decode the same prompt.
    """
    tokenizer, target = generator.target.tokenizer, generator.target
    eos = tokenizer.eos_token_id
    history, results = [], []

    for user in user_turns:
        history.append(("user", user))
        prompt = tokenizer(build_text(history, tokenizer.eos_token)[0], return_tensors="pt").input_ids

        eagle, eagle_s = timed(
            lambda: generator.generate(prompt, max_new_tokens, temperature, eos), device
        )
        result = TurnResult(eagle.tokens.shape[1], eagle.num_rounds, eagle_s)

        if baseline:
            baseline_out, base_s = timed(
                lambda: vanilla_generate(target, prompt, max_new_tokens, temperature, eos),
                device,
            )
            result.baseline_tokens, result.baseline_seconds = baseline_out.shape[1], base_s
            if temperature == 0:
                result.identical = torch.equal(baseline_out.cpu(), eagle.tokens.cpu())

        results.append(result)
        history.append(("assistant", tokenizer.decode(eagle.tokens[0], skip_special_tokens=True).strip()))
    return results


def summarize(turns: List[TurnResult]) -> dict:
    """
    mean_accepted_length: tokens produced per target-LLM pass (tau).
    speedup: EAGLE tokens/s divided by vanilla tokens/s, over all turns.
    """
    tokens = sum(t.tokens for t in turns)
    summary = {
        "num_turns": len(turns),
        "mean_accepted_length": tokens / max(sum(t.rounds for t in turns), 1),
        "eagle_tokens_per_s": tokens / sum(t.eagle_seconds for t in turns),
    }
    if turns[0].baseline_seconds is not None:
        base_rate = sum(t.baseline_tokens for t in turns) / sum(t.baseline_seconds for t in turns)
        summary["vanilla_tokens_per_s"] = base_rate
        summary["speedup"] = summary["eagle_tokens_per_s"] / base_rate
    if turns[0].identical is not None:
        summary["identical_to_vanilla"] = f"{sum(t.identical for t in turns)}/{len(turns)}"
    return summary


def print_table(summaries: dict) -> None:
    print(f"\n{'benchmark':<11}{'turns':>7}{'tau':>8}{'EAGLE tok/s':>13}{'vanilla tok/s':>15}{'speedup':>9}")
    for name, s in summaries.items():
        vanilla = f"{s['vanilla_tokens_per_s']:.1f}" if "vanilla_tokens_per_s" in s else "-"
        speedup = f"{s['speedup']:.2f}x" if "speedup" in s else "-"
        print(
            f"{name:<11}{s['num_turns']:>7}{s['mean_accepted_length']:>8.2f}"
            f"{s['eagle_tokens_per_s']:>13.1f}{vanilla:>15}{speedup:>9}"
        )


# Command line
def add_arguments(parser):
    parser.add_argument("--checkpoint", required=True, help="draft-head checkpoint from training")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--num-samples", type=int, default=80, help="conversations per benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", help="target LLM precision")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-baseline", action="store_true", help="skip vanilla decoding (no speedup)")
    parser.add_argument("--output", help="write per-turn results and summaries to this JSON file")


def run(args):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model_cfg = load_yaml(args.model_config)

    target = TargetLLM.from_config(model_cfg, device=device, dtype=getattr(torch, args.dtype))
    head = load_head(model_cfg, args.checkpoint, device)
    generator = EagleGenerator(target, head, DraftTreeStructure.from_config(model_cfg))

    summaries, records = {}, {}
    baseline = not args.no_baseline
    warmed_up = False
    for name in args.benchmarks:
        conversations = BENCHMARKS[name]()[: args.num_samples]
        if not warmed_up:  # keep kernel compilation and allocator growth out of the timings
            run_conversation(conversations[0][:1], generator, device, 8, args.temperature, baseline)
            warmed_up = True

        turns = []
        for i, conversation in enumerate(conversations, start=1):
            turns += run_conversation(
                conversation, generator, device, args.max_new_tokens, args.temperature, baseline
            )
            print(f"\r{name}: {i}/{len(conversations)}", end="", flush=True)
        print()
        summaries[name], records[name] = summarize(turns), [asdict(t) for t in turns]

    print_table(summaries)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            settings = {k: v for k, v in vars(args).items() if k != "func"}  # `func` is set by main.py
            json.dump({"args": settings, "summary": summaries, "turns": records}, f, indent=2)


if __name__ == "__main__":  # python -m evaluation.evaluate_benchmarks
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())
