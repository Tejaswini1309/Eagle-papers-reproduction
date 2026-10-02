# EAGLE-1 command-line entry point: `python main.py train|generate|evaluate [options]`

import argparse

from evaluation import evaluate_benchmarks
from scripts import run_generation
from training import train


def main():
    parser = argparse.ArgumentParser(description="EAGLE-1 speculative decoding")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser(
        "train",
        help="train the draft head (use scripts/launch_training.sh for multiple GPUs)",
    )
    train.add_arguments(train_parser)
    train_parser.set_defaults(func=train.run)

    generate_parser = commands.add_parser(
        "generate", help="generate text and measure the speedup over vanilla decoding"
    )
    run_generation.add_arguments(generate_parser)
    generate_parser.set_defaults(func=run_generation.run)

    evaluate_parser = commands.add_parser(
        "evaluate", help="measure speedup and acceptance length on MT-bench, HumanEval, GSM8K, Alpaca"
    )
    evaluate_benchmarks.add_arguments(evaluate_parser)
    evaluate_parser.set_defaults(func=evaluate_benchmarks.run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
