# Tests for evaluation/evaluate_benchmarks.py

from types import SimpleNamespace

import pytest
import torch

from evaluation import evaluate_benchmarks as ev
from inference.generate import EagleGenerator


class StubTokenizer:
    """Deterministic character-level tokenizer that records every prompt it sees."""

    eos_token = "</s>"
    eos_token_id = 2

    def __init__(self):
        self.prompts = []

    def __call__(self, text, return_tensors=None):
        self.prompts.append(text)
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 100 for c in text]]))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class TestSummarize:
    def test_aggregates_over_turns(self):
        turns = [
            ev.TurnResult(tokens=10, rounds=4, eagle_seconds=1.0,
                          baseline_tokens=10, baseline_seconds=2.0, identical=True),
            ev.TurnResult(tokens=30, rounds=6, eagle_seconds=3.0,
                          baseline_tokens=30, baseline_seconds=6.0, identical=False),
        ]
        summary = ev.summarize(turns)
        assert summary["num_turns"] == 2
        assert summary["mean_accepted_length"] == pytest.approx(40 / 10)
        assert summary["eagle_tokens_per_s"] == pytest.approx(40 / 4)
        assert summary["vanilla_tokens_per_s"] == pytest.approx(40 / 8)
        assert summary["speedup"] == pytest.approx(2.0)
        assert summary["identical_to_vanilla"] == "1/2"

    def test_speedup_uses_each_methods_own_token_count(self):
        turns = [ev.TurnResult(tokens=20, rounds=5, eagle_seconds=2.0,
                               baseline_tokens=10, baseline_seconds=2.0)]
        assert ev.summarize(turns)["speedup"] == pytest.approx(10 / 5)

    def test_without_baseline_has_no_speedup_fields(self):
        summary = ev.summarize([ev.TurnResult(tokens=8, rounds=4, eagle_seconds=1.0)])
        assert summary["mean_accepted_length"] == pytest.approx(2.0)
        assert not {"speedup", "vanilla_tokens_per_s", "identical_to_vanilla"} & summary.keys()


def test_print_table_shows_every_benchmark(capsys):
    ev.print_table({
        "gsm8k": {"num_turns": 3, "mean_accepted_length": 3.456,
                  "eagle_tokens_per_s": 90.0, "vanilla_tokens_per_s": 30.0, "speedup": 3.0},
        "alpaca": {"num_turns": 2, "mean_accepted_length": 2.0, "eagle_tokens_per_s": 50.0},
    })
    out = capsys.readouterr().out
    assert "gsm8k" in out and "3.46" in out and "3.00x" in out
    alpaca_row = next(line for line in out.splitlines() if line.startswith("alpaca"))
    assert alpaca_row.count("-") == 2  # no vanilla throughput, no speedup


class TestLoaders:
    @staticmethod
    def fake_rows(monkeypatch, rows):
        monkeypatch.setattr(ev, "_load", lambda name, config, split: rows)

    def test_mt_bench_keeps_both_turns(self, monkeypatch):
        self.fake_rows(monkeypatch, [{"prompt": ["first", "second"]}])
        assert ev.load_mt_bench() == [["first", "second"]]

    def test_single_turn_benchmarks(self, monkeypatch):
        self.fake_rows(monkeypatch, [{"prompt": "def f():", "question": "2+2?"}])
        assert ev.load_humaneval() == [["def f():"]]
        assert ev.load_gsm8k() == [["2+2?"]]

    def test_alpaca_appends_input_only_when_present(self, monkeypatch):
        self.fake_rows(monkeypatch, [
            {"instruction": "Summarise.", "input": "Some text."},
            {"instruction": "Say hi.", "input": ""},
        ])
        assert ev.load_alpaca() == [["Summarise.\n\nSome text."], ["Say hi."]]


class TestRunConversation:
    @staticmethod
    def make_generator(target, head, tree, monkeypatch):
        tokenizer = StubTokenizer()
        monkeypatch.setattr(target, "tokenizer", tokenizer, raising=False)
        return EagleGenerator(target, head, tree), tokenizer

    def test_greedy_matches_vanilla_and_later_turns_see_earlier_answers(
        self, target, head, tree, monkeypatch
    ):
        generator, tokenizer = self.make_generator(target, head, tree, monkeypatch)
        results = ev.run_conversation(
            ["first question", "second question"], generator, "cpu", max_new_tokens=6, temperature=0.0
        )

        assert len(results) == 2
        for result in results:
            assert 1 <= result.tokens <= 6 and result.rounds >= 1
            assert result.identical is True
            assert result.baseline_tokens == result.tokens
            assert result.eagle_seconds > 0 and result.baseline_seconds > 0

        # The second prompt holds an assistant answer (closed by EOS) between the two questions.
        second_prompt = tokenizer.prompts[1]
        assert "first question ASSISTANT: " in second_prompt
        assert second_prompt.endswith("</s>USER: second question ASSISTANT:")

    def test_baseline_can_be_skipped(self, target, head, tree, monkeypatch):
        generator, _ = self.make_generator(target, head, tree, monkeypatch)
        (result,) = ev.run_conversation(
            ["question"], generator, "cpu", max_new_tokens=4, temperature=0.0, baseline=False
        )
        assert result.baseline_seconds is None and result.identical is None
