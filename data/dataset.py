# ShareGPT conversations -> tokenized samples with an assistant-token loss mask

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Vicuna v1.1 conversation format (the format Vicuna-7B v1.5 was tuned on):
#   {SYSTEM} USER: {u1} ASSISTANT: {a1}</s>USER: {u2} ASSISTANT: {a2}</s>
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def load_conversations(path: str) -> list:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Dataset must be .json or .jsonl, got {path}")


def parse_turns(example: dict):
    """
    Return [(role, text), ...] strictly alternating user/assistant, or None if
    the conversation is malformed (unknown role, wrong order, empty turn).
    A trailing unanswered user turn is dropped.
    """
    turns = []
    for turn in example.get("conversations", []):
        role = ROLE_MAP.get(turn.get("from"))
        text = (turn.get("value") or "").strip()
        if role is None or not text:
            return None
        turns.append((role, text))

    if turns and turns[-1][0] == "user":
        turns.pop()
    expected = ("user", "assistant")
    if not turns or any(role != expected[i % 2] for i, (role, _) in enumerate(turns)):
        return None
    return turns


def build_text(turns, eos_token: str):
    """Render turns in Vicuna format; also return the character span of each assistant reply."""
    text, spans = SYSTEM_PROMPT, []
    for role, content in turns:
        if role == "user":
            sep = "" if text.endswith(eos_token) else " "
            text += f"{sep}USER: {content} ASSISTANT:"
        else:
            start = len(text)
            text += f" {content}{eos_token}"
            spans.append((start, len(text)))
    return text, spans


class ShareGPTDataset(Dataset):
    """
    Tokenizes every conversation once, up front.

    Each item holds:
        input_ids:  (S,) long
        loss_mask:  (S,) bool, True for tokens that belong to assistant replies
                    (including the closing EOS)

    Samples are not padded here; see `collate_fn`. Conversations that are
    malformed, or that have no assistant token left after truncation to
    `max_length`, are dropped.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for character offsets")

        self.samples = []
        conversations = load_conversations(data_path)

        for example in conversations:
            turns = parse_turns(example)
            if turns is None:
                continue
            sample = self._tokenize(turns, tokenizer, max_length)
            if sample is not None:
                self.samples.append(sample)

        print(f"ShareGPTDataset: kept {len(self.samples)}/{len(conversations)} conversations")

    @staticmethod
    def _tokenize(turns, tokenizer, max_length):
        text, spans = build_text(turns, tokenizer.eos_token)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )

        # A token is an assistant token if it starts inside an assistant span.
        loss_mask = [
            end > start and any(a <= start < b for a, b in spans)
            for start, end in enc["offset_mapping"]
        ]
        if len(loss_mask) < 2 or not any(loss_mask):
            return None

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(pad_token_id: int):
    """Right-pad a batch to its longest sequence."""

    def collate_fn(batch):
        max_len = max(len(s["input_ids"]) for s in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        loss_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

        for i, s in enumerate(batch):
            n = len(s["input_ids"])
            input_ids[i, :n] = s["input_ids"]
            attention_mask[i, :n] = 1
            loss_mask[i, :n] = s["loss_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }

    return collate_fn


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 1,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 2,
    distributed: bool = False,
) -> DataLoader:
    """With `distributed`, each process gets its own shard (call `sampler.set_epoch`)."""
    dataset = ShareGPTDataset(data_path, tokenizer, max_length)
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )
