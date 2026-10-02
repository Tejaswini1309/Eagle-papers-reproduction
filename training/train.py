# Training loop for the autoregressive head (Adam, gradient clipping, checkpointing)

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from models import AutoregressiveHead, TargetLLM
from training.loss import EagleLoss


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_distributed(device_name: str):
    """
    Return (rank, world_size, device). Under `torchrun` this joins the process
    group and binds the process to its GPU; otherwise it runs on one device.
    """
    if "RANK" not in os.environ:
        return 0, 1, resolve_device(device_name)

    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    device = f"cuda:{local_rank}" if use_cuda else "cpu"
    return dist.get_rank(), dist.get_world_size(), device


def save_checkpoint(path, head, optimizer, epoch, step):
    # Only the head is saved; the frozen target LLM is reloaded from its source.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        path,
    )


def compute_losses(target, head, criterion, batch, feature_noise=0.0, assistant_only=True):
    """One forward pass: target LLM -> draft head -> loss. Returns the loss dict."""
    input_ids = batch["input_ids"].to(target.device)
    attention_mask = batch["attention_mask"].to(target.device)

    inputs = target.extract(input_ids, attention_mask)

    # Position i predicts the feature of token i+1, so shift the token mask too.
    mask = inputs.mask
    if assistant_only:
        mask = mask & batch["loss_mask"][:, 1:].to(target.device)

    features = inputs.features
    if feature_noise > 0:
        features = features + torch.empty_like(features).uniform_(-feature_noise, feature_noise)

    predicted_features = head(features, inputs.shifted_embeddings)

    with torch.no_grad():
        target_logits = target.compute_logits(inputs.target_features)
    predicted_logits = target.compute_logits(predicted_features)

    return criterion(
        predicted_features,
        inputs.target_features,
        predicted_logits,
        target_logits,
        mask,
    )


def train(model_cfg: dict, train_cfg: dict):
    run, opt_cfg = train_cfg["runtime"], train_cfg["optimization"]

    rank, world_size, device = setup_distributed(run["device"])
    torch.manual_seed(run["seed"] + rank)  # distinct noise per process; DDP syncs head init
    is_main = rank == 0

    target = TargetLLM.from_config(
        model_cfg, device=device, dtype=getattr(torch, run["dtype"])
    )
    head = AutoregressiveHead.from_config(model_cfg).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    if is_main:
        print(f"Draft head: {n_params / 1e6:.1f}M trainable parameters, {world_size} process(es)")

    # Only the head is trained, so only it is replicated and synchronised;
    # every process holds its own frozen copy of the target LLM.
    model = head
    if world_size > 1:
        model = DistributedDataParallel(head, device_ids=[device] if device != "cpu" else None)

    # Expected to yield dicts with `input_ids`, `attention_mask` and `loss_mask`, shape (B, S).
    from data.dataset import create_dataloader

    dataloader = create_dataloader(
        data_path=train_cfg["data"]["path"],
        tokenizer=target.tokenizer,
        batch_size=opt_cfg["batch_size"],
        max_length=train_cfg["data"]["max_length"],
        shuffle=True,
        num_workers=train_cfg["data"]["num_workers"],
        distributed=world_size > 1,
    )

    criterion = EagleLoss(w_cls=train_cfg["loss"]["w_cls"])
    optimizer = torch.optim.Adam(
        head.parameters(), lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"])
    )

    accum = opt_cfg["grad_accum_steps"]
    noise = train_cfg["augmentation"]["feature_noise"]
    assistant_only = train_cfg["loss"]["assistant_only"]
    step = 0

    for epoch in range(opt_cfg["epochs"]):
        model.train()
        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{opt_cfg['epochs']}", disable=not is_main)
        for i, batch in enumerate(progress):
            sync_step = (i + 1) % accum == 0
            # Skip gradient all-reduce on accumulation steps that do not update.
            with nullcontext() if sync_step or world_size == 1 else model.no_sync():
                losses = compute_losses(target, model, criterion, batch, noise, assistant_only)
                (losses["loss"] / accum).backward()

            if not sync_step:
                continue

            torch.nn.utils.clip_grad_norm_(head.parameters(), opt_cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % run["log_every"] == 0:
                print(
                    f"step {step} loss {losses['loss'].item():.4f} "
                    f"reg {losses['reg_loss'].item():.4f} "
                    f"cls {losses['cls_loss'].item():.4f} "
                    f"top1 {losses['top1_acc'].item():.3f}"
                )
            if is_main and step % run["save_every"] == 0:
                save_checkpoint(
                    os.path.join(run["output_dir"], f"step_{step}.pt"),
                    head, optimizer, epoch, step,
                )

        if is_main:
            save_checkpoint(
                os.path.join(run["output_dir"], f"epoch_{epoch + 1}.pt"),
                head, optimizer, epoch, step,
            )

    if world_size > 1:
        dist.destroy_process_group()


def add_arguments(parser):
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")


def run(args):
    train(load_yaml(args.model_config), load_yaml(args.train_config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    run(parser.parse_args())
