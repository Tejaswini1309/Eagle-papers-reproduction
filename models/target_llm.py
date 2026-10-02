# Frozen target LLM: provides features and token embeddings to the draft head

import os
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class DraftInputs(NamedTuple):
    """
    Tensors aligned for one draft-head training step (S = sequence length).

    The head sees feature f_i and the embedding of token t_{i+1}, and must
    predict f_{i+1}. Position i therefore pairs:
        features[:, i]               = f_i
        shifted_embeddings[:, i]     = embed(t_{i+1})
        target_features[:, i]        = f_{i+1}
    """

    features: torch.Tensor            # (B, S-1, H)
    shifted_embeddings: torch.Tensor  # (B, S-1, E)
    target_features: torch.Tensor     # (B, S-1, H)
    mask: torch.Tensor                # (B, S-1) bool, True where t_{i+1} is a real token


class TargetLLM(nn.Module):
    """
    Frozen target LLM.

    "Feature" is the final decoder output after the last norm, i.e. the input
    of the LM head, so `compute_logits(feature)` equals the target LLM's logits.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.unk_token or self.tokenizer.eos_token
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype
        ).to(device)

        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_config(cls, cfg: dict, device: str = "cuda", dtype=torch.float16):
        """Build from the `target_model` section of model_config.yaml."""
        target_cfg = cfg["target_model"]
        source = (
            target_cfg["path"]
            if os.path.isdir(target_cfg["path"])
            else target_cfg["name"]
        )
        llm = cls(source, device=device, dtype=dtype)

        hidden_size = llm.model.config.hidden_size
        if hidden_size != target_cfg["hidden_size"]:
            raise ValueError(
                f"Config hidden_size={target_cfg['hidden_size']} but loaded "
                f"model has {hidden_size}"
            )
        return llm

    def train(self, mode: bool = True):
        # The target LLM must never leave eval mode (no dropout).
        return super().train(False)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @torch.no_grad()
    def get_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
    ) -> torch.Tensor:
        """
        Run the decoder only (no LM head) and return (B, S, H) features.

        For inference, pass a `DynamicCache` as `past_key_values` (it is
        appended to in place); `attention_mask` may then be a 4D boolean
        tree mask and `position_ids` the tree positions.
        """
        out = self.model.get_decoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        )
        return out.last_hidden_state

    @torch.no_grad()
    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings from the target LLM's embedding layer."""
        return self.model.get_input_embeddings()(token_ids)

    def compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Frozen LM head. Gradients flow to `features` but not to the head
        weights, so it is safe to call on draft-model predictions.
        """
        lm_head = self.model.get_output_embeddings()
        return lm_head(features.to(lm_head.weight.dtype))

    @torch.no_grad()
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DraftInputs:
        """Run the target LLM and build the shifted inputs for the draft head."""
        features = self.get_features(input_ids, attention_mask)

        if attention_mask is None:
            mask = torch.ones_like(input_ids[:, 1:], dtype=torch.bool)
        else:
            mask = attention_mask[:, 1:].bool()

        return DraftInputs(
            features=features[:, :-1],
            shifted_embeddings=self.embed_tokens(input_ids[:, 1:]),
            target_features=features[:, 1:],
            mask=mask,
        )
