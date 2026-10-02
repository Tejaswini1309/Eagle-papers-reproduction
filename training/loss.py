# EAGLE training loss: L = L_reg + w_cls * L_cls

import torch
import torch.nn as nn
import torch.nn.functional as F


class EagleLoss(nn.Module):
    """
    L_reg: Smooth L1 between predicted and target-LLM features.
    L_cls: cross-entropy between the target LLM's token distribution
           p = softmax(LMhead(target feature)) and the draft's
           p_hat = softmax(LMhead(predicted feature)).

    Both terms are averaged over valid (unmasked) token positions only, so the
    weight w_cls has the same meaning regardless of batch size and padding.
    """

    def __init__(self, w_cls: float = 0.1):
        super().__init__()
        self.w_cls = w_cls

    def forward(
        self,
        predicted_features: torch.Tensor,  # (B, S, H)
        target_features: torch.Tensor,     # (B, S, H)
        predicted_logits: torch.Tensor,    # (B, S, V)
        target_logits: torch.Tensor,       # (B, S, V)
        mask: torch.Tensor,                # (B, S) bool, True for valid positions
    ) -> dict:
        mask = mask.bool()
        n_valid = mask.sum().clamp(min=1)

        # Select valid positions first: (N, H) and (N, V).
        pred_f = predicted_features[mask].float()
        tgt_f = target_features[mask].float()
        pred_logits = predicted_logits[mask].float()
        tgt_logits = target_logits[mask].float()

        reg_loss = F.smooth_l1_loss(pred_f, tgt_f)

        # Soft-label cross-entropy; the target distribution is a constant.
        target_probs = F.softmax(tgt_logits, dim=-1)
        cls_loss = -(target_probs * F.log_softmax(pred_logits, dim=-1)).sum(-1).sum() / n_valid

        with torch.no_grad():
            top1_acc = (pred_logits.argmax(-1) == tgt_logits.argmax(-1)).float().sum() / n_valid

        return {
            "loss": reg_loss + self.w_cls * cls_loss,
            "reg_loss": reg_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "top1_acc": top1_acc,
        }
