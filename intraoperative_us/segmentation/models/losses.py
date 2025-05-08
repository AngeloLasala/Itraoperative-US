import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
        """
        Focal Loss as used in RetinaNet: https://arxiv.org/abs/1708.02002

        Args:
            alpha (float): Balances the importance of positive/negative examples. Default: 0.25
            gamma (float): Modulates the loss for hard/easy examples. Default: 2.0
            reduction (str): 'none' | 'mean' | 'sum'. Default: 'none'
        """
        super(FocalLoss, self).__init__()
        if not (0 <= alpha <= 1 or alpha == -1):
            raise ValueError(f"Invalid alpha value: {alpha}. Must be in [0, 1] or -1.")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigmoid focal loss.

        Args:
            inputs (Tensor): Raw logits of shape (N, *).
            targets (Tensor): Binary targets of same shape.

        Returns:
            torch.Tensor: Reduced or unreduced loss.
        """
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

            