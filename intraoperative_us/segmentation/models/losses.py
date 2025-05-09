import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
        """
        Focal Loss as used in RetinaNet: https://arxiv.org/abs/1708.02002

        Parameters
        ----------
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

        Parameters:
        ----------- 
            inputs (Tensor): Raw logits of shape (N, *).
            targets (Tensor): Binary targets of same shape.

        Returns:
        ---------
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

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """
        Dice Loss for binary segmentation.

        Parameters:
        ----------
            smooth (float): Smoothing factor to avoid division by zero. Default: 1e-6
            reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(DiceLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'.")

        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss.

        Parameters:
        -----------
            inputs (Tensor): Raw logits of shape (N, *).
            targets (Tensor): Binary targets of same shape.

        Returns:
        ---------
            torch.Tensor: Reduced or unreduced loss.
        """
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (total + self.smooth)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:  # 'none'
            return dice_loss

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1e-6, reduction: str = "mean",
                focal_weight: float = 1.0, dice_weight: float = 1.0):
        """
        Combined Focal and Dice Loss.

        Parameters:
        ----------
            alpha (float): Balances the importance of positive/negative examples for Focal Loss. Default: 0.25
            gamma (float): Modulates the loss for hard/easy examples for Focal Loss. Default: 2.0
            smooth (float): Smoothing factor to avoid division by zero for Dice Loss. Default: 1e-6
            reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.dice_loss = DiceLoss(smooth=smooth, reduction=reduction)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined Focal and Dice loss.

        Parameters:
        -----------
            inputs (Tensor): Raw logits of shape (N, *).
            targets (Tensor): Binary targets of same shape.

        Returns:
        ---------
            torch.Tensor: Reduced or unreduced loss.
        """
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return loss

            