"""
Segmentation losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)  # pt = p if y=1, else 1-p
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, apply_sigmoid=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs, targets):
        # Validate shapes
        if inputs.shape != targets.shape:
            raise ValueError(f"Inputs ({inputs.shape}) and targets ({targets.shape}) must have the same shape")
            
        # Apply sigmoid if needed
        if self.apply_sigmoid:
            inputs = inputs.sigmoid()

        # Reshape to (batch_size, num_pixels)
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        # Calculate per-sample Dice
        intersection = (inputs * targets).sum(-1)
        union = inputs.sum(-1) + targets.sum(-1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average across batch
        return 1 - dice.mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.5, focal_weight=1.0, dice_weight=1.0):
        """
        Folcal + Dice loss

        """
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal = FocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        fl = self.focal(inputs, targets)
        dl = self.dice(inputs, targets)
        return self.focal_weight * fl + self.dice_weight * dl