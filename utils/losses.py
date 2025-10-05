import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, num_classes=4):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(
            targets, num_classes).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self, num_classes=4, alpha=0.7):
        super(ComboLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets, self.num_classes)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
