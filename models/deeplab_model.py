"""
models/deeplab_model.py

Defines and configures a DeepLabV3+ segmentation model using the 
segmentation_models_pytorch library. Replaces all BatchNorm layers 
with GroupNorm for improved stability in small batch scenarios.

Functions:
    - replace_bn_with_gn: Recursively replaces BatchNorm2d with GroupNorm
    - get_deeplab_model: Returns a configured DeepLabV3+ model

Usage:
    from models.deeplab_model import get_deeplab_model
    model = get_deeplab_model(num_classes=4, encoder="resnet50", pretrained=True)
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


def replace_bn_with_gn(module, default_groups=32):
    """
    Recursively replace all BatchNorm2d layers with GroupNorm layers.

    Args:
        module (torch.nn.Module): The model or module to modify.
        default_groups (int): Target number of groups for GroupNorm. 
                              Will use largest divisor <= default_groups 
                              that divides num_channels evenly.

    Returns:
        torch.nn.Module: The modified module with GroupNorm replacing BatchNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Find largest divisor <= default_groups
            for g in reversed(range(1, default_groups + 1)):
                if num_channels % g == 0:
                    num_groups = g
                    break
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, default_groups)
    return module


def get_deeplab_model(num_classes=4, encoder="resnet50", pretrained=True):
    """
    Create a DeepLabV3+ segmentation model with GroupNorm.

    Args:
        num_classes (int): Number of output segmentation classes. Default is 4.
        encoder (str): Encoder backbone architecture (e.g., "resnet50", "resnet101").
        pretrained (bool): Whether to load ImageNet pretrained encoder weights.

    Returns:
        torch.nn.Module: Configured DeepLabV3+ model with GroupNorm normalization.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
    )
    # Replace BatchNorm with GroupNorm safely
    model = replace_bn_with_gn(model, default_groups=32)
    return model
