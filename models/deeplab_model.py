import segmentation_models_pytorch as smp
import torch.nn as nn

def replace_bn_with_gn(module, default_groups=32):
    """
    Recursively replace all BatchNorm2d layers with GroupNorm.
    Chooses a num_groups that divides num_channels.
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
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
    )
    # Replace BatchNorm with GroupNorm safely
    model = replace_bn_with_gn(model, default_groups=32)
    return model
