"""
models/unet_baseline.py

Provides a U-Net segmentation model using the segmentation_models_pytorch library.

The U-Net architecture is a popular choice for semantic segmentation tasks,
featuring an encoder-decoder structure with skip connections for combining
low-level and high-level features.

Usage:
    from models.unet_baseline import get_unet_model
    
    model = get_unet_model(num_classes=4, encoder="resnet34", pretrained=True)
"""

import segmentation_models_pytorch as smp


# TODO: filter and list out actually needed classes for the actual implementation
# TODO: explore better alternatives for segmentation_models_pytorch


def get_unet_model(num_classes=10, encoder="resnet34", pretrained=True):
    """
    Create and return a U-Net segmentation model.

    Args:
        num_classes (int): Number of output segmentation classes. Default is 10.
        encoder (str): Encoder backbone architecture (e.g., "resnet34", "resnet50").
                       Default is "resnet34".
        pretrained (bool): Whether to use ImageNet pretrained weights for the encoder.
                          Default is True.

    Returns:
        torch.nn.Module: U-Net model configured with the specified parameters.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
    )
    return model