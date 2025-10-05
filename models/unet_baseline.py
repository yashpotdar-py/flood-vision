"""
models/unet_baseline.py

UNet model definition using segmentation_models_pytorch library.
Usage:
    pip install segmentation-models-pytorch
    from models.unet_baseline import get_unet_model
    model = get_unet_model(num_classes=10, encoder="resnet34",
                            pretrained=True)
    output = model(input_tensor)  # input_tensor shape: (B, 3, H, W)
"""

import segmentation_models_pytorch as smp
# TODO: filter and list out actually needed classes for the actual implementation
# TODO: explore better alternatives for segmentation_models_pytorch
def get_unet_model(num_classes=10, encoder="resnet34",
                   pretrained=True):
    """
    Returns a UNet model from segmentation_models_pytorch with specified encoder and number of classes.
    Args:
        num_classes (int): Number of output segmentation classes.
        encoder (str): Backbone encoder architecture.
        pretrained (bool): Use ImageNet pre-trained
    Returns: 
        torch.nn.Module: UNet model
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
    )
    return model