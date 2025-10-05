import segmentation_models_pytorch as smp


def get_deeplab_model(num_classes=4, encoder="resnet50", pretrained=True):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
    )
    return model
