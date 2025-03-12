from pytorch_pretrained_vit import ViT

def vit_pre(num_classes):
    model = ViT('B_16_imagenet1k', pretrained=True, image_size=224, num_classes=num_classes)
    return model