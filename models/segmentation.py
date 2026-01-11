import torchvision.models.segmentation as segmentation_models
from utils.proxy_config import hpc_proxy


def deeplabv3_resnet50(pretrained=True, num_classes=21):
    with hpc_proxy():
        model = segmentation_models.deeplabv3_resnet50(
            weights='COCO_WITH_VOC_LABELS_V1' if pretrained else None,
            progress=False,
            num_classes=num_classes,
            aux_loss=None
        )
    return model