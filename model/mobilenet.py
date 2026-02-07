import torch.nn as nn
from torchvision import models




def build_model():
    model = models.mobilenet_v3_large(weights="DEFAULT")
    for p in model.parameters():
        p.requires_grad = False


    model.classifier = nn.Sequential(
        nn.Linear(960, 640), nn.Hardswish(), nn.Dropout(0.2),
        nn.Linear(640, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 5)
    )

    return model
