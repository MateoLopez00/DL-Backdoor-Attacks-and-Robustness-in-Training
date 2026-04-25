import torch
import torch.nn as nn
from torchvision import models


class TailWrapper(nn.Module):
    def __init__(self, body: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.body = body
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SimpleHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleTail(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.head = SimpleHead()
        self.backbone = SimpleBackbone()
        self.tail = SimpleTail(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.backbone(x)
        return self.tail(x)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

        self.head = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.backbone = nn.Sequential(model.layer1, model.layer2, model.layer3, model.layer4)
        self.tail = TailWrapper(nn.Sequential(model.avgpool), model.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.backbone(x)
        return self.tail(x)


def get_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        return ResNet18CIFAR(num_classes)
    if arch == "simple_cnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"Unsupported architecture: {arch}")
