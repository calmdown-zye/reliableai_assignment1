import torch
import torch.nn as nn
from torchvision import models

def get_cifar_model(device):
    """
    CIFAR-10용 모델
    torchvision의 ResNet18을 fine-tuning
    """
    # pretrained ResNet18 로드
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 마지막 fc layer를 CIFAR-10 클래스 수(10)에 맞게 교체
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model.to(device)