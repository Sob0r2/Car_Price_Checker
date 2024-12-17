import torch.nn as nn
from torchvision import models


class PretrainedResNet(nn.Module):
    def __init__(self):
        super(PretrainedResNet, self).__init__()
        self.image_model = models.resnet34(pretrained=True)
        num_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.image_model(x)
