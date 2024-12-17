import os

import torch
import torch.nn as nn

from .PretrainedResNet import PretrainedResNet


class CarPricePredictionModel(nn.Module):
    def __init__(self, tabular_input_dim):
        super(CarPricePredictionModel, self).__init__()

        # Load pre-trained ResNet model
        self.image_model = PretrainedResNet()
        resnet_weights_path = os.path.join(os.getenv("WEIGHTS"), "ResNet.pth")
        if os.path.exists(resnet_weights_path):
            self.image_model.image_model.load_state_dict(
                torch.load(resnet_weights_path), strict=False
            )
        else:
            raise FileNotFoundError(
                f"ResNet weights not found at {resnet_weights_path}"
            )

        # Freeze all layers except the fully connected (fc) layers
        for name, param in self.image_model.image_model.named_parameters():
            param.requires_grad = "fc" in name

        # Modify the fully connected layers of the pre-trained ResNet model
        num_features = self.image_model.image_model.fc.in_features
        self.image_model.image_model.fc = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.3)
        )

        # Define tabular data model
        self.tabular_model = nn.Sequential(
            nn.Linear(tabular_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Combine image and tabular data
        self.combined_model = nn.Sequential(
            nn.Linear(256 + 256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )

    def forward(self, image, tabular_data):
        # Extract features from both image and tabular data
        image_features = self.image_model(image)
        tabular_features = self.tabular_model(tabular_data)

        # Combine features and pass through final layers
        combined_features = torch.cat((image_features, tabular_features), dim=1)
        output = self.combined_model(combined_features)

        return output
