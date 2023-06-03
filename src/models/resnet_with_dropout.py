import torch.nn as nn
import torchvision.models as models


class ResNetWithDropout(nn.Module):
    def __init__(self, resnet, dropout_rate=0.1):
        super(ResNetWithDropout, self).__init__()
        self.resnet = resnet

        # Add dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Replace the original fc layer with (fc + dropout)
        self.resnet.fc = nn.Sequential(
            self.dropout,
            nn.Linear(self.resnet.fc.in_features, self.resnet.fc.out_features)
        )

    def forward(self, x):
        return self.resnet(x)
