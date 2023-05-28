from torch import nn
import torch


# class RegFNN(nn.Module):
#     '''
#       Multilayer Perceptron for regression.
#     '''

#     def __init__(self):
#         super().__init__()
#         self.linear_yq1_output = torch.nn.Linear(64, 1)
#         self.linear_yq2_output = torch.nn.Linear(64, 1)
#         self.linear_yq3_output = torch.nn.Linear(64, 1)

#         self.net = torch.nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#         )

#     def forward(self, x):
#         '''
#           Forward pass
#         '''
#         x = self.net(x)
#         q1 = self.linear_yq1_output(x)
#         q2 = self.linear_yq2_output(x)
#         q3 = self.linear_yq3_output(x)
#         return q2, (q1, q3)

class QuantileNet(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''

    def __init__(self, input_size: int = 8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q-1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
