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

class RegFNN(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 64),
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
