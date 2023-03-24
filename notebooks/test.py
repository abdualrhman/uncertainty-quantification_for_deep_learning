import torch
from src.models.GB_quantile_regressor import Conformalized_quantile_regression_logits


inputs = torch.tensor([[0.4630, 2.0476],
                       [2.2505, 3.0956],
                       [1.1385, 3.4338],
                       [1.5190, 3.5071],
                       [2.0850, 2.6518],
                       [1.4225, 2.4600],
                       [0.9652, 2.3409],
                       [1.1852, 4.8402],
                       [0.8857, 2.5145],
                       [1.7450, 3.5207],
                       [1.1903, 2.0947],
                       [1.1560, 2.0714],
                       [1.7541, 2.7718],
                       [1.5430, 2.9079],
                       [0.7063, 1.6242],
                       [0.9611, 2.4948],
                       [1.6540, 2.8775],
                       [0.9596, 1.6170],
                       [0.4651, 0.9665],
                       [1.2414, 1.8720],
                       [2.4025, 4.9447],
                       [1.2838, 3.1998],
                       [0.6027, 1.3405],
                       [0.9855, 1.7962],
                       [1.5674, 2.4610],
                       [0.9095, 1.2087],
                       [1.9868, 2.4648],
                       [1.9655, 3.0575],
                       [2.1278, 2.8847],
                       [1.9226, 3.1330],
                       [0.7184, 0.9206],
                       [2.1120, 3.3875]])
targets = torch.tensor([0, 2, 1, 3, 1, 2, 1, 1, 2, 2, 2, 1, 2, 3, 1, 0, 2, 0, 0, 2, 2, 2, 0, 1,
                        2, 0, 1, 2, 2, 3, 0, 2])

model = Conformalized_quantile_regression_logits()
model.fit_precomputed_logits(inputs, targets)
pred = model.predict(inputs)
print(pred)
