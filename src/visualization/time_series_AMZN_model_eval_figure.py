from src.data.make_amzn_stock_price_dataset import AMZN_SP
from src.models.conformal_forcast import ConformalForcast
from src.models.lstm_model import LSTM
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils.conformal_forcaster_utils import validate_forcaster


test_set = AMZN_SP(train=False, in_folder='data/raw',
                   out_folder='data/processed')

batch_size = 16
n_data_conf = 300

rand_int = torch.randint(low=0, high=len(
    test_set)-n_data_conf, size=(1,)).item()
cal_range = range(rand_int, rand_int+n_data_conf)
val_range = [i for i in range(len(test_set)) if i not in cal_range]

cal_set = torch.utils.data.Subset(test_set, cal_range)
val_set = torch.utils.data.Subset(test_set, val_range)

# assert non overlapping subsets
assert rand_int not in val_set.indices
assert len(cal_set.indices) == n_data_conf
assert len(val_set.indices) == len(test_set) - n_data_conf

calib_loader = torch.utils.data.DataLoader(
    cal_set, batch_size=batch_size, shuffle=False, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=len(val_set), shuffle=False, pin_memory=True)

model = LSTM(1, 4, 2)
model.load_state_dict(torch.load("models/trained_lstm_amzn.pt"))
model.eval()
model = torch.nn.DataParallel(model).cpu()

cmodel = ConformalForcast(model, calib_loader, 0.1)

validate_forcaster(val_loader, cmodel, print_bool=True)

pred_interval = []
predicted = []
with torch.no_grad():
    for x, y in val_loader:
        pred_interval = cmodel(x)
        predicted = model(x).to('cpu').numpy()


targets = [test_set.targets[i] for i in val_set.indices]
plt.plot(targets, color='red', label='Actual Close')
plt.plot(predicted, color='green', label='Predicted Close')


x = np.linspace(0, len(predicted), len(predicted))
plt.fill_between(x, pred_interval[0], pred_interval[1],
                 color='blue', alpha=0.15, label="Conformal interval")

plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.title("Predicted and actual close values from AMZN test dataset")
plt.savefig("reports/figures/time_series_AMZN_model_eval")
