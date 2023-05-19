from cProfile import label
from src.data.make_amzn_stock_price_dataset import AMZN_SP
from src.models.conformal_forcast import ConformalForcast
from src.models.lstm_model import LSTM
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train_set = AMZN_SP(train=True, in_folder='data/raw',
                    out_folder='data/processed')
test_set = AMZN_SP(train=False, in_folder='data/raw',
                   out_folder='data/processed')
batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


model = LSTM(1, 4, 2)
model.load_state_dict(torch.load("models/trained_lstm_amzn.pt"))
model.eval()
model = torch.nn.DataParallel(model).cpu()

cmodel = ConformalForcast(model, test_loader, 0.1)


with torch.no_grad():
    pred_interval = cmodel(test_set.data)
    predicted = model(test_set.data).to('cpu').numpy()

plt.plot(test_set.targets, color='red', label='Actual Close')
plt.plot(predicted, color='green', label='Predicted Close')


x = np.linspace(0, len(predicted), len(predicted))
plt.fill_between(x, pred_interval[0], pred_interval[1],
                 color='blue', alpha=0.15, label="Conformal interval")

plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.savefig("./time_seriers")
