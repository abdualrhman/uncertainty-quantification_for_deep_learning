import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.make_amzn_stock_price_dataset import AMZN_SP
from src.models.lstm_model import LSTM


train_set = AMZN_SP(train=True, in_folder='data/raw',
                    out_folder='data/processed')
test_set = AMZN_SP(train=False, in_folder='data/raw',
                   out_folder='data/processed')
batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description="Train file arguments")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--num_epochs", default=10)
args = parser.parse_args()

model = LSTM(1, 4, 2)
model.to(device)
learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(len(train_set))
print(len(test_set))
# print(test_set.data.shape)


def train_one_epoch(epoch):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()


def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


def save_model():
    torch.save(model.state_dict(), "models/trained_lstm_amzn.pt")


def train_model():
    for epoch in range(num_epochs):
        train_one_epoch(epoch)
        validate_one_epoch()


if __name__ == "__main__":
    train_model()
    save_model()
