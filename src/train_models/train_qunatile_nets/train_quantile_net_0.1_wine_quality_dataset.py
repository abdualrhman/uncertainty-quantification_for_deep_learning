import argparse
import torch
from src.data.make_wine_quality_dataset import WineQuality
from torch import nn
from src.models.quantile_net import QuantileNet, QuantileLoss

def training() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=5e-4)
    args = parser.parse_args()
    random_seed = 42
    torch.manual_seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = WineQuality(
        train=True, in_folder='data/raw', out_folder='data/processed')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)

    model = QuantileNet(input_size=11)
    model = model.to(device)
    criterion = QuantileLoss([0.1, 0.9])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-4, weight_decay=1e-6)

    num_epoch = 100
    for epoch in range(num_epoch):
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets.to(device))
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0
    print('Training process has finished.')
    torch.save(model.state_dict(),
               "models/trained_quantile_net0.1_wine_quality.pt")


if __name__ == "__main__":
    training()
