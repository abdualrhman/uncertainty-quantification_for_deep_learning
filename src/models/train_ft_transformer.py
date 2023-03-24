import argparse
import torch
from src.data.make_housing_dataset import CaliforniaHousing
import torch.optim as optim
from torch import nn
import numpy as np
from src.models.regFNN import RegFNN


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # assert not target.requires_grad
        # assert preds.size(0) == target.size(0)
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


def training() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=5e-4)
    args = parser.parse_args()
    random_seed = 42
    torch.manual_seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = CaliforniaHousing(
        split="train", in_folder='data/raw', out_folder='data/processed')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)

    model = RegFNN()
    model = model.to(device)
    criterion = QuantileLoss([0.05, 0.95])
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
    torch.save(model.state_dict(), "models/trained_reg_fnn.pt")


if __name__ == "__main__":
    training()
