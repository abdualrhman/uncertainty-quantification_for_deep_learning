import argparse
import torch
import numpy as np
from sklearn import metrics
import torch.nn as nn
from datetime import datetime
from tab_transformer_pytorch import FTTransformer, TabTransformer
from src.data.make_housing_dataset import CaliforniaHousing
import math


def accuracy(target, pred):
    return metrics.r2_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


device = "cuda" if torch.cuda.is_available() else "cpu"


def training() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=1e-2)
    parser.add_argument("--btz", default=64)
    parser.add_argument("--epochs", default=60)
    args = parser.parse_args()
    print('Start training')
    print(args)
    lr = float(args.lr)
    epochs = int(args.epochs)
    batch_size = int(args.btz)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset = CaliforniaHousing(
        split='train', in_folder='data/raw', out_folder='data/processed')
    spl_idx = math.ceil(0.85 * len(dataset))

    train_set, val_set = torch.utils.data.random_split(
        dataset, [spl_idx, len(dataset)-spl_idx], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size)

    model = FTTransformer(
        categories=0,
        num_continuous=8,                # number of continuous values
        dim=32,                           # dimension, paper set at 32
        dim_out=2,                        # binary prediction, but could be anything
        depth=6,                          # depth, paper recommended 6
        heads=8,                          # heads, paper recommends 8
        attn_dropout=0.2,                 # post-attention dropout
        ff_dropout=0.1                    # feed forward dropout
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.8)
    best_vloss = 1_000_000.
    for epoch in range(epochs):
        running_train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            cat_inputs, num_inputs = inputs[0].to(device), inputs[1].to(device)
            optimizer.zero_grad()
            output = model(cat_inputs, num_inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss_value = running_train_loss/len(train_loader)
        running_valid_loss = 0.0
        val_accuracy = []
        model.eval()
        with torch.no_grad():
            for i, (vinputs, vtargets) in enumerate(val_loader):
                vcat_inputs, vnum_inputs = vinputs[0], vinputs[1]
                voutputs = model(vcat_inputs, vnum_inputs)
                vloss = criterion(voutputs, vtargets)
                running_valid_loss += vloss.item()
                val_accuracy.append(np.mean(accuracy(vtargets, voutputs)))

        avg_vloss = running_valid_loss / (i + 1)
        if avg_vloss < best_vloss and epoch > 5:
            best_vloss = avg_vloss
            model_path = 'models/ft_transformer_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
        print(
            f'Epoch {epoch+1} \t\t Training Loss: { train_loss_value :.3f} \t\t Validation Loss: {avg_vloss :.3f} \t\t R^2: {val_accuracy[-1] :.3f} \t\t lr: {scheduler.get_last_lr()[0]:.4f}')
        scheduler.step()
        model.train()
    print("Finished training.")


if __name__ == "__main__":
    training()
