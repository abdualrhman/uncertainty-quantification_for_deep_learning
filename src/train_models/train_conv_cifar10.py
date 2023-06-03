import argparse
import numpy as np
from sklearn import metrics
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import random_split
from src.data.make_cifar10_dataset import CIFAR10, get_img_transformer
from src.models.cifar10_conv_model import Cifar10ConvModel


def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


def training() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CIFAR10(split="train", root='data/processed', download=True,
                      transform=get_img_transformer())
    random_seed = 42
    torch.manual_seed(random_seed)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128)
    model = Cifar10ConvModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("starting training loop:")
    validation_every_steps = 500
    step = 0
    n_epoch = 20
    train_accuracies = []
    valid_accuracies = []
    for epoch in range(n_epoch):
        print(f"epoch: {epoch+1}")
        loss_tracker = []
        train_accuracies_batches = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            output = model(inputs.to(device))
            loss = criterion(output, targets.to(device))
            loss.backward()
            optimizer.step()
            step += 1
            loss_tracker.append(loss.item())
            # Compute accuracy.
            predictions = output.max(1)[1]
            train_accuracies_batches.append(accuracy(targets, predictions))

            if step % validation_every_steps == 0:
                # Append average training accuracy to list.
                train_accuracies.append(np.mean(train_accuracies_batches))
                train_accuracies_batches = []
                # Compute accuracies on validation set.
                valid_accuracies_batches = []
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in val_dataloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        loss = criterion(output, targets)
                        predictions = output.max(1)[1]
                        # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
                        valid_accuracies_batches.append(
                            accuracy(targets, predictions) * len(inputs))
                    model.train()
                # Append average validation accuracy to list.
                valid_accuracies.append(
                    np.sum(valid_accuracies_batches) / len(val_dataset))
                print(
                    f"Step {step:<5}   training accuracy: {train_accuracies[-1]}")
                print(f"             val accuracy: {valid_accuracies[-1]}")
    print("Finished training.")
    torch.save(model.state_dict(), "models/trained_conv_cifar10_model.pt")

    plt.plot(loss_tracker, "-")
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.savefig(f"reports/figures/training_curve.png")


if __name__ == "__main__":
    training()
