# -------------------[ Imports ]-------------------
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
from matplotlib import pyplot as plt


# -------------------[ Define Dataset ]-------------------
class IRISDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).to(torch.float)
        target = torch.tensor(self.target[idx])
        return data, target


# -------------------[ Define Model ]-------------------
class IRISClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        return self.layers(x)


# -------------------[ Define Training step ]-------------------
def train_step(
        data_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: str,
) -> tuple[float, float]:
    model.train()

    total_loss = 0
    total_correct = 0

    for batch_of_data, batch_of_target in data_loader:
        batch_of_data = batch_of_data.to(device)
        batch_of_target = batch_of_target.to(device)

        optimizer.zero_grad()

        logits = model(batch_of_data)

        loss = loss_fn(logits, batch_of_target)
        total_loss += loss.item()

        predictions = logits.argmax(dim=1)
        total_correct += predictions.eq(batch_of_target).sum().item()

        loss.backward()

        optimizer.step()

    return total_loss / len(data_loader), total_correct / len(data_loader.dataset)


# -------------------[ Define Validation Step ]-------------------
def val_step(
        data_loader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str,
) -> tuple[float, float]:
    model.eval()

    with torch.inference_mode():
        total_loss = 0
        total_correct = 0

        for batch_of_data, batch_of_target in data_loader:
            batch_of_data = batch_of_data.to(device)
            batch_of_target = batch_of_target.to(device)

            logits = model(batch_of_data)

            loss = loss_fn(logits, batch_of_target)
            total_loss += loss.item()

            predictions = logits.argmax(dim=1)
            total_correct += predictions.eq(batch_of_target).sum().item()

        return total_loss / len(data_loader), total_correct / len(data_loader.dataset)


def main():
    # -------------------[ Find the accelerator ]-------------------
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"

    print(device)

    # -------------------[ Load the data ]-------------------
    iris = load_iris()

    # -------------------[ Split the data to train, validation, and test ]-------------------
    train_subset, val_subset, train_target, val_target = train_test_split(
        iris.data,
        iris.target,
        test_size=0.3,
        random_state=42,
        stratify=iris.target,
    )
    val_subset, test_subset, val_target, test_target = train_test_split(
        val_subset,
        val_target,
        test_size=0.33,
        random_state=42,
        stratify=val_target,
    )

    # -------------------[ Normalize subsets ]-------------------
    scaler = StandardScaler()
    scaler.fit(train_subset)

    train_subset_normalized = scaler.transform(train_subset)
    val_subset_normalized = scaler.transform(val_subset)
    test_subset_normalized = scaler.transform(test_subset)

    train_data = IRISDataset(train_subset_normalized, train_target)
    val_data = IRISDataset(val_subset_normalized, val_target)
    test_data = IRISDataset(test_subset_normalized, test_target)

    # -------------------[ Create data Loaders ]-------------------
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    # -------------------[ Create the model ]-------------------
    model = IRISClassifier()
    model.to(device)

    # -------------------[ Define loss function and optimizer ]-------------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # -------------------[ Train and evaluate the model ]-------------------
    train_losses = []
    train_accuracies = []

    val_accuracies = []
    val_losses = []

    for epoch in range(20):
        print("-" * 20)
        print(f"epoch: {epoch}")

        train_loss, train_accuracy = train_step(train_loader, model, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = val_step(val_loader, model, loss_fn, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"train: ")
        print(f"\tloss: {train_loss:.4f}")
        print(f"\taccuracy: {train_accuracy:.4f}")

        print(f"validation: ")
        print(f"\tloss: {val_loss:.4f}")
        print(f"\taccuracy: {val_accuracy:.4f}")

    print("-" * 20)
    test_loss, test_accuracy = val_step(test_loader, model, loss_fn, device)
    print(f"test: ")
    print(f"\tloss: {test_loss:.4f}")
    print(f"\taccuracy: {test_accuracy:.4f}")

    # -------------------[ Plot our results ]-------------------
    plt.figure()
    plt.title("loss")
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()

    plt.figure()
    plt.title("accuracy")
    plt.plot(train_accuracies, label="train")
    plt.plot(val_accuracies, label="val")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
