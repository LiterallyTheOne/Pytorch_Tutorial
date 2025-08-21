# -------------------[ Imports ]-------------------
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.datasets import load_iris


# -------------------[ Define Dataset ]-------------------
class IRISDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


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
) -> float:
    model.train()

    total_loss = 0

    for batch_of_data, batch_of_target in data_loader:
        batch_of_data = batch_of_data.to(device)
        batch_of_target = batch_of_target.to(device)

        optimizer.zero_grad()

        logits = model(batch_of_data)

        loss = loss_fn(logits, batch_of_target)
        total_loss += loss.item()

        loss.backward()

        optimizer.step()

    return total_loss / len(data_loader)


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

    data = torch.tensor(iris.data).to(torch.float)
    target = torch.tensor(iris.target)

    iris_dataset = IRISDataset(data, target)

    # -------------------[ Split the data to train, validation, and test ]-------------------
    g1 = torch.Generator().manual_seed(20)
    train_data, val_data, test_data = random_split(iris_dataset, [0.7, 0.2, 0.1], g1)

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
    for epoch in range(5):
        print("-" * 20)
        print(f"epoch: {epoch}")
        train_loss = train_step(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_accuracy = val_step(val_loader, model, loss_fn, device)
        print(f"train_loss: {train_loss}")
        print(f"val_loss: {val_loss}")
        print(f"val_accuracy: {val_accuracy}")

    print("-" * 20)
    test_loss, test_accuracy = val_step(test_loader, model, loss_fn, device)
    print(f"test_loss: {test_loss}")
    print(f"test_accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
