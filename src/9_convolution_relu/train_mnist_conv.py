# -------------------[ Imports ]-------------------
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import MNIST
from torchvision import transforms


# -------------------[ Define Model ]-------------------
class IRISClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),  # 64x7x7
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=3),  # 128x3x3
            nn.ReLU(),
        )

        self.classification_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classification_layers(x)
        return x


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
    train_data = MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
    test_data = MNIST("data/", train=False, download=True, transform=transforms.ToTensor())

    # -------------------[ Split the data to train, validation, and test ]-------------------
    g1 = torch.Generator().manual_seed(20)
    val_data, test_data = random_split(test_data, [0.7, 0.3], g1)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # -------------------[ Create the model ]-------------------
    model = IRISClassifier()
    model.to(device)

    # -------------------[ Define loss function and optimizer ]-------------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # -------------------[ Setup TensorBoard ]-------------------
    writer = SummaryWriter()

    # -------------------[ Train and evaluate the model ]-------------------
    for epoch in range(5):
        print("-" * 20)
        print(f"epoch: {epoch}")

        train_loss, train_accuracy = train_step(train_loader, model, optimizer, loss_fn, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("accuracy/train", train_accuracy, epoch)

        val_loss, val_accuracy = val_step(val_loader, model, loss_fn, device)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/val", val_accuracy, epoch)

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


if __name__ == "__main__":
    main()
