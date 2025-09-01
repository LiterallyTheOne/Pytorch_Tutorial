# -------------------[ Imports ]-------------------
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder
from torchvision import transforms

import kagglehub
from pathlib import Path

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


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
    path = kagglehub.dataset_download("balabaskar/tom-and-jerry-image-classification")
    path = Path(path) / "tom_and_jerry/tom_and_jerry"

    tom_and_jerry_transforms = transforms.Compose([transforms.Resize([90, 160]), transforms.ToTensor()])

    all_data = ImageFolder(path, transform=tom_and_jerry_transforms)

    # -------------------[ Split the data to train, validation, and test ]-------------------
    g1 = torch.Generator().manual_seed(20)
    train_data, val_data, test_data = random_split(all_data, [0.7, 0.2, 0.1], g1)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # -------------------[ Create the model ]-------------------
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # -------------------[ Freeze the model weights ]-------------------
    for name, param in model.named_parameters():
        if not ("18" in name or "17" in name):
            param.requires_grad = False

    # -------------------[ Change the classifier layer ]-------------------
    model.classifier = nn.Linear(in_features=1280, out_features=4)

    model.to(device)

    # -------------------[ Define loss function and optimizer ]-------------------
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    # -------------------[ Setup TensorBoard ]-------------------
    writer = SummaryWriter()

    # -------------------[ Train and evaluate the model ]-------------------
    for epoch in range(20):
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
