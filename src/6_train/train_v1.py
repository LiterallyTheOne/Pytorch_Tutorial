# -------------------[ Imports ]-------------------
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.datasets import load_iris

# -------------------[ Find the device ]-------------------
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = "cpu"

print(device)

# -------------------[ Load the data ]-------------------
iris = load_iris()

data = torch.tensor(iris.data).to(torch.float)
target = torch.tensor(iris.target)


class IRISDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


iris_dataset = IRISDataset(data, target)

# -------------------[ Split the data to train, validation, and test ]-------------------
g1 = torch.Generator().manual_seed(20)
train_data, val_data, test_data = random_split(iris_dataset, [0.7, 0.2, 0.1], g1)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


# -------------------[ Define model ]-------------------
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


# -------------------[ Train step ]-------------------
def train_step():
    model.train()

    total_loss = 0

    for batch_of_data, batch_of_target in train_loader:
        batch_of_data = batch_of_data.to(device)
        batch_of_target = batch_of_target.to(device)

        optimizer.zero_grad()

        logits = model(batch_of_data)

        loss = loss_fn(logits, batch_of_target)
        total_loss += loss.item()

        loss.backward()

        optimizer.step()

    print(f"training average_loss: {total_loss / len(train_loader)}")


# -------------------[ Validation step ]-------------------
def val_step():
    model.eval()

    with torch.inference_mode():
        total_loss = 0
        total_correct = 0

        for batch_of_data, batch_of_target in val_loader:
            batch_of_data = batch_of_data.to(device)
            batch_of_target = batch_of_target.to(device)

            logits = model(batch_of_data)

            loss = loss_fn(logits, batch_of_target)
            total_loss += loss.item()

            predictions = logits.argmax(dim=1)
            total_correct += predictions.eq(batch_of_target).sum().item()

        print(f"validation average_loss: {total_loss / len(val_loader)}")
        print(f"validation accuracy: {total_correct / len(val_loader.dataset)}")


# -------------------[ Create a model ]-------------------
model = IRISClassifier()
model.to(device)

# -------------------[ Define loss function and optimizer ]-------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# -------------------[ Train the model ]-------------------
for epoch in range(5):
    print("-" * 20)
    print(f"epoch: {epoch}")
    train_step()
    val_step()
