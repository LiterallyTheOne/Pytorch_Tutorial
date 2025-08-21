+++
date = '2025-08-21T08:00:00+03:30'
draft = false
title = 'Train 2'
description = "Training in PyTorch part 2"
weight = 70
tags = ["PyTorch", "Deep-Learning", "Python"]
image = "train.webp"
+++

# Train 2

## Introduction

In the previous tutorial, we have learned how to train our model.
But our model wasn't getting properly trained.
In this tutorial, we want to address that problem and try to solve it.

Code of this tutorial is available at:
[link to the code](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/6_train)

## Modular train step and validation step

In the previous tutorial, we wrote a code to train our model as below:

```python
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

```

Let's put that code in a file called
[train_v1.py](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/6_train/train_v1.py).
Right now, `train_step` and `val_step` only work with the global variables.
Let's make them more modular.

```python
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
```

As you can see, in the code above, we now give the needed arguments to `train_step` and `val_step` to work with.
Also, instead of printing the results in each function, now I return the results.
For both functions, I return `average loss` and `accuracy`.
Now, let's make our code more organized and put it in a file named
[train_v2.py](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/6_train/train_v2.py).

```python
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
        train_loss, train_accuracy = train_step(train_loader, model, optimizer, loss_fn, device)
        val_loss, val_accuracy = val_step(val_loader, model, loss_fn, device)
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

"""
--------
output: 
    mps
--------------------
epoch: 0
train: 
	loss: 1.0473
	accuracy: 0.3714
validation: 
	loss: 1.0471
	accuracy: 0.2333
--------------------
epoch: 1
train: 
	loss: 0.9799
	accuracy: 0.4857
validation: 
	loss: 0.9770
	accuracy: 0.6667
--------------------
epoch: 2
train: 
	loss: 0.9447
	accuracy: 0.6571
validation: 
	loss: 0.9077
	accuracy: 0.6667
--------------------
epoch: 3
train: 
	loss: 0.9004
	accuracy: 0.7143
validation: 
	loss: 0.8768
	accuracy: 0.6333
--------------------
epoch: 4
train: 
	loss: 0.8546
	accuracy: 0.6857
validation: 
	loss: 0.8063
	accuracy: 0.6667
--------------------
test: 
	loss: 0.8586
	accuracy: 0.6000
"""
```

In the code above, I organized the code.
I defined a main function, and separated the classes and functions with the code for running.
I made the logging look more appealing.
Also, at the end, I evaluated our model on `test` subset as well.
As you can see, `training loss` and `training accuracy` are improving,
but `validation loss` and `validation accuracy` might not necessarily.

## Better splitting

We have learned how to split our dataset into `3` subsets (`train`, `validation`, `test`),
using `random_split` in PyTorch, as below:

```python
class IRISDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


# -------------------[ Load the data ]-------------------
iris = load_iris()

data = torch.tensor(iris.data).to(torch.float)
target = torch.tensor(iris.target)

iris_dataset = IRISDataset(data, target)

# -------------------[ Split the data to train, validation, and test ]-------------------
g1 = torch.Generator().manual_seed(20)
train_data, val_data, test_data = random_split(iris_dataset, [0.7, 0.2, 0.1], g1)
```

Now, let's see how the labels are distributed.

```python
label_count = {
    0: 0,
    1: 0,
    2: 0,
}

for data, target in train_data:
    label_count[target.item()] += 1

print(f"train label count: {label_count}")

"""
--------
output: 

train label count: {0: 33, 1: 39, 2: 33}
"""

```

```python
label_count = {
    0: 0,
    1: 0,
    2: 0,
}

for data, target in val_data:
    label_count[target.item()] += 1

print(f"validation label count: {label_count}")

"""
--------
output: 

validation label count: {0: 13, 1: 6, 2: 11}
"""

```

```python
label_count = {
    0: 0,
    1: 0,
    2: 0,
}

for data, target in test_data:
    label_count[target.item()] += 1

print(f"test label count: {label_count}")

"""
--------
output: 

test label count: {0: 4, 1: 5, 2: 6}
"""
```

As you can see, the distribution of the labels isn't perfect.
Let's fix that by using the `train_test_split` function in `scikit-learn`.

```python
import numpy as np
from sklearn.model_selection import train_test_split

iris = load_iris()

data = iris.data
target = iris.target

train_data, val_data, train_target, val_target = train_test_split(
    data,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target,
)
val_data, test_data, val_target, test_target = train_test_split(
    val_data,
    val_target,
    test_size=0.33,
    random_state=42,
    stratify=val_target,
)

print("size of each subset: ")
print(f"\ttrain: {train_data.shape[0]}")
print(f"\tval: {val_data.shape[0]}")
print(f"\ttest: {test_data.shape[0]}")

print("target distribution:")
print(f"\ttrain: {np.unique(train_target, return_counts=True)}")
print(f"\tval: {np.unique(val_target, return_counts=True)}")
print(f"\ttest: {np.unique(test_target, return_counts=True)}")

"""
--------
output: 

size of each subset: 
	train: 105
	val: 30
	test: 15
target distribution:
	train: (array([0, 1, 2]), array([35, 35, 35]))
	val: (array([0, 1, 2]), array([10, 10, 10]))
	test: (array([0, 1, 2]), array([5, 5, 5]))
"""

```

In the code above, first, we split our data into `2` subsets (`train`, `val`).
As a result, our `train` would be $70%$ of the data, and `val` would be $30%$.
Then we split the `val` into `val` and `test`.
Then, our `test` would be $30% \times 33% = 9.9%$ of all data,
and `val` would be $30% - 9.9% = 20.1%$ of all the data.
As you can see, we used `stratify` argument as well.
This argument forces the splitting to have equal distribution.
As you can see, now we have `35` of each label for `train`,
`10` of each label for `val`,
and `5` of each label for `test`.
Now, let's make a dataset out of them.

```python
train_data = IRISDataset(train_data, train_target)
val_data = IRISDataset(val_data, val_target)
test_data = IRISDataset(test_data, test_target)
```

## Standard Scaler

One of the usual techniques in **Deep Learning** is to **Normalize** our data.
Right now, every feature has a different **average** and **standard deviation** (**std**).
Let's print them out.

```python
iris = load_iris()
data = iris.data

print(f"Mean of the features:\n\t {data.mean(axis=0)}")
print(f"Standard deviation of the features:\n\t {data.std(axis=0)}")

"""
--------
output: 
Mean of the features:
	 [5.84333333 3.05733333 3.758      1.19933333]
Standard deviation of the features:
	 [0.82530129 0.43441097 1.75940407 0.75969263]
"""
```

We want to change the **average** of each feature to `0` and their **std** to `1`.
To do so, we can
