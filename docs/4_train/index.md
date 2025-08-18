+++
date = '2025-08-18T12:09:00+03:30'
draft = false
title = 'Train'
description = "Training in PyTorch"
weight = 50
tags = ["PyTorch", "Deep-Learning", "Python"]
image = "train.webp"
+++

# Train

## Introduction

Training a model is one of the most important features in **PyTorch**.
In the previous tutorials, we prepared our **data** and our **model**.
Now, we are ready to train our model.

Code of this tutorial is available at:
[link to code](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/4_train.ipynb)

## Load the data and make the model

Let's go step by step and load our data, and make our model, like the previous tutorial, to train it.
First, let's load our data with the code below:

```python
iris = load_iris()

data = torch.tensor(iris.data).to(torch.float)
target = torch.tensor(iris.target).to(torch.float)
```

Now, let's make a `Dataset` for our data.

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


iris_dataset = IRISDataset(data, target)
```

Then, it is time to split it into `train`, `validation`, and `test`.

```python
g1 = torch.Generator().manual_seed(20)
train_data, val_data, test_data = random_split(iris_dataset, [0.7, 0.2, 0.1], g1)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
val_loader = DataLoader(val_data, batch_size=10, shuffle=False)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
```

Let's create our model as well.

```python
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


iris_classifier = IRISClassifier()
```

Now, we are ready to start learning how to train our model

## Loss function

## Optimizer

## Save and load our model

## Conclusion
