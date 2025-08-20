+++
date = '2025-08-18T12:09:00+03:30'
draft = false
title = 'Train'
description = "Training in PyTorch"
weight = 60
tags = ["PyTorch", "Deep-Learning", "Python"]
image = "train.webp"
+++

# Train

## Introduction

Code of this tutorial is available at:
[link to code](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/5_train.ipynb)

## Load the data and make the model

Let's go step by step and load our data, and make our model, like the previous tutorial, to train it.
First, let's load our data with the code below:

```python
iris = load_iris()

data = torch.tensor(iris.data).to(torch.float)
target = torch.tensor(iris.target)
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

Now, we are ready to start learning how to train our model.

## Train the model

Right now, we know how to train our model in `PyTorch`.
So, let's write an optimization step for our model.
First, we need to define `loss function` and `optimizer`.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(iris_classifier.parameters())
```

Now, let's write our training loop.

```python
for batch_of_data, batch_of_target in train_loader:
    optimizer.zero_grad()

    logits = iris_classifier(batch_of_data)

    loss = loss_fn(logits, batch_of_target)
    print(f"loss: {loss.item()}")

    loss.backward()

    optimizer.step()

"""
--------
output: 

loss: 1.181538462638855
loss: 1.1570122241973877
loss: 1.1441924571990967
loss: 1.1753343343734741
loss: 1.1002519130706787
loss: 1.1666862964630127
loss: 1.0838695764541626
loss: 1.1226308345794678
loss: 1.1205450296401978
loss: 1.1404510736465454
loss: 1.094001054763794
"""
```

As you can see, for each batch of data, we calculated the loss and the gradients and optimized the weights.
You might have noticed that the loss in each batch is not necessarily improving.
Don't worry about it, because we are going to address it pretty soon.

## Evaluate the model

## Save and load our model

## Conclusion
