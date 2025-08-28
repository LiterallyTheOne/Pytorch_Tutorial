+++
date = '2025-08-28T08:45:00+03:30'
draft = false
title = 'Work with image'
description = "Explaining how to work with images in PyTorch"
weight = 90
tags = ["PyTorch", "TorchVision", "Deep-Learning", "Python", "matplotlib"]
image = "work-with-image.webp"
+++

# Work with Images

## Introduction

In the previous tutorials, we have learned how to work with one-dimensional data.
In this tutorial, we are going to learn how to make a `dataloader` out of images.

Code of this tutorial is available at:
[link to the code](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/8_work_with_image.ipynb)

## Load a dataset

`PyTorch` has a built-in way to download and load some important datasets.
This functionality is available with their `TorchVision` package.
Let's download a minimal `Dataset` called `MNIST`.
This dataset contains `0` to `9` handwritten numbers.
To do so, we can use the code below:

```python
from torchvision.datasets import MNIST

train_data = MNIST("data/", train=True, download=True)
test_data = MNIST("data/", train=False, download=True)
```

In the code above, we loaded `MNIST` in two subsets: `train` and `test`.
The first argument is the path of the data that we want to load.
In our case, we set that to `data/`.
With the `train` argument, we can control whether we want to download `train` subset
or `test` subset.
When we set `download` to `True`, if the `data` is not available in the given path,
it would download it.
These subsets are the instances of `Dataset`.
To make sure, we can check them with the code below:

```python
print(isinstance(train_data, Dataset))

"""
--------
output: 

True
"""
```

So, knowing this, we can do all the things with `Dataset` that we would do before.
Let's now see the size of each dataset.

```python
print(f"train_data's size: {len(train_data)}")
print(f"test_data's size: {len(test_data)}")

"""
--------
output: 

train_data's size: 60000
test_data's size: 10000
"""
```

As you can see, we have `60000` data for training and `10000` data for testing.
Now let's display one of the images.

```python
from matplotlib import pyplot as plt

for image, label in train_data:
    plt.imshow(image, cmap="gray")
    print(label)
    break

"""
--------
output: 

5
"""
```

![mnist sample](mnist-sample.webp)

In the code above, we have displayed one sample of `MNIST` with its label.

## Transforms

As you recall, in the previous tutorials, we had created a `Dataset` like below:

```python
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
```

In the `__getitem__` function, we were transforming our `data` and `target` to tensors to make them ready for our
model.
In `PyTorch`, it is a good practice to implement two more arguments for our `Dataset` called:
`transform` and `target_transform`.
`transform` is being used for transforming each sample of data, and `target_transform` is being used for transforming
each target.
In the code above, we have:

* `transfrom`: `torch.tensor(self.data[idx]).to(torch.float)`
* `target_transform`: `torch.tensor(self.target[idx])`

If we want to change our dataset to have these two arguments, we can do something like below:

```python
class IRISDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        super().__init__()
        self.data = data
        self.target = target

        if transform is None:
            transform = lambda x: torch.tensor(x).to(torch.float)

        if target_transform is None:
            target_transform = lambda x: torch.tensor(x)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.transform(self.data[idx])
        target = self.target_transform(self.target[idx])
        return data, target
```

As you can see, in the code above, we have defined `transform` and `target_transform` as arguments.
If they were `None`, we would have defined them as they were, using `lambda` function.
`TorchVision` has provided us with some built-in transforms for images.
You can find all the transforms in this link:
[TorchVision Transforms](https://docs.pytorch.org/vision/0.9/transforms.html)
At first, we are going to use the `ToTensor` transform.
This module transforms the `image` to `Tensor`.
So, when we want to load our `MNIST`, we are going to add that as a transform.

```python
from torchvision import transforms

train_data = MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
```

Now, let's see if it's applied or not.

```python
for image, label in train_data:
    print(type(image))
    break

"""
--------
output: 

<class 'torch.Tensor'>
"""
```

As you can see, the type of our image is `Tensor`.

We can make a sequence of `transforms` using `transforms.Compose`.
For example, let's first resize each image to `[14, 14]` (our current size is `[28, 28]`).
Then, transform them into tensors.

```python
transform_compose = transforms.Compose(
    [
        transforms.Resize([14, 14]),
        transforms.ToTensor()
    ]
)
```

Now, let's test it to see if it works or not.

```python
# -------------------[ Before transform compose ]-------------------
for image, label in train_data:
    print(f"Before transform compose: {image.shape}")
    break

train_data = MNIST("data/", train=True, download=True, transform=transform_compose)
test_data = MNIST("data/", train=False, download=True, transform=transform_compose)

# -------------------[ After transform compose ]-------------------
for image, label in train_data:
    print(f"After transform compose: {image.shape}")
    break

"""
--------
output: 

Before transform compose: torch.Size([1, 28, 28])
After transform compose: torch.Size([1, 14, 14])
"""
```

As you can see in the code above, it works as intended.
