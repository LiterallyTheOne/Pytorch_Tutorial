+++
date = '2025-08-30T08:00:00+03:30'
draft = false
title = 'Convolution and ReLU'
description = "Explaning about convolution and ReLU"
weight = 100
tags = ["PyTorch", "TorchVision", "Deep-Learning", "Python", "matplotlib"]
image = "convolution-and-relu.webp"
+++

# Convolution and ReLU

## Introduction

In the previous tutorial, we learned how to work with images.
We learned how to load an image dataset and how to transform its images into tensors.
In this tutorial, we are going to learn about a layer that is being widely used for images
in **Deep Learning** called **Convolution**.
Also, we are going to talk about `ReLU` and make you more familiar with how to work with any **layer**.

Code of this tutorial is available at:
[link to the code](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/9_convolution_and_relu.ipynb)

## Load MNIST

At first, let's load **MNIST** again like we did in the previous tutorial.

```python
train_data = MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
```

Now let's make `train`, `validation`, and `test` data loaders and see the shape of a batch of our data.

```python
g1 = torch.Generator().manual_seed(20)
val_data, test_data = random_split(test_data, [0.7, 0.3], g1)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))

print(f"images shape : {images.shape}")
print(f"labels shape : {labels.shape}")

"""
--------
output: 

images shape : torch.Size([64, 1, 28, 28])
labels shape : torch.Size([64])
"""
```

As you can see, we have a batch of our data with a batch size of `64`.
Each image is grayscale, so it has `1` channel, and the size of the image is `28x28`.

## Convolution layer

Convolution is an operation in which we slide a smaller matrix (kernel) over a bigger matrix and calculate the
weighted sum.
