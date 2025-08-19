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
But first, let's learn more about the concepts of training in **PyTorch**.

Code of this tutorial is available at:
[link to code](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/4_train.ipynb)

## AutoGrad

One of the fundamental parts of each `Tensor` in `PyTorch` is that they can store gradients, using
`requires_grad` argument.
Let's define an equation with some tensors:

```python
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

y = a ** 2 + b
```

In the code above, we have tensor `a` and tensor `b` with the values of `3` and `2`.
As you can see, I set the `requires_grad` argument to true for both of them.
Then, I have defined an equation, where:

$$y=a^2+b$$

Now, let's calculate the gradient.
To do so, we can use a function called `.backward()`.
This function looks at the computational graph of the tensor and calculates
the gradient of the tensors that require gradient.
So, if I call the `.backward()` function for y, these gradients would be calculated
$\frac{\delta y}{\delta a}$ and $\frac{\delta y}{\delta b}$.
Before calling that function, let's calculate it ourselves.

$$\frac{\delta y}{\delta a}=\frac{\delta (a^2+b)}{\delta a}=2a \xrightarrow{a=3} 6$$
$$\frac{\delta y}{\delta b}=\frac{\delta (a^2+b)}{\delta b}=1$$

Now, let's see if we get the same results when we call the `.backward()`
function for `y`.

```python
y.backward()

print("dy/da: ", a.grad.item())  # d(a**2 + b)/da = 2*a ----a=3----> 6
print("dy/db: ", b.grad.item())  # d(a**2+b)/db = 1

"""
--------
output: 

dy/da:  6.0
dy/db:  1.0
"""
```

As you can see, our results are the same.
In **Deep Learning**, we use **gradient** to update the weights of our
model.
To do so, we can define a `loss function` as below:

$$l=(y-\hat{y})^2$$

* $l$: loss function
* $y$: true label
* $\hat{y}$: prediction

Now, let's have another example that is closer to what we want to do
in **Deep Learning**.

```python
w = torch.tensor(5.0, requires_grad=True)  # weight
b = torch.tensor(2.0, requires_grad=True)  # bias

x = 2  # input
y_true = 7  # true output

y_hat = w * x + b  # prediction

loss = (y_hat - y_true) ** 2  # calculate loss
loss.backward()  # calculate gradients

print(f"d(loss)/dw: {w.grad.item()}")
print(f"d(loss)/db: {b.grad.item()}")

"""
--------
output: 

d(loss)/dw: 20.0
d(loss)/db: 10.0
"""

```

In the example above, we have `w` that represents `weight`, and we also have
`b` that represents `bias`.
Our input is `2` and our expected output is `7`.
We predict the output by multiplying the input (`x`) by `w`, and then
add it to `b` to get the prediction that we want.
For our loss function, we have the difference between the prediction
and true output powered by 2.
Then, we calculate the gradient of `loss` with respect to `w` and `b` and
print them.
Let's calculate the gradients ourselves to be able to check the results.

$$
\frac{\delta l}{\delta w}
= \frac{\delta (wx + b - y)^2}{\delta w}
= \frac{\delta (wx + b -y)^2}{\delta (wx+b-y)}\frac{\delta (wx+b-y)}{\delta w}
= 2(wx + b -y)x
\xrightarrow{w=5, b=2, x=2, y=7}
2(5\times 2 + 2 - 7)\times 2
=4(10+2-7)
= 20
$$

$$
\frac{\delta l}{\delta b}
= \frac{\delta (wx + b - y)^2}{\delta b}
= \frac{\delta (wx + b -y)^2}{\delta (wx+b-y)}\frac{\delta (wx+b-y)}{\delta b}
= 2(wx + b -y)
\xrightarrow{w=5, b=2, x=2, y=7}
2(5\times 2 + 2 - 7)
=2(10+2-7)
= 10
$$

As you can see, the results are the same as our calculations.

## Loss function

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

## Optimizer

## Save and load our model

## Conclusion
