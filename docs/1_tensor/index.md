+++
date = '2025-08-12T11:52:00+03:30'
draft = false
title = 'Tensor'
description = "Tensor in Pytorch"
weight = 20
+++

# Tensor

## What is Tensor

`Tensor` is the fundamental of `PyTorch`.
Input, output, and the parameters of the model are all in `Tensors`.
`Tensor` is like an array (`Numpy array`) but with more power.

* It can be run on `GPU`
* It supports automatic gradients

`Tensor` operations in `Pytorch` are pretty similar to `Numpy array`.
So, if you have worked with `Numpy array` before, you are a step ahead.

In our `Hello world` example, we have created random data using
`torch.rand((3, 8))` also we got the index of the maximum probability
using `logits.argmax(1)`.
In this tutorial, we are going to explain more about the main operations
in `Tensor` and learn how to use them.

Code of this tutorial is available at:
[link to code](https://github.com/LiterallyTheOne/Pytorch_Tutorial/blob/main/src/1_tensor.ipynb)

## Create a Tensor

There are so many ways that we can create a `Tensor`.
One of the simplest ways to create a tensor is as below:

```python
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
t1 = torch.tensor(data)
print(t1)

# --------
# output: 
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
```

As you can see, we had a 2-dimensional matrix, and we gave it to `torch.tensor` as an argument and stored the result
in a variable called `t1`.
When we print `t1`, the output would be a `Tensor` of that matrix.

We can also create a `Tensor` by knowing its shape.
For example, in our `Hello World` example, we created a random dataset using `torch.rand` function.
We also have other functions that we can give the shape of `Tensor` to them and get a `Tensor`.
You can see the examples in the code below:

```python
s1 = torch.rand((3, 8))
print(s1)
print(s1.shape)

# --------
# output: 
# tensor([[0.6667, 0.7057, 0.7670, 0.7719, 0.7298, 0.5729, 0.8281, 0.5963],
#         [0.1056, 0.5377, 0.3380, 0.4923, 0.0246, 0.8192, 0.3945, 0.1150],
#         [0.3885, 0.4211, 0.2655, 0.6766, 0.5082, 0.6465, 0.9499, 0.2008]])
# torch.Size([3, 8])
```

```python
s2 = torch.zeros((3, 8))
print(s2)
print(s2.shape)

# --------
# output: 
# tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])
# torch.Size([3, 8])
```

```python
s3 = torch.ones((3, 8))
print(s3)
print(s3.shape)

# --------
# output: 
# tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1.],
#         [1., 1., 1., 1., 1., 1., 1., 1.]])
# torch.Size([3, 8])
```

In the above examples, we have used 3 functions:

* `torch.rand`: Creates random data
* `torch.ones`: Fills with one
* `torch.zeros`: Fills with zero

As you can see, the shape of all of them is `[3, 8]`, like the way that
we gave them. (You can access the shape of a tensor by `.shape` variable)

We can also create a `Tensor` from other `Tensors`.

```python
l1 = torch.zeros_like(t1)
print(l1)
print(l1.shape)

# --------
# output: 
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])
# torch.Size([3, 3])
```

The first `Tensor` that we created was called `t1` and its shape was `[3, 3]`.
In the example above, we created a `Tensor` like `t1`, which is filled with zeros.

## Attributes of a Tensor

`Tensor` has different attributes that define how it is stored.
We mentioned one of them, which was `shape`.
Now we learn two more of them, `dtype` and `device`.

* `shape`: shape of the tensor
* `dtype`: data type of the tensor
* `device`: device of the tensor, like `cpu` or `cuda` (for `gpu`)

```python
print(f"shape: {t1.shape}")
print(f"dtype: {t1.dtype}")
print(f"device: {t1.device}")

# --------
# output: 
# shape: torch.Size([3, 3])
# dtype: torch.int64
# device: cpu
```

## Control the device

To find if our system has any available accelerators, we can use the code below:

```python
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = "cpu"

print(device)

# --------
# output: 
# mps
```

The code above first checks if there are any accelerators like `cuda` or `mps` (for MacBook).
Then puts the current accelerator in a variable called `device`.
If there wasn't any available, the value of `device` would be `cpu`.
In my case, the output is `mps`.
If you run this code on `Google Colab` with the `GPU` on, you would get `cuda`.

We can change the device of any `Tensor` by using a function called `.to()`.
For example:

```python
t1 = t1.to(device)
print(t1.device)

# --------
# output: 
# mps:0
```

In the code above, we changed the device of the `Tensor` called `t1` to the current accelerator, which in
my case is `mps`.
