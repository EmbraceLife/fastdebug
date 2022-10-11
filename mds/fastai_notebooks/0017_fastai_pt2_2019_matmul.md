---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 0017_fastai_pt2_2019_matmul

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

## Matrix multiplication from foundations


The *foundations* we'll assume throughout this course are:

- Python
- Python modules (non-DL)
- pytorch indexable tensor, and tensor creation (including RNGs - random number generators)
- fastai.datasets


## Check imports

```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
```

[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=1850)

### [31:11](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1871) - how to build a test framework using the source code of `test`, `test_eq`, and run tests for all notebooks (fastforward to 2022, we have the test source code in [fastcore.test](https://nbviewer.org/github/fastai/fastcore/blob/master/nbs/00_test.ipynb) `nbdev_test` to run tests for all notebooks) [35:23](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2123) - why it is great to have a unit testing with jupyter



```python
#export
from exp.nb_00 import *
import operator

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def test_eq(a,b): test(a,b,operator.eq,'==')
```

```python
test_eq(TEST,'test')
```

```python
# To run tests in console:
# ! python run_notebook.py 01_matmul.ipynb
```

## Get data


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=2159)

### [35:59](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2159) - what are the basic libs needed to create our matrix multiplication [notebook](https://nbviewer.org/github/fastai/course-v3/blob/7fceebfd14d4f3bc7e0ec649834309b8cb786e40/nbs/dl2/01_matmul.ipynb#Get-data)/module 


```python
#export
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor


```

### [36:25](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2185) - how to [download and extract](https://nbviewer.org/github/fastai/course-v3/blob/7fceebfd14d4f3bc7e0ec649834309b8cb786e40/nbs/dl2/01_matmul.ipynb#Get-data) mnist dataset using the most basic libraries: `fastai.datasets`, `gzip`, `pickle`


```python
MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
path = datasets.download_data(MNIST_URL, ext='.gz'); path
```

```python
with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
```

### [36:57](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2217) - how to convert numpy array from mnist dataset into pytorch tensor using `map` and `tensor`; why Jeremy would like us to use pytorch tensor instead of numpy array; [37:42](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2262) - how to find out about the structure of the mnist dataset using `tensor.shape` and `min`, `max`


```python
x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
n,c = x_train.shape
x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()
```

### [38:15](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2295) - how to build a test to check the dataset has the structure we expect using `assert` and `test_eq`

```python
assert n==y_train.shape[0]==50000
test_eq(c,28*28)
test_eq(y_train.min(),0)
test_eq(y_train.max(),9)
```

### [38:39](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2319) - how to turn a long vector tensor into a 2d tensor using `img.view(28, 28)`; how to display image from a `torch.FloatTensor` using `plt.imshow(img.view(28,28))`


```python
mpl.rcParams['image.cmap'] = 'gray'
```

```python
img = x_train[0]
```

```python
img.view(28,28).type()
```

```python
plt.imshow(img.view((28,28)));
```

## Initial python model


 [Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=2342)


 [Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=2342)

### [39:04](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2344) - If we are to build a simplest linear model for mnist dataset, how to create the weights and biases for the model using `weights = torch.randn(784,10)` and `bias = torch.zeros(10)`. check the [notebook](https://nbviewer.org/github/fastai/course-v3/blob/7fceebfd14d4f3bc7e0ec649834309b8cb786e40/nbs/dl2/01_matmul.ipynb#Initial-python-model) 

```python
weights = torch.randn(784,10)
```

```python
bias = torch.zeros(10)
```

## Matrix multiplication

### [39:49](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2389) - how to understand the matrix multiplication calculation process (see [animation](http://matrixmultiplication.xyz/)); how to implement the matrix multiplication with 3 loops (see src code below); imagine an input matrix `rows=5, cols=28*28` and output matrix `rows=5, cols=10`, what would the weights matrix be? `(rows=28*28, cols=10)` In the src below, `a` would be the input matrix and `b` be the weights, we want to find out about the output matrix `c`. how to use `assert` (I found a useful link [here](https://www.programiz.com/python-programming/assert-statement))

```python
def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                c[i,j] += a[i,k] * b[k,j]
    return c
```

### [42:57](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2577) - run an example on `matmul` and test it and check how long does it take to calc a matrix of 5 rows; python is 1000 times slower than pytorch

```python
m1 = x_valid[:5]
m2 = weights
```

```python
m1.shape,m2.shape
```

```python
%time t1=matmul(m1, m2)
```

```python
t1.shape
```

This is kinda slow - what if we could speed it up by 50,000 times? Let's try!

```python
len(x_train)
```

## Elementwise ops

### [44:27](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2667) - how to speed up the matrix multiplication by 50000 times by using pytorch (which uses a different lib called aten (the [difference](https://discuss.pytorch.org/t/whats-the-difference-between-aten-and-c10/114034) between aten and c10) to replace each loop at a time [45:11](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2711) - what is elementwise operation [notebook](https://nbviewer.org/github/fastai/course-v3/blob/7fceebfd14d4f3bc7e0ec649834309b8cb786e40/nbs/dl2/01_matmul.ipynb#Elementwise-ops) from aten or c10 of pytorch; what does elementwise operation do between vectors and between matricies; 



Operators (+,-,\*,/,>,<,==) are usually element-wise.

Examples of element-wise operations:


 [Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=2682)

```python
a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
a,b
```

```python
a + b
```

```python
(a < b).float().mean()
```

```python
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]]); m
```

### [46:24](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2784) - how to translate equations into codes; how to read Frobenius norm equation; how often it appears in deep learning papers; [47:38](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2858) - how to get latex for math equations without actually writing them


Frobenius norm:

$$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$

*Hint*: you don't normally need to write equations in LaTeX yourself, instead, you can click 'edit' in Wikipedia and copy the LaTeX from there (which is what I did for the above equation). Or on arxiv.org, click "Download: Other formats" in the top right, then "Download source"; rename the downloaded file to end in `.tgz` if it doesn't already, and you should find the source there, including the equations to copy and paste.

```python
(m*m).sum().sqrt()
```

## Elementwise matmul

### [48:52](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2932) - how to use elementwise vector-vector multiplication to replace the last loop of scalar-scalar multiplication below, and how much faster  do we get (178 times); question: what does `%timeit -n 10` mean (doing `matmul(m1,m2` 10 times?); 




```python
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            # Any trailing ",:" can be removed
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c
```

```python
%timeit -n 10 _=matmul(m1, m2)
```

```python
890.1/5
```

### [50:59](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3059) - which language is this elementwise operation written in (c language); how does `test_near` and `torch.allclose` [check whether two numbers are real close to each other](https://pytorch.org/docs/stable/generated/torch.allclose.html) is not exact the same;


```python
#export
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)
```

```python
test_near(t1,matmul(m1, m2))
```

## Broadcasting

### [51:49](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3109) - how to get rid of the inner most loop now with broadcasting; what broadcasting does is to getting rid of all loops at the speed of Cuda written in C language; Where and when is broadcasting originated (APL in 1960s); What is this APL broadcasting (remove all the for loop and use implicit broadcasted loops)

The term **broadcasting** describes how arrays with different shapes are treated during arithmetic operations.  The term broadcasting was first used by Numpy.

From the [Numpy Documentation](https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html):

    The term broadcasting describes how numpy treats arrays with 
    different shapes during arithmetic operations. Subject to certain 
    constraints, the smaller array is “broadcast” across the larger 
    array so that they have compatible shapes. Broadcasting provides a 
    means of vectorizing array operations so that looping occurs in C
    instead of Python. It does this without making needless copies of 
    data and usually leads to efficient algorithm implementations.
    
In addition to the efficiency of broadcasting, it allows developers to write less code, which typically leads to fewer errors.

*This section was adapted from [Chapter 4](http://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra/blob/master/nbs/4.%20Compressed%20Sensing%20of%20CT%20Scans%20with%20Robust%20Regression.ipynb#4.-Compressed-Sensing-of-CT-Scans-with-Robust-Regression) of the fast.ai [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra) course.*


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=3110)


## Broadcasting with a scalar
### [52:58](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3178) - how to do broadcasting on a vector with a scalar or broadcasting a scalar to a tensor `a` which can be a vector or matrix  or more; and broadcasting is at speed of C or cuda;

```python
a
```

```python
a > 0
```

How are we able to do a > 0?  0 is being **broadcast** to have the same dimensions as a.

For instance you can normalize our dataset by subtracting the mean (a scalar) from the entire data set (a matrix) and dividing by the standard deviation (another scalar), using broadcasting.

Other examples of broadcasting with a scalar:

```python
a + 1
```

```python
m
```

```python
2*m
```

## Broadcasting a vector to a matrix
### [54:10](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3250) - how to broadcast a vector to matrix; Jeremy explains how to broadcast a vector to matrix without doing for loop; how to visualize a vector being broadcasted into a matrix using `t = c.expand_as(m)` (`c` as a data of column or row, `m` as matrix)



We can also broadcast a vector to a matrix:

```python
c = tensor([10.,20,30]); c
```

```python
m
```

```python
m.shape,c.shape
```

```python
m + c
```

```python
c + m
```

We don't really copy the rows, but it looks as if we did. In fact, the rows are given a *stride* of 0.

```python
t = c.expand_as(m)
```

```python
t
```

```python
m + t
```

### [55:51](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3351) - When broadcasting a vector to matrix, the vector is acting as a matrix but stored as a vector; how do we see this or interpret this using `t.storage()` and `t.stride()` and `t.shape`; 



```python
t.storage()
```

```python
t.stride(), t.shape
```

### [57:05](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3425) - how to turn a vector or 1d array into a 2d array or matrix using `c.unsqueeze(0).shape` (1, 3) or `c[None,:].shape` (1,3) or `c.unsqueeze(1).shape` (3,1) or `c[:,None].shape` (3,1), and turn a 1d array into a 3d array using `c[None,None,:]`(1,1,3) or `c[None,:,None]`(1,3,1); we use `None` over `unsqueeze`


You can index with the special value [None] or use `unsqueeze()` to convert a 1-dimensional array into a 2-dimensional array (although one of those dimensions has value 1).

```python
c.unsqueeze(0)
```

```python
c.unsqueeze(1)
```

```python
m
```

```python
c.shape, c.unsqueeze(0).shape,c.unsqueeze(1).shape
```

```python
c.shape, c[None].shape,c[:,None].shape
```

### [59:26](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3566) - using `None` or `unsqueeze`, `c + m` is the same to `c[None,:] + m` but very different to `c[:,None] + m` [1:00:25](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3625) - how to make sense/visualize of the broadcasting of `c[None]` and `c[...,None]` using excel

You can always skip trailling ':'s. And '...' means '*all preceding dimensions*'

```python
c[None].shape,c[...,None].shape
```

```python
c[:,None].expand_as(m)
```

```python
m + c[:,None]
```

```python
c[:,None]
```

## Matmul with broadcasting
### [1:02:05](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3725) - how to simplify the type of `c[None,:,:]` as `c[None]` and simplify `c[:,:,None]` as `c[...,None]` 

### [1:03:37](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3817) - how to write the broadcasting code to replace the for loop and how to thoroughly understand the code; what are the benefits of using broadcasting over many for loops (3700 times faster, less code less loops less error)


```python
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): # previously we have these two lines
            # Any trailing ",:" can be removed
            c[i,j] = (a[i,:] * b[:,j]).sum() # elementwise operation: row x column (both have same length)
    return c

def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        c[i]   = (a[i  ].unsqueeze(-1) * b).sum(dim=0) # the right side gives a row of summed values by smashing the tensor into one row
    return c
```

### how to understand the second inner most loop is replaced by broadcasting (homework assigned by Jeremy) I have written the following code blocks to understand it.

```python
from fastdebug.utils import *
from torch import tensor

a = tensor([[1,2,3], [4,5,6], [7,8,9]])

a
a.shape
```

```python
a[0]
a[0].shape
a[0,:].shape
a[0,...].shape
```

```python
a[0][None,:] # 2 dimensions, row dimension set as None, meaning 1 row
a[0][None,:].shape
a[0][None].shape # it can be 1 row, with unknown more dimensions carrying on e.g., [1,3,,,]
a[0, None].shape # even more simplified
a[0].unsqueeze(0).shape # make one row, but more columns
a[0,None].expand_as(a) # expand more rows 
a
a[0][None] * a
(a[0][None] * a).sum(dim=0) # sum up so that (3,3) tensor is smashed into a (3) tensor row
(a[0][None] * a).sum(dim=0).shape
(a[0][None] * a).sum(dim=1) # sum up so that (3,3) tensor is smashed into a (3) tensor column
(a[0][None] * a).sum(dim=1).shape
```

```python
a[0][:,None] # two dimensions, col dimension set None meaning 1 column
a[0][:,None].shape
a[0][...,None].shape
a[0].unsqueeze(1).shape
a[0].unsqueeze(-1).shape
a[0][:,None].expand_as(a)
a
a[0][:,None] * a
(a[0][:,None] * a).sum(dim=0) # sum up so that (3,3) tensor is smashed into a (3) tensor row
(a[0][:,None] * a).sum(dim=0).shape
(a[0][:,None] * a).sum(dim=1) # sum up so that (3,3) tensor is smashed into a (3) tensor column
(a[0][:,None] * a).sum(dim=1).shape
```

```python
%timeit -n 10 _=matmul(m1, m2)
```

```python
885000/277
```

```python
test_near(t1, matmul(m1, m2))
```

## Broadcasting Rules
### [1:06:21](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=3981) - How to understand broadcasting rules; two vector or matrix do some operations, we check their shapes side by side, e.g., `a.shape==[1,4,5] vs b.shape==[3,1,5]`, according to the 2 rules, a will broadcast to 3 rows, b will grow to 3 columns, and a and b change nothing on the 3rd dimention. and `(a*b).shape == [3,4,5]`


```python
c[None,:]
```

```python
c[None,:].shape
```

```python
c[:,None]
```

```python
c[:,None].shape
```

```python
c[None,:] * c[:,None]
```

```python
c[None] > c[:,None]
```

### Here is my own code for understanding the 2 rules of broadcasting

```python
c = tensor([1,2,3])
c
c.shape
c[:].shape
c[...].shape
```

```python
c[None,:]
c[None,:].shape
c[None,:,].shape
c[None].shape
c[None,None,:].shape
c[None,None].shape
```

```python
c[:,None]
c[:,None].shape
c[:,None].shape
c[:,None,None].shape 
```

```python
c[None,:].shape
c[:,None].shape
(c[None,:] * c[:,None]).shape
```

```python
c[None,:,None].shape
c[:,None,None].shape
(c[None,:,None] * c[:,None,None]).shape
```

```python
a = tensor([[1,2,3],[4,5,6],[7,8,9],[9,8,7]])
a.shape
a[None].shape
a[:,None].shape
(a[None] * a[:,None]).shape
```

When operating on two arrays/tensors, Numpy/PyTorch compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are **compatible** when

- they are equal, or
- one of them is 1, in which case that dimension is broadcasted to make it the same size

Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

    Image  (3d array): 256 x 256 x 3
    Scale  (1d array):             3
    Result (3d array): 256 x 256 x 3

The [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) includes several examples of what dimensions can and can not be broadcast together.

### [1:10:02](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4202) - why the broadcasting trick is the most important technique in creating fastai from scratch


## Einstein summation

### [1:10:43](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4243) - how to understand Einstein summation and how to use `torch.einsum` to do matrix multiplication with no loop at all and speed up 16000 times faster than pure python 3-for loop version; how to trick `torch.einsum` to do batch matrix multiplication and even more transforms and tweaks



Einstein summation (`einsum`) is a compact representation for combining products and sums in a general way. From the numpy docs:

"The subscripts string is a comma-separated list of subscript labels, where each label refers to a dimension of the corresponding operand. Whenever a label is repeated it is summed, so `np.einsum('i,i', a, b)` is equivalent to `np.inner(a,b)`. If a label appears only once, it is not summed, so `np.einsum('i', a)` produces a view of a with no changes."


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=4280)

```python
# c[i,j] += a[i,k] * b[k,j]
# c[i,j] = (a[i,:] * b[:,j]).sum()
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
def batch_matmul(a,b): return torch.einsum('bik,bkj->bij', a, b)
```

```python
%timeit -n 10 _=matmul(m1, m2)
```

```python
885000/55
```

```python
test_near(t1, matmul(m1, m2))
```

### [1:15:48](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4548) - what Jeremy does not like about `torch.einsum` and why APL, J and K are so great and what to expect from swift compiler, Julia



## pytorch op
### [1:18:23](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4703) - `torch.matmul` can do matrix multiplication without loops and 50000 times faster than the pure python 3-for loops version; but the reason why `torch.matmul` is so much faster is because it uses a lib like BLAS written by Nvdia (cuBLAS) or AMD or Intel (MKL)which split the large matricies into smaller ones and doing calc without using up all the ram; what are the problems of using these gpu libraries like MKL and cuBLAS;


### [1:21:48](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4908) - `torch.matmul` and `@` are the same thing, but they can handle [a lot more](https://pytorch.org/docs/stable/generated/torch.matmul.html) including batch matrix multiplication


We can use pytorch's function or operator directly for matrix multiplication.


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=4702)

```python
%timeit -n 10 t2 = m1.matmul(m2)
```

```python
# time comparison vs pure python:
885000/18
```

```python
t2 = m1@m2
```

```python
test_near(t1, t2)
```

```python
m1.shape,m2.shape
```

### [1:22:33](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4953) - What to do next; after having matrix multiplication fast enough, we need to initialize weights and biases, then create ReLU, then backward



## Export

```python
!python notebook2script.py 01_matmul.ipynb
```

```python

```
