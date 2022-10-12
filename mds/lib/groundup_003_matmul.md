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

# groundup_003_matmul

```python
#| default_exp delete0002
```

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

## Matrix multiplication from foundations


The *foundations* we'll assume throughout this course are:

- Python
- matplotlib
- The Python standard library
- Jupyter notebooks and nbdev


## imports

```python
from fastdebug.utils import *
from fastdebug.core import *
```

```python
from fastdebug.groundup import *
```

```python
from pathlib import Path
import pickle, gzip, math, os, time, shutil, matplotlib as mpl, matplotlib.pyplot as plt
```

## get_exp_data

```python
MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
path_data = Path('data')
path_data
```

```python
path_data.mkdir(exist_ok=True) # created a data folder in the current directory
```

```python
path_gz = path_data/'mnist.pkl.gz'
path_gz
```

[urlretrieve](https://docs.python.org/3/library/urllib.request.html#urllib.request.urlretrieve) - (read the docs!)

```python
from urllib.request import urlretrieve
```

```python
check(urlretrieve)
```

```python
if not path_gz.exists(): urlretrieve(MNIST_URL, path_gz)
```

```python
!ls -l data
```

```python
with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
```

```python
#| export
a = "todelete"
```

```python
#| export groundup
def get_exp_data():
    MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
    path_data = Path('data')
    path_data.mkdir(exist_ok=True) # created a data folder in the current directory
    path_gz = path_data/'mnist.pkl.gz'
    from urllib.request import urlretrieve
    if not path_gz.exists(): urlretrieve(MNIST_URL, path_gz)
    with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return x_train, y_train, x_valid, y_valid
```

```python
x_train, y_train, x_valid, y_valid = get_exp_data()
```

```python
x_train[0].shape
x_train[0].size
type(x_train[0])
```

### range, yield, chunks

```python
lst1 = list(x_train[0])
vals = lst1[200:210]
vals
```

```python
#| export groundup
def chunks(x, sz):
    for i in range(0, len(x), sz): 
        print(i)
        yield x[i:i+sz]
```

```python
vals
list(chunks(vals, 5))
```

```python
def chunks(x, sz):
    for i in range(0, len(x), sz): yield x[i:i+sz]
```

```python
type(chunks(lst1, 28))
```

```python
img = list(chunks(lst1, 28))
len(img)
```

```python
check(plt.imshow)
```

```python
mpl.rcParams['image.cmap'] = 'gray'
plt.imshow(list(chunks(lst1, 28)));
```

### [islice](https://docs.python.org/3/library/itertools.html#itertools.islice)

```python
from itertools import islice
```

```python
islice.__class__
```

```python
help(islice)
```

```python
vals
len(vals)
```

```python
it = iter(vals)
islice(it, 5)
```

```python
list(islice(it, 5))
```

```python
list(islice(it, 5))
```

```python
list(islice(it, 5))
```

```python
check(iter)
```

### islice, iter, chunks_faster
why using `islice` and `iter` over `chunks`

```python
%timeit -n 10 it = iter(lst1)
```

```python
%timeit -n 10 img = list(iter(lambda: list(islice(it, 28)), []))
```

```python
len(img)
```

```python
def chunks(x, sz):
    for i in range(0, len(x), sz): yield x[i:i+sz]
```

```python
%timeit -n 10 img = list(chunks(lst1, 28))
```

```python
len(img)
```

```python
type(x_train[0])
x_train[0].shape
x_train[0].size
```

```python
#| export groundup
def chunks_faster(x, sz):
    "if the data is numpy.ndarray and shape is 1 dimension, then we use chunks to make it a pseudo 2d"
    lst = list(x)
    it = iter(lst)
    img = list(iter(lambda: list(islice(it, sz)), []))
    print(f'len: {len(img)}')
    return img
```

```python
img = chunks_faster(x_train[0], 28)
```

```python
plt.imshow(img);
```

## Matrix and tensor


### list and Matrix

```python
type(img)
```

```python
img[20][15]
```

```python
#| export groundup
class Matrix:
    "turning a list of list into a maxtrix like object"
    def __init__(self, xs): self.xs = xs
    def __getitem__(self, idxs): return self.xs[idxs[0]][idxs[1]]
```

```python
m = Matrix(img)
m[20,15]
type(m)
```

### tensor, map, np.array

```python
import torch
from torch import tensor
```

```python
tensor([1,2,3])
```

```python
type(x_train)
x_train.shape
```

```python
check(tensor)
```

```python
x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
x_train.shape
```

### tensor.type, tensor.reshape

```python
check(x_train.type)
```

```python
x_train.type()
```

```python
check(x_train.reshape)
```

```python
check(torch.reshape)
```

```python
%whos Tensor
```

```python
imgs = x_train.reshape((-1,28,28))
```

```python
imgs.shape
```

```python
plt.imshow(imgs[0]);
```

```python
imgs[0,20,15]
```

### torch.shape

```python
check(x_train.shape)
```

```python
check(torch.Size)
```

```python
x_train.shape
n,c = x_train.shape
n,c
```

```python
y_train, y_train.shape
```

```python
min(y_train),max(y_train)
```

```python
y_train.min(), y_train.max()
```

```python

```

## Random numbers


Based on the Wichmann Hill algorithm used before Python 2.3.


### divmod, seed, rand
Create your own random number between 0 and 1

```python
divmod(10, 3)
```

```python
rnd_state = None
def seed(a):
    global rnd_state
    a, x = divmod(a, 30268)
    a, y = divmod(a, 30306)
    a, z = divmod(a, 30322)
    rnd_state = int(x)+1, int(y)+1, int(z)+1
```

```python
seed(457428938475)
rnd_state
```

```python
5%2
5%3
```

```python
#| export groundup
def rand():
    "create a random number between 0 and 1"
    global rnd_state
    x, y, z = rnd_state
    x = (171 * x) % 30269
    y = (172 * y) % 30307
    z = (170 * z) % 30323
    rnd_state = x,y,z
    return (x/30269 + y/30307 + z/30323) % 1.0
```

```python
rand(),rand(),rand()
```

```python
check(os.fork)
```

```python
check(os._exit)
```

```python
check(os.EX_OK)
```

```python
if os.fork(): print(f'In parent: {rand()}')
else:
    print(f'In child: {rand()}')
    os._exit(os.EX_OK)
```

```python
if os.fork(): print(f'In parent: {torch.rand(1)}')
else:
    print(f'In child: {torch.rand(1)}')
    os._exit(os.EX_OK)
```

```python
plt.plot([rand() for _ in range(50)]);
```

```python
plt.hist([rand() for _ in range(10000)]);
```

### torch.randn
much faster than rand from scratch

```python
check(torch.randn)
```

```python
%timeit -n 10 list(chunks([rand() for _ in range(7840)], 10))
```

```python
%timeit -n 10 torch.randn(784,10)
```

```python
rd = torch.randn(1, 1,784,10)
rd.shape
rd[:5].shape
```

## Matrix multiplication


### matmul_3loops, torch.zeros

```python
weights = torch.randn(784,10)
bias = torch.zeros(10)
```

```python
weights[0,:]
len(weights[0,:])
bias
len(bias)
```

```python
m1 = x_valid[:5] # as input
m2 = weights # as layer1 weights
```

```python
m1.shape,m2.shape
```

```python
ar,ac = m1.shape # n_rows * n_cols
br,bc = m2.shape
(ar,ac),(br,bc)
```

```python
t1 = torch.zeros(ar, bc)
t1.shape
```

```python
for i in range(ar):         # 5
    for j in range(bc):     # 10
        for k in range(ac): # 784
            t1[i,j] += m1[i,k] * m2[k,j]
```

```python
t1
```

```python
t1.shape
```

```python
#| export groundup
def matmul_3loops(a, b):
    (ar,ac),(br,bc) = a.shape,b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar): # ar == 5
        for j in range(bc): # bc == 10
            for k in range(ac): c[i,j] += a[i,k] * b[k,j] # ac == 784

    print(f'shapes => a: {a.shape}, b: {b.shape}, res: {c.shape}')
    return c
```

```python
matmul_3loops(m1,m2)
```

### torch.set_printoptions, np.set_printoptions

```python
import numpy as np
```

```python
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
np.set_printoptions(precision=2, linewidth=140)
t1
```

```python
%time _=matmul_3loops(m1, m2)
```

```python

```

## Numba
Numba is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.


### njit, dot, np.array

```python
import numba
```

```python
check(numba)
```

```python
whatinside(numba)
```

```python
from numba import njit, jit
```

```python
check(njit)
```

```python
check(jit)
```

```python
#| export groundup
@njit
def dot(a,b):
    res = 0.
    for i in range(len(a)): res+=a[i]*b[i]
    return res
```

```python
from numpy import array
```

```python
%time dot(array([1.,2,3]),array([2.,3,4]))
```

```python
%time dot(array([1.,2,3]),array([2.,3,4]))
```

```python
%time dot(array([1.,2,3]),array([2.,3,4]))
```

### matmul_2loops_njit, m1.numpy()


Now only two of our loops are running in Python, and the third loop is running in machine code

```python
#| export groundup
def matmul_2loops_njit(a,b):
    "doing matrix multiplication with 2 python loops and 1 loop in machine code"
    a,b = a.numpy(),b.numpy() # njit or numba don't work with torch.tensor but numpy array
    (ar,ac),(br,bc) = a.shape,b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): c[i,j] = dot(a[i,:], b[:,j])
    return c
```

```python
matmul_2loops_njit(m1,m2)
```

```python
check(m1.numpy)
```

```python
m1a,m2a = m1.numpy(),m2.numpy()
```

```python
m1.shape, m1a.shape
```

### test_close, %timeit, %time

```python
from fastcore.test import *
```

```python
test_close(matmul_3loops(m1, m2),matmul_2loops_njit(m1, m2))
```

```python
%timeit -n 10 matmul_2loops_njit(m1,m2)
```

```python
%time _=matmul_2loops_njit(m1,m2)
```

```python

```

## Elementwise ops


[TryAPL](https://tryapl.org/)


### elementwise by tensor

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

<!-- #region -->
Frobenius norm:

$$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$

*Hint*: you don't normally need to write equations in LaTeX yourself, instead, you can click 'edit' in Wikipedia and copy the LaTeX from there (which is what I did for the above equation). Or on arxiv.org, click "Download: Other formats" in the top right, then "Download source"; rename the downloaded file to end in `.tgz` if it doesn't already, and you should find the source there, including the equations to copy and paste. This is the source LaTeX that I pasted to render the equation above:

```latex
$$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$
```
<!-- #endregion -->

```python
(m*m).sum().sqrt()
```

```python
check(m.sum)
```

```python
check(torch.sum)
```

### matmul_2loops_elementwise

```python
#| export groundup
def matmul_2loops_elementwise(a,b):
    (ar,ac),(br,bc) = a.shape,b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): c[i,j] = (a[i,:] * b[:,j]).sum()
    return c
```

```python
test_close(matmul_3loops(m1,m2), matmul_2loops_elementwise(m1, m2))
```

```python
%time _=matmul_3loops(m1, m2)
```

```python
%time _=matmul_2loops_njit(m1, m2)
```

```python
%time _=matmul_2loops_elementwise(m1, m2)
```

### matmul_2loops_dotproduct

```python
#| export groundup
def matmul_2loops_dotproduct(a,b):
    (ar,ac),(br,bc) = a.shape,b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc): c[i,j] = torch.dot(a[i,:], b[:,j])
    return c
```

```python
test_close(t1,matmul_2loops_dotproduct(m1, m2))
```

```python
%timeit -n 10 _=matmul_2loops_dotproduct(m1, m2)
```

```python
%time _=matmul_2loops_njit(m1, m2)
```

```python
%time _=matmul_2loops_elementwise(m1, m2)
```

```python
%time _=matmul_2loops_dotproduct(m1, m2)
```

```python

```

## Broadcasting


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


### Broadcasting with a scalar
How a scalar is broadcasted to do operation with a matrix

```python
a
```

```python
a > 0
```

How are we able to do `a > 0`?  0 is being **broadcast** to have the same dimensions as a.

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

### Broadcasting a vector to a matrix


Although broadcasting a scalar is an idea that dates back to APL, the more powerful idea of broadcasting across higher rank tensors [comes from](https://mail.python.org/pipermail/matrix-sig/1995-November/000143.html) a little known language called [Yorick](https://software.llnl.gov/yorick-doc/manual/yorick_50.html).

We can also broadcast a vector to a matrix, when the vector's shape is a scalar

```python
c = tensor([10.,20,30]); 
c
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

#### c.expand_as(m), t.storage(), t.stride()


We don't really copy the rows, but it looks as if we did. In fact, the rows are given a *stride* of 0.

```python
check(c.expand_as)
```

```python
t = c.expand_as(m)
```

```python
t
```

```python
m + t
```

```python
check(t.storage)
```

```python
t.storage()
```

```python
check(t.stride)
```

```python
t.stride(), t.shape
```

### c.unsqueeze(0), c[None,:], c.unsqueeze(1), c[:, None]


You can index with the special value [None] or use `unsqueeze()` to convert a 1-dimensional array into a 2-dimensional array (although one of those dimensions has value 1).

```python
c
c.shape
```

```python
c.unsqueeze(0), c[None, :]
```

```python
c.unsqueeze(0).shape
c[None, :].shape
```

```python
c.unsqueeze(1), c[:, None]
```

```python
c.shape
c.unsqueeze(1).shape
c[:, None].shape
```

#### c[None], c[..., None]
You can always skip trailling ':'s. And '...' means '*all preceding dimensions*'

```python
c[None].shape,c[...,None].shape
```

### Broadcast a one-row/col matrix

```python
# c[None,:].expand_as(m)
c[None].expand_as(m)
```

```python
c[:,None].expand_as(m)
```

```python
m + c[:,None]
```

```python
m + c[None,:]
```

### Broadcasting Rules
how to broadcast two 1-row/col matricies

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

When operating on two arrays/tensors, Numpy/PyTorch compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are **compatible** when

- they are equal, or
- one of them is 1, in which case that dimension is broadcasted to make it the same size

Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

    Image  (3d array): 256 x 256 x 3
    Scale  (1d array):             3
    Result (3d array): 256 x 256 x 3


### Practice Broadcasting
Very helpful [examples](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) to solidify my practical understanding of Broadcasting.


```python

```

```python

```

## Matmul with broadcasting


### matmul_1loop_broadcast

```python
rowZero = m1[0]
rowZero.shape, m2.shape
```

```python
rowZero[:,None].shape # make rowZero flip from horizontal to vertical
```

```python
rowZero[:,None].expand_as(m2).shape # broadcast from one column to 10 columns to match m2
```

```python
(rowZero[:,None]*m2).shape # (broadcast vertically) vector * matrix 
```

```python
(rowZero[:,None]*m2).sum(dim=0) 
# dim=0, smash vertically to the ground so that we just have one row as output
# dim=1, smash horizontal to the left/right so that we just have one column as output
```

```python
#| export groundup
def matmul_1loop_broadcast(a,b):
    (ar,ac),(br,bc) = a.shape,b.shape
    c = torch.zeros(ar, bc)
    for i in range(ar):
        c[i]   = (a[i,:,None] * b).sum(dim=0) # broadcast version
    return c
```

```python
test_close(matmul_3loops(m1,m2),matmul_1loop_broadcast(m1, m2))
```

```python
%timeit -n 10 _=matmul_1loop_broadcast(m1, m2)
```

```python
%time _=matmul_2loops_njit(m1,m2)
%time _=matmul_2loops_elementwise(m1,m2)
%time _=matmul_2loops_dotproduct(m1,m2)
%time _=matmul_1loop_broadcast(m1,m2)
```

### matmul on x_train and weights


Our time has gone from ~500ms to <0.1ms, an over 5000x improvement! We can run on the whole dataset now.

```python
tr = matmul_1loop_broadcast(x_train, weights)
```

```python
tr.shape
```

```python
%time _=matmul_2loops_njit(x_train, weights)
%time _=matmul_2loops_elementwise(x_train, weights)
%time _=matmul_2loops_dotproduct(x_train, weights)
%time _=matmul_1loop_broadcast(x_train, weights)
```

```python

```

## matmul_einsum_noloop


[Einstein summation](https://ajcr.net/Basic-guide-to-einsum/) ([`einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)) is a compact representation for combining products and sums in a general way. The key rules are:

- Repeating letters between input arrays means that values along those axes will be multiplied together.
- Omitting a letter from the output means that values along that axis will be summed.

```python
#| export groundup
def matmul_einsum_noloop(a,b): return torch.einsum('ik,kj->ij', a, b)
# c[i,j] += a[i,k] * b[k,j]
# c[i,j] = (a[i,:] * b[:,j]).sum()
```

```python
test_close(tr, matmul_einsum_noloop(x_train, weights), eps=1e-3) # question: why 1e-3
```

```python
%time _=matmul_2loops_njit(x_train, weights)
%time _=matmul_2loops_elementwise(x_train, weights)
%time _=matmul_2loops_dotproduct(x_train, weights)
%time _=matmul_1loop_broadcast(x_train, weights)
%time _=matmul_einsum_noloop(x_train, weights)
```

```python

```

## torch.matmul or x_train@weights


We can use pytorch's function or operator directly for matrix multiplication.

```python
test_close(tr, x_train@weights, eps=1e-3)
```

```python
%time _=matmul_2loops_njit(x_train, weights)
%time _=matmul_2loops_elementwise(x_train, weights)
%time _=matmul_2loops_dotproduct(x_train, weights)
%time _=matmul_1loop_broadcast(x_train, weights)
%time _=matmul_einsum_noloop(x_train, weights)
%time _=torch.matmul(x_train, weights)
```

```python

```

## CUDA

```python
def matmul(grid, a,b,c):
    i,j = grid 
    if i < c.shape[0] and j < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]): tmp += a[i, k] * b[k, j]  # this is the 3rd loop
        c[i,j] = tmp
```

```python
res = torch.zeros(ar, bc)
matmul((0,0), m1, m2, res)
res
```

```python
def launch_kernel(kernel, grid_x, grid_y, *args, **kwargs):
    for i in range(grid_x): # the 1st loop
        for j in range(grid_y): kernel((i,j), *args, **kwargs) # the 2nd loop
```

```python
res = torch.zeros(ar, bc)
launch_kernel(matmul, ar, bc, m1, m2, res)
res
```

```python
from numba import cuda
```

```python
@cuda.jit
def matmul(a,b,c):
    "the 3rd loop from python to machine code"
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]): tmp += a[i, k] * b[k, j]
        c[i,j] = tmp
    cuda.syncthreads()
```

```python
tr.shape
```

```python
r = np.zeros(tr.shape) # prepare dataset in cuda
m1g,m2g,rg = cuda.to_device(x_train),cuda.to_device(weights),cuda.to_device(r)
```

```python
r.shape
```

```python
TPB = 16
rr,rc = r.shape
blockspergrid = (math.ceil(rr / TPB), math.ceil(rc / TPB))
blockspergrid
```

```python
matmul[blockspergrid, (TPB,TPB)](m1g,m2g,rg)
r = rg.copy_to_host()
test_close(tr, r, eps=1.03)
```

```python
%%timeit -n 1
matmul[blockspergrid, (TPB,TPB)](m1g,m2g,rg)
r = rg.copy_to_host()
```

```python
m1c,m2c = x_train.cuda(),weights.cuda()
```

```python
%timeit -n 1 r==(m1c@m2c).cpu()
```

Our broadcasting version was >500ms, and our CUDA version is around 0.5ms, which is another 1000x improvement compared to broadcasting. So our total speedup is around 5 million times!

```python

```

```python

```

```python
# |hide
import nbdev
nbdev.nbdev_export()
```

```python

```
