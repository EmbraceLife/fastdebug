```python

```

# groundup_003_matmul


```python
#| default_exp delete0002
```
---
skip_exec: true
---
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


<style>.container { width:100% !important; }</style>



```python
# from fastdebug.groundup import *
```


```python
#| export groundup
from pathlib import Path
import pickle, gzip, math, os, time, shutil, matplotlib as mpl, matplotlib.pyplot as plt
```

## get_exp_data


```python
MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
path_data = Path('data')
path_data
```




    PosixPath('data')




```python
path_data.mkdir(exist_ok=True) # created a data folder in the current directory
```


```python
path_gz = path_data/'mnist.pkl.gz'
path_gz
```




    PosixPath('data/mnist.pkl.gz')



[urlretrieve](https://docs.python.org/3/library/urllib.request.html#urllib.request.urlretrieve) - (read the docs!)


```python
from urllib.request import urlretrieve
```


```python
check(urlretrieve)
```

    signature: (url, filename=None, reporthook=None, data=None)
    __class__: <class 'function'>
    __repr__: <function urlretrieve>
    
    __doc__:
    Retrieve a URL into a temporary location on disk.
    
    Requires a URL argument. If a filename is passed, it is used as
    the temporary file location. The reporthook argument should be
    a callable that accepts a block number, a read size, and the
    total file size of the URL target. The data argument should be
    valid URL encoded data.
    
    If a filename is passed and the URL points to a local resource,
    the result is a copy from local file to new file.
    
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: True
    method: False



```python
if not path_gz.exists(): urlretrieve(MNIST_URL, path_gz)
```


```python
!ls -l data
```

    total 33312
    -rw-r--r--  1 Natsume  staff  17051982 Oct 11 18:46 mnist.pkl.gz



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
    from pathlib import Path
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




    (784,)






    784






    numpy.ndarray



### range, yield, chunks


```python
lst1 = list(x_train[0])
vals = lst1[200:210]
vals
```




    [0.0,
     0.0,
     0.0,
     0.19140625,
     0.9296875,
     0.98828125,
     0.98828125,
     0.98828125,
     0.98828125,
     0.98828125]




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




    [0.0,
     0.0,
     0.0,
     0.19140625,
     0.9296875,
     0.98828125,
     0.98828125,
     0.98828125,
     0.98828125,
     0.98828125]



    0
    5





    [[0.0, 0.0, 0.0, 0.19140625, 0.9296875],
     [0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125]]




```python
def chunks(x, sz):
    for i in range(0, len(x), sz): yield x[i:i+sz]
```


```python
type(chunks(lst1, 28))
```




    generator




```python
img = list(chunks(lst1, 28))
len(img)
```




    28




```python
check(plt.imshow)
```

    signature: (X, cmap=None, norm=None, *, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None, data=None, **kwargs)
    __class__: <class 'function'>
    __repr__: <function imshow>
    
    __doc__:
    Display data as an image, i.e., on a 2D regular raster.
    
    The input may either be actual RGB(A) data, or 2D scalar data, which
    will be rendered as a pseudocolor image. For displaying a grayscale
    image set up the colormapping using the parameters
    ``cmap='gray', vmin=0, vmax=255``.
    
    The number of pixels used to render an image is set by the Axes size
    and the *dpi* of the figure. This can lead to aliasing artifacts when
    the image is resampled because the displayed image size will usually
    not match the size of *X* (see
    __dict__: 
    {'__signature__': <Signature (X, cmap=None, norm=None, *, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None, data=None, **kwargs)>,
     '__wrapped__': <function imshow>}
    metaclass: False
    class: False
    decorator: False
    function: True
    method: False



```python
mpl.rcParams['image.cmap'] = 'gray'
plt.imshow(list(chunks(lst1, 28)));
```


    
![png](groundup_003_matmul_files/groundup_003_matmul_32_0.png)
    


### [islice](https://docs.python.org/3/library/itertools.html#itertools.islice)


```python
from itertools import islice
```


```python
islice.__class__
```




    type




```python
help(islice)
```

    Help on class islice in module itertools:
    
    class islice(builtins.object)
     |  islice(iterable, stop) --> islice object
     |  islice(iterable, start, stop[, step]) --> islice object
     |  
     |  Return an iterator whose next() method returns selected values from an
     |  iterable.  If start is specified, will skip all preceding elements;
     |  otherwise, start defaults to zero.  Step defaults to one.  If
     |  specified as another value, step determines how many values are
     |  skipped between successive calls.  Works like a slice() on a list
     |  but returns an iterator.
     |  
     |  Methods defined here:
     |  
     |  __getattribute__(self, name, /)
     |      Return getattr(self, name).
     |  
     |  __iter__(self, /)
     |      Implement iter(self).
     |  
     |  __next__(self, /)
     |      Implement next(self).
     |  
     |  __reduce__(...)
     |      Return state information for pickling.
     |  
     |  __setstate__(...)
     |      Set state information for unpickling.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from builtins.type
     |      Create and return a new object.  See help(type) for accurate signature.
    



```python
vals
len(vals)
```




    [0.0,
     0.0,
     0.0,
     0.19140625,
     0.9296875,
     0.98828125,
     0.98828125,
     0.98828125,
     0.98828125,
     0.98828125]






    10




```python
it = iter(vals)
islice(it, 5)
```




    <itertools.islice>




```python
list(islice(it, 5))
```




    [0.0, 0.0, 0.0, 0.19140625, 0.9296875]




```python
list(islice(it, 5))
```




    [0.98828125, 0.98828125, 0.98828125, 0.98828125, 0.98828125]




```python
list(islice(it, 5))
```




    []




```python
check(iter)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in function iter>
    
    __doc__:
    iter(iterable) -> iterator
    iter(callable, sentinel) -> iterator
    
    Get an iterator from an object.  In the first form, the argument must
    supply its own iterator, or be a sequence.
    In the second form, the callable is called until it returns the sentinel.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False


### islice, iter, chunks_faster
why using `islice` and `iter` over `chunks`


```python
%timeit -n 10 it = iter(lst1)
```

    54.2 ns ± 17.1 ns per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit -n 10 img = list(iter(lambda: list(islice(it, 28)), []))
```

    449 ns ± 151 ns per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
len(img)
```




    28




```python
def chunks(x, sz):
    for i in range(0, len(x), sz): yield x[i:i+sz]
```


```python
%timeit -n 10 img = list(chunks(lst1, 28))
```

    3.48 µs ± 189 ns per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
len(img)
```




    28




```python
type(x_train[0])
x_train[0].shape
x_train[0].size
```




    numpy.ndarray






    (784,)






    784




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

    len: 28



```python
plt.imshow(img);
```


    
![png](groundup_003_matmul_files/groundup_003_matmul_53_0.png)
    


## Matrix and tensor

### list and Matrix


```python
type(img)
```




    list




```python
img[20][15]
```




    0.98828125




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




    0.98828125






    __main__.Matrix



### tensor, map, np.array


```python
import torch
from torch import tensor
```


```python
tensor([1,2,3])
```




    tensor([1, 2, 3])




```python
type(x_train)
x_train.shape
```




    numpy.ndarray






    (50000, 784)




```python
check(tensor)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method tensor of type object>
    
    __doc__:
    tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor
    
    Constructs a tensor with no autograd history (also known as a "leaf tensor", see :doc:`/notes/autograd`) by copying :attr:`data`.
    
    .. warning::
    
        When working with tensors prefer using :func:`torch.Tensor.clone`,
        :func:`torch.Tensor.detach`, and :func:`torch.Tensor.requires_grad_` for
        readability. Letting `t` be a tensor, ``torch.tensor(t)`` is equivalent to
        ``t.clone().detach()``, and ``torch.tensor(t, requires_grad=True)``
        is equivalent to ``t.clone().detach().requires_grad_(True)``.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
x_train.shape
```




    torch.Size([50000, 784])



### tensor.type, tensor.reshape


```python
check(x_train.type)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method type of Tensor object>
    
    __doc__:
    type(dtype=None, non_blocking=False, **kwargs) -> str or Tensor
    Returns the type if `dtype` is not provided, else casts this object to
    the specified type.
    
    If this is already of the correct type, no copy is performed and the
    original object is returned.
    
    Args:
        dtype (dtype or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
x_train.type()
```




    'torch.FloatTensor'




```python
check(x_train.reshape)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method reshape of Tensor object>
    
    __doc__:
    reshape(*shape) -> Tensor
    
    Returns a tensor with the same data and number of elements as :attr:`self`
    but with the specified shape. This method returns a view if :attr:`shape` is
    compatible with the current shape. See :meth:`torch.Tensor.view` on when it is
    possible to return a view.
    
    See :func:`torch.reshape`
    
    Args:
        shape (tuple of ints or int...): the desired shape
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
check(torch.reshape)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method reshape of type object>
    
    __doc__:
    reshape(input, shape) -> Tensor
    
    Returns a tensor with the same data and number of elements as :attr:`input`,
    but with the specified shape. When possible, the returned tensor will be a view
    of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs
    with compatible strides can be reshaped without copying, but you should not
    depend on the copying vs. viewing behavior.
    
    See :meth:`torch.Tensor.view` on when it is possible to return a view.
    
    A single dimension may be -1, in which case it's inferred from the remaining
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
%whos Tensor
```

    Variable   Type      Data/Info
    ------------------------------
    x_train    Tensor    tensor([[0., 0., 0.,  ...<...>, 0.,  ..., 0., 0., 0.]])
    x_valid    Tensor    tensor([[0., 0., 0.,  ...<...>, 0.,  ..., 0., 0., 0.]])
    y_train    Tensor    tensor([5, 0, 4,  ..., 8, 4, 8])
    y_valid    Tensor    tensor([3, 8, 6,  ..., 5, 6, 8])



```python
imgs = x_train.reshape((-1,28,28))
```


```python
imgs.shape
```




    torch.Size([50000, 28, 28])




```python
plt.imshow(imgs[0]);
```


    
![png](groundup_003_matmul_files/groundup_003_matmul_74_0.png)
    



```python
imgs[0,20,15]
```




    tensor(0.9883)



### torch.shape


```python
check(x_train.shape)
```

    signature: None
    __class__: <class 'torch.Size'>
    __repr__: torch.Size([50000, 784])
    
    __doc__: not exist
    
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
check(torch.Size)
```

    signature: (iterable=(), /)
    __class__: <class 'type'>
    __repr__: <class 'torch.Size'>
    
    __doc__: not exist
    
    __dict__: 
    mappingproxy({'__add__': <slot wrapper '__add__' of 'torch.Size' objects>,
                  '__doc__': None,
                  '__getitem__': <slot wrapper '__getitem__' of 'torch.Size' objects>,
                  '__mul__': <slot wrapper '__mul__' of 'torch.Size' objects>,
                  '__new__': <built-in method __new__ of type object>,
                  '__reduce__': <method '__reduce__' of 'torch.Size' objects>,
                  '__repr__': <slot wrapper '__repr__' of 'torch.Size' objects>,
                  '__rmul__': <slot wrapper '__rmul__' of 'torch.Size' objects>,
                  'numel': <method 'numel' of 'torch.Size' objects>})
    metaclass: False
    class: True
    decorator: False
    function: False
    method: False



```python
x_train.shape
n,c = x_train.shape
n,c
```




    torch.Size([50000, 784])






    (50000, 784)




```python
y_train, y_train.shape
```




    (tensor([5, 0, 4,  ..., 8, 4, 8]), torch.Size([50000]))




```python
min(y_train),max(y_train)
```




    (tensor(0), tensor(9))




```python
y_train.min(), y_train.max()
```




    (tensor(0), tensor(9))




```python

```

## Random numbers

Based on the Wichmann Hill algorithm used before Python 2.3.

### divmod, seed, rand
Create your own random number between 0 and 1


```python
divmod(10, 3)
```




    (3, 1)




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




    (4976, 20238, 499)




```python
5%2
5%3
```




    1






    2




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




    (0.7645251082582081, 0.7920889799553945, 0.06912886811267205)




```python
check(os.fork)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in function fork>
    
    __doc__:
    Fork a child process.
    
    Return 0 to child process and PID of child to parent process.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
check(os._exit)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in function _exit>
    
    __doc__:
    Exit to the system with specified status, without normal exit processing.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
check(os.EX_OK)
```

    signature: None
    __class__: <class 'int'>
    __repr__: 0
    
    __doc__:
    int([x]) -> integer
    int(x, base=10) -> integer
    
    Convert a number or string to an integer, or return 0 if no arguments
    are given.  If x is a number, return x.__int__().  For floating point
    numbers, this truncates towards zero.
    
    If x is not a number or if base is given, then x must be a string,
    bytes, or bytearray instance representing an integer literal in the
    given base.  The literal can be preceded by '+' or '-' and be surrounded
    by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
if os.fork(): print(f'In parent: {rand()}')
else:
    print(f'In child: {rand()}')
    os._exit(os.EX_OK)
```

    In parent: 0.9559050644103264
    In child: 0.9559050644103264



```python
if os.fork(): print(f'In parent: {torch.rand(1)}')
else:
    print(f'In child: {torch.rand(1)}')
    os._exit(os.EX_OK)
```

    In parent: tensor([0.2364])
    In child: tensor([0.2364])



```python
plt.plot([rand() for _ in range(50)]);
```


    
![png](groundup_003_matmul_files/groundup_003_matmul_98_0.png)
    



```python
plt.hist([rand() for _ in range(10000)]);
```


    
![png](groundup_003_matmul_files/groundup_003_matmul_99_0.png)
    


### torch.randn
much faster than rand from scratch


```python
check(torch.randn)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method randn of type object>
    
    __doc__:
    randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    
    Returns a tensor filled with random numbers from a normal distribution
    with mean `0` and variance `1` (also called the standard normal
    distribution).
    
    .. math::
        \text{out}_{i} \sim \mathcal{N}(0, 1)
    
    The shape of the tensor is defined by the variable argument :attr:`size`.
    
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
%timeit -n 10 list(chunks([rand() for _ in range(7840)], 10))
```

    2.26 ms ± 26.2 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit -n 10 torch.randn(784,10)
```

    99.2 µs ± 4.28 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
rd = torch.randn(1, 1,784,10)
rd.shape
rd[:5].shape
```




    torch.Size([1, 1, 784, 10])






    torch.Size([1, 1, 784, 10])



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




    tensor([-1.3237,  1.1737,  0.3392,  0.2210,  0.6607,  1.8192,  0.5331,  1.0146,
             0.7724, -0.2601])






    10






    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])






    10




```python
m1 = x_valid[:5] # as input
m2 = weights # as layer1 weights
```


```python
m1.shape,m2.shape
```




    (torch.Size([5, 784]), torch.Size([784, 10]))




```python
ar,ac = m1.shape # n_rows * n_cols
br,bc = m2.shape
(ar,ac),(br,bc)
```




    ((5, 784), (784, 10))




```python
t1 = torch.zeros(ar, bc)
t1.shape
```




    torch.Size([5, 10])




```python
for i in range(ar):         # 5
    for j in range(bc):     # 10
        for k in range(ac): # 784
            t1[i,j] += m1[i,k] * m2[k,j]
```


```python
t1
```




    tensor([[ 13.9416,  -2.0733,   7.3798,   4.8941,   9.7936,  11.2293,  -5.6609,
               2.7170,   1.6722,   0.6240],
            [ -3.9407,   8.6393,   1.4139,   3.4698,  -0.4061,  -6.5177,  -2.3376,
               8.8883,   8.4269,   2.1366],
            [  3.9033,   5.0146,  12.8683,  -8.9402,  11.4366,  31.3686,   3.2794,
               2.2839,  11.1379,   0.6423],
            [  6.6159,   1.9588,   2.2625,  12.5532,   5.5336,   0.8998,   0.5374,
               7.7319,  -0.2104,   0.8249],
            [ -5.4877,  19.7717,  12.5163,  -2.7179,   7.1121,   5.9905,  -2.4888,
               6.5958,  22.2492, -10.1361]])




```python
t1.shape
```




    torch.Size([5, 10])




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

    shapes => a: torch.Size([5, 784]), b: torch.Size([784, 10]), res: torch.Size([5, 10])





    tensor([[ 13.9416,  -2.0733,   7.3798,   4.8941,   9.7936,  11.2293,  -5.6609,
               2.7170,   1.6722,   0.6240],
            [ -3.9407,   8.6393,   1.4139,   3.4698,  -0.4061,  -6.5177,  -2.3376,
               8.8883,   8.4269,   2.1366],
            [  3.9033,   5.0146,  12.8683,  -8.9402,  11.4366,  31.3686,   3.2794,
               2.2839,  11.1379,   0.6423],
            [  6.6159,   1.9588,   2.2625,  12.5532,   5.5336,   0.8998,   0.5374,
               7.7319,  -0.2104,   0.8249],
            [ -5.4877,  19.7717,  12.5163,  -2.7179,   7.1121,   5.9905,  -2.4888,
               6.5958,  22.2492, -10.1361]])



### torch.set_printoptions, np.set_printoptions


```python
import numpy as np
```


```python
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
np.set_printoptions(precision=2, linewidth=140)
t1
```




    tensor([[ 13.94,  -2.07,   7.38,   4.89,   9.79,  11.23,  -5.66,   2.72,   1.67,   0.62],
            [ -3.94,   8.64,   1.41,   3.47,  -0.41,  -6.52,  -2.34,   8.89,   8.43,   2.14],
            [  3.90,   5.01,  12.87,  -8.94,  11.44,  31.37,   3.28,   2.28,  11.14,   0.64],
            [  6.62,   1.96,   2.26,  12.55,   5.53,   0.90,   0.54,   7.73,  -0.21,   0.82],
            [ -5.49,  19.77,  12.52,  -2.72,   7.11,   5.99,  -2.49,   6.60,  22.25, -10.14]])




```python
%time _=matmul_3loops(m1, m2)
```

    shapes => a: torch.Size([5, 784]), b: torch.Size([784, 10]), res: torch.Size([5, 10])
    CPU times: user 334 ms, sys: 2.33 ms, total: 336 ms
    Wall time: 336 ms



```python

```

## Numba
Numba is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.

### njit, dot, np.array


```python
#| export groundup
import numba
```


```python
check(numba)
```

    signature: None
    __class__: <class 'module'>
    __repr__: <module 'numba' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/__init__.py'>
    
    __doc__:
    Expose top-level symbols that are safe for import *
    __dict__: 
    {'ByteCodeSupportError': <class 'numba.core.errors.ByteCodeSupportError'>,
     'CompilerError': <class 'numba.core.errors.CompilerError'>,
     'ConstantInferenceError': <class 'numba.core.errors.ConstantInferenceError'>,
     'DeprecationError': <class 'numba.core.errors.DeprecationError'>,
     'ForbiddenConstruct': <class 'numba.core.errors.ForbiddenConstruct'>,
     'ForceLiteralArg': <class 'numba.core.errors.ForceLiteralArg'>,
     'IRError': <class 'numba.core.errors.IRError'>,
     'InternalError': <class 'numba.core.errors.InternalError'>,
     'InternalTargetMismatchError': <class 'numba.core.errors.InternalTargetMismatchError'>,
     'LiteralTypingError': <class 'numba.core.errors.LiteralTypingError'>,
     'LoweringError': <class 'numba.core.errors.LoweringError'>,
     'NotDefinedError': <class 'numba.core.errors.NotDefinedError'>,
     'NumbaAssertionError': <class 'numba.core.errors.NumbaAssertionError'>,
     'NumbaAttributeError': <class 'numba.core.errors.NumbaAttributeError'>,
     'NumbaDebugInfoWarning': <class 'numba.core.errors.NumbaDebugInfoWarning'>,
     'NumbaDeprecationWarning': <class 'numba.core.errors.NumbaDeprecationWarning'>,
     'NumbaError': <class 'numba.core.errors.NumbaError'>,
     'NumbaExperimentalFeatureWarning': <class 'numba.core.errors.NumbaExperimentalFeatureWarning'>,
     'NumbaIRAssumptionWarning': <class 'numba.core.errors.NumbaIRAssumptionWarning'>,
     'NumbaIndexError': <class 'numba.core.errors.NumbaIndexError'>,
     'NumbaInvalidConfigWarning': <class 'numba.core.errors.NumbaInvalidConfigWarning'>,
     'NumbaKeyError': <class 'numba.core.errors.NumbaKeyError'>,
     'NumbaNotImplementedError': <class 'numba.core.errors.NumbaNotImplementedError'>,
     'NumbaParallelSafetyWarning': <class 'numba.core.errors.NumbaParallelSafetyWarning'>,
     'NumbaPedanticWarning': <class 'numba.core.errors.NumbaPedanticWarning'>,
     'NumbaPendingDeprecationWarning': <class 'numba.core.errors.NumbaPendingDeprecationWarning'>,
     'NumbaPerformanceWarning': <class 'numba.core.errors.NumbaPerformanceWarning'>,
     'NumbaRuntimeError': <class 'numba.core.errors.NumbaRuntimeError'>,
     'NumbaTypeError': <class 'numba.core.errors.NumbaTypeError'>,
     'NumbaTypeSafetyWarning': <class 'numba.core.errors.NumbaTypeSafetyWarning'>,
     'NumbaValueError': <class 'numba.core.errors.NumbaValueError'>,
     'NumbaWarning': <class 'numba.core.errors.NumbaWarning'>,
     'RedefinedError': <class 'numba.core.errors.RedefinedError'>,
     'RequireLiteralValue': <class 'numba.core.errors.RequireLiteralValue'>,
     'TypingError': <class 'numba.core.errors.TypingError'>,
     'UnsupportedError': <class 'numba.core.errors.UnsupportedError'>,
     'UnsupportedParforsError': <class 'numba.core.errors.UnsupportedParforsError'>,
     'UnsupportedRewriteError': <class 'numba.core.errors.UnsupportedRewriteError'>,
     'UntypedAttributeError': <class 'numba.core.errors.UntypedAttributeError'>,
     'VerificationError': <class 'numba.core.errors.VerificationError'>,
     '__all__': ['cfunc',
                 'from_dtype',
                 'guvectorize',
                 'jit',
                 'experimental',
                 'njit',
                 'stencil',
                 'jit_module',
                 'typeof',
                 'prange',
                 'gdb',
                 'gdb_breakpoint',
                 'gdb_init',
                 'vectorize',
                 'objmode',
                 'literal_unroll',
                 'get_num_threads',
                 'set_num_threads',
                 'set_parallel_chunksize',
                 'get_parallel_chunksize',
                 'parallel_chunksize',
                 'int8',
                 'int16',
                 'int32',
                 'int64',
                 'uint8',
                 'uint16',
                 'uint32',
                 'uint64',
                 'intp',
                 'uintp',
                 'intc',
                 'uintc',
                 'ssize_t',
                 'size_t',
                 'boolean',
                 'float32',
                 'float64',
                 'complex64',
                 'complex128',
                 'bool_',
                 'byte',
                 'char',
                 'uchar',
                 'short',
                 'ushort',
                 'int_',
                 'uint',
                 'long_',
                 'ulong',
                 'longlong',
                 'ulonglong',
                 'float_',
                 'double',
                 'void',
                 'none',
                 'b1',
                 'i1',
                 'i2',
                 'i4',
                 'i8',
                 'u1',
                 'u2',
                 'u4',
                 'u8',
                 'f4',
                 'f8',
                 'c8',
                 'c16',
                 'optional',
                 'ffi_forced_object',
                 'ffi',
                 'deferred_type',
                 'NumbaWarning',
                 'NumbaPerformanceWarning',
                 'NumbaDeprecationWarning',
                 'NumbaPendingDeprecationWarning',
                 'NumbaParallelSafetyWarning',
                 'NumbaTypeSafetyWarning',
                 'NumbaExperimentalFeatureWarning',
                 'NumbaInvalidConfigWarning',
                 'NumbaPedanticWarning',
                 'NumbaIRAssumptionWarning',
                 'NumbaDebugInfoWarning',
                 'NumbaError',
                 'UnsupportedError',
                 'UnsupportedRewriteError',
                 'IRError',
                 'RedefinedError',
                 'NotDefinedError',
                 'VerificationError',
                 'DeprecationError',
                 'LoweringError',
                 'UnsupportedParforsError',
                 'ForbiddenConstruct',
                 'TypingError',
                 'UntypedAttributeError',
                 'ByteCodeSupportError',
                 'CompilerError',
                 'ConstantInferenceError',
                 'InternalError',
                 'InternalTargetMismatchError',
                 'RequireLiteralValue',
                 'ForceLiteralArg',
                 'LiteralTypingError',
                 'NumbaValueError',
                 'NumbaTypeError',
                 'NumbaAttributeError',
                 'NumbaAssertionError',
                 'NumbaNotImplementedError',
                 'NumbaKeyError',
                 'NumbaIndexError',
                 'NumbaRuntimeError'],
     '__builtins__': {'ArithmeticError': <class 'ArithmeticError'>,
                      'AssertionError': <class 'AssertionError'>,
                      'AttributeError': <class 'AttributeError'>,
                      'BaseException': <class 'BaseException'>,
                      'BlockingIOError': <class 'BlockingIOError'>,
                      'BrokenPipeError': <class 'BrokenPipeError'>,
                      'BufferError': <class 'BufferError'>,
                      'BytesWarning': <class 'BytesWarning'>,
                      'ChildProcessError': <class 'ChildProcessError'>,
                      'ConnectionAbortedError': <class 'ConnectionAbortedError'>,
                      'ConnectionError': <class 'ConnectionError'>,
                      'ConnectionRefusedError': <class 'ConnectionRefusedError'>,
                      'ConnectionResetError': <class 'ConnectionResetError'>,
                      'DeprecationWarning': <class 'DeprecationWarning'>,
                      'EOFError': <class 'EOFError'>,
                      'Ellipsis': Ellipsis,
                      'EnvironmentError': <class 'OSError'>,
                      'Exception': <class 'Exception'>,
                      'False': False,
                      'FileExistsError': <class 'FileExistsError'>,
                      'FileNotFoundError': <class 'FileNotFoundError'>,
                      'FloatingPointError': <class 'FloatingPointError'>,
                      'FutureWarning': <class 'FutureWarning'>,
                      'GeneratorExit': <class 'GeneratorExit'>,
                      'IOError': <class 'OSError'>,
                      'ImportError': <class 'ImportError'>,
                      'ImportWarning': <class 'ImportWarning'>,
                      'IndentationError': <class 'IndentationError'>,
                      'IndexError': <class 'IndexError'>,
                      'InterruptedError': <class 'InterruptedError'>,
                      'IsADirectoryError': <class 'IsADirectoryError'>,
                      'KeyError': <class 'KeyError'>,
                      'KeyboardInterrupt': <class 'KeyboardInterrupt'>,
                      'LookupError': <class 'LookupError'>,
                      'MemoryError': <class 'MemoryError'>,
                      'ModuleNotFoundError': <class 'ModuleNotFoundError'>,
                      'NameError': <class 'NameError'>,
                      'None': None,
                      'NotADirectoryError': <class 'NotADirectoryError'>,
                      'NotImplemented': NotImplemented,
                      'NotImplementedError': <class 'NotImplementedError'>,
                      'OSError': <class 'OSError'>,
                      'OverflowError': <class 'OverflowError'>,
                      'PendingDeprecationWarning': <class 'PendingDeprecationWarning'>,
                      'PermissionError': <class 'PermissionError'>,
                      'ProcessLookupError': <class 'ProcessLookupError'>,
                      'RecursionError': <class 'RecursionError'>,
                      'ReferenceError': <class 'ReferenceError'>,
                      'ResourceWarning': <class 'ResourceWarning'>,
                      'RuntimeError': <class 'RuntimeError'>,
                      'RuntimeWarning': <class 'RuntimeWarning'>,
                      'StopAsyncIteration': <class 'StopAsyncIteration'>,
                      'StopIteration': <class 'StopIteration'>,
                      'SyntaxError': <class 'SyntaxError'>,
                      'SyntaxWarning': <class 'SyntaxWarning'>,
                      'SystemError': <class 'SystemError'>,
                      'SystemExit': <class 'SystemExit'>,
                      'TabError': <class 'TabError'>,
                      'TimeoutError': <class 'TimeoutError'>,
                      'True': True,
                      'TypeError': <class 'TypeError'>,
                      'UnboundLocalError': <class 'UnboundLocalError'>,
                      'UnicodeDecodeError': <class 'UnicodeDecodeError'>,
                      'UnicodeEncodeError': <class 'UnicodeEncodeError'>,
                      'UnicodeError': <class 'UnicodeError'>,
                      'UnicodeTranslateError': <class 'UnicodeTranslateError'>,
                      'UnicodeWarning': <class 'UnicodeWarning'>,
                      'UserWarning': <class 'UserWarning'>,
                      'ValueError': <class 'ValueError'>,
                      'Warning': <class 'Warning'>,
                      'ZeroDivisionError': <class 'ZeroDivisionError'>,
                      '__IPYTHON__': True,
                      '__build_class__': <built-in function __build_class__>,
                      '__debug__': True,
                      '__doc__': 'Built-in functions, exceptions, and other '
                                 'objects.\n'
                                 '\n'
                                 "Noteworthy: None is the `nil' object; Ellipsis "
                                 "represents `...' in slices.",
                      '__import__': <built-in function __import__>,
                      '__loader__': <class '_frozen_importlib.BuiltinImporter'>,
                      '__name__': 'builtins',
                      '__package__': '',
                      '__pybind11_internals_v4_clang_libcpp_cxxabi1002__': <capsule object NULL>,
                      '__spec__': ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>, origin='built-in'),
                      'abs': <built-in function abs>,
                      'all': <built-in function all>,
                      'any': <built-in function any>,
                      'ascii': <built-in function ascii>,
                      'bin': <built-in function bin>,
                      'bool': <class 'bool'>,
                      'breakpoint': <built-in function breakpoint>,
                      'bytearray': <class 'bytearray'>,
                      'bytes': <class 'bytes'>,
                      'callable': <built-in function callable>,
                      'chr': <built-in function chr>,
                      'classmethod': <class 'classmethod'>,
                      'compile': <built-in function compile>,
                      'complex': <class 'complex'>,
                      'copyright': Copyright (c) 2001-2022 Python Software Foundation.
    All Rights Reserved.
    
    Copyright (c) 2000 BeOpen.com.
    All Rights Reserved.
    
    Copyright (c) 1995-2001 Corporation for National Research Initiatives.
    All Rights Reserved.
    
    Copyright (c) 1991-1995 Stichting Mathematisch Centrum, Amsterdam.
    All Rights Reserved.,
                      'credits':     Thanks to CWI, CNRI, BeOpen.com, Zope Corporation and a cast of thousands
        for supporting Python development.  See www.python.org for more information.,
                      'delattr': <built-in function delattr>,
                      'dict': <class 'dict'>,
                      'dir': <built-in function dir>,
                      'display': <function display>,
                      'divmod': <built-in function divmod>,
                      'enumerate': <class 'enumerate'>,
                      'eval': <built-in function eval>,
                      'exec': <built-in function exec>,
                      'execfile': <function execfile>,
                      'filter': <class 'filter'>,
                      'float': <class 'float'>,
                      'format': <built-in function format>,
                      'frozenset': <class 'frozenset'>,
                      'get_ipython': <bound method InteractiveShell.get_ipython of <ipykernel.zmqshell.ZMQInteractiveShell object>>,
                      'getattr': <built-in function getattr>,
                      'globals': <built-in function globals>,
                      'hasattr': <built-in function hasattr>,
                      'hash': <built-in function hash>,
                      'help': Type help() for interactive help, or help(object) for help about object.,
                      'hex': <built-in function hex>,
                      'id': <built-in function id>,
                      'input': <bound method Kernel.raw_input of <ipykernel.ipkernel.IPythonKernel object>>,
                      'int': <class 'int'>,
                      'isinstance': <built-in function isinstance>,
                      'issubclass': <built-in function issubclass>,
                      'iter': <built-in function iter>,
                      'len': <built-in function len>,
                      'license': Type license() to see the full license text,
                      'list': <class 'list'>,
                      'locals': <built-in function locals>,
                      'map': <class 'map'>,
                      'max': <built-in function max>,
                      'memoryview': <class 'memoryview'>,
                      'min': <built-in function min>,
                      'next': <built-in function next>,
                      'object': <class 'object'>,
                      'oct': <built-in function oct>,
                      'open': <built-in function open>,
                      'ord': <built-in function ord>,
                      'pow': <built-in function pow>,
                      'print': <built-in function print>,
                      'property': <class 'property'>,
                      'range': <class 'range'>,
                      'repr': <built-in function repr>,
                      'reversed': <class 'reversed'>,
                      'round': <built-in function round>,
                      'runfile': <function runfile>,
                      'set': <class 'set'>,
                      'setattr': <built-in function setattr>,
                      'slice': <class 'slice'>,
                      'sorted': <built-in function sorted>,
                      'staticmethod': <class 'staticmethod'>,
                      'str': <class 'str'>,
                      'sum': <built-in function sum>,
                      'super': <class 'super'>,
                      'tuple': <class 'tuple'>,
                      'type': <class 'type'>,
                      'vars': <built-in function vars>,
                      'zip': <class 'zip'>},
     '__cached__': '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/__pycache__/__init__.cpython-39.pyc',
     '__doc__': '\nExpose top-level symbols that are safe for import *\n',
     '__file__': '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/__init__.py',
     '__loader__': <_frozen_importlib_external.SourceFileLoader object>,
     '__name__': 'numba',
     '__package__': 'numba',
     '__path__': ['/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba'],
     '__spec__': ModuleSpec(name='numba', loader=<_frozen_importlib_external.SourceFileLoader object>, origin='/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/__init__.py', submodule_search_locations=['/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba']),
     '__version__': '0.56.2',
     '_devicearray': <module 'numba._devicearray' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/_devicearray.cpython-39-darwin.so'>,
     '_dispatcher': <module 'numba._dispatcher' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/_dispatcher.cpython-39-darwin.so'>,
     '_dynfunc': <module 'numba._dynfunc' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/_dynfunc.cpython-39-darwin.so'>,
     '_ensure_critical_deps': <function _ensure_critical_deps>,
     '_ensure_llvm': <function _ensure_llvm>,
     '_helperlib': <module 'numba._helperlib' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/_helperlib.cpython-39-darwin.so'>,
     '_min_llvm_version': (11, 0, 0),
     '_min_llvmlite_version': (0, 39, 0),
     '_try_enable_svml': <function _try_enable_svml>,
     '_version': <module 'numba._version' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/_version.py'>,
     'b1': bool,
     'bool_': bool,
     'boolean': bool,
     'byte': uint8,
     'c16': complex128,
     'c8': complex64,
     'carray': <function carray>,
     'cfunc': <function cfunc>,
     'char': int8,
     'cloudpickle': <module 'numba.cloudpickle' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/cloudpickle/__init__.py'>,
     'complex128': complex128,
     'complex64': complex64,
     'config': <module 'numba.core.config' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/core/config.py'>,
     'core': <module 'numba.core' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/core/__init__.py'>,
     'cpython': <module 'numba.cpython' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/cpython/__init__.py'>,
     'deferred_type': <class 'numba.core.types.misc.DeferredType'>,
     'double': float64,
     'errors': <module 'numba.core.errors' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/core/errors.py'>,
     'experimental': <module 'numba.experimental' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/experimental/__init__.py'>,
     'extending': <module 'numba.extending' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/extending.py'>,
     'f4': float32,
     'f8': float64,
     'farray': <function farray>,
     'ffi': ffi,
     'ffi_forced_object': ffi_forced_object,
     'float32': float32,
     'float64': float64,
     'float_': float32,
     'from_dtype': <function from_dtype>,
     'gdb': <function gdb>,
     'gdb_breakpoint': <function gdb_breakpoint>,
     'gdb_init': <function gdb_init>,
     'generated_jit': <function generated_jit>,
     'get_num_threads': <function get_num_threads>,
     'get_parallel_chunksize': <function get_parallel_chunksize>,
     'get_thread_id': <function get_thread_id>,
     'guvectorize': <function guvectorize>,
     'i1': int8,
     'i2': int16,
     'i4': int32,
     'i8': int64,
     'int16': int16,
     'int32': int32,
     'int64': int64,
     'int8': int8,
     'int_': int64,
     'intc': int32,
     'intp': int64,
     'jit': <function jit>,
     'jit_module': <function jit_module>,
     'literal_unroll': <function literal_unroll>,
     'literally': <function literally>,
     'llvmlite': <module 'llvmlite' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/llvmlite/__init__.py'>,
     'long_': int64,
     'longlong': int64,
     'misc': <module 'numba.misc' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/misc/__init__.py'>,
     'njit': <function njit>,
     'none': none,
     'np': <module 'numba.np' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/np/__init__.py'>,
     'numba': <module 'numba' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/__init__.py'>,
     'objmode': <numba.core.withcontexts._ObjModeContextType object>,
     'optional': <class 'numba.core.types.misc.Optional'>,
     'parallel_chunksize': <numba.core.withcontexts._ParallelChunksize object>,
     'parfors': <module 'numba.parfors' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/parfors/__init__.py'>,
     'platform': <module 'platform' from '/Users/Natsume/mambaforge/lib/python3.9/platform.py'>,
     'pndindex': <function pndindex>,
     'prange': <class 'numba.misc.special.prange'>,
     're': <module 're' from '/Users/Natsume/mambaforge/lib/python3.9/re.py'>,
     'set_num_threads': <function set_num_threads>,
     'set_parallel_chunksize': <function set_parallel_chunksize>,
     'short': int16,
     'size_t': uint64,
     'ssize_t': int64,
     'stencil': <function stencil>,
     'stencils': <module 'numba.stencils' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/stencils/__init__.py'>,
     'sys': <module 'sys' (built-in)>,
     'test': <function test>,
     'threading_layer': <function threading_layer>,
     'typed': <module 'numba.typed' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/typed/__init__.py'>,
     'typeof': <function typeof>,
     'types': <module 'numba.core.types' from '/Users/Natsume/mambaforge/lib/python3.9/site-packages/numba/core/types/__init__.py'>,
     'u1': uint8,
     'u2': uint16,
     'u4': uint32,
     'u8': uint64,
     'uchar': uint8,
     'uint': uint64,
     'uint16': uint16,
     'uint32': uint32,
     'uint64': uint64,
     'uint8': uint8,
     'uintc': uint32,
     'uintp': uint64,
     'ulong': uint64,
     'ulonglong': uint64,
     'ushort': uint16,
     'vectorize': <function vectorize>,
     'version_info': version_info(major=0, minor=56, patch=2, short=(0, 56), full=(0, 56, 2), string='0.56.2', tuple=('0', '56', '2'), git_revision=None),
     'void': none,
     'warnings': <module 'warnings' from '/Users/Natsume/mambaforge/lib/python3.9/warnings.py'>}
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
whatinside(numba)
```

    numba has: 
    113 items in its __all__, and 
    28 user defined functions, 
    43 classes or class objects, 
    0 builtin funcs and methods, and
    123 callables.
    
    Expose top-level symbols that are safe for import *



```python
#| export groundup
from numba import njit, jit
```


```python
check(njit)
```

    signature: (*args, **kws)
    __class__: <class 'function'>
    __repr__: <function njit>
    
    __doc__:
    Equivalent to jit(nopython=True)
    
    See documentation for jit function/decorator for full description.
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: True
    method: False



```python
check(jit)
```

    signature: (signature_or_function=None, locals={}, cache=False, pipeline_class=None, boundscheck=None, **options)
    __class__: <class 'function'>
    __repr__: <function jit>
    
    __doc__:
    This decorator is used to compile a Python function into native code.
    
    Args
    -----
    signature_or_function:
        The (optional) signature or list of signatures to be compiled.
        If not passed, required signatures will be compiled when the
        decorated function is called, depending on the argument values.
        As a convenience, you can directly pass the function to be compiled
        instead.
    
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: True
    method: False



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

    CPU times: user 174 ms, sys: 20.8 ms, total: 195 ms
    Wall time: 219 ms





    20.0




```python
%time dot(array([1.,2,3]),array([2.,3,4]))
```

    CPU times: user 20 µs, sys: 1 µs, total: 21 µs
    Wall time: 21 µs





    20.0




```python
%time dot(array([1.,2,3]),array([2.,3,4]))
```

    CPU times: user 29 µs, sys: 1 µs, total: 30 µs
    Wall time: 31 µs





    20.0



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




    tensor([[ 13.94,  -2.07,   7.38,   4.89,   9.79,  11.23,  -5.66,   2.72,   1.67,   0.62],
            [ -3.94,   8.64,   1.41,   3.47,  -0.41,  -6.52,  -2.34,   8.89,   8.43,   2.14],
            [  3.90,   5.01,  12.87,  -8.94,  11.44,  31.37,   3.28,   2.28,  11.14,   0.64],
            [  6.62,   1.96,   2.26,  12.55,   5.53,   0.90,   0.54,   7.73,  -0.21,   0.82],
            [ -5.49,  19.77,  12.52,  -2.72,   7.11,   5.99,  -2.49,   6.60,  22.25, -10.14]])




```python
check(m1.numpy)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method numpy of Tensor object>
    
    __doc__:
    numpy() -> numpy.ndarray
    
    Returns :attr:`self` tensor as a NumPy :class:`ndarray`. This tensor and the
    returned :class:`ndarray` share the same underlying storage. Changes to
    :attr:`self` tensor will be reflected in the :class:`ndarray` and vice versa.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
m1a,m2a = m1.numpy(),m2.numpy()
```


```python
m1.shape, m1a.shape
```




    (torch.Size([5, 784]), (5, 784))



### test_close, %timeit, %time


```python
from fastcore.test import *
```


```python
test_close(matmul_3loops(m1, m2),matmul_2loops_njit(m1, m2))
```

    shapes => a: torch.Size([5, 784]), b: torch.Size([784, 10]), res: torch.Size([5, 10])



```python
%timeit -n 10 matmul_2loops_njit(m1,m2)
```

    259 µs ± 160 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%time _=matmul_2loops_njit(m1,m2)
```

    CPU times: user 376 µs, sys: 160 µs, total: 536 µs
    Wall time: 379 µs



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




    (tensor([10.,  6., -4.]), tensor([2., 8., 7.]))




```python
a + b
```




    tensor([12., 14.,  3.])




```python
(a < b).float().mean()
```




    tensor(0.67)




```python
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]]); m
```




    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])



Frobenius norm:

$$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$

*Hint*: you don't normally need to write equations in LaTeX yourself, instead, you can click 'edit' in Wikipedia and copy the LaTeX from there (which is what I did for the above equation). Or on arxiv.org, click "Download: Other formats" in the top right, then "Download source"; rename the downloaded file to end in `.tgz` if it doesn't already, and you should find the source there, including the equations to copy and paste. This is the source LaTeX that I pasted to render the equation above:

```latex
$$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$
```


```python
(m*m).sum().sqrt()
```




    tensor(16.88)




```python
check(m.sum)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method sum of Tensor object>
    
    __doc__:
    sum(dim=None, keepdim=False, dtype=None) -> Tensor
    
    See :func:`torch.sum`
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
check(torch.sum)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method sum of type object>
    
    __doc__:
    sum(input, *, dtype=None) -> Tensor
    
    Returns the sum of all elements in the :attr:`input` tensor.
    
    Args:
        input (Tensor): the input tensor.
    
    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False


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

    shapes => a: torch.Size([5, 784]), b: torch.Size([784, 10]), res: torch.Size([5, 10])



```python
%time _=matmul_3loops(m1, m2)
```

    shapes => a: torch.Size([5, 784]), b: torch.Size([784, 10]), res: torch.Size([5, 10])
    CPU times: user 336 ms, sys: 3.59 ms, total: 340 ms
    Wall time: 340 ms



```python
%time _=matmul_2loops_njit(m1, m2)
```

    CPU times: user 335 µs, sys: 198 µs, total: 533 µs
    Wall time: 322 µs



```python
%time _=matmul_2loops_elementwise(m1, m2)
```

    CPU times: user 635 µs, sys: 370 µs, total: 1.01 ms
    Wall time: 613 µs


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

    393 µs ± 20.6 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%time _=matmul_2loops_njit(m1, m2)
```

    CPU times: user 380 µs, sys: 230 µs, total: 610 µs
    Wall time: 389 µs



```python
%time _=matmul_2loops_elementwise(m1, m2)
```

    CPU times: user 654 µs, sys: 410 µs, total: 1.06 ms
    Wall time: 686 µs



```python
%time _=matmul_2loops_dotproduct(m1, m2)
```

    CPU times: user 584 µs, sys: 323 µs, total: 907 µs
    Wall time: 576 µs



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




    tensor([10.,  6., -4.])




```python
a > 0
```




    tensor([ True,  True, False])



How are we able to do `a > 0`?  0 is being **broadcast** to have the same dimensions as a.

For instance you can normalize our dataset by subtracting the mean (a scalar) from the entire data set (a matrix) and dividing by the standard deviation (another scalar), using broadcasting.

Other examples of broadcasting with a scalar:


```python
a + 1
```




    tensor([11.,  7., -3.])




```python
m
```




    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])




```python
2*m
```




    tensor([[ 2.,  4.,  6.],
            [ 8., 10., 12.],
            [14., 16., 18.]])



### Broadcasting a vector to a matrix

Although broadcasting a scalar is an idea that dates back to APL, the more powerful idea of broadcasting across higher rank tensors [comes from](https://mail.python.org/pipermail/matrix-sig/1995-November/000143.html) a little known language called [Yorick](https://software.llnl.gov/yorick-doc/manual/yorick_50.html).

We can also broadcast a vector to a matrix, when the vector's shape is a scalar


```python
c = tensor([10.,20,30]); 
c
```




    tensor([10., 20., 30.])




```python
m
```




    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])




```python
m.shape,c.shape
```




    (torch.Size([3, 3]), torch.Size([3]))




```python
m + c
```




    tensor([[11., 22., 33.],
            [14., 25., 36.],
            [17., 28., 39.]])




```python
c + m
```




    tensor([[11., 22., 33.],
            [14., 25., 36.],
            [17., 28., 39.]])



#### c.expand_as(m), t.storage(), t.stride()

We don't really copy the rows, but it looks as if we did. In fact, the rows are given a *stride* of 0.


```python
check(c.expand_as)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method expand_as of Tensor object>
    
    __doc__:
    expand_as(other) -> Tensor
    
    Expand this tensor to the same size as :attr:`other`.
    ``self.expand_as(other)`` is equivalent to ``self.expand(other.size())``.
    
    Please see :meth:`~Tensor.expand` for more information about ``expand``.
    
    Args:
        other (:class:`torch.Tensor`): The result tensor has the same size
            as :attr:`other`.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
t = c.expand_as(m)
```


```python
t
```




    tensor([[10., 20., 30.],
            [10., 20., 30.],
            [10., 20., 30.]])




```python
m + t
```




    tensor([[11., 22., 33.],
            [14., 25., 36.],
            [17., 28., 39.]])




```python
check(t.storage)
```

    signature: ()
    __class__: <class 'method'>
    __repr__: <bound method Tensor.storage of tensor([[10., 20., 30.],
            [10., 20., 30.],
            [10., 20., 30.]])>
    
    __doc__:
    storage() -> torch.Storage
    
    Returns the underlying storage.
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: False
    method: True



```python
t.storage()
```




     10.0
     20.0
     30.0
    [torch.storage._TypedStorage(dtype=torch.float32, device=cpu) of size 3]




```python
check(t.stride)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method stride of Tensor object>
    
    __doc__:
    stride(dim) -> tuple or int
    
    Returns the stride of :attr:`self` tensor.
    
    Stride is the jump necessary to go from one element to the next one in the
    specified dimension :attr:`dim`. A tuple of all strides is returned when no
    argument is passed in. Otherwise, an integer value is returned as the stride in
    the particular dimension :attr:`dim`.
    
    Args:
        dim (int, optional): the desired dimension in which stride is required
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```python
t.stride(), t.shape
```




    ((0, 1), torch.Size([3, 3]))



### c.unsqueeze(0), c[None,:], c.unsqueeze(1), c[:, None]

You can index with the special value [None] or use `unsqueeze()` to convert a 1-dimensional array into a 2-dimensional array (although one of those dimensions has value 1).


```python
c
c.shape
```




    tensor([10., 20., 30.])






    torch.Size([3])




```python
c.unsqueeze(0), c[None, :]
```




    (tensor([[10., 20., 30.]]), tensor([[10., 20., 30.]]))




```python
c.unsqueeze(0).shape
c[None, :].shape
```




    torch.Size([1, 3])






    torch.Size([1, 3])




```python
c.unsqueeze(1), c[:, None]
```




    (tensor([[10.],
             [20.],
             [30.]]),
     tensor([[10.],
             [20.],
             [30.]]))




```python
c.shape
c.unsqueeze(1).shape
c[:, None].shape
```




    torch.Size([3])






    torch.Size([3, 1])






    torch.Size([3, 1])



#### c[None], c[..., None]
You can always skip trailling ':'s. And '...' means '*all preceding dimensions*'


```python
c[None].shape,c[...,None].shape
```




    (torch.Size([1, 3]), torch.Size([3, 1]))



### Broadcast a one-row/col matrix


```python
# c[None,:].expand_as(m)
c[None].expand_as(m)
```




    tensor([[10., 20., 30.],
            [10., 20., 30.],
            [10., 20., 30.]])




```python
c[:,None].expand_as(m)
```




    tensor([[10., 10., 10.],
            [20., 20., 20.],
            [30., 30., 30.]])




```python
m + c[:,None]
```




    tensor([[11., 12., 13.],
            [24., 25., 26.],
            [37., 38., 39.]])




```python
m + c[None,:]
```




    tensor([[11., 22., 33.],
            [14., 25., 36.],
            [17., 28., 39.]])



### Broadcasting Rules
how to broadcast two 1-row/col matricies


```python
c[None,:]
```




    tensor([[10., 20., 30.]])




```python
c[None,:].shape
```




    torch.Size([1, 3])




```python
c[:,None]
```




    tensor([[10.],
            [20.],
            [30.]])




```python
c[:,None].shape
```




    torch.Size([3, 1])




```python
c[None,:] * c[:,None]
```




    tensor([[100., 200., 300.],
            [200., 400., 600.],
            [300., 600., 900.]])




```python
c[None] > c[:,None]
```




    tensor([[False,  True,  True],
            [False, False,  True],
            [False, False, False]])



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




    (torch.Size([784]), torch.Size([784, 10]))




```python
rowZero[:,None].shape # make rowZero flip from horizontal to vertical
```




    torch.Size([784, 1])




```python
rowZero[:,None].expand_as(m2).shape # broadcast from one column to 10 columns to match m2
```




    torch.Size([784, 10])




```python
(rowZero[:,None]*m2).shape # (broadcast vertically) vector * matrix 
```




    torch.Size([784, 10])




```python
(rowZero[:,None]*m2).sum(dim=0) 
# dim=0, smash vertically to the ground so that we just have one row as output
# dim=1, smash horizontal to the left/right so that we just have one column as output
```




    tensor([13.94, -2.07,  7.38,  4.89,  9.79, 11.23, -5.66,  2.72,  1.67,  0.62])




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

    shapes => a: torch.Size([5, 784]), b: torch.Size([784, 10]), res: torch.Size([5, 10])



```python
%timeit -n 10 _=matmul_1loop_broadcast(m1, m2)
```

    57.4 µs ± 8.95 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%time _=matmul_2loops_njit(m1,m2)
%time _=matmul_2loops_elementwise(m1,m2)
%time _=matmul_2loops_dotproduct(m1,m2)
%time _=matmul_1loop_broadcast(m1,m2)
```

    CPU times: user 364 µs, sys: 237 µs, total: 601 µs
    Wall time: 370 µs
    CPU times: user 600 µs, sys: 32 µs, total: 632 µs
    Wall time: 639 µs
    CPU times: user 397 µs, sys: 1e+03 ns, total: 398 µs
    Wall time: 399 µs
    CPU times: user 79 µs, sys: 0 ns, total: 79 µs
    Wall time: 80.8 µs


### matmul on x_train and weights

Our time has gone from ~500ms to <0.1ms, an over 5000x improvement! We can run on the whole dataset now.


```python
tr = matmul_1loop_broadcast(x_train, weights)
```


```python
tr.shape
```




    torch.Size([50000, 10])




```python
%time _=matmul_2loops_njit(x_train, weights)
%time _=matmul_2loops_elementwise(x_train, weights)
%time _=matmul_2loops_dotproduct(x_train, weights)
%time _=matmul_1loop_broadcast(x_train, weights)
```

    CPU times: user 2.25 s, sys: 513 ms, total: 2.76 s
    Wall time: 1.87 s
    CPU times: user 4.34 s, sys: 500 ms, total: 4.84 s
    Wall time: 4.06 s
    CPU times: user 4.09 s, sys: 517 ms, total: 4.61 s
    Wall time: 3.56 s
    CPU times: user 1 s, sys: 507 ms, total: 1.51 s
    Wall time: 501 ms



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

    CPU times: user 2.2 s, sys: 490 ms, total: 2.69 s
    Wall time: 1.82 s
    CPU times: user 4.36 s, sys: 499 ms, total: 4.86 s
    Wall time: 4.01 s
    CPU times: user 4.13 s, sys: 525 ms, total: 4.65 s
    Wall time: 3.71 s
    CPU times: user 898 ms, sys: 467 ms, total: 1.36 s
    Wall time: 513 ms
    CPU times: user 160 ms, sys: 16.1 ms, total: 176 ms
    Wall time: 25 ms



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

    CPU times: user 2.23 s, sys: 521 ms, total: 2.75 s
    Wall time: 1.83 s
    CPU times: user 4.34 s, sys: 502 ms, total: 4.84 s
    Wall time: 3.99 s
    CPU times: user 4.12 s, sys: 497 ms, total: 4.62 s
    Wall time: 3.62 s
    CPU times: user 1.12 s, sys: 528 ms, total: 1.64 s
    Wall time: 505 ms
    CPU times: user 174 ms, sys: 11.8 ms, total: 186 ms
    Wall time: 25 ms
    CPU times: user 171 ms, sys: 11.1 ms, total: 182 ms
    Wall time: 25.3 ms



```python

```

## CUDA
run from kaggle [notebook](https://www.kaggle.com/code/danielliao/course22p2-0001-matmul/edit/run/107870903)


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




    tensor([[-17.70,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
            [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
            [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
            [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00],
            [  0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00,   0.00]])




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




    tensor([[-17.70,  -8.61,  12.17, -15.78,   5.78, -14.75,  -8.66,  -4.28,  13.34,   7.52],
            [  2.51,  -0.46,  10.73, -11.62,  12.98,   2.54,  -9.66,   8.52,   6.11,  15.23],
            [ -4.33, -15.79,   1.87,   5.80,   0.38,   5.14,   3.61,  -1.90,   7.30,   9.60],
            [ -9.27,  -3.22,   7.59,   2.54,   3.17,  -2.14,  -2.51,   5.58,   8.43,   4.03],
            [ -2.47,  -2.54, -11.92,  -4.81,   0.89,   2.14,  -5.88,   9.96,  10.68,  15.94]])




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




    torch.Size([50000, 10])




```python
r = np.zeros(tr.shape)
m1g,m2g,rg = cuda.to_device(x_train),cuda.to_device(weights),cuda.to_device(r)
```


```python
m1g
```




    <numba.cuda.cudadrv.devicearray.DeviceNDArray>




```python
m1g.shape
```




    (50000, 784)




```python
r.shape
```




    (50000, 10)




```python
TPB = 16
rr,rc = r.shape
blockspergrid = (math.ceil(rr / TPB), math.ceil(rc / TPB))
blockspergrid
```




    (3125, 1)




```python
matmul[blockspergrid, (TPB,TPB)](m1g,m2g,rg) # not sure about [blockspergrid, (TPB,TPB)] part
r = rg.copy_to_host()
test_close(tr, r, eps=1.03)
```


```python
%%timeit -n 1
matmul[blockspergrid, (TPB,TPB)](m1g,m2g,rg)
r = rg.copy_to_host()
```

    6.91 ms ± 45.2 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
m1c,m2c = x_train.cuda(),weights.cuda()
```


```python
%timeit -n 1 rr = (m1c@m2c).cpu()
```

    The slowest run took 939.39 times longer than the fastest. This could mean that an intermediate result is being cached.
    117 ms ± 283 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%time rr = (m1c@m2c).cpu()
```

    CPU times: user 1.61 ms, sys: 0 ns, total: 1.61 ms
    Wall time: 1.05 ms


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
