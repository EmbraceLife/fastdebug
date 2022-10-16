# groundup_004_forward_backward_passes
---
skip_exec: true
---

```python
#| default_exp delete0004
```


```python
#| export
a = "to delete"
```

## imports


```python
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>



```python
from fastdebug.groundup import *
```


```python
%whos function
```

    Variable                    Type        Data/Info
    -------------------------------------------------
    check_data_directories      function    <function check_data_dire<...>tories at 0x7f99547f73a0>
    chunks                      function    <function chunks at 0x7f99261531f0>
    chunks_faster               function    <function chunks_faster at 0x7f9926153280>
    fastcodes                   function    <function fastcodes at 0x7f99547f48b0>
    fastlistnbs                 function    <function fastlistnbs at 0x7f99547f49d0>
    fastlistsrcs                function    <function fastlistsrcs at 0x7f99547f4a60>
    fastnbs                     function    <function fastnbs at 0x7f99547f4820>
    fastnotes                   function    <function fastnotes at 0x7f99547f4940>
    fastsrcs                    function    <function fastsrcs at 0x7f99547f4310>
    fastview                    function    <function fastview at 0x7f99547df280>
    get_all_nbs                 function    <function get_all_nbs at 0x7f99547f44c0>
    get_exp_data                function    <function get_exp_data at 0x7f99547f7700>
    get_img_paths               function    <function get_img_paths at 0x7f99547f7430>
    get_labels                  function    <function get_labels at 0x7f99547f74c0>
    imgs2tensor                 function    <function imgs2tensor at 0x7f99547f7670>
    inspect_class               function    <function inspect_class at 0x7f9964055a60>
    ipy2md                      function    <function ipy2md at 0x7f9964055790>
    isdecorator                 function    <function isdecorator at 0x7f99547df0d0>
    ismetaclass                 function    <function ismetaclass at 0x7f99547dde50>
    match_pct                   function    <function match_pct at 0x7f99547f7280>
    matmul_1loop_broadcast      function    <function matmul_1loop_br<...>adcast at 0x7f99202c4b80>
    matmul_2loops_dotproduct    function    <function matmul_2loops_d<...>roduct at 0x7f99202c4310>
    matmul_2loops_elementwise   function    <function matmul_2loops_e<...>ntwise at 0x7f99202e3dc0>
    matmul_2loops_njit          function    <function matmul_2loops_njit at 0x7f9920365c10>
    matmul_3loops               function    <function matmul_3loops at 0x7f99261534c0>
    matmul_einsum_noloop        function    <function matmul_einsum_noloop at 0x7f99202c4ca0>
    mean_std                    function    <function mean_std at 0x7f99547f7550>
    nb_name                     function    <function nb_name at 0x7f9964055700>
    nb_path                     function    <function nb_path at 0x7f9964055670>
    nb_url                      function    <function nb_url at 0x7f99640555e0>
    normalize                   function    <function normalize at 0x7f99547f75e0>
    openNB                      function    <function openNB at 0x7f99547f4550>
    openNBKaggle                function    <function openNBKaggle at 0x7f99547f45e0>
    rand                        function    <function rand at 0x7f9926153310>
    search_data_url             function    <function search_data_url at 0x7f99547f7310>
    test                        function    <function test at 0x7f99547f7160>
    test_eq                     function    <function test_eq at 0x7f99547f71f0>
    test_is                     function    <function test_is at 0x7f99547ddb80>
    whatinside                  function    <function whatinside at 0x7f99547df160>
    whichversion                function    <function whichversion at 0x7f99547df1f0>



```python
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl, numpy as np
from pathlib import Path
from torch import tensor
```


```python
mpl.rcParams['image.cmap'] = 'gray'
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
np.set_printoptions(precision=2, linewidth=140)
```

## get_exp_data, map, tensor


```python
path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'
```


```python
Path
```




    pathlib.Path




```python
x_train, y_train, x_valid, y_valid = get_exp_data()
```


```python
x_train.shape
type(x_train)
```




    (50000, 784)






    numpy.ndarray




```python
x_train, y_train, x_valid, y_valid = map(tensor, [x_train, y_train, x_valid, y_valid])
```


```python
x_train.shape
type(x_train)
```




    torch.Size([50000, 784])






    torch.Tensor



## Foundations version

### Basic architecture


```python
# n,m = x_train.shape
# c = y_train.max()+1
# n,m,c
```


```python
type(x_train)
```




    torch.Tensor




```python
r,c = x_train.shape
l = y_train.max()+1
r,c,l
```




    (50000, 784, tensor(10))




```python
# num hidden activations
nh = 50
```


```python
# w1 = torch.randn(m,nh)
# b1 = torch.zeros(nh)
# w2 = torch.randn(nh,1)
# b2 = torch.zeros(1)
```


```python
w1 = torch.randn(c,nh)
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1)
b2 = torch.zeros(1)
w1.shape
b1.shape
w2.shape
b2.shape
```




    torch.Size([784, 50])






    torch.Size([50])






    torch.Size([50, 1])






    torch.Size([1])




```python
#| export groundup
def lin(x, w, b): 
    "use torch.matmul (faster version of einsum) to create a linear model"
    return x@w + b
```


```python
t = lin(x_valid, w1, b1)
t.shape
```




    torch.Size([10000, 50])




```python
#| export groundup
def relu(x): 
    "basic relu with max in torch"
    return x.clamp_min(0.)
```


```python
t = relu(lin(x_valid, w1, b1)) # add relu unto the linear model
t
```




    tensor([[0.00, 0.00, 2.25,  ..., 0.00, 9.30, 0.00],
            [0.00, 0.00, 0.00,  ..., 0.00, 8.57, 0.00],
            [0.00, 2.04, 0.00,  ..., 0.00, 5.76, 0.00],
            ...,
            [0.00, 0.00, 0.00,  ..., 0.00, 0.00, 3.62],
            [0.00, 0.73, 0.00,  ..., 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00,  ..., 0.00, 0.00, 0.00]])




```python
(t >= 0).shape
(t >= 0).count_nonzero()
(t < 0).count_nonzero()
```




    torch.Size([10000, 50])






    tensor(500000)






    tensor(0)




```python
def model(xb):
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    return lin(l2, w2, b2)
```


```python
res = model(x_valid)
res.shape
```




    torch.Size([10000, 1])



### Loss function: MSE

We need to get rid of that trailing (,1), in order to use `mse`.


```python
res[:,0].shape
```




    torch.Size([10000])



(Of course, `mse` is not a suitable loss function for multi-class classification; we'll use a better loss function soon. We'll use `mse` for now to keep things simple.)


```python
def mse(output, targ): return (output[:,0]-targ).pow(2).mean()
```


```python
y_train,y_valid = y_train.float(),y_valid.float()
```


```python
preds = model(x_train)
preds.shape
```




    torch.Size([50000, 1])




```python
mse(preds, y_train)
```




    tensor(6871.75)



### Gradients and backward pass


```python
from sympy import symbols,diff
x,y = symbols('x y')
diff(x**2, x)
```




$\displaystyle 2 x$




```python
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
    b.g = out.g.sum(0)
```


```python
def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    diff = out[:,0]-targ
    loss = res.pow(2).mean()
    
    # backward pass:
    out.g = 2.*diff[:,None] / inp.shape[0]
    lin_grad(l2, out, w2, b2)
    l1.g = (l1>0).float() * l2.g
    lin_grad(inp, l1, w1, b1)
```


```python
forward_and_backward(x_train, y_train)
```


```python
# Save for testing against later
w1g = w1.g.clone()
w2g = w2.g.clone()
b1g = b1.g.clone()
b2g = b2.g.clone()
ig  = x_train.g.clone()
```

We cheat a little bit and use PyTorch autograd to check our results.


```python
xt2 = x_train.clone().requires_grad_(True)
w12 = w1.clone().requires_grad_(True)
w22 = w2.clone().requires_grad_(True)
b12 = b1.clone().requires_grad_(True)
b22 = b2.clone().requires_grad_(True)
```


```python
def forward(inp, targ):
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    out = l2 @ w22 + b22
    return mse(out, targ)
```


```python
loss = forward(xt2, y_train)
loss.backward()
```


```python
from fastcore.test import test_close
```


```python
test_close(w22.grad, w2g, eps=0.01)
test_close(b22.grad, b2g, eps=0.01)
test_close(w12.grad, w1g, eps=0.01)
test_close(b12.grad, b1g, eps=0.01)
test_close(xt2.grad, ig , eps=0.01)
```

## Refactor model

### Layers as classes


```python
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out
    
    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g
```


```python
class Lin():
    def __init__(self, w, b): self.w,self.b = w,b
        
    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out

    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = self.inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
```


```python
class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out
    
    def backward(self):
        self.inp.g = 2. * (self.inp.squeeze() - self.targ).unsqueeze(-1) / self.targ.shape[0]
```


```python
class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)
    
    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()
```


```python
model = Model(w1, b1, w2, b2)
```


```python
%time loss = model(x_train, y_train)
```

    CPU times: user 326 ms, sys: 3.48 ms, total: 329 ms
    Wall time: 56.3 ms



```python
%time model.backward()
```

    CPU times: user 780 ms, sys: 206 ms, total: 986 ms
    Wall time: 171 ms



```python
test_close(w2g, w2.g, eps=0.01)
test_close(b2g, b2.g, eps=0.01)
test_close(w1g, w1.g, eps=0.01)
test_close(b1g, b1.g, eps=0.01)
test_close(ig, x_train.g, eps=0.01)
```

### Module.forward()


```python
class Module():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self): raise Exception('not implemented')
    def bwd(self): raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)
```


```python
class Relu(Module):
    def forward(self, inp): return inp.clamp_min(0.)
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g
```


```python
class Lin(Module):
    def __init__(self, w, b): self.w,self.b = w,b
    def forward(self, inp): return inp@self.w + self.b
    def bwd(self, out, inp):
        inp.g = self.out.g @ self.w.t()
        self.w.g = inp.t() @ self.out.g
        self.b.g = self.out.g.sum(0)
```


```python
class Mse(Module):
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    def bwd(self, out, inp, targ): inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```


```python
model = Model(w1, b1, w2, b2)
```


```python
%time loss = model(x_train, y_train)
```

    CPU times: user 167 ms, sys: 0 ns, total: 167 ms
    Wall time: 29.9 ms



```python
%time model.backward()
```

    CPU times: user 370 ms, sys: 287 ms, total: 657 ms
    Wall time: 114 ms



```python
test_close(w2g, w2.g, eps=0.01)
test_close(b2g, b2.g, eps=0.01)
test_close(w1g, w1.g, eps=0.01)
test_close(b1g, b1.g, eps=0.01)
test_close(ig, x_train.g, eps=0.01)
```

### Autograd


```python
from torch import nn
import torch.nn.functional as F
```


```python
class Linear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = torch.randn(n_in,n_out).requires_grad_()
        self.b = torch.zeros(n_out).requires_grad_()
    def forward(self, inp): return inp@self.w + self.b
```


```python
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [Linear(n_in,nh), nn.ReLU(), Linear(nh,n_out)]
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return F.mse_loss(x, targ[:,None])
```


```python
model = Model(m, nh, 1)
loss = model(x_train, y_train)
loss.backward()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [62], in <cell line: 1>()
    ----> 1 model = Model(m, nh, 1)
          2 loss = model(x_train, y_train)
          3 loss.backward()


    NameError: name 'm' is not defined



```python
l0 = model.layers[0]
l0.b.grad
```


```python

```
