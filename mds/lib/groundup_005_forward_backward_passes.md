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

# groundup_005_forward_backward_passes

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

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

```python
from fastdebug.groundup import *
```

```python
# %whos function
```

```python
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl, numpy as np
from pathlib import Path
from torch import tensor
```

```python
# fastnbs("set_printoptions")
```

```python
mpl.rcParams['image.cmap'] = 'gray'
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
np.set_printoptions(precision=2, linewidth=140)
```

## get data: get_exp_data, map, tensor

```python
# fastnbs("idx check")
# check(get_exp_data)
```

```python
path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'
```

```python
Path
```

```python
# get_exp_data??
```

```python
x_train, y_train, x_valid, y_valid = map(tensor, get_exp_data())
```

```python
x_train.shape
type(x_train)
```

## exploratory version

```python
# fastlistnbs("groundup")

```

### w1, b1, w2, b2

```python
# n,m = x_train.shape
# c = y_train.max()+1
# n,m,c
```

```python
type(x_train)
```

```python
input_r,input_c = x_train.shape
label_num = y_train.max()+1
input_r
input_c
label_num
```

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
w1 = torch.randn(input_c,nh) # weights or coefficients for input
b1 = torch.zeros(nh) # biases
w2 = torch.randn(nh,1) # weights for hidden activations 
b2 = torch.zeros(1)
w1.shape
b1.shape
w2.shape
b2.shape
```

### lin(x,w,b)

```python
#| export groundup
def lin(x, w, b): 
    "build a single layer linear model. use torch.matmul (faster version of einsum) to create a linear model"
    return x@w + b
```

```python
hidden_layer_activations = lin(x_train, w1, b1)
hidden_layer_activations.shape
```

```python
# t = lin(x_valid, w1, b1)
# t.shape
```

### relu(x)

```python
check(torch.clamp_min)
```

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

```python
t.shape
```

```python
test_eq((t >= 0).count_nonzero(), 500000)
test_eq((t < 0).count_nonzero(), 0)
```

### model(x_valid)

```python
def model(xb):
    "build a model of 2 layers (1 hidden layer) using lin and relu"
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    res = lin(l2, w2, b2)
    return res
```

```python
res = model(x_valid)
res.shape
```

### MSE (mean of the squared error)


We need to get rid of that trailing (,1), in order to use `mse`.

```python
res[:,0].shape
```

(Of course, `mse` is not a suitable loss function for multi-class classification; we'll use a better loss function soon. We'll use `mse` for now to keep things simple.)

```python
def mse(output, targ): return (output[:,0]-targ).pow(2).mean()
```

```python
y_train, y_valid
```

```python
y_train,y_valid = y_train.float(),y_valid.float()
```

```python
preds = model(x_train)
preds.shape
```

```python
mse(preds, y_train)
```

### Gradients and backward pass


### question: gradients of input, w and b

```python
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
    b.g = out.g.sum(0)
```

#### derivaties on scalar

```python
from sympy import symbols, diff, Function
```

```python
w,b,inp = symbols('w,b,inp', real=True)
```

```python
f = Function('f')
```

```python
f = f(w,b,inp)
f
```

```python
expr = w*inp + b
expr
```

```python
import sympy
```

```python
sympy.Eq(f, expr)
```

```python
#| export groundup
def print_derivaties(func, expr, *variables):
    import sympy
    from fastdebug.utils import display_md
    display_md("$"+sympy.latex(sympy.Eq(func, expr))+"$")
    func = expr
    lst_derivatives = []
    for i in variables:
        display_md("$\\frac{\\partial f}{\\partial " + str(i) + "} =" + sympy.latex(sympy.simplify(func.diff(i))) + "$")
        lst_derivatives.append(func.diff(i))
    return lst_derivatives
```

```python
_ = print_derivaties(f, expr, inp, w, b)
```

#### derivaties on vector

```python
from sympy import symbols, Matrix, diff, Function
```

```python
u1, u2, u3, v1, v2, v3, t, b = symbols('u_1 u_2 u_3 v_1 v_2 v_3  t b', real=True)
f = Function('f')
g = Function('g')
inp = Matrix([u1,u2,u3])
inp.shape
w = Matrix([v1,v2,v3])
f = f(inp, w, b)
g = g(f)
f
g
```

```python
expr = inp.dot(w) + b
```

```python
expr
```

```python
lst = print_derivaties(f, expr, inp, w, b)
```

### forward_backward

```python
def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    diff = out[:,0]-targ
#     loss = res.pow(2).mean()
    loss = diff.pow(2).mean()
    
    # backward pass:
    out.g = 2.*diff[:,None] / inp.shape[0] # d_loss/d_diff
    # d_diff/d_out = 1
    lin_grad(l2, out, w2, b2)
    l1.g = (l1>0).float() * l2.g # derivate of relu(l1) with respect to l1
    lin_grad(inp, l1, w1, b1)
```

```python
# forward_and_backward(x_train, y_train)
```

```python
# # Save for testing against later
# w1g = w1.g.clone()
# w2g = w2.g.clone()
# b1g = b1.g.clone()
# b2g = b2.g.clone()
# ig  = x_train.g.clone()
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
# test_close(w22.grad, w2g, eps=0.01)
# test_close(b22.grad, b2g, eps=0.01)
# test_close(w12.grad, w1g, eps=0.01)
# test_close(b12.grad, b1g, eps=0.01)
# test_close(xt2.grad, ig , eps=0.01)
```

## Refactor model


### Layers as classes

```python
from snoop import snoop
# https://github.com/alexmojaki/snoop
```

```python
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out
    
    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g
```

```python
Relu()(x_train)
```

```python
class Lin():
    def __init__(self, w, b): self.w,self.b = w,b
        
    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out

    @snoop
    def backward(self):
        pp(self.inp.shape, self.out.g.shape, self.w.shape, self.w.t().shape)
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

```python
%time model.backward()
```

```python
# test_close(w2g, w2.g, eps=0.01)
# test_close(b2g, b2.g, eps=0.01)
# test_close(w1g, w1.g, eps=0.01)
# test_close(b1g, b1.g, eps=0.01)
# test_close(ig, x_train.g, eps=0.01)
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
    
#     @snoop(watch=('self.out.g.shape, self.w.shape, self.w.t().shape, inp.t().shape, self.w.g.shape, self.b.g.shape'))
    def bwd(self, out, inp):

        from snoop import pp
        with snoop: #(watch_explode=('self')):
            pp(self.out.g.shape, self.w.t().shape)
            inp.g = self.out.g @ self.w.t()
            pp(inp.t().shape)        
            self.w.g = inp.t() @ self.out.g
            pp(self.out.g.sum(0), self.out.g.sum(0).shape)
            self.b.g = self.out.g.sum(0)
```

### use snoop

```python
class Mse(Module):
    @snoop
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    
    @snoop
    def bwd(self, out, inp, targ): 
        from snoop import pp
        pp(inp.shape, inp.squeeze().shape, 2*(inp.squeeze()-targ).unsqueeze(-1).shape, targ.shape[0])
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```

```python

model = Model(w1, b1, w2, b2)
```

```python
%time loss = model(x_train, y_train)
```

```python
%time model.backward()
```

```python
# test_close(w2g, w2.g, eps=0.01)
# test_close(b2g, b2.g, eps=0.01)
# test_close(w1g, w1.g, eps=0.01)
# test_close(b1g, b1.g, eps=0.01)
# test_close(ig, x_train.g, eps=0.01)
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
model = Model(input_c, nh, 1)
loss = model(x_train, y_train)
loss.backward()
```

```python
l0 = model.layers[0]
l0.b.grad
```

```python

```
