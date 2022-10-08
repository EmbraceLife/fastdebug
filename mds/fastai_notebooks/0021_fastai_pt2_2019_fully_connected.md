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

# 0021_fastai_pt2_2019_fully_connected

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
```

## The forward and backward passes


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=4960)

```python
#export
from exp.nb_01 import *

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): return (x-m)/s
```

```python
x_train,y_train,x_valid,y_valid = get_data()
```

```python
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std
```

```python
x_train = normalize(x_train, train_mean, train_std)
# NB: Use training, not validation mean for validation set
x_valid = normalize(x_valid, train_mean, train_std)
```

```python
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std
```

```python
#export
def test_near_zero(a,tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"
```

```python
test_near_zero(x_train.mean())
test_near_zero(1-x_train.std())
```

```python
n,m = x_train.shape
c = y_train.max()+1
n,m,c
```

## Foundations version


### Basic architecture


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=5128)

```python
# num hidden
nh = 50
```

[Tinker practice](https://course19.fast.ai/videos/?lesson=8&t=5255)

```python
# standard xavier init
w1 = torch.randn(m,nh)/math.sqrt(m)
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1)/math.sqrt(nh)
b2 = torch.zeros(1)
```

```python
test_near_zero(w1.mean())
test_near_zero(w1.std()-1/math.sqrt(m))
```

```python
# This should be ~ (0,1) (mean,std)...
x_valid.mean(),x_valid.std()
```

```python
def lin(x, w, b): return x@w + b
```

```python
t = lin(x_valid, w1, b1)
```

```python
#...so should this, because we used xavier init, which is designed to do this
t.mean(),t.std()
```

```python
def relu(x): return x.clamp_min(0.)
```

```python
t = relu(lin(x_valid, w1, b1))
```

```python
#...actually it really should be this!
t.mean(),t.std()
```

From pytorch docs: `a: the negative slope of the rectifier used after this layer (0 for ReLU by default)`

$$\text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}$$

This was introduced in the paper that described the Imagenet-winning approach from *He et al*: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852), which was also the first paper that claimed "super-human performance" on Imagenet (and, most importantly, it introduced resnets!)


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=5128)

```python
# kaiming init / he init for relu
w1 = torch.randn(m,nh)*math.sqrt(2/m)
```

```python
w1.mean(),w1.std()
```

```python
t = relu(lin(x_valid, w1, b1))
t.mean(),t.std()
```

```python
#export
from torch.nn import init
```

```python
w1 = torch.zeros(m,nh)
init.kaiming_normal_(w1, mode='fan_out')
t = relu(lin(x_valid, w1, b1))
```

```python
init.kaiming_normal_??
```

```python
w1.mean(),w1.std()
```

```python
t.mean(),t.std()
```

```python
w1.shape
```

```python
import torch.nn
```

```python
torch.nn.Linear(m,nh).weight.shape
```

```python
torch.nn.Linear.forward??
```

```python
torch.nn.functional.linear??
```

```python
torch.nn.Conv2d??
```

```python
torch.nn.modules.conv._ConvNd.reset_parameters??
```

```python
# what if...?
def relu(x): return x.clamp_min(0.) - 0.5
```

```python
# kaiming init / he init for relu
w1 = torch.randn(m,nh)*math.sqrt(2./m )
t1 = relu(lin(x_valid, w1, b1))
t1.mean(),t1.std()
```

```python
def model(xb):
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3
```

```python
%timeit -n 10 _=model(x_valid)
```

```python
assert model(x_valid).shape==torch.Size([x_valid.shape[0],1])
```

### Loss function: MSE


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=6372)

```python
model(x_valid).shape
```

We need `squeeze()` to get rid of that trailing (,1), in order to use `mse`. (Of course, `mse` is not a suitable loss function for multi-class classification; we'll use a better loss function soon. We'll use `mse` for now to keep things simple.)

```python
#export
def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()
```

```python
y_train,y_valid = y_train.float(),y_valid.float()
```

```python
preds = model(x_train)
```

```python
preds.shape
```

```python
mse(preds, y_train)
```

### Gradients and backward pass


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=6493)

```python
def mse_grad(inp, targ): 
    # grad of loss with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
```

```python
def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.g = (inp>0).float() * out.g
```

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
    # we don't actually need the loss in backward!
    loss = mse(out, targ)
    
    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
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
    # forward pass:
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    out = l2 @ w22 + b22
    # we don't actually need the loss in backward!
    return mse(out, targ)
```

```python
loss = forward(xt2, y_train)
```

```python
loss.backward()
```

```python
test_near(w22.grad, w2g)
test_near(b22.grad, b2g)
test_near(w12.grad, w1g)
test_near(b12.grad, b1g)
test_near(xt2.grad, ig )
```

## Refactor model


### Layers as classes


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=7112)

```python
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)-0.5
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
        # Creating a giant outer product, just to sum it, is inefficient!
        self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0)
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
w1.g,b1.g,w2.g,b2.g = [None]*4
model = Model(w1, b1, w2, b2)
```

```python
%time loss = model(x_train, y_train)
```

```python
%time model.backward()
```

```python
test_near(w2g, w2.g)
test_near(b2g, b2.g)
test_near(w1g, w1.g)
test_near(b1g, b1.g)
test_near(ig, x_train.g)
```

### Module.forward()

```python
class Module():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out
    
    def forward(self): raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)
```

```python
class Relu(Module):
    def forward(self, inp): return inp.clamp_min(0.)-0.5
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g
```

```python
class Lin(Module):
    def __init__(self, w, b): self.w,self.b = w,b
        
    def forward(self, inp): return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = torch.einsum("bi,bj->ij", inp, out.g)
        self.b.g = out.g.sum(0)
```

```python
class Mse(Module):
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    def bwd(self, out, inp, targ): inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```

```python
class Model():
    def __init__(self):
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
w1.g,b1.g,w2.g,b2.g = [None]*4
model = Model()
```

```python
%time loss = model(x_train, y_train)
```

```python
%time model.backward()
```

```python
test_near(w2g, w2.g)
test_near(b2g, b2.g)
test_near(w1g, w1.g)
test_near(b1g, b1.g)
test_near(ig, x_train.g)
```

### Without einsum


[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=7484)

```python
class Lin(Module):
    def __init__(self, w, b): self.w,self.b = w,b
        
    def forward(self, inp): return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g
        self.b.g = out.g.sum(0)
```

```python
w1.g,b1.g,w2.g,b2.g = [None]*4
model = Model()
```

```python
%time loss = model(x_train, y_train)
```

```python
%time model.backward()
```

```python
test_near(w2g, w2.g)
test_near(b2g, b2.g)
test_near(w1g, w1.g)
test_near(b1g, b1.g)
test_near(ig, x_train.g)
```

### nn.Linear and nn.Module

```python
#export
from torch import nn
```

```python
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        self.loss = mse
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x.squeeze(), targ)
```

```python
model = Model(m, nh, 1)
```

```python
%time loss = model(x_train, y_train)
```

```python
%time loss.backward()
```

## Export

```python
!./notebook2script.py 02_fully_connected.ipynb
```

```python

```
