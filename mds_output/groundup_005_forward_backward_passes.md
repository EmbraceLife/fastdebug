# groundup_005_forward_backward_passes
---
skip_exec: true
---

```
#| default_exp delete0004
```


```
#| export
a = "to delete"
```

## imports


```
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>



```
from fastdebug.groundup import *
```


```
# %whos function
```


```
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl, numpy as np
from pathlib import Path
from torch import tensor
```


```
# fastnbs("set_printoptions")
```


```
mpl.rcParams['image.cmap'] = 'gray'
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
np.set_printoptions(precision=2, linewidth=140)
```

## get data: get_exp_data, map, tensor


```
# fastnbs("idx check")
# check(get_exp_data)
```


```
path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'
```


```
Path
```




    pathlib.Path




```
# get_exp_data??
```


```
x_train, y_train, x_valid, y_valid = map(tensor, get_exp_data())
```


```
x_train.shape
type(x_train)
```




    torch.Size([50000, 784])






    torch.Tensor



## exploratory version


```
# fastlistnbs("groundup")

```

### w1, b1, w2, b2


```
# n,m = x_train.shape
# c = y_train.max()+1
# n,m,c
```


```
type(x_train)
```




    torch.Tensor




```
input_r,input_c = x_train.shape
label_num = y_train.max()+1
input_r
input_c
label_num
```




    50000






    784






    tensor(10)




```
# num hidden activations
nh = 50
```


```
# w1 = torch.randn(m,nh)
# b1 = torch.zeros(nh)
# w2 = torch.randn(nh,1)
# b2 = torch.zeros(1)
```


```
w1 = torch.randn(input_c,nh) # weights or coefficients for input
b1 = torch.zeros(nh) # biases
w2 = torch.randn(nh,1) # weights for hidden activations 
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



### lin(x,w,b)


```
#| export groundup
def lin(x, w, b): 
    "build a single layer linear model. use torch.matmul (faster version of einsum) to create a linear model"
    return x@w + b
```


```
hidden_layer_activations = lin(x_train, w1, b1)
hidden_layer_activations.shape
```




    torch.Size([50000, 50])




```
# t = lin(x_valid, w1, b1)
# t.shape
```

### relu(x)


```
check(torch.clamp_min)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method clamp_min of type object>
    
    __module__: torch
    __doc__: not exist
    
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```
#| export groundup
def relu(x): 
    "basic relu with max in torch"
    return x.clamp_min(0.)
```


```
t = relu(lin(x_valid, w1, b1)) # add relu unto the linear model
t
```




    tensor([[ 0.00,  0.00,  1.79,  ...,  0.00,  4.86,  3.40],
            [ 0.00,  1.57,  6.56,  ...,  0.00,  6.72,  0.00],
            [ 0.25,  4.06,  0.69,  ...,  0.00,  0.00,  0.00],
            ...,
            [11.80,  0.00, 18.88,  ...,  3.64,  7.60,  0.00],
            [ 0.00, 13.53,  7.00,  ...,  5.06,  1.91,  0.00],
            [ 4.54,  4.09, 11.58,  ...,  2.35,  0.00,  0.00]])




```
t.shape
```




    torch.Size([10000, 50])




```
test_eq((t >= 0).count_nonzero(), 500000)
test_eq((t < 0).count_nonzero(), 0)
```

### model(x_valid)


```
def model(xb):
    "build a model of 2 layers (1 hidden layer) using lin and relu"
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    res = lin(l2, w2, b2)
    return res
```


```
res = model(x_valid)
res.shape
```




    torch.Size([10000, 1])



### MSE (mean of the squared error)

We need to get rid of that trailing (,1), in order to use `mse`.


```
res[:,0].shape
```




    torch.Size([10000])



(Of course, `mse` is not a suitable loss function for multi-class classification; we'll use a better loss function soon. We'll use `mse` for now to keep things simple.)


```
def mse(output, targ): return (output[:,0]-targ).pow(2).mean()
```


```
y_train, y_valid
```




    (tensor([5, 0, 4,  ..., 8, 4, 8]), tensor([3, 8, 6,  ..., 5, 6, 8]))




```
y_train,y_valid = y_train.float(),y_valid.float()
```


```
preds = model(x_train)
preds.shape
```




    torch.Size([50000, 1])




```
mse(preds, y_train)
```




    tensor(877.31)



### Gradients and backward pass

### question: gradients of input, w and b


```
def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.g = out.g @ w.t()
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0)
    b.g = out.g.sum(0)
```

#### derivaties on scalar


```
from sympy import symbols, diff, Function
```


```
w,b,inp = symbols('w,b,inp', real=True)
```


```
f = Function('f')
```


```
f = f(w,b,inp)
f
```




$\displaystyle f{\left(w,b,inp \right)}$




```
expr = w*inp + b
expr
```




$\displaystyle b + inp w$




```
import sympy
```


```
sympy.Eq(f, expr)
```




$\displaystyle f{\left(w,b,inp \right)} = b + inp w$




```
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


```
_ = print_derivaties(f, expr, inp, w, b)
```


$f{\left(w,b,inp \right)} = b + inp w$



$\frac{\partial f}{\partial inp} =w$



$\frac{\partial f}{\partial w} =inp$



$\frac{\partial f}{\partial b} =1$


#### derivaties on vector


```
from sympy import symbols, Matrix, diff, Function
```


```
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




    (3, 1)






$\displaystyle f{\left(\left[\begin{matrix}u_{1}\\u_{2}\\u_{3}\end{matrix}\right],\left[\begin{matrix}v_{1}\\v_{2}\\v_{3}\end{matrix}\right],b \right)}$






$\displaystyle g{\left(f{\left(\left[\begin{matrix}u_{1}\\u_{2}\\u_{3}\end{matrix}\right],\left[\begin{matrix}v_{1}\\v_{2}\\v_{3}\end{matrix}\right],b \right)} \right)}$




```
expr = inp.dot(w) + b
```


```
expr
```




$\displaystyle b + u_{1} v_{1} + u_{2} v_{2} + u_{3} v_{3}$




```
lst = print_derivaties(f, expr, inp, w, b)
```


$f{\left(\left[\begin{matrix}u_{1}\\u_{2}\\u_{3}\end{matrix}\right],\left[\begin{matrix}v_{1}\\v_{2}\\v_{3}\end{matrix}\right],b \right)} = b + u_{1} v_{1} + u_{2} v_{2} + u_{3} v_{3}$



$\frac{\partial f}{\partial Matrix([[u_1], [u_2], [u_3]])} =\left[\begin{matrix}v_{1}\\v_{2}\\v_{3}\end{matrix}\right]$



$\frac{\partial f}{\partial Matrix([[v_1], [v_2], [v_3]])} =\left[\begin{matrix}u_{1}\\u_{2}\\u_{3}\end{matrix}\right]$



$\frac{\partial f}{\partial b} =1$


### forward_backward


```
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


```
# forward_and_backward(x_train, y_train)
```


```
# # Save for testing against later
# w1g = w1.g.clone()
# w2g = w2.g.clone()
# b1g = b1.g.clone()
# b2g = b2.g.clone()
# ig  = x_train.g.clone()
```

We cheat a little bit and use PyTorch autograd to check our results.


```
xt2 = x_train.clone().requires_grad_(True)
w12 = w1.clone().requires_grad_(True)
w22 = w2.clone().requires_grad_(True)
b12 = b1.clone().requires_grad_(True)
b22 = b2.clone().requires_grad_(True)
```


```
def forward(inp, targ):
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    out = l2 @ w22 + b22
    return mse(out, targ)
```


```
loss = forward(xt2, y_train)
loss.backward()
```


```
from fastcore.test import test_close
```


```
# test_close(w22.grad, w2g, eps=0.01)
# test_close(b22.grad, b2g, eps=0.01)
# test_close(w12.grad, w1g, eps=0.01)
# test_close(b12.grad, b1g, eps=0.01)
# test_close(xt2.grad, ig , eps=0.01)
```

## Refactor model

### Layers as classes


```
from snoop import snoop
# https://github.com/alexmojaki/snoop
```


```
class Relu():
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)
        return self.out
    
    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g
```


```
Relu()(x_train)
```




    tensor([[0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            ...,
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.]])




```
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


```
class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out
    
    def backward(self):
        self.inp.g = 2. * (self.inp.squeeze() - self.targ).unsqueeze(-1) / self.targ.shape[0]
```


```
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


```
model = Model(w1, b1, w2, b2)
```


```
%time loss = model(x_train, y_train)
```

    CPU times: user 383 ms, sys: 25.7 ms, total: 409 ms
    Wall time: 60.4 ms



```
%time model.backward()
```

    23:05:56.75 >>> Call to Lin.backward in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_49026/878164743.py", line 10
    23:05:56.75 .......... self = <__main__.Lin object>
    23:05:56.75   10 |     def backward(self):
    23:05:56.75   11 |         pp(self.inp.shape, self.out.g.shape, self.w.shape, self.w.t().shape)
    23:05:56.75 LOG:
    23:05:57.11 .... self.inp.shape = torch.Size([50000, 50])
    23:05:57.11 .... self.out.g.shape = torch.Size([50000, 1])
    23:05:57.11 .... self.w.shape = torch.Size([50, 1])
    23:05:57.11 .... self.w.t().shape = torch.Size([1, 50])
    23:05:57.11   12 |         self.inp.g = self.out.g @ self.w.t()
    23:05:57.11   13 |         self.w.g = self.inp.t() @ self.out.g
    23:05:57.14   14 |         self.b.g = self.out.g.sum(0)
    23:05:57.14 <<< Return value from Lin.backward: None
    23:05:57.14 >>> Call to Lin.backward in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_49026/878164743.py", line 10
    23:05:57.14 .......... self = <__main__.Lin object>
    23:05:57.14   10 |     def backward(self):
    23:05:57.14   11 |         pp(self.inp.shape, self.out.g.shape, self.w.shape, self.w.t().shape)
    23:05:57.14 LOG:
    23:05:57.14 .... self.inp.shape = torch.Size([50000, 784])
    23:05:57.14 .... self.out.g.shape = torch.Size([50000, 50])
    23:05:57.14 .... self.w.shape = torch.Size([784, 50])
    23:05:57.14 .... self.w.t().shape = torch.Size([50, 784])
    23:05:57.14   12 |         self.inp.g = self.out.g @ self.w.t()
    23:05:57.21   13 |         self.w.g = self.inp.t() @ self.out.g
    23:05:57.30   14 |         self.b.g = self.out.g.sum(0)
    23:05:57.30 <<< Return value from Lin.backward: None


    CPU times: user 1.45 s, sys: 857 ms, total: 2.31 s
    Wall time: 556 ms



```
# test_close(w2g, w2.g, eps=0.01)
# test_close(b2g, b2.g, eps=0.01)
# test_close(w1g, w1.g, eps=0.01)
# test_close(b1g, b1.g, eps=0.01)
# test_close(ig, x_train.g, eps=0.01)
```

### Module.forward()


```
class Module():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self): raise Exception('not implemented')
    def bwd(self): raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)
```


```
class Relu(Module):
    def forward(self, inp): return inp.clamp_min(0.)
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g
```


```
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


```
class Mse(Module):
    @snoop
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    
    @snoop
    def bwd(self, out, inp, targ): 
        from snoop import pp
        pp(inp.shape, inp.squeeze().shape, 2*(inp.squeeze()-targ).unsqueeze(-1).shape, targ.shape[0])
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
```


```

model = Model(w1, b1, w2, b2)
```


```
%time loss = model(x_train, y_train)
```

    23:05:57.49 >>> Call to Mse.forward in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_49026/1757170846.py", line 3
    23:05:57.49 .......... self = <__main__.Mse object>
    23:05:57.49 .......... inp = tensor([[ 11.52],
    23:05:57.49                          [ 28.00],
    23:05:57.49                          [-37...   [-24.99],
    23:05:57.49                          [ -6.92],
    23:05:57.49                          [ -5.51]])
    23:05:57.49 .......... inp.shape = (50000, 1)
    23:05:57.49 .......... inp.dtype = torch.float32
    23:05:57.49 .......... targ = tensor([5., 0., 4.,  ..., 8., 4., 8.])
    23:05:57.49 .......... targ.shape = (50000,)
    23:05:57.49 .......... targ.dtype = torch.float32
    23:05:57.49    3 |     def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    23:05:57.49    3 |     def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    23:05:57.49 <<< Return value from Mse.forward: tensor(877.31)


    CPU times: user 389 ms, sys: 64.6 ms, total: 453 ms
    Wall time: 61.7 ms



```
%time model.backward()
```

    23:05:57.51 >>> Call to Mse.bwd in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_49026/1757170846.py", line 6
    23:05:57.51 .......... self = <__main__.Mse object>
    23:05:57.51 .......... out = tensor(877.31)
    23:05:57.51 .......... out.shape = ()
    23:05:57.51 .......... out.dtype = torch.float32
    23:05:57.51 .......... inp = tensor([[ 11.52],
    23:05:57.51                          [ 28.00],
    23:05:57.51                          [-37...   [-24.99],
    23:05:57.51                          [ -6.92],
    23:05:57.51                          [ -5.51]])
    23:05:57.51 .......... inp.shape = (50000, 1)
    23:05:57.51 .......... inp.dtype = torch.float32
    23:05:57.51 .......... targ = tensor([5., 0., 4.,  ..., 8., 4., 8.])
    23:05:57.51 .......... targ.shape = (50000,)
    23:05:57.51 .......... targ.dtype = torch.float32
    23:05:57.51    6 |     def bwd(self, out, inp, targ): 
    23:05:57.51    7 |         from snoop import pp
    23:05:57.51 .............. pp = <snoop.pp_module.PP object>
    23:05:57.51    8 |         pp(inp.shape, inp.squeeze().shape, 2*(inp.squeeze()-targ).unsqueeze(-1).shape, targ.shape[0])
    23:05:57.51 LOG:
    23:05:57.52 .... inp.shape = torch.Size([50000, 1])
    23:05:57.52 .... inp.squeeze().shape = torch.Size([50000])
    23:05:57.52 .... 2*(inp.squeeze()-targ).unsqueeze(-1).shape = torch.Size([50000, 1, 50000, 1])
    23:05:57.52 .... targ.shape[0] = 50000
    23:05:57.52    9 |         inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]
    23:05:57.53 <<< Return value from Mse.bwd: None
    23:05:57.53 >>> Enter with block in Lin.bwd in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_49026/3860237156.py", line 9
    23:05:57.53 .............. self = <__main__.Lin object>
    23:05:57.53 .............. out = tensor([[ 11.52],
    23:05:57.53                              [ 28.00],
    23:05:57.53                              [-37...   [-24.99],
    23:05:57.53                              [ -6.92],
    23:05:57.53                              [ -5.51]])
    23:05:57.53 .............. out.shape = (50000, 1)
    23:05:57.53 .............. out.dtype = torch.float32
    23:05:57.53 .............. inp = tensor([[10.66,  0.00, 16.42,  ...,  0.00,  2.49... 3.75,  0.00,  0.00,  ...,  0.00,  3.01,  0.00]])
    23:05:57.53 .............. inp.shape = (50000, 50)
    23:05:57.53 .............. inp.dtype = torch.float32
    23:05:57.53 .............. pp = <snoop.pp_module.PP object>
    23:05:57.53   10 |             pp(self.out.g.shape, self.w.t().shape)
    23:05:57.53 LOG:
    23:05:57.54 .... self.out.g.shape = torch.Size([50000, 1])
    23:05:57.54 .... self.w.t().shape = torch.Size([1, 50])
    23:05:57.54   11 |             inp.g = self.out.g @ self.w.t()
    23:05:57.55   12 |             pp(inp.t().shape)        
    23:05:57.55 LOG:
    23:05:57.56 .... inp.t().shape = torch.Size([50, 50000])
    23:05:57.56   13 |             self.w.g = inp.t() @ self.out.g
    23:05:57.57   14 |             pp(self.out.g.sum(0), self.out.g.sum(0).shape)
    23:05:57.57 LOG:
    23:05:57.57 .... self.out.g.sum(0) = tensor([-12.42])
    23:05:57.57 .... self.out.g.sum(0).shape = torch.Size([1])
    23:05:57.57   15 |             self.b.g = self.out.g.sum(0)
    23:05:57.58 <<< Exit with block in Lin.bwd
    23:05:57.58 >>> Enter with block in Lin.bwd in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_49026/3860237156.py", line 9
    23:05:57.58 .............. self = <__main__.Lin object>
    23:05:57.58 .............. out = tensor([[ 10.66, -10.51,  16.42,  ...,  -1.19,  ...,  -8.48,  -1.09,  ...,  -3.13,   3.01,  -1.09]])
    23:05:57.58 .............. out.shape = (50000, 50)
    23:05:57.58 .............. out.dtype = torch.float32
    23:05:57.58 .............. inp = tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    23:05:57.58                              ...0., 0.],
    23:05:57.58                              [0., 0., 0.,  ..., 0., 0., 0.]])
    23:05:57.58 .............. inp.shape = (50000, 784)
    23:05:57.58 .............. inp.dtype = torch.float32
    23:05:57.58 .............. pp = <snoop.pp_module.PP object>
    23:05:57.58   10 |             pp(self.out.g.shape, self.w.t().shape)
    23:05:57.58 LOG:
    23:05:57.58 .... self.out.g.shape = torch.Size([50000, 50])
    23:05:57.58 .... self.w.t().shape = torch.Size([50, 784])
    23:05:57.58   11 |             inp.g = self.out.g @ self.w.t()
    23:05:57.65   12 |             pp(inp.t().shape)        
    23:05:57.66 LOG:
    23:05:57.66 .... inp.t().shape = torch.Size([784, 50000])
    23:05:57.66   13 |             self.w.g = inp.t() @ self.out.g
    23:05:57.74   14 |             pp(self.out.g.sum(0), self.out.g.sum(0).shape)
    23:05:57.74 LOG:
    23:05:57.74 .... self.out.g.sum(0) = tensor([ -2.48,  18.34,  -3.39,  -3.04,   1.27,  10.89,   3.33,  -1.18,  -0.03,   0.56,  13.46,   6.57,  -7.00, -10.49,  -3.84,  23.47,
    23:05:57.74                                   -0.33,   6.83,  -0.17,  -2.58,   7.33,  17.98,  10.49,   0.82,   0.63,  -2.49,   2.80,   4.78,  -0.60, -10.30,  -9.16,   8.34,
    23:05:57.74                                    0.31,   1.92,  -6.47,  -4.49,   0.05,  -3.37,  -1.52,  -1.90,  -1.55,   3.13,  -0.05,  -4.54,  -0.34,   1.22,  13.38,   0.24,
    23:05:57.74                                   -2.23,  -0.24])
    23:05:57.74 .... self.out.g.sum(0).shape = torch.Size([50])
    23:05:57.74   15 |             self.b.g = self.out.g.sum(0)
    23:05:57.74 <<< Exit with block in Lin.bwd


    CPU times: user 958 ms, sys: 474 ms, total: 1.43 s
    Wall time: 236 ms



```
# test_close(w2g, w2.g, eps=0.01)
# test_close(b2g, b2.g, eps=0.01)
# test_close(w1g, w1.g, eps=0.01)
# test_close(b1g, b1.g, eps=0.01)
# test_close(ig, x_train.g, eps=0.01)
```

### Autograd


```
from torch import nn
import torch.nn.functional as F
```


```
class Linear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = torch.randn(n_in,n_out).requires_grad_()
        self.b = torch.zeros(n_out).requires_grad_()
    def forward(self, inp): return inp@self.w + self.b
```


```
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [Linear(n_in,nh), nn.ReLU(), Linear(nh,n_out)]
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return F.mse_loss(x, targ[:,None])
```


```
model = Model(input_c, nh, 1)
loss = model(x_train, y_train)
loss.backward()
```


```
l0 = model.layers[0]
l0.b.grad
```




    tensor([ 169.15,   17.12,   23.60,   30.69,   85.86,   27.71,  213.83,  159.29, -199.09,   41.94,    1.28,   58.02,   -1.48,   45.44,
              43.16,  134.84,   52.21,  -24.13,   10.31,    3.67,  -77.47,  -43.81,   -3.78, -157.99,  -80.99,  -25.59,    0.67,   70.84,
             108.06,   49.26,   -9.78,    6.48,  -51.64,   71.07,  -44.97,   23.96,  -42.41,   56.45,    1.68,  -50.10,    2.48,   27.53,
              64.56,  -21.02,   15.59,   -1.81,  -10.04,   87.41,  -13.01,  -72.34])




```

```
