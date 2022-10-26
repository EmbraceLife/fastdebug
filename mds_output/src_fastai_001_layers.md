```
#|hide
#| eval: false
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab
```


```
#|default_exp layers
#|default_cls_lvl 3
```


```
#|export
from __future__ import annotations
from fastai.imports import *
from fastai.torch_imports import *
from fastai.torch_core import *
from torch.nn.utils import weight_norm, spectral_norm
```


```
#|hide
from nbdev.showdoc import *
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>



```
import fastai.layers as fl
import fastai.torch_core as ft
```


```
whatinside(ft)
whatinside(fl)
```

    fastai.torch_core has: 
    85 items in its __all__, and 
    316 user defined functions, 
    137 classes or class objects, 
    4 builtin funcs and methods, and
    476 callables.
    
    None
    fastai.layers has: 
    61 items in its __all__, and 
    342 user defined functions, 
    172 classes or class objects, 
    4 builtin funcs and methods, and
    541 callables.
    
    None


# Layers
> Custom fastai layers and basic functions to grab them.

## Basic manipulations and resize

### ```module(*flds, **defaults)```
- Decorator to create an `nn.Module` using `f` as `forward` method
- create parameters from `flds` and `defaults` and make fileds of args from their keys or names
- make the decorated function eg. `Identity` a subclass of ```nn.Module``` and 
- make `Identity` function itself to be the `forward` function of the subclass of ```nn.Module```



```
#|export
# @snoop
# @pysnoop()
def module(*flds, **defaults):
    "Decorator to create an `nn.Module` using `f` as `forward` method"
    pa = [inspect.Parameter(o, inspect.Parameter.POSITIONAL_OR_KEYWORD) for o in flds]
    pb = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=v)
          for k,v in defaults.items()]
    params = pa+pb
    all_flds = [*flds,*defaults.keys()]
    
#     @snoop
#     @pysnoop()
    def _f(f):
        class c(nn.Module):
#             @snoop # to enable debug for Identity()
            def __init__(self, *args, **kwargs):
                super().__init__()
                for i,o in enumerate(args): kwargs[all_flds[i]] = o
                kwargs = merge(defaults,kwargs)
                for k,v in kwargs.items(): setattr(self,k,v)
            __repr__ = basic_repr(all_flds)
            forward = f # making Identity's own function to be the forward function?
        c.__signature__ = inspect.Signature(params)
        c.__name__ = c.__qualname__ = f.__name__
        c.__doc__  = f.__doc__
        return c
    return _f
```


```

```


```
#|export
@module() # running module() and return _f
# @snoop
def Identity(self, x): # running _f(Identify) and return c, c has __name__ as `Identity` which is a subclass of nn.Module
    "Do nothing at all"
    return x
```


```
# doc(module)
# doc(Identity)
# fastnbs("module(*", "src")
# module?
```


```
pp(module)
# ic(Identity()(1)) # running Identity's own function
```

    14:32:14.54 LOG:
    14:32:14.66 .... module = <function module>





    <function __main__.module(*flds, **defaults)>




```
test_eq(Identity()(1), 1)
```

### ```Lambda(self, x)```
- An easy way to create a pytorch layer for a simple `func`
- using ```module``` decorator, and make ```Lambda``` a subclass of ```nn.Module``` and create a parameter `func` for ```Lambda```
- run ```Lambda(func)``` to make the `func` a pytorch layer


```
#|export
@module('func')
# @snoop
def Lambda(self, x):
    "An easy way to create a pytorch layer for a simple `func`"
    return self.func(x)
```


```
def _add2(x): return x+2
tst = Lambda(_add2)
tst
```




    __main__.Lambda(func=<function _add2>)




```
x = torch.randn(10,20)
test_eq(tst(x), x+2) # running foward function?
```


```
tst2 = pickle.loads(pickle.dumps(tst)) # question: why dumps and then loads again (check the rubostness of the func?)
test_eq(tst2(x), x+2)
tst
```




    __main__.Lambda(func=<function _add2>)



### ```PartialLambda(Lambda)```
- Layer that applies `partial(func, **kwargs)`"


```
# fastnbs("module(*flds", filter_folder="src")
```

### ```PartialLambda(Lambda)```
- a subclass of Lambda, which is a subclass of module, which wrap around nn.Module
- Layer that applies `partial(func, **kwargs)` which can custom the `func` of ```Lambda```


```
#|export
class PartialLambda(Lambda):
    "Layer that applies `partial(func, **kwargs)`"
    def __init__(self, func, **kwargs):
        super().__init__(partial(func, **kwargs))
        self.repr = f'{func.__name__}, {kwargs}'

    def forward(self, x): return self.func(x)
    def __repr__(self): return f'{self.__class__.__name__}({self.repr})'
```


```
def test_func(a,b=2): return a+b
tst = PartialLambda(test_func, b=5)
x.shape
test_eq(tst(x), x+5)
```




    torch.Size([10, 20])



### ```view(self:Tensor-1), x.view(x.size(0), -1)```
- flatten x into a 1d tensor
- flatten x into a 2d tensor, keep the 1dim unchanged, but flatten the rest dims



```
# Tensor.view?
```


```
x = torch.randn(4, 4)
x.size()
y = x.view(16)
y.size()
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
z.size()
z1 = x.view(-1)
z1.shape
```




    torch.Size([4, 4])






    torch.Size([16])






    torch.Size([2, 8])






    torch.Size([16])



### ```Flatten(self, x)```
- Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor"
Logic: 
- use decorator ```module(full=False)``` to make ```Flatten``` a layer and create a parameter ```full=False```
- ```Flatten(self, x)``` works as the `forward` function
- `self.full` can be access in the `forward` function above
- ```Flatten(full=True)```: to flatten all dims of a tensor into a 1d tensor
- ```Flatten(full=False)```: to keep 1st dim and flatten the rest dims, so only 2 dims remains



```
#|export
@module(full=False)
# @snoop
def Flatten(self, x):
    "Flatten `x` to a single dimension, e.g. at end of a model. `full` for rank-1 tensor"
#     pp(x.shape, x.view(-1).shape, x.size(0), x.size(), x.view(x.size(0), -1).shape)
    return TensorBase(x.view(-1) if self.full else x.view(x.size(0), -1))
```


```
tst = Flatten() # this is running __init__
x = torch.randn(10,5,4)
```


```
test_eq(tst(x).shape, [10,20])
tst = Flatten(full=True)
test_eq(tst(x).shape, [200])
```

### ```ToTensorBase(self, x)```
- make ```ToTensorBase``` a subclass of `module` which is a subclass of `nn.Module`
- ```ToTensorBase(tensor_cls=TensorBase)``` initialize itself with a tensor class (default to TensorBase) as a parameter
- after initialization, the output function can take in `x` to turn `x` into an instance of ```TensorBase```


```
#|export
@module(tensor_cls=TensorBase) # the args here are for ToTensorBase.__init__
def ToTensorBase(self, x):
    "Convert x to TensorBase"
    return self.tensor_cls(x)
```


```
# fastnbs("def module(")
```


```
ttb = ToTensorBase()
timg = TensorImage(torch.rand(1,3,32,32))
test_eq(type(ttb(timg)), TensorBase)
```

### ```View(Module)```
- ```View``` is a subclass of ```Module```, which inherites from ```nn.Module``` and ```metaclass=PrePostInitMeta```
- so, ```View``` is to create a layer for Viewing data
- ```View(*size)``` can initialize itself by setting values for `self.size`
- ```View.forward(x)``` can run `x.view(self.size)` to create a new tensor based on `x` but with different shape


```
# fastnbs("class Module(") # to remind me of `Module`
```


```
#|export
class View(Module):
    "Reshape `x` to `size`"
    def __init__(self, *size): self.size = size
    def forward(self, x): return x.view(self.size)
```


```
x = torch.randn(4,5,10)
tst = View(10,5,4)
test_eq(tst(x).shape, [10,5,4])
```

### ```ResizeBatch(Module)```
- ```ResizeBatch``` is a subclass of nn.Module and no need to run `super().__init__`
- ```ResizeBatch(*size)``` can initialize itself with a specific shape/size for tensors
- ```rb(x)``` can reshape `x` so that the batch size dim is unchanged but other dims is changed based on `*size`


```
(3,) + (1,2,3)
```




    (3, 1, 2, 3)




```
#|export
class ResizeBatch(Module):
    "Reshape `x` to `size`, keeping batch dim the same size"
    def __init__(self, *size): self.size = size
    def forward(self, x): return x.view((x.size(0),) + self.size)
```


```
tst = ResizeBatch(5,4)
x = torch.randn(10,20)
test_eq(tst(x).shape, [10,5,4])
```

### ```Debugger(self,x)```
- ```Debugger``` is made into a layer by decorator `module` using `nn.Module`
- after initialization, `db(x)` will run `set_trace()` and return `x` which is a model object


```
# fastnbs("module(*")
```


```
#|export
@module()
def Debugger(self,x):
    "A module to debug inside a model."
    set_trace()
    return x
```


```
tst = nn.Sequential(nn.Linear(4,5), nn.Sequential(nn.Linear(4,5), nn.Linear(4,5)))
tst
```




    Sequential(
      (0): Linear(in_features=4, out_features=5, bias=True)
      (1): Sequential(
        (0): Linear(in_features=4, out_features=5, bias=True)
        (1): Linear(in_features=4, out_features=5, bias=True)
      )
    )




```
# Debugger()(tst) # run this code to activate the ipdb
```

### ```sigmoid_range(x, low, high)```
- calculate sigmoid on tensor `x` and also keep the sigmoid values within `[low, high]`


```
#|export
def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low
```


```
test = tensor([-10.,0.,10.])
torch.sigmoid(test)
sigmoid_range(test, -1,  2)
```




    tensor([4.5398e-05, 5.0000e-01, 9.9995e-01])






    tensor([-0.9999,  0.5000,  1.9999])




```
assert torch.allclose(sigmoid_range(test, -1,  2), tensor([-1.,0.5, 2.]), atol=1e-4, rtol=1e-4)
assert torch.allclose(sigmoid_range(test, -5, -1), tensor([-5.,-3.,-1.]), atol=1e-4, rtol=1e-4)
assert torch.allclose(sigmoid_range(test,  2,  4), tensor([2.,  3., 4.]), atol=1e-4, rtol=1e-4)
```

### ```SigmoidRange(self, x)```
- ```SigmoidRange``` is a subclass of `nn.Module`, and the func defined under `SigmoidRange` is used as `forward` func, thanks to ```module``` decorator
- ```sr = SigmoidRange(low, high)``` initialize an instance with the value range `[low, high]`, with `low` and `high` as args for `__init__`
- `sr(x)` to calc sigmoid on `x` and put it within the range `[low, high]`


```
#|export
@module('low','high')
def SigmoidRange(self, x):
    "Sigmoid module with range `(low, high)`"
    return sigmoid_range(x, self.low, self.high)
```


```
# fastlistnbs("src")
# fastnbs("module(*f", "src")
```


```
tst = SigmoidRange(-1, 2)
assert torch.allclose(tst(test), tensor([-1.,0.5, 2.]), atol=1e-4, rtol=1e-4)
```

## Pooling layers

### ```AdaptiveConcatPool1d(Module)```
- becomes a layer, which is a subclass of `Module`, which inherits from `nn.Module` and `PrePostInitMeta` to avoid `super().__init__`
- this layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d` side by side with each other
- ```acp = AdaptiveConcatPool1d(size)``` to initialize the layer with size or num of activations
- `acp(x)` is to run tensor `x` through the layer, and if `x.shape` is (5, 10), then the output shape is (5, 2)


```
#|export
class AdaptiveConcatPool1d(Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
```


```
AdaptiveConcatPool1d()
```




    AdaptiveConcatPool1d(
      (ap): AdaptiveAvgPool1d(output_size=1)
      (mp): AdaptiveMaxPool1d(output_size=1)
    )




```
type(AdaptiveConcatPool1d())
```




    __main__.AdaptiveConcatPool1d




```
x.shape
```




    torch.Size([10, 20])




```
list(AdaptiveConcatPool1d().children())[0](x).shape
```




    torch.Size([10, 1])




```
AdaptiveConcatPool1d()(x).shape
```




    torch.Size([10, 2])



### ```torch.max(a, dim, keepdim)```


```
a = torch.randn(4, 4)
a
```




    tensor([[ 0.5431,  0.4291,  0.6668, -0.1806],
            [ 1.7736, -1.4782, -0.7680, -0.4568],
            [-1.7035, -0.4617, -1.7458, -0.2594],
            [-1.2322, -1.4792,  0.8141, -0.7532]])




```
torch.max(a, 1)
torch.max(a, 1)[0].shape
torch.max(a, 1)[1].shape
torch.max(a, dim=1, keepdim=True)
torch.max(a, dim=1, keepdim=True)[0].shape
torch.max(a, dim=1, keepdim=True)[1].shape
```




    torch.return_types.max(
    values=tensor([ 0.6668,  1.7736, -0.2594,  0.8141]),
    indices=tensor([2, 0, 3, 2]))






    torch.Size([4])






    torch.Size([4])






    torch.return_types.max(
    values=tensor([[ 0.6668],
            [ 1.7736],
            [-0.2594],
            [ 0.8141]]),
    indices=tensor([[2],
            [0],
            [3],
            [2]]))






    torch.Size([4, 1])






    torch.Size([4, 1])




```
torch.max(a, 0)
torch.max(a, 0)[0].shape
torch.max(a, 0)[1].shape
torch.max(a, dim=0, keepdim=True)
torch.max(a, dim=0, keepdim=True)[0].shape
torch.max(a, dim=0, keepdim=True)[1].shape
```




    torch.return_types.max(
    values=tensor([ 1.7736,  0.4291,  0.8141, -0.1806]),
    indices=tensor([1, 0, 3, 0]))






    torch.Size([4])






    torch.Size([4])






    torch.return_types.max(
    values=tensor([[ 1.7736,  0.4291,  0.8141, -0.1806]]),
    indices=tensor([[1, 0, 3, 0]]))






    torch.Size([1, 4])






    torch.Size([1, 4])



### ```AdaptiveConcatPool2d(Module)```
- it is like ```AdaptiveConcatPoold(Module)```, but deal with 2d
- Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
- If the input is `bs x nf x h x h`, the output will be `bs x 2*nf x 1 x 1` if no size is passed or `bs x 2*nf x size x size` (nf: num of filters)


```
#|export
class AdaptiveConcatPool2d(Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`"
    def __init__(self, size=None):
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.size)
        self.mp = nn.AdaptiveMaxPool2d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
```


```
tst = AdaptiveConcatPool2d()
x = torch.randn(10,5,4,4)
test_eq(tst(x).shape, [10,10,1,1])
```


```
max1 = torch.max(x,    dim=2, keepdim=True)[0]
max2 = torch.max(x,    dim=2, keepdim=False)[0]
maxp = torch.max(max1, dim=3, keepdim=True)[0]
max1.shape
maxp.shape
x.shape
tst(x).shape
test_eq(tst(x)[:,:5], maxp)
test_eq(tst(x)[:,5:], x.mean(dim=[2,3], keepdim=True))
```




    torch.Size([10, 5, 1, 4])






    torch.Size([10, 5, 1, 1])






    torch.Size([10, 5, 4, 4])






    torch.Size([10, 10, 1, 1])




```
tst = AdaptiveConcatPool2d(2)
x.shape
test_eq(tst(x).shape, [10,10,2,2])
```




    torch.Size([10, 5, 4, 4])



### ```PoolType.Avg, PoolType.Max, PoolType.Cat```
- they are class properties, which are strings `Avg`, `Max` and `Cat`


```
#|export
class PoolType: Avg,Max,Cat = 'Avg','Max','Cat'
```

### ```adaptive_pool(pool_type)```
- `pool_type` can be `Avg`, `Max` or `Cat`
- return `nn.AdaptiveAvgPool2d`, `nn.AdaptiveMaxPool2d`, `nn.AdaptiveConcatPool2d`


```
#|export
def adaptive_pool(pool_type):
    return nn.AdaptiveAvgPool2d if pool_type=='Avg' else nn.AdaptiveMaxPool2d if pool_type=='Max' else AdaptiveConcatPool2d
```

### ```nn.AdaptiveAvgPool2d((*output_size))```
- to initialize the layer using output_size such as `(5,7)`, `7`, `(None, 7)`


```
# nn.AdaptiveAvgPool2d??
# target output size of 5x7
m = nn.AdaptiveAvgPool2d((5,7))
m
input = torch.randn(1, 64, 8, 9)
output = m(input)
output.shape
```




    AdaptiveAvgPool2d(output_size=(5, 7))






    torch.Size([1, 64, 5, 7])




```
# target output size of 7x7 (square)
m = nn.AdaptiveAvgPool2d(7)
m
input = torch.randn(1, 64, 10, 9)
output = m(input)
output.shape
```




    AdaptiveAvgPool2d(output_size=7)






    torch.Size([1, 64, 7, 7])




```
# target output size of 10x7
m = nn.AdaptiveAvgPool2d((None, 7))
m
input = torch.randn(1, 64, 10, 9)
output = m(input)
output.shape
```




    AdaptiveAvgPool2d(output_size=(None, 7))






    torch.Size([1, 64, 10, 7])



### ```PoolFlatten(nn.Sequential)```
- it inherits from `nn.Sequential`, so it can be a layer
- it combines `nn.AdaptiveAvgPool2d` and `Flatten`
- its `nn.AdaptiveAvgPool2d` layer has the last 2 dims to be (1,1)
- its `Flatten` layer only keeps two dims


```
#|export
class PoolFlatten(nn.Sequential):
    "Combine `nn.AdaptiveAvgPool2d` and `Flatten`."
    def __init__(self, pool_type=PoolType.Avg): super().__init__(adaptive_pool(pool_type)(1), Flatten())
```


```
# fastnbs("Flatten(")
```


```
tst = PoolFlatten()
tst
x.shape
nn.AdaptiveAvgPool2d(1)(x).shape
Flatten()(nn.AdaptiveAvgPool2d(1)(x)).shape
test_eq(tst(x).shape, [10,5])
test_eq(tst(x), x.mean(dim=[2,3]))
```




    PoolFlatten(
      (0): AdaptiveAvgPool2d(output_size=1)
      (1): __main__.Flatten(full=False)
    )






    torch.Size([10, 5, 4, 4])






    torch.Size([10, 5, 1, 1])






    torch.Size([10, 5])




```
ic(x)
```

    ic| x: class=<class 'torch.Tensor'>, shape=torch.Size([10, 5, 4, 4]), dtype=torch.float32





    tensor([[[[ 7.1276e-02,  6.2517e-01, -2.2132e+00, -1.7874e-01],
              [-3.2758e-01, -1.2376e+00,  2.0178e+00, -1.0494e+00],
              [-1.8578e+00, -7.9698e-01, -2.1171e+00, -5.7127e-01],
              [ 3.5175e-01, -9.9976e-01, -4.4960e-02, -5.3514e-01]],
    
             [[-1.5330e+00, -6.3157e-01, -1.9691e-01,  1.5494e+00],
              [-1.8135e-01,  8.2392e-01,  6.2565e-01,  2.6549e+00],
              [-1.0391e+00, -7.7068e-01, -4.1978e-01, -8.8519e-01],
              [ 6.2824e-01,  7.7295e-03,  1.1718e+00,  1.7979e-01]],
    
             [[ 1.3124e+00, -5.9516e-01, -1.5508e+00,  2.4107e+00],
              [ 1.9538e+00, -7.1425e-02, -1.7505e+00, -3.5783e-01],
              [ 1.8213e+00,  4.5503e-01,  4.4617e-01, -6.1684e-01],
              [ 4.9954e-01, -1.5206e-02,  4.4372e-01,  7.6883e-01]],
    
             [[ 5.1626e-01, -2.0851e+00,  6.3266e-01, -6.9705e-01],
              [ 3.8760e-01,  4.9122e-01, -8.1681e-01, -5.4353e-01],
              [-3.0987e-01,  8.5079e-01, -1.4253e+00, -1.8715e+00],
              [-4.7243e-01,  1.1677e+00,  3.2027e-01, -3.9756e-01]],
    
             [[ 6.6441e-01, -2.7888e-01, -9.1732e-01,  5.2301e-02],
              [ 2.4174e-01, -3.8863e-01,  2.5564e+00, -1.8568e+00],
              [-1.4144e+00,  1.6601e+00, -6.2815e-01,  2.1165e-01],
              [-8.8526e-01,  9.9137e-01, -9.1870e-01,  9.7471e-01]]],
    
    
            [[[ 8.5411e-01, -1.4079e-01, -4.5996e-01,  1.9226e+00],
              [ 4.4002e-01, -4.7346e-01, -6.6095e-01, -1.2353e+00],
              [ 2.8805e-01,  2.1752e-01,  1.2827e+00,  2.0792e+00],
              [-1.5756e+00,  5.8671e-03,  1.7082e+00, -6.7465e-01]],
    
             [[-1.6744e-02, -5.8505e-01, -9.6304e-01,  9.8751e-01],
              [-7.0186e-01,  9.4427e-01, -1.2360e+00, -1.1824e+00],
              [-1.8416e-01,  2.0272e+00, -9.7819e-01,  1.2365e+00],
              [ 1.6972e+00,  1.3064e+00,  4.6088e-01,  1.1749e+00]],
    
             [[ 1.2804e+00,  1.2733e+00,  9.1404e-01, -5.6902e-01],
              [-3.8190e-01, -9.2348e-01,  1.2402e+00,  3.2749e-01],
              [ 3.9479e-01, -1.3263e-01,  7.4480e-02,  3.8929e-01],
              [ 1.0001e+00,  3.9114e-01,  6.6925e-01, -3.6250e-01]],
    
             [[ 1.4952e+00,  1.4829e-01, -6.7750e-01,  1.3470e+00],
              [ 2.8724e-01, -4.1802e-01, -1.0474e+00,  9.9734e-01],
              [-4.9104e-01,  1.1030e+00,  1.4318e-01,  9.4340e-01],
              [ 4.9610e-01, -9.8606e-01,  5.2903e-01, -1.5625e+00]],
    
             [[-2.0098e-01, -5.0178e-01,  2.1208e+00, -1.9780e-01],
              [ 2.3420e+00,  4.0106e-01, -9.6023e-01,  1.4530e-01],
              [-9.4389e-01, -4.7905e-01,  8.4070e-02, -7.7703e-01],
              [ 6.7524e-01,  6.4435e-01,  1.5233e+00, -1.6391e+00]]],
    
    
            [[[-1.4528e-01, -1.2144e+00,  1.0056e+00,  2.0253e+00],
              [-1.1327e+00,  3.5422e-01, -1.0266e-01, -1.0216e+00],
              [-6.0221e-01, -1.0852e-01,  2.0836e-01,  1.7681e+00],
              [-1.6425e+00, -5.7871e-01,  3.4348e-02, -5.0475e-01]],
    
             [[ 2.6669e-01,  3.9647e-02,  2.4831e-02,  1.7188e+00],
              [-9.4571e-01, -7.1550e-02, -1.7187e+00,  2.1262e+00],
              [-2.5101e-01, -4.2545e-01, -8.3281e-01,  1.1361e+00],
              [ 8.4980e-01,  1.7948e+00, -9.1904e-01, -4.5934e-01]],
    
             [[ 2.1983e-01, -1.7947e+00, -2.5603e+00, -1.9525e-01],
              [-1.3886e+00,  3.7017e-01,  1.6856e-01, -1.1704e+00],
              [-2.0337e+00, -9.0179e-01, -1.6478e-02,  1.0710e+00],
              [-1.1617e+00, -2.9705e-01, -3.8188e-01, -1.4006e-01]],
    
             [[ 1.7651e+00, -1.5793e-01, -1.6319e-01, -7.5060e-01],
              [-3.2650e-01, -7.6616e-01,  1.0208e+00, -1.6736e-01],
              [ 1.1899e+00, -1.8894e+00,  1.0354e+00,  1.1029e+00],
              [-2.8614e-01,  1.6467e-01, -1.0387e-01, -2.3302e-01]],
    
             [[ 1.2813e-01,  9.1702e-01,  2.0176e+00, -7.2630e-02],
              [ 6.4941e-01, -3.0242e-02, -7.6090e-01,  1.3246e+00],
              [-8.3618e-01,  1.0300e+00, -7.1378e-01, -9.9908e-01],
              [ 1.3054e+00,  8.5191e-01, -1.1092e+00,  4.9509e-01]]],
    
    
            [[[-8.5098e-01,  2.9827e-02, -8.0978e-01,  8.1100e-01],
              [ 7.1080e-03, -5.7866e-01, -9.2739e-01, -1.4993e+00],
              [-6.0394e-01, -4.3079e-01,  6.7823e-02,  3.6457e-01],
              [ 1.5380e+00,  9.7004e-01, -1.1404e+00, -1.8412e+00]],
    
             [[-4.1824e-01, -1.2558e+00, -7.2759e-01,  3.9580e-01],
              [-7.5232e-01,  7.9374e-02, -3.1311e-01,  3.8938e-02],
              [-6.2674e-02,  8.4168e-01,  6.2125e-02,  9.0617e-01],
              [ 1.0313e+00, -1.2334e+00, -1.8282e+00,  7.1337e-01]],
    
             [[-9.5536e-01,  1.1021e+00, -4.0973e-01, -2.0955e+00],
              [-9.6094e-02, -1.3681e+00, -5.6422e-01,  5.7673e-01],
              [-6.6747e-02,  5.1652e-01, -2.1394e-01, -7.3456e-01],
              [ 5.1635e-01,  3.9295e-01, -2.7970e+00,  1.5529e+00]],
    
             [[-7.2798e-01,  1.1982e+00, -1.2034e+00, -4.4999e-01],
              [ 3.4875e-01, -8.5097e-01, -1.6014e-01,  7.4394e-01],
              [ 8.3235e-01, -9.6810e-01, -6.6616e-01,  5.3891e-01],
              [-1.3793e-01,  1.0814e+00, -1.3625e+00, -3.1790e+00]],
    
             [[ 2.8055e-01, -3.2892e-01,  1.0078e+00,  2.7217e-01],
              [-1.1820e+00,  1.3828e+00,  1.8733e-01,  7.5096e-01],
              [-4.4867e-01,  5.7199e-01, -9.5139e-01,  1.1595e+00],
              [-9.1969e-01, -2.1889e+00, -2.1795e-01, -6.7772e-01]]],
    
    
            [[[-1.4230e+00, -1.2827e+00, -6.4773e-01,  5.3069e-01],
              [-9.4603e-01,  1.3846e+00,  1.9679e-01, -1.0207e+00],
              [-1.9975e+00,  5.5610e-01,  9.6428e-01, -1.5693e+00],
              [-5.8120e-01, -1.6824e+00,  1.3976e+00,  2.1085e+00]],
    
             [[ 1.0263e+00,  1.5669e+00,  9.8732e-02, -4.8245e-01],
              [ 6.9305e-01, -2.8643e-01,  1.0906e+00, -1.3857e+00],
              [ 8.2051e-01, -1.1270e+00, -6.5998e-01,  8.2600e-01],
              [-4.8435e-01,  5.8733e-01, -1.4307e+00,  6.3304e-01]],
    
             [[-1.3710e+00, -2.1436e+00,  1.2375e+00, -5.8922e-01],
              [-1.3469e-01, -2.5069e-01,  4.5243e-01,  1.1170e+00],
              [ 1.2932e+00, -8.1922e-01,  5.4539e-01, -9.0520e-01],
              [ 5.3671e-01, -1.9746e-01, -1.6610e+00,  5.1803e-01]],
    
             [[-7.3704e-01,  8.3018e-02, -9.0494e-01,  5.1647e-01],
              [ 8.9332e-01, -6.0252e-01,  1.0689e+00,  3.8105e-01],
              [-7.6559e-02,  1.4634e+00, -8.5146e-02,  5.6729e-01],
              [-5.8978e-02,  2.9077e-02,  2.8729e-01, -1.1237e-01]],
    
             [[ 9.1113e-01, -4.2713e-01,  1.1678e+00,  5.5829e-02],
              [-5.5915e-01, -7.5248e-01,  4.8039e-01, -1.7192e-01],
              [ 2.3167e+00, -8.1557e-01,  7.7049e-01, -8.0069e-01],
              [ 2.6477e-01,  1.3390e+00, -5.8780e-01,  1.2042e+00]]],
    
    
            [[[-4.7799e-02,  2.5133e+00, -6.5872e-01,  1.2875e+00],
              [-1.4708e-01,  1.6382e-02, -8.0350e-01,  7.2154e-01],
              [ 1.0486e+00,  1.4450e+00, -1.1485e+00,  1.1143e+00],
              [-8.4803e-01,  1.2372e-02, -1.8934e+00,  6.2387e-01]],
    
             [[-5.8095e-01, -1.1230e+00,  1.1767e+00,  6.7539e-01],
              [-1.0221e+00, -8.7211e-01,  2.2970e-01, -5.1538e-01],
              [ 2.1181e+00,  1.7756e+00, -6.9463e-01, -1.5532e-01],
              [-1.5628e-01,  4.0780e-02, -3.5777e-01, -7.9609e-01]],
    
             [[-3.5896e-01, -7.2146e-01,  5.8672e-01,  5.9493e-01],
              [-8.1050e-01, -2.0215e+00, -8.9023e-01, -2.5140e-01],
              [-5.9221e-01,  3.2656e-01, -1.9404e+00, -1.1726e+00],
              [-7.3787e-01, -2.3600e+00, -1.1365e+00,  1.2152e-01]],
    
             [[ 6.3743e-01,  1.1481e+00,  6.8125e-01,  4.4055e-01],
              [ 2.8476e-01, -1.2878e+00,  9.3413e-01, -1.4156e+00],
              [ 1.3514e-01,  4.4844e-02, -2.5468e-01,  6.0548e-01],
              [ 1.3280e+00, -9.0382e-01,  8.0037e-01, -1.8158e+00]],
    
             [[-4.3236e-01,  9.5112e-01,  2.6078e-02, -8.3347e-01],
              [ 1.0782e+00,  1.0826e+00,  4.5012e-01,  1.3466e-01],
              [ 1.0359e-01,  5.4636e-01,  1.1700e+00,  1.8953e-01],
              [ 4.9144e-01,  1.4636e-01,  1.4520e+00,  9.4950e-01]]],
    
    
            [[[-2.1518e-01,  3.8338e-02, -8.0790e-01,  8.2000e-01],
              [ 4.4115e-01,  1.0701e+00,  9.7729e-01,  1.1446e+00],
              [ 4.7575e-01,  1.7557e-01, -9.1124e-01,  1.9234e-01],
              [-1.8970e+00,  1.9903e-01, -4.7748e-01,  2.4640e+00]],
    
             [[-4.1412e-01,  1.1166e+00, -1.9670e+00, -9.3118e-02],
              [ 1.0180e+00,  1.2118e+00,  1.0237e+00, -2.0243e+00],
              [-2.8915e-01,  1.8342e+00, -6.9385e-01,  1.0136e+00],
              [ 1.7862e-01,  5.0190e-01,  9.1958e-01,  1.7784e-01]],
    
             [[ 8.6724e-01,  5.1904e-01,  1.8337e+00,  2.4936e-01],
              [-2.3488e-02, -7.6858e-01,  3.5942e-01,  5.4792e-02],
              [ 7.5185e-01, -9.0920e-01, -6.9346e-01, -8.9782e-02],
              [-1.8352e+00,  1.1977e+00, -1.5806e-01,  1.0841e+00]],
    
             [[-2.0023e+00,  9.7873e-01, -2.3510e-01, -1.2064e-01],
              [ 3.2778e-01,  1.0221e-02, -1.5931e+00, -3.8158e-01],
              [-1.4134e+00, -1.8253e+00,  8.3570e-01, -1.9947e-03],
              [ 1.4017e+00,  3.1338e-01, -1.3754e+00,  1.1153e+00]],
    
             [[-4.3511e-01, -8.6626e-01,  2.9169e-01, -2.1154e+00],
              [-1.3038e+00,  1.2810e+00, -1.0570e+00,  9.8999e-01],
              [-1.9927e+00,  1.3580e+00, -1.3337e+00,  9.6701e-01],
              [-9.1993e-01, -1.5865e+00, -1.2045e+00, -3.3228e-01]]],
    
    
            [[[ 5.6998e-01, -8.2750e-01,  2.8479e-01,  3.1080e-01],
              [-4.8005e-01,  1.0459e+00, -3.4289e-01,  8.3390e-01],
              [-1.1884e+00, -2.3039e-01,  7.1923e-01, -4.2334e-01],
              [-1.1472e+00, -9.1375e-01, -1.2548e+00,  1.2815e+00]],
    
             [[ 1.4915e-01, -9.6142e-02, -4.1310e-01,  4.5889e-02],
              [ 5.4780e-01, -3.3337e-01, -7.3116e-01, -2.9561e-01],
              [ 1.0882e+00,  6.5868e-02,  1.6538e-01,  3.4795e-01],
              [-3.7871e-01,  1.0249e+00,  1.7850e+00,  1.3535e+00]],
    
             [[-1.8688e+00, -1.6023e+00,  9.5732e-01,  2.6037e-01],
              [ 4.3183e-01, -1.6706e+00, -4.6861e-01, -2.1245e+00],
              [-2.6538e-01,  5.7490e-01, -1.5282e+00,  1.6099e+00],
              [ 4.8484e-01, -3.3317e-01, -8.2136e-01, -1.8570e+00]],
    
             [[ 2.3119e+00,  2.0212e-01,  1.2814e-01, -3.7424e-01],
              [ 2.6964e-01, -2.1957e-01, -1.2322e+00, -1.4888e-02],
              [-2.2166e+00, -1.9315e-01,  2.0875e+00, -9.9234e-01],
              [-7.2910e-02,  1.1604e+00, -8.2046e-01,  2.2093e+00]],
    
             [[ 1.4975e+00,  6.5460e-01, -1.6791e+00,  5.7435e-01],
              [-9.5246e-01, -2.0844e-01, -7.0477e-01, -1.7094e+00],
              [ 1.1892e+00, -1.1758e+00, -5.4757e-03,  5.9981e-01],
              [-9.0864e-01,  1.1139e+00,  1.7115e+00,  3.4402e-02]]],
    
    
            [[[ 4.0445e-01, -8.7644e-01, -2.0054e-01, -1.2665e+00],
              [ 2.2018e+00,  1.5964e+00,  1.6571e+00, -4.7530e-01],
              [-1.2392e+00,  2.8490e-01,  3.8575e-01, -9.6754e-01],
              [-2.7140e-02, -6.5488e-01, -1.2949e-01,  1.0551e+00]],
    
             [[-9.0379e-01, -1.1746e-01,  8.6639e-03, -1.9275e-02],
              [-5.7271e-01,  5.4609e-01,  1.9900e-02, -1.1271e+00],
              [-9.3153e-01, -6.4726e-01, -2.9966e-01, -3.0680e-01],
              [ 2.0507e-01, -1.0066e+00, -7.9067e-01, -4.8482e-01]],
    
             [[-9.4636e-01, -5.8670e-01,  1.2884e+00, -1.1309e+00],
              [-4.2506e-02,  1.2170e-01,  1.1384e+00,  1.3565e+00],
              [ 3.1184e-01,  1.6576e-01, -3.6241e-01, -7.3425e-01],
              [ 1.1574e-01, -1.6562e+00,  3.8258e-02, -3.4908e-02]],
    
             [[-4.5905e-01,  9.9646e-01,  8.2948e-02,  1.4327e+00],
              [ 2.0450e+00, -1.3238e+00,  1.0999e+00,  1.2909e-01],
              [ 1.6498e+00,  1.5863e+00,  1.3788e+00, -1.1040e+00],
              [ 1.2618e-01,  1.3593e+00,  1.3069e+00,  9.2345e-02]],
    
             [[-9.3038e-01, -5.0516e-01, -8.3015e-01,  1.1183e+00],
              [ 9.8178e-01,  1.1917e-02, -3.7628e-01, -2.0041e-02],
              [ 9.2492e-01, -1.1911e-01, -1.1774e+00, -5.4928e-03],
              [-3.9663e-01, -1.9163e+00,  1.8818e+00, -3.5154e-01]]],
    
    
            [[[ 1.9737e+00, -9.4061e-03, -1.0948e+00,  4.8829e-01],
              [-1.2299e+00,  2.8213e-01, -8.2643e-01, -9.7332e-01],
              [-3.7367e-02,  3.0301e+00,  6.9472e-01, -3.7477e-01],
              [-2.6270e-01, -6.6064e-01, -6.5184e-01,  1.4470e+00]],
    
             [[-1.0854e-01,  7.7681e-01, -7.2263e-02, -1.5369e+00],
              [ 3.9759e-01,  6.7642e-01, -2.0758e+00,  1.1063e-01],
              [-1.6925e-01, -2.3881e-02,  9.7442e-02, -5.7205e-01],
              [ 3.0789e-01,  5.3282e-01, -6.8828e-01, -1.0591e+00]],
    
             [[ 8.8402e-02, -2.6184e-01,  4.4935e-02, -7.1720e-01],
              [ 5.5470e-01, -1.4828e+00,  2.4339e-01, -4.4988e-01],
              [-1.0658e+00,  2.1370e-01,  1.9690e-01, -1.3239e+00],
              [-4.2803e-01, -1.5254e+00,  7.5737e-02,  9.9495e-01]],
    
             [[-1.2079e+00,  2.0546e-01, -2.7789e+00, -6.1982e-01],
              [-6.2037e-01,  1.4004e-01,  8.3062e-01,  3.5284e-01],
              [ 1.4910e+00,  1.3343e+00, -7.4336e-01, -8.0683e-01],
              [ 3.7078e-01,  3.9941e-01,  1.8679e+00, -1.0093e+00]],
    
             [[ 9.2598e-01,  1.1377e+00, -2.0298e-01, -6.4767e-01],
              [ 8.0823e-01, -4.7065e-01,  1.0002e+00, -1.0118e-01],
              [-9.3336e-01, -5.2469e-01,  5.1067e-01, -1.9203e-01],
              [-3.2204e-01, -7.3693e-01,  2.3474e-01, -1.8345e+00]]]])



## BatchNorm layers

### ```NormType``` with `Enum`


```
#|export
NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

```


```
# help(Enum)
list(NormType)
```




    [<NormType.Batch: 1>,
     <NormType.BatchZero: 2>,
     <NormType.Weight: 3>,
     <NormType.Spectral: 4>,
     <NormType.Instance: 5>,
     <NormType.InstanceZero: 6>]




```
list(NormType)[0].name
list(NormType)[0].value
```




    'Batch'






    1




```
check(NormType)
```

    signature: (value, names=None, *, module=None, qualname=None, type=None, start=1)
    __class__: <class 'enum.EnumMeta'>
    __repr__: <enum 'NormType'>
    
    __module__: __main__
    __doc__:
    An enumeration.
    __dict__: 
    mappingproxy({'Batch': <NormType.Batch: 1>,
                  'BatchZero': <NormType.BatchZero: 2>,
                  'Instance': <NormType.Instance: 5>,
                  'InstanceZero': <NormType.InstanceZero: 6>,
                  'Spectral': <NormType.Spectral: 4>,
                  'Weight': <NormType.Weight: 3>,
                  '__doc__': 'An enumeration.',
                  '__module__': '__main__',
                  '__new__': <function Enum.__new__>,
                  '_generate_next_value_': <function Enum._generate_next_value_>,
                  '_member_map_': {'Batch': <NormType.Batch: 1>,
                                   'BatchZero': <NormType.BatchZero: 2>,
                                   'Instance': <NormType.Instance: 5>,
                                   'InstanceZero': <NormType.InstanceZero: 6>,
                                   'Spectral': <NormType.Spectral: 4>,
                                   'Weight': <NormType.Weight: 3>},
                  '_member_names_': ['Batch',
                                     'BatchZero',
                                     'Weight',
                                     'Spectral',
                                     'Instance',
                                     'InstanceZero'],
                  '_member_type_': <class 'object'>,
                  '_value2member_map_': {1: <NormType.Batch: 1>,
                                         2: <NormType.BatchZero: 2>,
                                         3: <NormType.Weight: 3>,
                                         4: <NormType.Spectral: 4>,
                                         5: <NormType.Instance: 5>,
                                         6: <NormType.InstanceZero: 6>}})
    metaclass: False
    class: True
    decorator: False
    function: False
    method: False



```
check(list(NormType)[0])
```

    signature: None
    __class__: <enum 'NormType'>
    __repr__: NormType.Batch
    
    __module__: __main__
    __doc__:
    An enumeration.
    __dict__: 
    {'__objclass__': <enum 'NormType'>, '_name_': 'Batch', '_value_': 1}
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False


### ```_get_norm(prefix, nf, ndim=2, zero=False, **kwargs)```
official doc: Norm layer with `nf` features and `ndim` initialized depending on `norm_type`.

My doc: to create a `nn.BatchNorm` between 1d to 3d, and output `nf` activation, and can set `weight.data` to either 0 or 1
- to get normalization layer
- `prefix`: tell which type of normalization layer, like 'BatchNorm'
- `ndim=2`: default to 2d, so we get `BatchNorm2d`
- `nf`: like 15, to return 15 output or activation at the end of the BatchNorm2d layer
- `zero`: True or False, to set BatchNorm layer's weight to be either 0 or 1
- `bn.affine`: when it is False, then weight and bias will be None


```
#|export
# @snoop(watch=('bn.bias.data', 'bn.weight.data'))
def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
#     pp.deep(lambda: getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs))
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn
```

### ```BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs)```
Official doc:  BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`.

My doc: create a BatchNorm layer (2d, by default) by wrapping around `_get_norm`
- use kwargs from `nn.BatchNorm2d`
- `ndim=2`: by default to create a `nn.BatchNorm2d`
- `nf`: like 15, to output 15 activations
- `norm_type`: if not `NormType.BatchZero`, then make `wegith.data` all equals 1; otherwise, equals 0


```
NormType.Batch
NormType.BatchZero
```




    <NormType.Batch: 1>






    <NormType.BatchZero: 2>




```
#|export
@delegates(nn.BatchNorm2d) # pass its args to BatchNorm
def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)
```


```
# help(torch.nn.modules.batchnorm.BatchNorm2d) # to check the meaning of variables
```


```
BatchNorm # receive kwargs from nn.BatchNorm2d
```




    <function __main__.BatchNorm(nf, ndim=2, norm_type=<NormType.Batch: 1>, *, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None)>




```
tst = BatchNorm(15)
assert isinstance(tst, nn.BatchNorm2d)
test_eq(tst.weight, torch.ones(15))
tst = BatchNorm(15, norm_type=NormType.BatchZero)
test_eq(tst.weight, torch.zeros(15))
tst = BatchNorm(15, ndim=1)
assert isinstance(tst, nn.BatchNorm1d)
tst = BatchNorm(15, ndim=3)
assert isinstance(tst, nn.BatchNorm3d)
test_eq(BatchNorm(15, affine=False).weight, None)
```

### ```InstanceNorm(nf, ndim=2, norm_type=NormType.Instance, affine=True, **kwargs)```
official doc: InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`.

mydoc: to create a InstanceNorm layer (1d-3d), any num of activations, set weight.data to 0 or 1, set `affine` True by default
- wrapping around `_get_norm`
- using kwargs from `nn.InstanceNorm2d`; 
- default to `NormType.Instance` and `weight.data` will be set to 1; if `NormType.InstanceZero` then `weight.data` is set to 0


```
#|export
@delegates(nn.InstanceNorm2d)
def InstanceNorm(nf, ndim=2, norm_type=NormType.Instance, affine=True, **kwargs):
    "InstanceNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('InstanceNorm', nf, ndim, zero=norm_type==NormType.InstanceZero, affine=affine, **kwargs)
```

`kwargs` are passed to `nn.BatchNorm` and can be `eps`, `momentum`, `affine` and `track_running_stats`.


```
tst = InstanceNorm(15)
assert isinstance(tst, nn.InstanceNorm2d)
test_eq(tst.weight, torch.ones(15))
tst = InstanceNorm(15, norm_type=NormType.InstanceZero)
test_eq(tst.weight, torch.zeros(15))
tst = InstanceNorm(15, ndim=1)
assert isinstance(tst, nn.InstanceNorm1d)
tst = InstanceNorm(15, ndim=3)
assert isinstance(tst, nn.InstanceNorm3d)
```

If `affine` is false the weight should be `None`


```
test_eq(BatchNorm(15, affine=False).weight, None)
test_eq(InstanceNorm(15, affine=False).weight, None)
```

### ```BatchNorm1dFlat(nn.BatchNorm1d)```, `running_mean`, `running_var`, `contiguous`
official doc: `nn.BatchNorm1d`, but first flattens leading dimensions

mydoc: allow high dim `x` to run through `nn.BatchNorm1d` by flattening leading dims first, and return `x` in its original shape
- how to use `torch.Tensor.contiguous`: stackoverflow [answer](https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch)
- how to access `bn.running_mean` and `bn.running_var`


```
#|export
class BatchNorm1dFlat(nn.BatchNorm1d):
    "`nn.BatchNorm1d`, but first flattens leading dimensions"
#     @snoop(watch=('snp.shape', 'help(x.contiguous)'))
    def forward(self, x):
        if x.dim()==2: 
            return super().forward(x)
        *f,l = x.shape
#         snp = x.contiguous()
#         snp = snp.view(-1,1)
        x = x.contiguous().view(-1,l)
        return super().forward(x).view(*f,l)
```


```
# check(BatchNorm1dFlat)
# help(BatchNorm1dFlat)
# help(torch.nn.modules.batchnorm._NormBase)
# help(torch.nn.modules.module.Module)
```


```
tst = BatchNorm1dFlat(15)
tst
tst.running_mean
tst.running_var
```




    BatchNorm1dFlat(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)






    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])






    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```
x = torch.randn(32, 64, 15)
y = tst(x)
y.shape
tst.running_mean
tst.running_var
```




    torch.Size([32, 64, 15])






    tensor([ 3.4016e-03,  8.1919e-04, -9.7062e-05, -2.5036e-03,  3.2423e-04,
            -2.1983e-03, -1.6188e-03,  2.6421e-04, -2.3959e-03,  1.6011e-03,
             1.6295e-03, -3.4547e-03,  1.4706e-03,  1.4990e-03, -6.8275e-04])






    tensor([0.9978, 0.9982, 1.0006, 1.0035, 1.0039, 0.9943, 0.9998, 1.0038, 0.9978,
            1.0001, 1.0034, 1.0029, 1.0004, 1.0056, 0.9969])




```
mean = x.mean(dim=[0,1])
test_close(tst.running_mean, 0*0.9 + mean*0.1)
var = (x-mean).pow(2).mean(dim=[0,1])
test_close(tst.running_var, 1*0.9 + var*0.1, eps=1e-4)
test_close(y, (x-mean)/torch.sqrt(var+1e-5) * tst.weight + tst.bias, eps=1e-4)
```

### ```LinBnDrop(nn.Sequential)```
official doc: Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"

mydoc: create a block of layers (BatchNorm1d, Dropout, Linear) together 
- `lin_first=False`: default to put linear layer to the end of the block
- `act=None`: default to None, adding a something (None, or a layer like nn.ReLu, maybe) behind linear layer
- `p=0.`: default to 0., as num of dropouts
- `bn=True`: default to True, to have a BatchNorm layer or not; if True, the linear layer removes bias
- `n_in, n_out`: num of input and output activations


```
#|export
class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [BatchNorm(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)
```

The `BatchNorm` layer is skipped if `bn=False`, as is the dropout if `p=0.`. Optionally, you can add an activation for after the linear layer with `act`.


```
tst = LinBnDrop(10, 20)
tst
mods = list(tst.children())
mods
test_eq(len(mods), 2)
assert isinstance(mods[0], nn.BatchNorm1d)
assert isinstance(mods[1], nn.Linear)
```




    LinBnDrop(
      (0): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Linear(in_features=10, out_features=20, bias=False)
    )






    [BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Linear(in_features=10, out_features=20, bias=False)]




```
tst = LinBnDrop(10, 20, p=0.1)
tst
mods = list(tst.children())
mods
test_eq(len(mods), 3)
assert isinstance(mods[0], nn.BatchNorm1d)
assert isinstance(mods[1], nn.Dropout)
assert isinstance(mods[2], nn.Linear)
```




    LinBnDrop(
      (0): BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Dropout(p=0.1, inplace=False)
      (2): Linear(in_features=10, out_features=20, bias=False)
    )






    [BatchNorm1d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     Dropout(p=0.1, inplace=False),
     Linear(in_features=10, out_features=20, bias=False)]




```
tst = LinBnDrop(10, 20, act=nn.ReLU(), lin_first=True)
tst
mods = list(tst.children())
mods
test_eq(len(mods), 3)
assert isinstance(mods[0], nn.Linear)
assert isinstance(mods[1], nn.ReLU)
assert isinstance(mods[2], nn.BatchNorm1d)
```




    LinBnDrop(
      (0): Linear(in_features=10, out_features=20, bias=False)
      (1): ReLU()
      (2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )






    [Linear(in_features=10, out_features=20, bias=False),
     ReLU(),
     BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]




```
tst = LinBnDrop(10, 20, bn=False)
tst
mods = list(tst.children())
mods
test_eq(len(mods), 1)
assert isinstance(mods[0], nn.Linear)
```




    LinBnDrop(
      (0): Linear(in_features=10, out_features=20, bias=True)
    )






    [Linear(in_features=10, out_features=20, bias=True)]



## Inits

### ```clamp(min, max)```


```
help(x.clamp)
```

    Help on built-in function clamp:
    
    clamp(...) method of torch.Tensor instance
        clamp(min=None, max=None) -> Tensor
        
        See :func:`torch.clamp`
    



```
x = torch.randn(2,3)
x
x.sigmoid()
x.sigmoid().clamp(0,0.5)
```




    tensor([[ 1.5912,  0.3023, -0.6173],
            [-0.2088,  0.6322,  0.3258]])






    tensor([[0.8308, 0.5750, 0.3504],
            [0.4480, 0.6530, 0.5807]])






    tensor([[0.5000, 0.5000, 0.3504],
            [0.4480, 0.5000, 0.5000]])



### ```sigmoid(input, eps=1e-7)```
official docs: Same as `torch.sigmoid`, plus clamping to `(eps,1-eps)`

mydoc: wrap around `torch.sigmoid` and clamping values to be within `[eps, 1-eps]`


```
#|export
def sigmoid(input, eps=1e-7):
    "Same as `torch.sigmoid`, plus clamping to `(eps,1-eps)"
    return input.sigmoid().clamp(eps,1-eps)
```


```
x = torch.randn(2,3)
x
x.sigmoid()
x.sigmoid().clamp(0,0.5)
```




    tensor([[-0.2419, -1.7405, -0.4980],
            [ 1.6586, -0.2702,  1.0093]])






    tensor([[0.4398, 0.1493, 0.3780],
            [0.8401, 0.4329, 0.7329]])






    tensor([[0.4398, 0.1493, 0.3780],
            [0.5000, 0.4329, 0.5000]])




```
sigmoid(x)
x
```




    tensor([[0.4398, 0.1493, 0.3780],
            [0.8401, 0.4329, 0.7329]])






    tensor([[-0.2419, -1.7405, -0.4980],
            [ 1.6586, -0.2702,  1.0093]])



### ```sigmoid_(input, eps=1e-7)```
official docs: Same as `torch.sigmoid_`, plus clamping to `(eps,1-eps)`

mydoc: inplace version of `sigmoid`


```
#|export
def sigmoid_(input, eps=1e-7):
    "Same as `torch.sigmoid_`, plus clamping to `(eps,1-eps)"
    return input.sigmoid_().clamp_(eps,1-eps)
```


```
x = torch.randn(2,3)
x
sigmoid_(x)
x
```




    tensor([[ 0.3758, -0.4757, -0.3201],
            [ 1.6543, -2.6505, -1.0499]])






    tensor([[0.5929, 0.3833, 0.4206],
            [0.8395, 0.0660, 0.2592]])






    tensor([[0.5929, 0.3833, 0.4206],
            [0.8395, 0.0660, 0.2592]])



### ```kaiming_uniform_,uniform_,xavier_uniform_,normal_``` from `torch.nn.init`


```
#|export
from torch.nn.init import kaiming_uniform_,uniform_,xavier_uniform_,normal_
```

### ```vleaky_relu(input, inplace=True)```
original docs: `F.leaky_relu` with 0.3 slope

mydoc: wrap `F.leaky_rely` and set `negative_slop` to 0.3 and set `inplace` True


```
#|export
def vleaky_relu(input, inplace=True):
    "`F.leaky_relu` with 0.3 slope"
    return F.leaky_relu(input, negative_slope=0.3, inplace=inplace)
```


```
x = torch.randn(2,3)
x
F.leaky_relu(x)
vleaky_relu(x)
```




    tensor([[-0.6327,  0.6887,  0.2330],
            [ 0.7055, -0.4074, -0.8898]])






    tensor([[-0.0063,  0.6887,  0.2330],
            [ 0.7055, -0.0041, -0.0089]])






    tensor([[-0.1898,  0.6887,  0.2330],
            [ 0.7055, -0.1222, -0.2669]])



### ```__default_init__``` of all ReLus are set to ```kaiming_uniform_```


```
#|export
for o in F.relu,nn.ReLU,F.relu6,nn.ReLU6,F.leaky_relu,nn.LeakyReLU:
    o.__default_init__ = kaiming_uniform_
```

### ```__default_init__``` of all sigmoid are set to ```xavier_uniform_```


```
#|export
for o in F.sigmoid,nn.Sigmoid,F.tanh,nn.Tanh,sigmoid,sigmoid_:
    o.__default_init__ = xavier_uniform_
```

### ```nested_callable(m, 'bias.fill_')```


```
m = nn.Linear(3,2)
m.bias.data
with torch.no_grad(): nested_callable(m, 'bias.fill_')(0.)
```




    tensor([ 0.1619, -0.1003])






    Parameter containing:
    tensor([0., 0.], requires_grad=True)



### ```init_default(m, func=nn.init.kaiming_normal_)```
official docs:Initialize `m` weights with `func` and set `bias` to 0.

mydoc: 
- initialize a model `m.weight` with `func` which default to `kaiming_normal_`
- initialize `m.bias` with 0


```
#|export
def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad(): nested_callable(m, 'bias.fill_')(0.)
    return m
```


```
m = nn.Linear(3,2)
m.weight
m.bias
init_default(m)
m.weight
m.bias
```




    Parameter containing:
    tensor([[-0.4545,  0.3245, -0.0279],
            [-0.2206, -0.4405,  0.5584]], requires_grad=True)






    Parameter containing:
    tensor([-0.3999,  0.1045], requires_grad=True)






    Linear(in_features=3, out_features=2, bias=True)






    Parameter containing:
    tensor([[ 1.0997, -0.6984,  0.2521],
            [-0.0327,  0.4805,  0.2035]], requires_grad=True)






    Parameter containing:
    tensor([0., 0.], requires_grad=True)



### ```init_linear(m, act_func=None, init='auto', bias_std=0.01)```
mydoc: initialize a linear layer or any layer's weight and bias
- normalize bias with 0 mean and bias_std=0.01 by default; if bias is not available or bias_std is None, then set biase to be 0
- normalize weight with `kaiming_uniform_`



```
#|export
def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m,'bias',None) is not None and bias_std is not None:
        if bias_std != 0: normal_(m.bias, 0, bias_std)
        else: m.bias.data.zero_()
    if init=='auto':
        if act_func in (F.relu_,F.leaky_relu_): init = kaiming_uniform_
        else: init = nested_callable(act_func, '__class__.__default_init__')
        if init == noop: init = getcallable(act_func, '__default_init__')
    if callable(init): init(m.weight)
```


```
normal_
```




    <function torch.nn.init.normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor>



## Convolutions

### ```_conv_func(ndim=2, transpose=False)```
official: Return the proper conv `ndim` function, potentially a `transposed`

mydoc: return a conv layer with 1d to 3d, can be transposed if set True


```
#|export
def _conv_func(ndim=2, transpose=False):
    "Return the proper conv `ndim` function, potentially `transposed`."
    assert 1 <= ndim <=3
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')
```


```
#|hide
test_eq(_conv_func(ndim=1),torch.nn.modules.conv.Conv1d)
test_eq(_conv_func(ndim=2),torch.nn.modules.conv.Conv2d)
test_eq(_conv_func(ndim=3),torch.nn.modules.conv.Conv3d)
test_eq(_conv_func(ndim=1, transpose=True),torch.nn.modules.conv.ConvTranspose1d)
test_eq(_conv_func(ndim=2, transpose=True),torch.nn.modules.conv.ConvTranspose2d)
test_eq(_conv_func(ndim=3, transpose=True),torch.nn.modules.conv.ConvTranspose3d)
```

### ```defaults.activation``` is set to `nn.ReLU`


```
#|export
defaults.activation=nn.ReLU
```

### ```weight_norm```


```
# help(weight_norm)
nn.Linear(20, 40)
m = weight_norm(nn.Linear(20, 40), name='weight')
m
m.weight_g.size()
m.weight_v.size()
```




    Linear(in_features=20, out_features=40, bias=True)






    Linear(in_features=20, out_features=40, bias=True)






    torch.Size([40, 1])






    torch.Size([40, 20])



### ```ConvLayer(nn.Sequential)```
official:    Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers.

mydoc: create a block/sequence of layers including convolutional, ReLU and norm_type layers
- use `padding` and `transpose` to set padding to be `(ks-1)/2` or 0
- set `bn` True if either `NormType.Batch` or `NormType.BatchZero`
- set `inn` True if either `NormType.Instance` or `NormType.InstanceZero`
- set `bias` True, if `bn` or `inn` is False and `bias` is given as None
- `conv_func` is assigned to a conv layer class created by `_conv_func` with `ndim` dimension and `transpose` or not
- `conv` is assigned to an actual conv layer object by running `conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)`
- `act` is assigned to None or a layer class by calling `act_cls()` which gives us ReLU
- use `init_linear(conv, act, init=init, bias_std=bias_std)` to initialize weight and bias of `conv` 
- use `weight_norm` or `spectral_norm` to normalize the weight of `conv` if `norm_type == NormType.Weight` or `==NormType.Spectral`
- create a list `act_bn` to store `act` layer, `BatchNorm(nf, norm_type=norm_type, ndim=ndim)`, `InstanceNorm(nf, norm_type=norm_type, ndim=ndim)` if `act is not None`, `bn, inn` are True respectively; and reverse the list order if `bn_1st` True
- put `conv` layer in the front of the `act_bn` list and assign the new list to `layers`
- if there is `xtra` layer, then add it to the end of the list `layers`
- finally asking the `super()` i.e., `nn.Sequential` initialze all the layers inside `layers`


```
#|export
class ConvLayer(nn.Sequential):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and `norm_type` layers."
    @delegates(nn.Conv2d)
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, ndim=2, norm_type=NormType.Batch, bn_1st=True,
                 act_cls=defaults.activation, transpose=False, init='auto', xtra=None, bias_std=0.01, **kwargs):
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        layers = [conv]
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn: act_bn.append(InstanceNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st: act_bn.reverse()
        layers += act_bn
        if xtra: layers.append(xtra)
        super().__init__(*layers)
```

The convolution uses `ks` (kernel size) `stride`, `padding` and `bias`. `padding` will default to the appropriate value (`(ks-1)//2` if it's not a transposed conv) and `bias` will default to `True` the `norm_type` is `Spectral` or `Weight`, `False` if it's `Batch` or `BatchZero`. Note that if you don't want any normalization, you should pass `norm_type=None`.

This defines a conv layer with `ndim` (1,2 or 3) that will be a ConvTranspose if `transpose=True`. `act_cls` is the class of the activation function to use (instantiated inside). Pass `act=None` if you don't want an activation function. If you quickly want to change your default activation, you can change the value of `defaults.activation`.

`init` is used to initialize the weights (the bias are initialized to 0) and `xtra` is an optional layer to add at the end.


```
tst = ConvLayer(16, 32)
tst
```




    ConvLayer(
      (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )




```
mods = list(tst.children())
mods
```




    [Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
     BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     ReLU()]




```
test_eq(len(mods), 3)
test_eq(mods[1].weight, torch.ones(32))
test_eq(mods[0].padding, (1,1))
```


```
x = torch.randn(64, 16, 8, 8)#.cuda()
```


```
#Padding is selected to make the shape the same if stride=1
test_eq(tst(x).shape, [64,32,8,8])
```


```
#Padding is selected to make the shape half if stride=2
tst = ConvLayer(16, 32, stride=2)
test_eq(tst(x).shape, [64,32,4,4])
```


```
#But you can always pass your own padding if you want
tst = ConvLayer(16, 32, padding=0)
test_eq(tst(x).shape, [64,32,6,6])
```


```
#No bias by default for Batch NormType
assert mods[0].bias is None
#But can be overridden with `bias=True`
tst = ConvLayer(16, 32, bias=True)
assert first(tst.children()).bias is not None
#For no norm, or spectral/weight, bias is True by default
for t in [None, NormType.Spectral, NormType.Weight]:
    tst = ConvLayer(16, 32, norm_type=t)
    assert first(tst.children()).bias is not None
```


```
#Various n_dim/tranpose
tst = ConvLayer(16, 32, ndim=3)
assert isinstance(list(tst.children())[0], nn.Conv3d)
```


```
tst = ConvLayer(16, 32, ndim=1, transpose=True)
assert isinstance(list(tst.children())[0], nn.ConvTranspose1d)
```


```
#No activation/leaky
tst = ConvLayer(16, 32, ndim=3, act_cls=None)
tst
mods = list(tst.children())
mods
test_eq(len(mods), 2)
```




    ConvLayer(
      (0): Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )






    [Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
     BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]




```
tst = ConvLayer(16, 32, ndim=3, act_cls=partial(nn.LeakyReLU, negative_slope=0.1))
tst
mods = list(tst.children())
mods
test_eq(len(mods), 3)
assert isinstance(mods[2], nn.LeakyReLU)
```




    ConvLayer(
      (0): Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
      (1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.1)
    )






    [Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
     BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
     LeakyReLU(negative_slope=0.1)]




```
# #export
# def linear(in_features, out_features, bias=True, act_cls=None, init='auto'):
#     "Linear layer followed by optional activation, with optional auto-init"
#     res = nn.Linear(in_features, out_features, bias=bias)
#     if act_cls: act_cls = act_cls()
#     init_linear(res, act_cls, init=init)
#     if act_cls: res = nn.Sequential(res, act_cls)
#     return res
```


```
# #export
# @delegates(ConvLayer)
# def conv1d(ni, nf, ks, stride=1, ndim=1, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)
```


```
# #export
# @delegates(ConvLayer)
# def conv2d(ni, nf, ks, stride=1, ndim=2, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)
```


```
# #export
# @delegates(ConvLayer)
# def conv3d(ni, nf, ks, stride=1, ndim=3, norm_type=None, **kwargs):
#     "Convolutional layer followed by optional activation, with optional auto-init"
#     return ConvLayer(ni, nf, ks, stride=stride, ndim=ndim, norm_type=norm_type, **kwargs)
```

### ```AdaptiveAvgPool(sz=1, ndim=2)```
official: nn.AdaptiveAvgPool layer for `ndim`

instantiate an AdaptiveAvgPool2d layer object with 1 activation output by default
- it can be 1d to 3d
- it can output any number of activations with `sz` arg


```
#|export
def AdaptiveAvgPool(sz=1, ndim=2):
    "nn.AdaptiveAvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)
```


```
AdaptiveAvgPool(3, 3)
```




    AdaptiveAvgPool3d(output_size=3)



### ```MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False)```
official: nn.MaxPool layer for `ndim`

instantiate an nn.MaxPool2d layer object with kernel size 2, stride 2, padding 0, no ceil_mode by default
- it can be 1d to 3d
- according to `nn.MaxPool2d`, by default `stride` is equal to `ks`


```
#|export
def MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.MaxPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"MaxPool{ndim}d")(ks, stride=stride, padding=padding)
```


```
# help(nn.MaxPool2d)
```


```
MaxPool()
```




    MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)




```
MaxPool(3, ndim=3)
```




    MaxPool3d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)



### ```AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False)```
official: nn.AvgPool layer for `ndim`

instantiate an nn.AvgPool2d layer object with kernel size 2, stride 2, padding 0, no ceil_mode by default
- it can be 1d to 3d
- according to `nn.AvgPool2d`, by default `stride` is equal to `ks`


```
#|export
def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    "nn.AvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)
```


```
AvgPool()
AvgPool(3, 5, 2, 3)
```




    AvgPool2d(kernel_size=2, stride=2, padding=0)






    AvgPool3d(kernel_size=3, stride=5, padding=2)



## Embeddings

### ```trunc_normal_(x, mean=0., std=1.)```
official: Truncated normal initialization (approximation)

This is to implement a finding from a paper. There is discussion on how to implement it. https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12


```
#|export
def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)
```

### ```Embedding(nn.Embedding)```
official: Embedding layer with truncated normal initialization

- is a subclass of `nn.Embedding`
- instantiate an embedding layer with `nn.Embedding(num_input, n_features, std=0.01)`
- then apply truncated normalization on the weight using std=0.01 by default


```
#|export
class Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    def __init__(self, ni, nf, std=0.01):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)
```


```
Embedding(10, 5)
```




    Embedding(10, 5)



Truncated normal initialization bounds the distribution to avoid large value. For a given standard deviation `std`, the bounds are roughly `-2*std`, `2*std`.


```
std = 0.02
tst = Embedding(10, 30, std)
assert tst.weight.min() > -2*std
assert tst.weight.max() < 2*std
test_close(tst.weight.mean(), 0, 1e-2)
test_close(tst.weight.std(), std, 0.1)
```

## Self attention

### ```SelfAttention(Module)```
official: Self attention layer for `n_channels`.

To build SelfAttention from scratch, key implementation details is discussed below
- `sa = SelfAttention(n_channels)` to instantiate a SelfAttention layer
- during instantiation, 3 conv1d layers are created with `n_in`, `n_out` calculated based on `n_channels`
- the forward function is to implement the paper in the link below


```
#|export
class SelfAttention(Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
```

Self-attention layer as introduced in [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318).

Initially, no change is done to the input. This is controlled by a trainable parameter named `gamma` as we return `x + gamma * out`.


```
tst = SelfAttention(16)
tst
tst.gamma.data
```




    SelfAttention(
      (query): ConvLayer(
        (0): Conv1d(16, 2, kernel_size=(1,), stride=(1,), bias=False)
      )
      (key): ConvLayer(
        (0): Conv1d(16, 2, kernel_size=(1,), stride=(1,), bias=False)
      )
      (value): ConvLayer(
        (0): Conv1d(16, 16, kernel_size=(1,), stride=(1,), bias=False)
      )
    )






    tensor([0.])




```
x = torch.randn(32, 16, 8, 8)
test_eq(tst(x),x)
```

Then during training `gamma` will probably change since it's a trainable parameter. Let's see what's happening when it gets a nonzero value.


```
tst.gamma.data.fill_(1.)
y = tst(x)
test_eq(y.shape, [32,16,8,8])
test_ne(y, x)
```




    tensor([1.])



The attention mechanism requires three matrix multiplications (here represented by 1x1 convs). The multiplications are done on the channel level (the second dimension in our tensor) and we flatten the feature map (which is 8x8 here). As in the paper, we note `f`, `g` and `h` the results of those multiplications.


```
tst.query
tst.query[0]
```




    ConvLayer(
      (0): Conv1d(16, 2, kernel_size=(1,), stride=(1,), bias=False)
    )






    Conv1d(16, 2, kernel_size=(1,), stride=(1,), bias=False)




```
q,k,v = tst.query[0].weight.data,tst.key[0].weight.data,tst.value[0].weight.data
test_eq([q.shape, k.shape, v.shape], [[2, 16, 1], [2, 16, 1], [16, 16, 1]])
f,g,h = map(lambda m: x.view(32, 16, 64).transpose(1,2) @ m.squeeze().t(), [q,k,v])
test_eq([f.shape, g.shape, h.shape], [[32,64,2], [32,64,2], [32,64,16]])
```

The key part of the attention layer is to compute attention weights for each of our location in the feature map (here 8x8 = 64). Those are positive numbers that sum to 1 and tell the model to pay attention to this or that part of the picture. We make the product of `f` and the transpose of `g` (to get something of size bs by 64 by 64) then apply a softmax on the first dimension (to get the positive numbers that sum up to 1). The result can then be multiplied with `h` transposed to get an output of size bs by channels by 64, which we can then be viewed as an output the same size as the original input. 

The final result is then `x + gamma * out` as we saw before.


```
beta = F.softmax(torch.bmm(f, g.transpose(1,2)), dim=1)
test_eq(beta.shape, [32, 64, 64])
out = torch.bmm(h.transpose(1,2), beta)
test_eq(out.shape, [32, 16, 64])
test_close(y, x + out.view(32, 16, 8, 8), eps=1e-4)
```

### ```PooledSelfAttention2d(Module)```
official: Pooled self attention layer for 2d.

Implemented from scratch and build with the template of `SelfAttention`, and the difference between `SelfAttention` is discussed below


```
#|export
class PooledSelfAttention2d(Module):
    "Pooled self attention layer for 2d."
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels//2)]
        self.out   = self._conv(n_channels//2, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        return ConvLayer(n_in, n_out, ks=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        n_ftrs = x.shape[2]*x.shape[3]
        f = self.query(x).view(-1, self.n_channels//8, n_ftrs)
        g = F.max_pool2d(self.key(x),   [2,2]).view(-1, self.n_channels//8, n_ftrs//4)
        h = F.max_pool2d(self.value(x), [2,2]).view(-1, self.n_channels//2, n_ftrs//4)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)
        o = self.out(torch.bmm(h, beta.transpose(1,2)).view(-1, self.n_channels//2, x.shape[2], x.shape[3]))
        return self.gamma * o + x
```

Self-attention layer used in the [Big GAN paper](https://arxiv.org/abs/1809.11096).

It uses the same attention as in `SelfAttention` but adds a max pooling of stride 2 before computing the matrices `g` and `h`: the attention is ported on one of the 2x2 max-pooled window, not the whole feature map. There is also a final matrix product added at the end to the output, before retuning `gamma * out + x`.


```
PooledSelfAttention2d(8)
```




    PooledSelfAttention2d(
      (query): ConvLayer(
        (0): Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (key): ConvLayer(
        (0): Conv2d(8, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (value): ConvLayer(
        (0): Conv2d(8, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
      (out): ConvLayer(
        (0): Conv2d(4, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
    )



### ```_conv1d_spect(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False)```
official : Create and initialize a `nn.Conv1d` layer with spectral normalization.

- create a conv1d layer with `nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)`
- initialize it with `nn.init.kaiming_normal_(conv.weight)`
- if `bias=True`, make them zero
- run spectral normalization on this conv layer and return it


```
#|export
def _conv1d_spect(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)
```


```
_conv1d_spect(3,2)
```




    Conv1d(3, 2, kernel_size=(1,), stride=(1,), bias=False)



### ```SimpleSelfAttention(self, n_in:int, ks=1, sym=False)```


```
#|export
class SimpleSelfAttention(Module):
    def __init__(self, n_in:int, ks=1, sym=False):
        self.sym,self.n_in = sym,n_in
        self.conv = _conv1d_spect(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self,x):
        if self.sym:
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)

        size = x.size()
        x = x.view(*size[:2],-1)

        convx = self.conv(x)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())
        o = torch.bmm(xxT, convx)
        o = self.gamma * o + x
        return o.view(*size).contiguous()
```

## PixelShuffle

PixelShuffle introduced in [this article](https://arxiv.org/pdf/1609.05158.pdf) to avoid checkerboard artifacts when upsampling images. If we want an output with `ch_out` filters, we use a convolution with `ch_out * (r**2)` filters, where `r` is the upsampling factor. Then we reorganize those filters like in the picture below:

<img src="images/pixelshuffle.png" alt="Pixelshuffle" width="800" />

### ```icnr_init(x, scale=2, init=nn.init.kaiming_normal_)```
official: ICNR init of `x`, with `scale` and `init` function


```
#|export
# @snoop
def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
#     pp(x.new_zeros([ni2,nf,h,w]).shape, init(x.new_zeros([ni2,nf,h,w])).shape)
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
```

ICNR init was introduced in [this article](https://arxiv.org/abs/1707.02937). It suggests to initialize the convolution that will be used in PixelShuffle so that each of the `r**2` channels get the same weight (so that in the picture above, the 9 colors in a 3 by 3 window are initially the same).

> Note: This is done on the first dimension because PyTorch stores the weights of a convolutional layer in this format: `ch_out x ch_in x ks x ks`. 


```
tst = torch.randn(16*4, 32, 1, 1)
tst = icnr_init(tst)
```


```
for i in range(0,16*4,4):
    test_eq(tst[i],tst[i+1])
    test_eq(tst[i],tst[i+2])
    test_eq(tst[i],tst[i+3])
```

### ```PixelShuffle_ICNR(nn.Sequential)```
official: Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`.

- subclass of `nn.Sequential`
- if `nf` is None, set it to be `ni`
- create a list of layers, by default they are Conv2d, ReLU, PixelShuffle
- if NormType.Weight, apply ICNR init to Conv2d's weight_v, and weight_g
- if blur, add nn.ReplicationPad2d, nn.AvgPool2d to the layers list
- finally, put all the layers into the Sequential block


```
#|export
class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
#     @snoop
    def __init__(self, ni, nf=None, scale=2, blur=False, norm_type=NormType.Weight, act_cls=defaults.activation):
        super().__init__()
        nf = ifnone(nf, ni)
        layers = [ConvLayer(ni, nf*(scale**2), ks=1, norm_type=norm_type, act_cls=act_cls, bias_std=0),
                  nn.PixelShuffle(scale)]
        if norm_type == NormType.Weight:
            layers[0][0].weight_v.data.copy_(icnr_init(layers[0][0].weight_v.data))
            layers[0][0].weight_g.data.copy_(((layers[0][0].weight_v.data**2).sum(dim=[1,2,3])**0.5)[:,None,None,None])
        else:
            layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)
```

The convolutional layer is initialized with `icnr_init` and passed `act_cls` and `norm_type` (the default of weight normalization seemed to be what's best for super-resolution problems, in our experiments). 

The `blur` option comes from [Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts](https://arxiv.org/abs/1806.02658) where the authors add a little bit of blur to completely get rid of checkerboard artifacts.


```
psfl = PixelShuffle_ICNR(16)
psfl
psfl[0][0]
psfl[0][1]
psfl[1]
```




    PixelShuffle_ICNR(
      (0): ConvLayer(
        (0): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU()
      )
      (1): PixelShuffle(upscale_factor=2)
    )






    Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))






    ReLU()






    PixelShuffle(upscale_factor=2)




```
x = torch.randn(64, 16, 8, 8)
y = psfl(x)
ic(psfl(x).shape)
ic(psfl[0][0](x).shape)
layer1 = psfl[0][0](x)
ic(psfl[0][1](layer1).shape)
layer2 = psfl[0][1](layer1)
ic(psfl[1](layer2).shape)

test_eq(y.shape, [64, 16, 16, 16])
```

    ic| psfl(x).shape: torch.Size([64, 16, 16, 16])





    torch.Size([64, 16, 16, 16])



    ic| psfl[0][0](x).shape: torch.Size([64, 64, 8, 8])





    torch.Size([64, 64, 8, 8])



    ic| psfl[0][1](layer1).shape: torch.Size([64, 64, 8, 8])





    torch.Size([64, 64, 8, 8])



    ic| psfl[1](layer2).shape: torch.Size([64, 16, 16, 16])





    torch.Size([64, 16, 16, 16])




```
#ICNR init makes every 2x2 window (stride 2) have the same elements
for i in range(0,16,2):
    for j in range(0,16,2):
        test_eq(y[:,:,i,j],y[:,:,i+1,j])
        test_eq(y[:,:,i,j],y[:,:,i  ,j+1])
        test_eq(y[:,:,i,j],y[:,:,i+1,j+1])
```


```
psfl = PixelShuffle_ICNR(16, norm_type=None)
x = torch.randn(64, 16, 8, 8)
y = psfl(x)
test_eq(y.shape, [64, 16, 16, 16])
#ICNR init makes every 2x2 window (stride 2) have the same elements
for i in range(0,16,2):
    for j in range(0,16,2):
        test_eq(y[:,:,i,j],y[:,:,i+1,j])
        test_eq(y[:,:,i,j],y[:,:,i  ,j+1])
        test_eq(y[:,:,i,j],y[:,:,i+1,j+1])
```


```
psfl = PixelShuffle_ICNR(16, norm_type=NormType.Spectral)
x = torch.randn(64, 16, 8, 8)
y = psfl(x)
test_eq(y.shape, [64, 16, 16, 16])
#ICNR init makes every 2x2 window (stride 2) have the same elements
for i in range(0,16,2):
    for j in range(0,16,2):
        test_eq(y[:,:,i,j],y[:,:,i+1,j])
        test_eq(y[:,:,i,j],y[:,:,i  ,j+1])
        test_eq(y[:,:,i,j],y[:,:,i+1,j+1])
```

## Sequential extensions

### ```sequential(*args)```
official: Create an `nn.Sequential`, wrapping items with `Lambda` if needed"


```
# help(Lambda)
# help(nn.ReLU)
```


```
#|export
def sequential(*args):
    "Create an `nn.Sequential`, wrapping items with `Lambda` if needed"
    if len(args) != 1 or not isinstance(args[0], OrderedDict):
        args = list(args)
        for i,o in enumerate(args):
            if not isinstance(o,nn.Module): args[i] = Lambda(o)
    return nn.Sequential(*args)
```


```
Lambda(nn.ReLU)
```




    __main__.Lambda(func=<class 'torch.nn.modules.activation.ReLU'>)




```
sequential()
sequential([nn.ReLU, nn.Linear])
```




    Sequential()






    Sequential(
      (0): __main__.Lambda(func=[<class 'torch.nn.modules.activation.ReLU'>, <class 'torch.nn.modules.linear.Linear'>])
    )



### ```SequentialEx(Module)```
official: Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

To build a block of layers and let x pass through them one after another and each layer's input remembers the original input
- This is useful to write layers that require to remember the input (like a resnet block) in a sequential way.
- the input is remembered as `x.orig` or `res.orig` before running `l(res)` so that `MergeLayer.forward(res)` defined below can utilize `res.orig` before setting to None


```
#|export
class SequentialEx(Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"
    def __init__(self, *layers): self.layers = nn.ModuleList(layers)

#     @snoop
    def forward(self, x):
        res = x
        for l in self.layers:
            res.orig = x
            nres = l(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig, nres.orig = None, None
            res = nres
        return res

    def __getitem__(self,i): return self.layers[i]
    def append(self,l):      return self.layers.append(l)
    def extend(self,l):      return self.layers.extend(l)
    def insert(self,i,l):    return self.layers.insert(i,l)
```

This is useful to write layers that require to remember the input (like a resnet block) in a sequential way.

### ```MergeLayer(Module)```
official: Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`.

- MergeLayer() turns to be the last layer of the layer block, so `x` for MergeLayer.forward is usually the output of last layer
- since MergeLayer is used inside SequentialEx, `x` will bring the original input `x.orig` into `MergeLayer.forward` to process
- if `dense=False`, the output shape won't change as `x + x.orig`
- if `dense=True`, the output shape (2nd dim) will double due to `torch.concat([x, x.orig], dim=1)`


```
#|export
class MergeLayer(Module):
    "Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`."
    def __init__(self, dense:bool=False): self.dense=dense
#     @snoop
    def forward(self, x): 
#         return torch.cat([x,x.orig], dim=1) if self.dense else (x+x.orig)
        if self.dense:
            return torch.cat([x,x.orig], dim=1) 
        else: 
            return (x+x.orig)        
```


```
x = torch.randn(32, 16, 8, 8)
res_block = SequentialEx(ConvLayer(16, 16), ConvLayer(16,16))
y = res_block(x)
test_eq(y.shape, (32, 16, 8, 8))
test_eq(y.orig, None)
```


```
res_block.append(MergeLayer()) # just to test append - normally it would be in init params
y1 = res_block(x)
test_eq(y1.shape, [32, 16, 8, 8])
test_eq(y1.orig, None)
```




    ModuleList(
      (0): ConvLayer(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ConvLayer(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): MergeLayer()
    )




```
x = torch.randn(32, 16, 8, 8)
res_block = SequentialEx(ConvLayer(16, 16), ConvLayer(16,16))
res_block.append(MergeLayer()) # just to test append - normally it would be in init params
y = res_block(x)
test_eq(y, x + res_block[1](res_block[0](x)))
```




    ModuleList(
      (0): ConvLayer(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ConvLayer(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): MergeLayer()
    )




```
res_block.append(MergeLayer(True)) # just to test append - normally it would be in init params
y = res_block(x)
test_eq(y.shape, [32, 32, 8, 8])
```




    ModuleList(
      (0): ConvLayer(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ConvLayer(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): MergeLayer()
      (3): MergeLayer()
    )



## Concat

### ```Cat(nn.ModuleList)```
official: Concatenate layers outputs over a given dim

by default, the outputs of all layers inside the ModuleList will be concatenated on 2nd dim

Equivalent to keras.layers.Concatenate, it will concat the outputs of a ModuleList over a given dimension (default the filter dimension)


```
#|export 
class Cat(nn.ModuleList):
    "Concatenate layers outputs over a given dim"
    def __init__(self, layers, dim=1):
        self.dim=dim
        super().__init__(layers)
    def forward(self, x): return torch.cat([l(x) for l in self], dim=self.dim)
```


```
layers = [ConvLayer(2,4), ConvLayer(2,4), ConvLayer(2,4)] 
x = torch.rand(1,2,8,8) 
cat = Cat(layers) 
cat
```




    Cat(
      (0): ConvLayer(
        (0): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ConvLayer(
        (0): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): ConvLayer(
        (0): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )




```
test_eq(cat(x).shape, [1,12,8,8]) 
test_eq(cat(x), torch.cat([ic(l(x)) for l in layers], dim=1)) # a good use case for ic
```

    ic| l(x): class=<class 'torch.Tensor'>, shape=torch.Size([1, 4, 8, 8]), dtype=torch.float32
    ic| l(x): class=<class 'torch.Tensor'>, shape=torch.Size([1, 4, 8, 8]), dtype=torch.float32
    ic| l(x): class=<class 'torch.Tensor'>, shape=torch.Size([1, 4, 8, 8]), dtype=torch.float32


## Ready-to-go models

### ```SimpleCNN(nn.Sequential)```
Create a simple CNN with `filters`.

- use `filters` like `[8, 16, 32]` to define `kernel_szs` and `strides`, and the number of Conv layers to create
- then add a PoolFlatten layer
- finally put them all into a Sequential block


```
#|export
class SimpleCNN(nn.Sequential):
    "Create a simple CNN with `filters`."
#     @snoop
    def __init__(self, filters, kernel_szs=None, strides=None, bn=True):
        nl = len(filters)-1
        kernel_szs = ifnone(kernel_szs, [3]*nl)
        strides    = ifnone(strides   , [2]*nl)
        layers = [ConvLayer(filters[i], filters[i+1], kernel_szs[i], stride=strides[i],
                  norm_type=(NormType.Batch if bn and i<nl-1 else None)) for i in range(nl)]
        layers.append(PoolFlatten())
        super().__init__(*layers)
```

The model is a succession of convolutional layers from `(filters[0],filters[1])` to `(filters[n-2],filters[n-1])` (if `n` is the length of the `filters` list) followed by a `PoolFlatten`. `kernel_szs` and `strides` defaults to a list of 3s and a list of 2s. If `bn=True` the convolutional layers are successions of conv-relu-batchnorm, otherwise conv-relu.


```
tst = SimpleCNN([8,16,32])
tst
```




    SimpleCNN(
      (0): ConvLayer(
        (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ConvLayer(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): ReLU()
      )
      (2): PoolFlatten(
        (0): AdaptiveAvgPool2d(output_size=1)
        (1): __main__.Flatten(full=False)
      )
    )




```
mods = list(tst.children())
```


```
test_eq(len(mods), 3)
test_eq([[m[0].in_channels, m[0].out_channels] for m in mods[:2]], [[8,16], [16,32]])
```

Test kernel sizes


```
tst = SimpleCNN([8,16,32], kernel_szs=[1,3])
mods = list(tst.children())
test_eq([m[0].kernel_size for m in mods[:2]], [(1,1), (3,3)])
```

Test strides


```
tst = SimpleCNN([8,16,32], strides=[1,2])
mods = list(tst.children())
test_eq([m[0].stride for m in mods[:2]], [(1,1),(2,2)])
```

### ```ProdLayer(Module)```
official: Merge a shortcut with the result of the module by multiplying them.

check ```MergeLayer``` doc for better understanding of ProdLayer


```
#|export
class ProdLayer(Module):
    "Merge a shortcut with the result of the module by multiplying them."
    def forward(self, x): return x * x.orig
```

### ```inplace_relu```


```
#|export
inplace_relu = partial(nn.ReLU, inplace=True)
```

### ```SEModule(ch, reduction, act_cls=defaults.activation)```
Use SequentialEx to put AdaptiveAvgPool2d, 2 ConvLayer, ProdLayer together


```
#|export
def SEModule(ch, reduction, act_cls=defaults.activation):
    nf = math.ceil(ch//reduction/8)*8
    return SequentialEx(nn.AdaptiveAvgPool2d(1),
                        ConvLayer(ch, nf, ks=1, norm_type=None, act_cls=act_cls),
                        ConvLayer(nf, ch, ks=1, norm_type=None, act_cls=nn.Sigmoid),
                        ProdLayer())
```

### ```ResBlock(Module)```
official: Resnet block from `ni` to `nh` with `stride`

- user inputs without default: `expansion`, `ni`, `nf`
- `norm2` is to choose between `BatchZero`, `InstanceZero`, or other `norm_type`
- `nh1` and `nh2` are defined by `nf`
- `nf` and `ni` is multiplied with `expansion`
- `k0`, `k1` are two dicts of norm_type, act_cls, ndim, and `**kwargs`
- `convpath`: a list of ConvLayers; if expansion == 1, 2 ConvLayers; otherwise, 3 ConvLayers
- if reduction, then add SEModule layer block
- if sa: 



```
#|export
class ResBlock(Module):
    "Resnet block from `ni` to `nh` with `stride`"
    @snoop
    @delegates(ConvLayer.__init__)
    def __init__(self, expansion, ni, nf, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
                 sa=False, sym=False, norm_type=NormType.Batch, act_cls=defaults.activation, ndim=2, ks=3,
                 pool=AvgPool, pool_first=True, **kwargs):
        norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
                 NormType.InstanceZero if norm_type==NormType.Instance else norm_type)
        if nh2 is None: nh2 = nf
        if nh1 is None: nh1 = nh2
        nf,ni = nf*expansion,ni*expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
                     ConvLayer(nh2,  nf, ks, groups=g2, **k1)
        ] if expansion == 1 else [
                     ConvLayer(ni,  nh1, 1, **k0),
                     ConvLayer(nh1, nh2, ks, stride=stride, groups=nh1 if dw else groups, **k0),
                     ConvLayer(nh2,  nf, 1, groups=g2, **k1)]
        if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
        if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
        self.convpath = nn.Sequential(*convpath)
        idpath = []
        if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride!=1: idpath.insert((1,0)[pool_first], pool(stride, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()

    def forward(self, x): return self.act(self.convpath(x) + self.idpath(x))
```

This is a resnet block (normal or bottleneck depending on `expansion`, 1 for the normal block and 4 for the traditional bottleneck) that implements the tweaks from [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187). In particular, the last batchnorm layer (if that is the selected `norm_type`) is initialized with a weight (or gamma) of zero to facilitate the flow from the beginning to the end of the network. It also implements optional [Squeeze and Excitation](https://arxiv.org/abs/1709.01507) and grouped convs for [ResNeXT](https://arxiv.org/abs/1611.05431) and similar models (use `dw=True` for depthwise convs).

The `kwargs` are passed to `ConvLayer` along with `norm_type`.


```
ResBlock(1, 4, 2)
```

    08:25:10.38 >>> Call to ResBlock.__init__ in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_57140/1783655079.py", line 6
    08:25:10.38 .......... self = ResBlock()
    08:25:10.38 .......... type(self) = <class '__main__.ResBlock'>
    08:25:10.38 .......... expansion = 1
    08:25:10.38 .......... type(expansion) = <class 'int'>
    08:25:10.38 .......... ni = 4
    08:25:10.38 .......... type(ni) = <class 'int'>
    08:25:10.38 .......... nf = 2
    08:25:10.38 .......... type(nf) = <class 'int'>
    08:25:10.38 .......... stride = 1
    08:25:10.38 .......... type(stride) = <class 'int'>
    08:25:10.38 .......... groups = 1
    08:25:10.38 .......... type(groups) = <class 'int'>
    08:25:10.38 .......... reduction = None
    08:25:10.38 .......... nh1 = None
    08:25:10.38 .......... nh2 = None
    08:25:10.38 .......... dw = False
    08:25:10.38 .......... type(dw) = <class 'bool'>
    08:25:10.38 .......... g2 = 1
    08:25:10.38 .......... type(g2) = <class 'int'>
    08:25:10.38 .......... sa = False
    08:25:10.38 .......... type(sa) = <class 'bool'>
    08:25:10.38 .......... sym = False
    08:25:10.38 .......... type(sym) = <class 'bool'>
    08:25:10.38 .......... norm_type = <NormType.Batch: 1>
    08:25:10.38 .......... type(norm_type) = <enum 'NormType'>
    08:25:10.38 .......... act_cls = <class 'torch.nn.modules.activation.ReLU'>
    08:25:10.38 .......... type(act_cls) = <class 'type'>
    08:25:10.38 .......... ndim = 2
    08:25:10.38 .......... type(ndim) = <class 'int'>
    08:25:10.38 .......... ks = 3
    08:25:10.38 .......... type(ks) = <class 'int'>
    08:25:10.38 .......... pool = <function AvgPool>
    08:25:10.38 .......... type(pool) = <class 'function'>
    08:25:10.38 .......... sig(pool) = <Signature (ks=2, stride=None, padding=0, ndim=2, ceil_mode=False)>
    08:25:10.38 .......... pool_first = True
    08:25:10.38 .......... type(pool_first) = <class 'bool'>
    08:25:10.38 .......... kwargs = {}
    08:25:10.38 .......... type(kwargs) = <class 'dict'>
    08:25:10.38    6 |     def __init__(self, expansion, ni, nf, stride=1, groups=1, reduction=None, nh1=None, nh2=None, dw=False, g2=1,
    08:25:10.38    9 |         norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
    08:25:10.38    9 |         norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
    08:25:10.39 .............. norm2 = <NormType.BatchZero: 2>
    08:25:10.39 .............. type(norm2) = <enum 'NormType'>
    08:25:10.39   11 |         if nh2 is None: nh2 = nf
    08:25:10.39 ...... nh2 = 2
    08:25:10.39 ...... type(nh2) = <class 'int'>
    08:25:10.39   12 |         if nh1 is None: nh1 = nh2
    08:25:10.39 ...... nh1 = 2
    08:25:10.39 ...... type(nh1) = <class 'int'>
    08:25:10.39   13 |         nf,ni = nf*expansion,ni*expansion
    08:25:10.39   14 |         k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
    08:25:10.39 .............. k0 = {'norm_type': <NormType.Batch: 1>, 'act_cls': <class 'torch.nn.modules.activation.ReLU'>, 'ndim': 2}
    08:25:10.39 .............. type(k0) = <class 'dict'>
    08:25:10.39 .............. len(k0) = 3
    08:25:10.39   15 |         k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
    08:25:10.39 .............. k1 = {'norm_type': <NormType.BatchZero: 2>, 'act_cls': None, 'ndim': 2}
    08:25:10.39 .............. type(k1) = <class 'dict'>
    08:25:10.39 .............. len(k1) = 3
    08:25:10.39   16 |         convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
    08:25:10.39   17 |                      ConvLayer(nh2,  nf, ks, groups=g2, **k1)
    08:25:10.39   18 |         ] if expansion == 1 else [
    08:25:10.39   16 |         convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
    08:25:10.39   17 |                      ConvLayer(nh2,  nf, ks, groups=g2, **k1)
    08:25:10.39   16 |         convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
    08:25:10.39   16 |         convpath  = [ConvLayer(ni,  nh2, ks, stride=stride, groups=ni if dw else groups, **k0),
    08:25:10.39 .............. convpath = [ConvLayer(
    08:25:10.39                             (0): Conv2d(4, 2, kernel_size=(3, 3...e=True, track_running_stats=True)
    08:25:10.39                             (2): ReLU()
    08:25:10.39                           ), ConvLayer(
    08:25:10.39                             (0): Conv2d(2, 2, kernel_size=(3, 3...tum=0.1, affine=True, track_running_stats=True)
    08:25:10.39                           )]
    08:25:10.39 .............. type(convpath) = <class 'list'>
    08:25:10.39 .............. len(convpath) = 2
    08:25:10.39   22 |         if reduction: convpath.append(SEModule(nf, reduction=reduction, act_cls=act_cls))
    08:25:10.39   23 |         if sa: convpath.append(SimpleSelfAttention(nf,ks=1,sym=sym))
    08:25:10.39   24 |         self.convpath = nn.Sequential(*convpath)
    08:25:10.39 .............. self = ResBlock(
    08:25:10.39                         (convpath): Sequential(
    08:25:10.39                           (0): Con...ffine=True, track_running_stats=True)
    08:25:10.39                           )
    08:25:10.39                         )
    08:25:10.39                       )
    08:25:10.39   25 |         idpath = []
    08:25:10.39 .............. type(idpath) = <class 'list'>
    08:25:10.39   26 |         if ni!=nf: idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
    08:25:10.39 ...... idpath = [ConvLayer(
    08:25:10.39                   (0): Conv2d(4, 2, kernel_size=(1, 1...tum=0.1, affine=True, track_running_stats=True)
    08:25:10.39                 )]
    08:25:10.39 ...... len(idpath) = 1
    08:25:10.39   27 |         if stride!=1: idpath.insert((1,0)[pool_first], pool(stride, ndim=ndim, ceil_mode=True))
    08:25:10.40   28 |         self.idpath = nn.Sequential(*idpath)
    08:25:10.40   29 |         self.act = defaults.activation(inplace=True) if act_cls is defaults.activation else act_cls()
    08:25:10.40 .............. self = ResBlock(
    08:25:10.40                         (convpath): Sequential(
    08:25:10.40                           (0): Con...ats=True)
    08:25:10.40                           )
    08:25:10.40                         )
    08:25:10.40                         (act): ReLU(inplace=True)
    08:25:10.40                       )
    08:25:10.40 <<< Return value from ResBlock.__init__: None





    ResBlock(
      (convpath): Sequential(
        (0): ConvLayer(
          (0): Conv2d(4, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): ConvLayer(
          (0): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (idpath): Sequential(
        (0): ConvLayer(
          (0): Conv2d(4, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (act): ReLU(inplace=True)
    )




```
#|export
def SEBlock(expansion, ni, nf, groups=1, reduction=16, stride=1, **kwargs):
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, reduction=reduction, nh1=nf*2, nh2=nf*expansion, **kwargs)
```


```
#|export
def SEResNeXtBlock(expansion, ni, nf, groups=32, reduction=16, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, reduction=reduction, nh2=w, **kwargs)
```


```
#|export
def SeparableBlock(expansion, ni, nf, reduction=16, stride=1, base_width=4, **kwargs):
    return ResBlock(expansion, ni, nf, stride=stride, reduction=reduction, nh2=nf*2, dw=True, **kwargs)
```

## Time Distributed Layer

Equivalent to Keras `TimeDistributed` Layer, enables computing pytorch `Module` over an axis.


```
#|export
def _stack_tups(tuples, stack_dim=1):
    "Stack tuple of tensors along `stack_dim`"
    return tuple(torch.stack([t[i] for t in tuples], dim=stack_dim) for i in range_of(tuples[0]))
```


```
#|export
class TimeDistributed(Module):
    "Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time." 
    def __init__(self, module, low_mem=False, tdim=1):
        store_attr()
        
    def forward(self, *tensors, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim!=1: 
            return self.low_mem_forward(*tensors, **kwargs)
        else:
            #only support tdim=1
            inp_shape = tensors[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]   
            out = self.module(*[x.view(bs*seq_len, *x.shape[2:]) for x in tensors], **kwargs)
        return self.format_output(out, bs, seq_len)
    
    def low_mem_forward(self, *tensors, **kwargs):                                           
        "input x with shape:(bs,seq_len,channels,width,height)"
        seq_len = tensors[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in tensors]
        out = []
        for i in range(seq_len):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        if isinstance(out[0], tuple):
            return _stack_tups(out, stack_dim=self.tdim)
        return torch.stack(out, dim=self.tdim)
    
    def format_output(self, out, bs, seq_len):
        "unstack from batchsize outputs"
        if isinstance(out, tuple):
            return tuple(out_i.view(bs, seq_len, *out_i.shape[1:]) for out_i in out)
        return out.view(bs, seq_len,*out.shape[1:])
    
    def __repr__(self):
        return f'TimeDistributed({self.module})'
```


```
bs, seq_len = 2, 5
x, y = torch.rand(bs,seq_len,3,2,2), torch.rand(bs,seq_len,3,2,2)
```


```
tconv = TimeDistributed(nn.Conv2d(3,4,1))
test_eq(tconv(x).shape, (2,5,4,2,2))
tconv.low_mem=True
test_eq(tconv(x).shape, (2,5,4,2,2))
```


```
class Mod(Module):
    def __init__(self):
        self.conv = nn.Conv2d(3,4,1)
    def forward(self, x, y):
        return self.conv(x) + self.conv(y)
tmod = TimeDistributed(Mod())
```


```
out = tmod(x,y)
test_eq(out.shape, (2,5,4,2,2))
tmod.low_mem=True
out_low_mem = tmod(x,y)
test_eq(out_low_mem.shape, (2,5,4,2,2))
test_eq(out, out_low_mem)
```


```
class Mod2(Module):
    def __init__(self):
        self.conv = nn.Conv2d(3,4,1)
    def forward(self, x, y):
        return self.conv(x), self.conv(y)
tmod2 = TimeDistributed(Mod2())
```


```
out = tmod2(x,y)
test_eq(len(out), 2)
test_eq(out[0].shape, (2,5,4,2,2))
tmod2.low_mem=True
out_low_mem = tmod2(x,y)
test_eq(out_low_mem[0].shape, (2,5,4,2,2))
test_eq(out, out_low_mem)
```


```
show_doc(TimeDistributed)
```




---

### TimeDistributed

>      TimeDistributed (module, low_mem=False, tdim=1)

Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time.



This module is equivalent to [Keras TimeDistributed Layer](https://keras.io/api/layers/recurrent_layers/time_distributed/). This wrapper allows to apply a layer to every temporal slice of an input. By default it is assumed the time axis (`tdim`) is the 1st one (the one after the batch size). A typical usage would be to encode a sequence of images using an image encoder.

The `forward` function of `TimeDistributed` supports `*args` and `**kkwargs` but only `args` will be split and passed to the underlying module independently for each timestep, `kwargs` will be passed as they are. This is useful when you have module that take multiple arguments as inputs, this way, you can put all tensors you need spliting as `args` and other arguments that don't need split as `kwargs`.

> This module is heavy on memory, as it will try to pass mutiple timesteps at the same time on the batch dimension, if you get out of memorey errors, try first reducing your batch size by the number of timesteps.


```
from fastai.vision.all import *
```


```
encoder = create_body(resnet18())
```

A resnet18 will encode a feature map of 512 channels. Height and Width will be divided by 32.


```
time_resnet = TimeDistributed(encoder)
```

a synthetic batch of 2 image-sequences of lenght 5. `(bs, seq_len, ch, w, h)`


```
image_sequence = torch.rand(2, 5, 3, 64, 64)
```


```
time_resnet(image_sequence).shape
```




    torch.Size([2, 5, 512, 2, 2])



This way, one can encode a sequence of images on feature space.
There is also a `low_mem_forward` that will pass images one at a time to reduce GPU memory consumption.


```
time_resnet.low_mem_forward(image_sequence).shape
```




    torch.Size([2, 5, 512, 2, 2])



## Swish and Mish


```
#|export
from torch.jit import script
```


```
#|export
@script
def _swish_jit_fwd(x): return x.mul(torch.sigmoid(x))

@script
def _swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))

class _SwishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _swish_jit_bwd(x, grad_output)
```


```
#|export
def swish(x, inplace=False): return _SwishJitAutoFn.apply(x)
```


```
#|export
class Swish(Module):
    def forward(self, x): return _SwishJitAutoFn.apply(x)
```


```
#|export
@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))

@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)
```


```
#|export
def mish(x): return F.mish(x) if torch.__version__ >= '1.9' else MishJitAutoFn.apply(x)
```


```
#|export
class Mish(Module):
    def forward(self, x): return MishJitAutoFn.apply(x)
```


```
#|export
if ismin_torch('1.9'): Mish = nn.Mish
```


```
#|export
for o in swish,Swish,mish,Mish: o.__default_init__ = kaiming_uniform_
```

## Helper functions for submodules

It's easy to get the list of all parameters of a given model. For when you want all submodules (like linear/conv layers) without forgetting lone parameters, the following class wraps those in fake modules.


```
#|export
class ParameterModule(Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p): self.val = p
    def forward(self, x): return x
```


```
#|export
def children_and_parameters(m):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children
```


```
class TstModule(Module):
    def __init__(self): self.a,self.lin = nn.Parameter(torch.randn(1)),nn.Linear(5,10)

tst = TstModule()
children = children_and_parameters(tst)
test_eq(len(children), 2)
test_eq(children[0], tst.lin)
assert isinstance(children[1], ParameterModule)
test_eq(children[1].val, tst.a)
```


```
#|export
def has_children(m):
    try: next(m.children())
    except StopIteration: return False
    return True
```


```
class A(Module): pass
assert not has_children(A())
assert has_children(TstModule())
```


```
#|export
def flatten_model(m):
    "Return the list of all submodules and parameters of `m`"
    return sum(map(flatten_model,children_and_parameters(m)),[]) if has_children(m) else [m]
```


```
tst = nn.Sequential(TstModule(), TstModule())
children = flatten_model(tst)
test_eq(len(children), 4)
assert isinstance(children[1], ParameterModule)
assert isinstance(children[3], ParameterModule)
```


```
#|export
class NoneReduce():
    "A context manager to evaluate `loss_func` with none reduce."
    def __init__(self, loss_func): self.loss_func,self.old_red = loss_func,None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = self.loss_func.reduction
            self.loss_func.reduction = 'none'
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None: self.loss_func.reduction = self.old_red
```


```
x,y = torch.randn(5),torch.randn(5)
loss_fn = nn.MSELoss()
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn.reduction, 'mean')

loss_fn = F.mse_loss
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn, F.mse_loss)
```


```
#|export
def in_channels(m):
    "Return the shape of the first weight layer in `m`."
    try: return next(l.weight.shape[1] for l in flatten_model(m) if nested_attr(l,'weight.ndim',-1)==4)
    except StopIteration as e: e.args = ["No weight layer"]; raise
```


```
test_eq(in_channels(nn.Sequential(nn.Conv2d(5,4,3), nn.Conv2d(4,3,3))), 5)
test_eq(in_channels(nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(BatchNorm(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(InstanceNorm(4), nn.Conv2d(4,3,3))), 4)
test_eq(in_channels(nn.Sequential(InstanceNorm(4, affine=False), nn.Conv2d(4,3,3))), 4)
test_fail(lambda : in_channels(nn.Sequential(nn.AvgPool2d(4))))
```

## Export -


```
#|hide
from nbdev import *
nbdev_export()
```

    /Users/Natsume/mambaforge/lib/python3.9/site-packages/nbdev/export.py:54: UserWarning: Notebook '/Users/Natsume/Documents/fastai/nbs/09c_vision.widgets.ipynb' uses `#|export` without `#|default_exp` cell.
    Note nbdev2 no longer supports nbdev1 syntax. Run `nbdev_migrate` to upgrade.
    See https://nbdev.fast.ai/getting_started.html for more information.
      warn(f"Notebook '{nbname}' uses `#|export` without `#|default_exp` cell.\n"



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Input In [140], in <cell line: 3>()
          1 #|hide
          2 from nbdev import *
    ----> 3 nbdev_export()


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/script.py:110, in call_parse.<locals>._f(*args, **kwargs)
        107 @wraps(func)
        108 def _f(*args, **kwargs):
        109     mod = inspect.getmodule(inspect.currentframe().f_back)
    --> 110     if not mod: return func(*args, **kwargs)
        111     if not SCRIPT_INFO.func and mod.__name__=="__main__": SCRIPT_INFO.func = func.__name__
        112     if len(sys.argv)>1 and sys.argv[1]=='': sys.argv.pop(1)


    File ~/mambaforge/lib/python3.9/site-packages/nbdev/doclinks.py:135, in nbdev_export(path, **kwargs)
        133 for f in files: nb_export(f)
        134 add_init(get_config().lib_path)
    --> 135 _build_modidx()


    File ~/mambaforge/lib/python3.9/site-packages/nbdev/doclinks.py:97, in _build_modidx(dest, nbs_path, skip_exists)
         95 code_root = dest.parent.resolve()
         96 for file in globtastic(dest, file_glob="*.py", skip_file_re='^_', skip_folder_re="\.ipynb_checkpoints"):
    ---> 97     res['syms'].update(_get_modidx((dest.parent/file).resolve(), code_root, nbs_path=nbs_path))
         98 idxfile.write_text("# Autogenerated by nbdev\n\nd = "+pformat(res, width=140, indent=2, compact=True))


    File ~/mambaforge/lib/python3.9/site-packages/nbdev/doclinks.py:71, in _get_modidx(py_path, code_root, nbs_path)
         69 for cell in _iter_py_cells(py_path):
         70     if cell.nb == 'auto': continue
    ---> 71     loc = _nbpath2html(cell.nb_path.relative_to(nbs_path))
         73     def _stor(nm):
         74         for n in L(nm): d[f'{mod_name}.{n}'] = f'{loc.as_posix()}#{n.lower()}',rel_name


    File ~/mambaforge/lib/python3.9/pathlib.py:939, in PurePath.relative_to(self, *other)
        937 if (root or drv) if n == 0 else cf(abs_parts[:n]) != cf(to_abs_parts):
        938     formatted = self._format_parsed_parts(to_drv, to_root, to_parts)
    --> 939     raise ValueError("{!r} is not in the subpath of {!r}"
        940             " OR one path is relative and the other is absolute."
        941                      .format(str(self), str(formatted)))
        942 return self._from_parsed_parts('', root if n == 1 else '',
        943                                abs_parts[n:])


    ValueError: '/Users/Natsume/Documents/fastai/fastai/nbs/09c_vision.widgets.ipynb' is not in the subpath of '/Users/Natsume/Documents/fastai/nbs' OR one path is relative and the other is absolute.



```

```


```

```
