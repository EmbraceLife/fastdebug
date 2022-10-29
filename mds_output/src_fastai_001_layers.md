```
#|hide
#| eval: false
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab
```


```
#|default_exp todeletelayers
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

    14:52:40.08 LOG:
    14:52:40.23 .... module = <function module>





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




    tensor([[-0.1961, -0.8795,  0.9460,  2.0959],
            [-0.9706, -1.1677,  1.4113, -0.5296],
            [ 1.6900, -2.2192,  0.9794, -1.7867],
            [-0.4907,  0.6063, -1.1281, -0.2069]])




```
torch.max(a, 1)
torch.max(a, 1)[0].shape
torch.max(a, 1)[1].shape
torch.max(a, dim=1, keepdim=True)
torch.max(a, dim=1, keepdim=True)[0].shape
torch.max(a, dim=1, keepdim=True)[1].shape
```




    torch.return_types.max(
    values=tensor([2.0959, 1.4113, 1.6900, 0.6063]),
    indices=tensor([3, 2, 0, 1]))






    torch.Size([4])






    torch.Size([4])






    torch.return_types.max(
    values=tensor([[2.0959],
            [1.4113],
            [1.6900],
            [0.6063]]),
    indices=tensor([[3],
            [2],
            [0],
            [1]]))






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
    values=tensor([1.6900, 0.6063, 1.4113, 2.0959]),
    indices=tensor([2, 3, 1, 0]))






    torch.Size([4])






    torch.Size([4])






    torch.return_types.max(
    values=tensor([[1.6900, 0.6063, 1.4113, 2.0959]]),
    indices=tensor([[2, 3, 1, 0]]))






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





    tensor([[[[-7.1134e-01, -9.9607e-01, -1.2362e+00,  2.2192e+00],
              [-9.2149e-01, -4.7613e-01, -1.9622e+00, -7.8476e-01],
              [-6.2538e-01,  1.0842e+00,  5.9557e-01, -1.9869e-01],
              [-3.3655e-01, -1.4927e+00,  6.3933e-01,  2.5596e-01]],
    
             [[ 1.5941e+00, -2.6400e-01,  3.3970e-02, -1.8899e+00],
              [ 9.0067e-01, -5.4662e-02, -4.3492e-01,  2.2304e+00],
              [-1.1263e+00, -8.9024e-01,  6.3741e-01,  1.8211e-01],
              [ 1.5691e+00, -6.5804e-02,  1.4386e+00,  5.7467e-01]],
    
             [[-1.5395e+00,  3.8009e-01, -1.0150e+00,  4.0274e-02],
              [ 5.4112e-01,  1.1350e+00,  6.7048e-01,  1.6456e+00],
              [-4.8748e-01, -1.2841e+00,  6.1856e-01, -9.4902e-01],
              [-1.4359e+00,  8.3798e-01, -2.2432e+00, -2.6160e-01]],
    
             [[-9.5391e-02,  3.7970e-01, -3.5172e-02,  8.3178e-01],
              [-5.8331e-01, -2.6988e-01, -2.7612e-01,  1.3224e+00],
              [ 4.1869e-01,  3.3830e-01,  1.6538e-02, -5.4941e-01],
              [-4.4819e-01, -1.3498e-01,  4.1665e-01,  1.5720e-01]],
    
             [[ 4.9426e-01, -1.3869e+00,  1.3001e+00,  5.2034e-01],
              [-6.0255e-01,  3.8579e-01,  3.5533e-01, -5.8241e-01],
              [-1.9062e+00, -9.3887e-01,  1.5800e-01, -1.4945e+00],
              [-1.9333e+00,  5.2670e-01, -5.1293e-01, -1.6049e+00]]],
    
    
            [[[-1.0157e+00, -2.3455e+00, -4.2475e-01, -3.7559e-01],
              [ 1.1944e+00,  4.0143e-01, -4.3259e-01, -8.4820e-01],
              [-7.4481e-02, -2.1816e-01,  8.3423e-02,  1.0652e+00],
              [ 1.6184e+00,  1.3076e+00,  1.6673e-01,  4.0178e-01]],
    
             [[-3.7414e-01, -2.6123e-02, -5.4650e-01, -1.8188e-01],
              [-1.2025e+00,  1.3948e+00,  7.2542e-01,  4.6323e-01],
              [-2.2198e+00, -7.2702e-01, -2.6050e-01, -2.6035e-01],
              [ 5.8270e-01, -1.4014e+00, -1.6328e-02,  5.3239e-01]],
    
             [[ 1.9573e+00, -1.9700e+00, -4.2815e-01, -5.5273e-01],
              [ 4.9112e-01,  1.0814e+00, -5.5603e-01, -3.5645e-01],
              [-7.9357e-01,  1.1546e+00,  2.0966e+00, -2.9573e-01],
              [ 9.7181e-01, -1.0400e+00,  1.2679e+00, -1.2343e+00]],
    
             [[-8.3901e-01, -1.6174e+00, -1.0974e+00, -1.7761e+00],
              [ 8.6959e-01, -1.0474e+00,  7.4540e-01,  1.6417e+00],
              [ 5.9619e-01, -2.9758e-01,  1.6667e+00, -9.0589e-01],
              [ 3.2092e-01, -1.8596e-01,  4.4537e-01, -5.1966e-02]],
    
             [[-2.5424e-01, -1.1967e+00,  5.7911e-01,  1.8584e-01],
              [ 1.6691e+00,  3.2990e-01,  6.1244e-01, -7.7749e-01],
              [ 8.8799e-01,  5.5377e-01, -1.2193e+00,  2.4213e-02],
              [ 1.5533e-01, -1.1091e+00,  4.4351e-01, -2.0776e+00]]],
    
    
            [[[-1.1319e+00,  2.0076e-01,  1.0232e+00, -1.5897e+00],
              [-2.0357e-01, -2.8251e+00,  3.8959e-01,  1.0232e+00],
              [-1.0184e+00,  2.5289e-01,  7.3592e-01,  5.7374e-01],
              [-1.4978e+00,  6.5334e-02, -6.5226e-01,  1.0083e+00]],
    
             [[ 2.9369e-01, -3.3525e-01, -8.4944e-01,  1.8558e+00],
              [-1.5380e+00,  1.0600e+00,  3.3181e-01,  1.2682e+00],
              [ 1.4941e+00,  7.8667e-01, -1.3419e+00,  1.2280e+00],
              [-7.7970e-02, -5.9542e-01, -2.6168e+00,  1.0010e+00]],
    
             [[-7.3543e-01,  2.6170e+00, -1.0913e+00,  1.6529e+00],
              [ 1.9372e-01, -2.6608e+00, -1.4914e-01,  6.7004e-01],
              [ 1.4315e-01, -5.4096e-01, -4.0494e-01,  1.5426e+00],
              [ 1.1460e+00, -3.3811e-01, -3.2829e-01, -1.1516e+00]],
    
             [[ 2.1251e-01, -2.5890e-01, -7.5580e-01, -9.8821e-01],
              [ 3.1658e-01, -2.5991e-01,  1.0066e+00,  1.2465e+00],
              [ 1.5539e+00, -4.9563e-01, -1.8770e+00, -7.2246e-01],
              [-9.9305e-01,  8.0784e-01,  1.0672e-02,  4.7153e-01]],
    
             [[ 2.5651e-01, -2.1439e-01, -2.6471e-01,  2.0702e+00],
              [-1.2004e+00,  1.3182e+00, -8.6991e-01, -1.0670e+00],
              [ 6.9865e-01,  1.9459e+00,  1.9976e+00, -2.0958e-01],
              [-7.7282e-01, -1.6011e+00,  3.9619e-01, -7.7430e-01]]],
    
    
            [[[ 1.2093e+00, -8.1950e-01, -2.5434e-02, -6.8060e-01],
              [-4.9330e-01, -2.8020e-01,  9.8972e-01,  9.7297e-01],
              [ 6.4830e-01,  2.5073e-01,  2.6888e-01, -9.3854e-01],
              [-3.2507e-01,  4.9615e-01,  7.9890e-01,  1.5446e+00]],
    
             [[-8.1331e-01, -3.6518e-01,  7.7109e-01,  6.9970e-01],
              [-2.3026e+00, -9.1488e-01, -7.1835e-01,  1.0142e+00],
              [ 2.6651e-02, -2.7526e-01,  5.2323e-01, -3.3234e-02],
              [ 1.0167e+00,  1.1015e+00,  2.1942e+00,  8.0602e-01]],
    
             [[ 1.4669e+00, -3.4106e-01, -6.5527e-01, -4.7170e-01],
              [-1.0834e+00, -2.1852e+00, -1.8642e+00, -1.4551e+00],
              [ 6.1257e-01, -1.7239e+00,  1.6451e+00,  5.2325e-04],
              [ 1.3523e+00, -2.2505e-01,  1.6659e-01,  1.1650e+00]],
    
             [[ 4.4918e-01, -1.0368e+00,  1.1763e+00,  1.1114e-02],
              [-1.6019e+00,  9.3257e-02,  1.0813e+00, -1.3974e+00],
              [-1.3091e+00,  1.3476e+00,  1.2510e+00, -4.7561e-02],
              [ 8.6856e-01, -2.5905e-01,  1.7444e-01, -8.4461e-01]],
    
             [[ 3.3506e-01, -1.3883e+00, -5.9094e-01,  1.8419e-01],
              [ 1.1352e+00,  1.3375e+00,  1.0208e+00, -6.4435e-01],
              [-9.0285e-01,  3.0084e-01, -6.0607e-01,  4.8166e-01],
              [ 5.3981e-01,  8.8007e-01,  3.7355e-01,  4.6180e-01]]],
    
    
            [[[-9.0571e-01,  7.9029e-01, -1.1794e+00, -8.0096e-02],
              [ 8.7670e-01, -2.4142e-02, -4.0806e-01, -8.3221e-02],
              [-4.8874e-01, -2.1448e-01,  5.8343e-01,  1.9045e+00],
              [-1.1570e-01,  1.2363e+00, -9.0250e-01,  3.3993e-01]],
    
             [[-4.7792e-01,  6.7091e-01,  8.2414e-01, -5.1808e-01],
              [ 1.7345e+00,  3.2963e-01, -8.4986e-01, -2.1885e-01],
              [ 6.0303e-01,  7.6727e-01, -1.1845e+00,  9.7677e-02],
              [-3.2229e-01,  5.6926e-01,  3.9369e-01, -3.2665e-01]],
    
             [[-2.0682e+00,  3.9969e-01, -7.2576e-01,  1.2333e+00],
              [-1.7421e-01, -3.2267e-01, -1.2077e+00,  1.4455e+00],
              [ 7.7890e-01,  7.7048e-01,  1.2610e-01, -6.0329e-01],
              [ 9.0818e-01, -8.4406e-01, -2.9910e-01, -3.0544e-01]],
    
             [[-4.7576e-01,  7.9694e-01,  8.1631e-02,  2.5017e-02],
              [ 3.7546e-01,  9.1021e-01,  1.5452e-01,  9.6281e-01],
              [ 1.5576e+00,  3.3024e-01, -1.5683e+00,  3.0350e-01],
              [-1.8896e+00, -1.3941e+00, -2.8641e-01, -3.5747e-01]],
    
             [[ 7.2289e-01,  2.0987e+00, -3.3702e-01,  2.1694e+00],
              [-9.8494e-01, -2.2353e+00, -1.7367e+00,  1.4042e+00],
              [-2.1083e+00, -6.4073e-01, -2.0975e+00, -3.6848e-01],
              [-1.8404e-01, -1.5693e+00, -6.0938e-01,  1.6194e-01]]],
    
    
            [[[-1.5645e+00,  1.3684e+00,  1.6816e-01, -7.6315e-01],
              [ 1.1518e+00,  4.8765e-01, -6.4365e-01, -1.6646e-01],
              [-6.5168e-01,  1.4889e+00, -3.5659e-01, -6.6924e-01],
              [-9.4892e-01, -5.2736e-01, -3.8948e-01, -8.7832e-02]],
    
             [[ 1.3596e+00, -2.1955e-01,  2.2069e-01, -1.7030e+00],
              [-5.0489e-01,  1.7579e-01,  3.9333e-01, -3.5957e-01],
              [-1.4848e+00, -1.1054e-01, -3.0408e-01, -2.0735e+00],
              [-1.7488e+00,  1.5904e-02, -6.6630e-01, -1.5875e-01]],
    
             [[-4.1066e-01,  1.1661e+00,  9.2669e-01, -3.7516e-01],
              [ 1.0622e+00, -7.9659e-01, -1.1568e-01,  1.9987e+00],
              [ 7.3940e-01, -1.6579e-01, -1.3167e+00,  2.2857e-01],
              [-1.4001e+00,  6.3376e-01,  9.4178e-01,  4.6071e-01]],
    
             [[-6.7605e-01,  9.8846e-01,  9.7990e-01,  1.0766e+00],
              [-1.8742e+00, -6.0104e-01, -1.0522e+00,  2.1566e+00],
              [ 3.7905e-01,  1.5747e-01, -4.2352e-01, -1.1492e+00],
              [ 1.4159e+00, -7.3884e-01, -3.6326e-01, -4.5874e-01]],
    
             [[-7.0899e-03,  7.5393e-01, -6.6801e-01, -2.3622e-01],
              [ 1.6022e-01,  4.4819e-01,  1.1378e+00, -1.0354e+00],
              [-1.2523e+00,  8.1739e-01,  2.1197e+00,  3.8932e-01],
              [-1.8869e+00, -1.6848e+00, -1.5382e+00,  8.5648e-01]]],
    
    
            [[[ 4.4108e-01, -5.5777e-01, -9.5274e-02, -1.8150e+00],
              [-1.3210e-01, -1.2753e-01,  1.5278e-01,  9.4364e-02],
              [-6.3930e-02,  2.9066e+00, -1.4014e+00,  6.1019e-01],
              [-1.3037e+00, -7.9879e-01,  1.2175e-01, -9.5121e-01]],
    
             [[-3.2208e-01, -2.0547e+00,  9.8241e-01,  1.1295e+00],
              [ 5.0798e-01,  6.6591e-02, -1.8618e-01,  6.1505e-01],
              [ 1.2628e+00,  5.0201e-01, -2.5604e-01, -1.6296e+00],
              [-8.9917e-01,  7.2231e-01, -3.8776e-01, -1.0471e-02]],
    
             [[-9.4769e-01,  9.4623e-01, -1.0016e+00, -2.3024e-02],
              [ 1.1242e+00, -7.8809e-02, -1.6865e+00,  1.1589e+00],
              [-1.6647e+00,  1.6115e+00, -9.2662e-02,  3.5332e-01],
              [-5.0983e-02,  1.1632e+00, -1.2704e+00, -1.7280e-01]],
    
             [[-7.3661e-01,  1.0124e+00,  6.6844e-01,  1.6411e-02],
              [ 1.5428e-01,  9.9501e-01, -3.4633e-01, -8.6644e-01],
              [-6.0005e-01,  2.5242e-01, -1.0435e+00, -3.4366e-01],
              [ 3.5180e-01,  2.1368e-01, -7.7975e-01, -9.7630e-01]],
    
             [[-5.1927e-01,  2.9015e+00,  2.1524e-01,  7.1904e-01],
              [ 1.0004e+00, -1.3255e+00,  3.0691e-01, -1.0313e+00],
              [ 1.3765e+00,  4.3625e-01,  7.7203e-01,  8.8723e-01],
              [ 1.2455e-01, -2.7815e-01,  4.4031e-01,  1.2033e+00]]],
    
    
            [[[-5.2607e-01, -1.9414e-01, -1.5587e+00, -5.8147e-01],
              [ 1.1435e+00, -1.4865e+00, -4.0082e-01, -2.7370e-01],
              [ 4.5247e-01,  9.3737e-01, -2.0299e+00, -3.9682e-04],
              [-8.3269e-01, -1.5331e-01, -1.1885e+00,  4.5343e-01]],
    
             [[ 4.9789e-01, -1.1788e+00,  1.1917e+00, -1.2961e+00],
              [ 6.9166e-01,  1.4038e+00,  1.4819e+00,  5.3665e-01],
              [-1.2591e+00, -4.5341e-01, -7.5515e-01, -7.2794e-01],
              [ 3.0047e-01, -6.8652e-02,  1.5926e-02, -2.3904e-01]],
    
             [[ 1.5157e+00,  4.7548e-01,  6.2731e-02, -7.1596e-02],
              [ 5.6397e-01, -1.3262e+00, -1.1214e+00,  1.7090e+00],
              [-1.1481e+00, -5.5349e-01,  1.2108e-01, -1.2124e+00],
              [-8.0474e-01, -4.0069e-01,  8.5053e-02,  1.6795e+00]],
    
             [[-3.4908e-01, -1.3580e+00,  1.9778e+00, -5.6038e-01],
              [ 5.9964e-02,  1.3697e+00,  2.2952e-01,  1.2338e+00],
              [ 2.6539e-01,  9.5732e-02,  2.6049e-01, -3.0509e-01],
              [ 1.7603e+00, -1.0335e+00,  1.4180e-01, -1.9307e+00]],
    
             [[-1.7344e+00, -3.6356e-01,  1.7767e+00,  1.2724e+00],
              [ 2.6266e+00,  1.6071e-01, -5.7090e-01,  2.0255e+00],
              [ 1.3758e+00, -5.8043e-01,  3.3445e-01, -4.9063e-01],
              [-8.4320e-01,  4.1818e-01,  3.1326e-01, -7.5748e-01]]],
    
    
            [[[-1.1371e+00,  1.0132e+00,  8.8670e-01,  3.9788e-01],
              [ 7.3021e-01,  7.9143e-01,  2.9210e-01,  4.0294e-02],
              [-4.0587e-01, -1.0899e+00, -6.9569e-01,  1.2365e+00],
              [ 1.0535e+00,  7.7362e-01,  3.4564e-01,  6.5072e-01]],
    
             [[-2.1455e+00,  1.4068e+00,  8.1426e-02, -8.9781e-01],
              [ 6.3621e-01, -4.0519e-01,  7.8058e-01, -7.6835e-03],
              [-1.6004e+00,  1.4019e+00, -1.0040e+00, -7.5760e-01],
              [ 7.9327e-01,  4.4595e-02,  4.5663e-01,  4.2593e-02]],
    
             [[-8.3540e-04, -1.5268e+00,  3.2712e-01, -1.8549e-01],
              [-1.1860e+00, -4.4049e-01, -8.9372e-01, -3.7910e-01],
              [-2.6682e-01, -5.9872e-01, -1.5979e+00, -1.4854e+00],
              [ 8.4257e-01,  1.1993e+00,  1.9692e+00,  5.2776e-01]],
    
             [[-7.2501e-01, -4.9189e-01, -1.6312e+00, -7.4351e-01],
              [ 6.3122e-01, -4.8949e-02,  3.3338e-01, -7.0890e-01],
              [ 7.8867e-01, -6.7354e-01, -4.0503e-02,  2.0207e+00],
              [ 9.0374e-01,  2.8125e-01, -9.0018e-02,  4.5729e-01]],
    
             [[-9.1603e-02,  5.6410e-01, -8.5473e-01, -8.4376e-01],
              [ 9.3179e-01,  1.1740e+00, -1.8299e-01, -1.9015e-01],
              [-9.5030e-01,  2.9436e-01, -2.0207e-01, -4.7721e-01],
              [-8.5018e-02,  1.1248e+00,  1.6164e-01,  2.4080e-01]]],
    
    
            [[[-8.6686e-01, -2.9960e-01,  8.0157e-01, -8.6211e-01],
              [ 2.4365e+00, -4.1328e-01, -3.5045e-01, -1.3216e+00],
              [-6.2284e-02,  1.3707e+00, -1.6568e-01,  9.1302e-01],
              [-1.8815e+00, -1.1336e+00, -5.3247e-01, -3.9840e-01]],
    
             [[-2.4079e+00,  1.2210e-01, -6.6324e-01, -7.1417e-01],
              [-4.5398e-01,  1.5027e+00,  1.1016e+00, -6.0969e-01],
              [ 9.1224e-01,  3.9161e-01, -7.3733e-01, -1.2696e+00],
              [-2.1931e-01,  1.8106e+00,  7.9962e-01, -7.0053e-01]],
    
             [[-9.3824e-01, -6.3403e-01,  7.5732e-01,  1.4464e+00],
              [-1.6821e+00,  1.2188e+00,  1.6393e+00,  8.9150e-02],
              [-1.3109e+00, -5.7144e-01, -1.3480e+00, -6.9490e-01],
              [-2.4465e-01, -3.8417e-01,  2.6887e-01,  6.6231e-01]],
    
             [[ 1.3556e+00,  2.1708e-01, -1.3076e+00,  1.9358e+00],
              [ 9.3762e-01,  8.5401e-01,  6.5951e-01, -4.5369e-01],
              [-1.5324e+00,  3.4974e-02, -1.3442e+00,  2.3610e+00],
              [-8.7915e-01, -1.2860e+00,  4.5144e-01, -9.8597e-01]],
    
             [[-1.2221e-02, -9.5632e-01, -2.1982e-01,  1.2271e+00],
              [ 1.2993e+00, -5.9235e-01, -4.2703e-01,  1.0354e+00],
              [ 2.8287e-02,  1.1631e+00,  3.9992e-01,  3.1695e-01],
              [ 5.2600e-01, -1.1039e+00, -8.1282e-01,  6.4756e-02]]]])



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






    tensor([-0.0008, -0.0007, -0.0008, -0.0011, -0.0030,  0.0044, -0.0006,  0.0005,
            -0.0027, -0.0016, -0.0008, -0.0001, -0.0014, -0.0006, -0.0017])






    tensor([0.9983, 0.9983, 1.0020, 1.0052, 1.0031, 0.9971, 0.9996, 1.0019, 1.0006,
            0.9978, 0.9978, 0.9990, 0.9989, 1.0028, 1.0056])




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




    tensor([[ 1.3040, -0.8230,  1.9708],
            [-0.7702,  0.2772, -1.4552]])






    tensor([[0.7865, 0.3051, 0.8777],
            [0.3164, 0.5689, 0.1892]])






    tensor([[0.5000, 0.3051, 0.5000],
            [0.3164, 0.5000, 0.1892]])



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




    tensor([[-1.0752, -0.0275, -0.4274],
            [ 0.1253,  0.2610,  0.0779]])






    tensor([[0.2544, 0.4931, 0.3947],
            [0.5313, 0.5649, 0.5195]])






    tensor([[0.2544, 0.4931, 0.3947],
            [0.5000, 0.5000, 0.5000]])




```
sigmoid(x)
x
```




    tensor([[0.2544, 0.4931, 0.3947],
            [0.5313, 0.5649, 0.5195]])






    tensor([[-1.0752, -0.0275, -0.4274],
            [ 0.1253,  0.2610,  0.0779]])



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




    tensor([[ 0.8335, -0.1666,  0.6176],
            [ 1.5064,  0.5305, -1.0481]])






    tensor([[0.6971, 0.4584, 0.6497],
            [0.8185, 0.6296, 0.2596]])






    tensor([[0.6971, 0.4584, 0.6497],
            [0.8185, 0.6296, 0.2596]])



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




    tensor([[-0.9864,  2.3052,  0.5506],
            [ 2.5464, -0.7592, -0.4969]])






    tensor([[-0.0099,  2.3052,  0.5506],
            [ 2.5464, -0.0076, -0.0050]])






    tensor([[-0.2959,  2.3052,  0.5506],
            [ 2.5464, -0.2277, -0.1491]])



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




    tensor([ 0.4230, -0.0635])






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
    tensor([[ 0.2378,  0.3854, -0.2254],
            [-0.2904,  0.4997, -0.1770]], requires_grad=True)






    Parameter containing:
    tensor([-0.1925, -0.3759], requires_grad=True)






    Linear(in_features=3, out_features=2, bias=True)






    Parameter containing:
    tensor([[ 0.5843,  1.3520, -0.3061],
            [-0.8992, -0.6008, -1.3654]], requires_grad=True)






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

How ResBlock is initialized
- user inputs without default: `expansion`, `ni`, `nf`
- `norm2` is to choose between `BatchZero`, `InstanceZero`, or other `norm_type`
- `nh1` and `nh2` are defined by `nf`
- `nf` and `ni` is multiplied with `expansion`
- `k0`, `k1` are two dicts of norm_type, act_cls, ndim, and `**kwargs`
- `convpath`: a list of ConvLayers; if expansion == 1, 2 ConvLayers; otherwise, 3 ConvLayers
- if reduction, then add SEModule layer block
- if sa: add SimpleSelfAttention 
- self.convpath: take all the layers above into a single Sequential
- self.idpath: create a Sequential block in which there may or may be be a AvgPool and ConvLayer (position can switch too)
- self.act: default.activation or act_cls()

How ResBlock transforms input in `forward`
- self.act(self.convpath(x) + self.idpath(x))



```
#|export
class ResBlock(Module):
    "Resnet block from `ni` to `nh` with `stride`"
#     @snoop
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


```

```

This is a resnet block (normal or bottleneck depending on `expansion`, 1 for the normal block and 4 for the traditional bottleneck) that implements the tweaks from [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187). In particular, the last batchnorm layer (if that is the selected `norm_type`) is initialized with a weight (or gamma) of zero to facilitate the flow from the beginning to the end of the network. It also implements optional [Squeeze and Excitation](https://arxiv.org/abs/1709.01507) and grouped convs for [ResNeXT](https://arxiv.org/abs/1611.05431) and similar models (use `dw=True` for depthwise convs).

The `kwargs` are passed to `ConvLayer` along with `norm_type`.


```
ResBlock(1, 4, 2)
```




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



### ```SEBlock(expansion, ni, nf, groups=1, reduction=16, stride=1, **kwargs)```


```
#|export
def SEBlock(expansion, ni, nf, groups=1, reduction=16, stride=1, **kwargs):
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, reduction=reduction, nh1=nf*2, nh2=nf*expansion, **kwargs)
```

### ```SEResNeXtBlock(expansion, ni, nf, groups=32, reduction=16, stride=1, base_width=4, **kwargs)```


```
#|export
def SEResNeXtBlock(expansion, ni, nf, groups=32, reduction=16, stride=1, base_width=4, **kwargs):
    w = math.floor(nf * (base_width / 64)) * groups
    return ResBlock(expansion, ni, nf, stride=stride, groups=groups, reduction=reduction, nh2=w, **kwargs)
```

### ```SeparableBlock(expansion, ni, nf, reduction=16, stride=1, base_width=4, **kwargs)```


```
#|export
def SeparableBlock(expansion, ni, nf, reduction=16, stride=1, base_width=4, **kwargs):
    return ResBlock(expansion, ni, nf, stride=stride, reduction=reduction, nh2=nf*2, dw=True, **kwargs)
```


```
# fastnbs("AvgPool(")
```

## Time Distributed Layer

Equivalent to Keras `TimeDistributed` Layer, enables computing pytorch `Module` over an axis.

### ```_stack_tups(tuples, stack_dim=1)```
official: Stack tuple of tensors along `stack_dim`


```
#|export
# @snoop
def _stack_tups(tuples, stack_dim=1):
    "Stack tuple of tensors along `stack_dim`"
    return tuple(torch.stack([t[i] for t in tuples], dim=stack_dim) for i in range_of(tuples[0]))
#     lst = []
#     res = []
#     pp(range_of(tuples[0]))
#     for i in range_of(tuples[0]):
#         for t in tuples:
#             lst.append(t[i])
#         res.append(torch.stack(lst, dim=stack_dim))
#     return tuple(res)

```

### ```TimeDistributed(Module)```
official: Applies `module` over `tdim` identically for each step, use `low_mem` to compute one at a time.

- apply on individual layer: `tconv = TimeDistributed(nn.Conv2d(3,4,1))`
- on a user defined layer: `TimeDistributed(Mod())`
- how to use low_memory: out = tmod(x,y)，tmod.low_mem=True， out_low_mem = tmod(x,y)


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
x.shape, y.shape
```




    (torch.Size([2, 5, 3, 2, 2]), torch.Size([2, 5, 3, 2, 2]))




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
# show_doc(TimeDistributed)
```

This module is equivalent to [Keras TimeDistributed Layer](https://keras.io/api/layers/recurrent_layers/time_distributed/). This wrapper allows to apply a layer to every temporal slice of an input. By default it is assumed the time axis (`tdim`) is the 1st one (the one after the batch size). A typical usage would be to encode a sequence of images using an image encoder.

The `forward` function of `TimeDistributed` supports `*args` and `**kkwargs` but only `args` will be split and passed to the underlying module independently for each timestep, `kwargs` will be passed as they are. This is useful when you have module that take multiple arguments as inputs, this way, you can put all tensors you need spliting as `args` and other arguments that don't need split as `kwargs`.

> This module is heavy on memory, as it will try to pass mutiple timesteps at the same time on the batch dimension, if you get out of memorey errors, try first reducing your batch size by the number of timesteps.


```
create_body??
```

    Object `create_body` not found.



```
from fastai.vision.all import *
```


```
encoder = create_body(resnet18())
encoder
```




    Sequential(
      (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (5): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (6): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (7): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )



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
autograd in fastai

### ```script, _swish_ji_fwd, _SwishJitAutoFn, swish, Swish, _mish_jit_fwd```


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

### ```ParameterModule(Module)```
Register a lone parameter `p` in a module.


```
#|export
class ParameterModule(Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p): self.val = p
    def forward(self, x): return x
```

### ```children_and_parameters(m)```
Return the children of `m` and its direct parameters not registered in modules.

The direct parameters are those not inside those parameters of `m.children()`


```
sum([[1],[2]],[])
```




    [1, 2]




```
#|export
# @snoop
def children_and_parameters(m):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: 
            children.append(ParameterModule(p))
    return children
```


```
class TstModule(Module):
    def __init__(self): self.a,self.lin = nn.Parameter(torch.randn(1)),nn.Linear(5,10)

tst = TstModule()
children = children_and_parameters(tst)
```


```
test_eq(len(children), 2)
test_eq(children[0], tst.lin)
assert isinstance(children[1], ParameterModule)
test_eq(children[1].val, tst.a)
```

### ```has_children(m)```
Whether a model has children layers


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
tst
children = flatten_model(tst)
children
```




    Sequential(
      (0): TstModule(
        (lin): Linear(in_features=5, out_features=10, bias=True)
      )
      (1): TstModule(
        (lin): Linear(in_features=5, out_features=10, bias=True)
      )
    )






    [Linear(in_features=5, out_features=10, bias=True),
     ParameterModule(),
     Linear(in_features=5, out_features=10, bias=True),
     ParameterModule()]




```
test_eq(len(children), 4)
assert isinstance(children[1], ParameterModule)
assert isinstance(children[3], ParameterModule)
```

### ```NoneReduce()```
A context manager to evaluate `loss_func` with none reduce.

within this context, the `loss_func.reduction` is set to None or its `reduction` arg is set to None, before applying data to the `loss_func`


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
test_eq("reduction" in str(inspect.signature(loss_fn)), False)
test_eq(loss_fn.reduction, 'mean')
```


```
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn.reduction, 'mean')
```


```
loss_fn = F.mse_loss
test_eq("reduction" in str(inspect.signature(loss_fn)), True)
```


```
with NoneReduce(loss_fn) as loss_func:
    loss = loss_func(x,y)
test_eq(loss.shape, [5])
test_eq(loss_fn, F.mse_loss)
```

### ```in_channels(m)```
Return the shape of the first weight layer in `m`.


```
#|export
# @snoop
def in_channels(m):
    "Return the shape of the first weight layer in `m`."
#     pp.deep(lambda: next(l.weight.shape[1] for l in flatten_model(m) if nested_attr(l,'weight.ndim',-1)==4))
    try: return next(l.weight.shape[1] for l in flatten_model(m) if nested_attr(l,'weight.ndim',-1)==4)
    except StopIteration as e: e.args = ["No weight layer"]; raise
```


```
nn.Conv2d(5,4,3).weight.ndim
```




    4




```
# help(nested_attr)
```


```
in_channels(nn.Sequential(nn.Conv2d(5,4,3)))
```




    5




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


```

```


```

```
