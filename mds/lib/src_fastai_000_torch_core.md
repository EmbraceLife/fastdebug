---
skip_exec: true
jupyter:
  jupytext:
    split_at_heading: true
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

## imports

```python
#|hide
#| eval: false
! [ -e /content ] && pip install -Uqq fastai  # upgrade fastai on colab
```

```python
#| default_exp to_delete_torch_core
```

```python
#|export
from __future__ import annotations
from fastai.imports import *
from fastai.torch_imports import *
from packaging.version import parse
```

```python
from PIL import Image
```

```python
from fastdebug.utils import *
from fastdebug.core import *
```

```python
#|hide
from nbdev.showdoc import *
```

## fastai.imports

```python
import fastai.imports as fi
import fastai.torch_imports as fti
```

```python
fi
fti
```

```python
# %cat /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastai/imports.py
# %cat /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastai/torch_imports.py
```

## \_all_, defaults

```python
#|export
_all_ = ['progress_bar','master_bar']
```

```python
import fastai.torch_core as ft
```

```python
ft._all_
```

```python
#|export
defaults.benchmark = True
```

## setup_cuda
not used in this notebook

```python
#|export
def setup_cuda(benchmark=defaults.benchmark):
    "Sets the main cuda device and sets `cudnn.benchmark` to `benchmark`"
    if torch.cuda.is_available():
        if torch.cuda.current_device()==0:
            def_gpu = int(os.environ.get('DEFAULT_GPU') or 0)
            if torch.cuda.device_count()>=def_gpu: torch.cuda.set_device(def_gpu)
        torch.backends.cudnn.benchmark = benchmark
```

# Torch Core

> Basic pytorch functions used in the fastai library

## Arrays and show

### `subplots(nrows, ncols,...)`
A wrapper around plt.subplots(...); only used in show_images; not in show_image, nor in show_image_batch;to create/display and return a fig with specified size and a specified num of empty subplots;

```python
# fastview("subplots")
```

```python
#|export
# @snoop
@delegates(plt.subplots, keep=True)
def subplots(
    nrows:int=1, # Number of rows in returned axes grid
    ncols:int=1, # Number of columns in returned axes grid
    figsize:tuple=None, # Width, height in inches of the returned figure 
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure
    suptitle:str=None, # Title to be set to returned figure
    **kwargs
) -> (plt.Figure, plt.Axes): # Returns both fig and ax as a tuple 
    "Returns a figure and set of subplots to display images of `imsize` inches"
    if figsize is None: 
#         pp.deep(lambda: nrows*imsize if suptitle is None or imsize>2 else nrows*imsize+0.6)
        h=nrows*imsize if suptitle is None or imsize>2 else nrows*imsize+0.6 #https://github.com/matplotlib/matplotlib/issues/5355
        figsize=(ncols*imsize, h)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows*ncols==1: ax = array([ax])
    return fig,ax
```

This is used in `get_grid`. `suptitle`, `sharex`, `sharey`, `squeeze`, `subplot_kw` and `gridspec_kw` are all passed down to [plt.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib-pyplot-subplots).

```python
# check(subplots)
```

```python
# fdb = Fastdb(subplots)
# fdb.eg = """
# _,axs = subplots()
# test_eq(axs.shape,[1])
# plt.close()
# """
# fdb.print()
# fdb.docsrc(10, "A wrapper around plt.subplots(...); only used in show_images; not in show_image, nor in show_image_batch;\
# to create/display and return a fig with specified size and a specified num of empty subplots")
# fdb.print()
```

```python
# fastview('subplots')
```

```python
#|hide
_,axs = subplots()
test_eq(axs.shape,[1])
plt.close()
```

```python
_,axs = subplots(2,3)
test_eq(axs.shape,[2,3])
plt.close()
```

### `_fig_bounds(x)`
- make sure figsize is properly defined; 
- the bounds (either width or height) is between 1 and 5; 

```python
# fastview("_fig_bounds")
```

```python
[min(5, max(1,r)) for r in range(10)] # only allow value from 1 to 5; 
```

```python
#|export
# @snoop
def _fig_bounds(x):
#     pp.deep(lambda: x//32)
    r = x//32
#     pp.deep(lambda: min(5, max(1,r)))
    return min(5, max(1,r))
# to be used in show_image
```

```python
# fdb = Fastdb(_fig_bounds)
# fdb.print()
# fdb.docsrc(0, "make sure figsize is properly defined; the bounds (either width or height) is between 1 and 5")
# fdb.docsrc(5, "only allow value from 1 to 5")
# fdb.print()
```

```python
# fastcodes("allow value")
```

```python
# fastview("_fig_bounds")
```

### `show_image(im, ax...)`
- im can be a pytorch tensor, np.array or pure a PIL image object which will then be turned into np.array; 
- if im is a pytorch tensor and 1st dim is < 5, meaning if color channel is not the last dim, then the dims will be permuted; 
- if im has only one channel, remove the dim for the channel; 
- ax and its options are used to display; 

```python
# fastview("show_image")
```

```python
# torch.permute: to reorganise a tensor's dim
```

```python
#|export
# @snoop
@delegates(plt.Axes.imshow, keep=True, but=['shape', 'imlim'])
def show_image(im, ax=None, figsize=None, title=None, ctx=None, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    # Handle pytorch axis order
    if hasattrs(im, ('data','cpu','permute')):
        im = im.data.cpu()
        if im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=array(im)
    # Handle 1-channel images
    if im.shape[-1]==1: im=im[...,0]

    ax = ifnone(ax,ctx)
    if figsize is None: figsize = (_fig_bounds(im.shape[0]), _fig_bounds(im.shape[1]))
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.axis('off')
    return ax
```

```python
# fdb = Fastdb(show_image)
# fdb.eg = """
# # pp.deep(lambda: Image.open(TEST_IMAGE_BW))
# im = Image.open(TEST_IMAGE_BW)
# ax = show_image(im, cmap="Greys")
# """
# fdb.docsrc(3, "im can be a pytorch tensor, np.array or pure a PIL image object which will then be turned into np.array; \
# if im is a pytorch tensor and 1st dim is < 5, meaning if color channel is not the last dim, then the dims will be permuted; \
# if im has only one channel, remove the dim for the channel; ax and its options are used to display")
# fdb.print()
```

```python
# fastview("show_image")
```

`show_image` can show PIL images...

```python
# check(show_image) # to see its signature
```

```python
%%snoop # activated by %load_ext snoop, but pp, snoop, @snoop are not available
# pp.deep(lambda: Image.open(TEST_IMAGE_BW))
im = Image.open(TEST_IMAGE_BW)
# im = array(im)
# pp.deep(lambda: show_image(im, cmap="Greys"))
ax = show_image(im, cmap="Greys")
```

...and color images with standard `CHW` dim order...

```python
# %%snoop
# pp.deep(lambda: np.array(Image.open(TEST_IMAGE)))
im2 = np.array(Image.open(TEST_IMAGE))
im2.shape
ax = show_image(im2, figsize=(5,5))
```

...and color images with `HWC` dim order...

```python
torch.as_tensor(im2).shape
im3 = torch.as_tensor(im2).permute(2,0,1) # how to turn an array to a tensor and permute its dims
im3.shape
ax = show_image(im3, figsize=(2,2))
```

```python
# fastview("show_image")
```

### `show_titled_image(o, ...)`
use a tuple o which is (img, title) to make show_image to display the image with title printed on top

```python
# fastview("show_titled_image")
```

```python
#|export
# @snoop
@delegates(show_image, keep=True)
def show_titled_image(o, **kwargs):
    "Call `show_image` destructuring `o` to `(img,title)`"
    show_image(o[0], title=str(o[1]), **kwargs)
```

```python
# fdb = Fastdb(show_titled_image)
# fdb.eg = """
# %%snoop
# show_titled_image((im3,'A puppy'), figsize=(2,2))
# """
# fdb.docsrc(3, "use a tuple o which is (img, title) to make show_image to display the image with title printed on top")
# fdb.print()
```

```python
# %%snoop
show_titled_image((im3,'A puppy'), figsize=(2,2))
```

### `show_images(ims, nrows, ncols, ...)`
- to display multiple images with/wihtout titles in rows and cols; 
- first input 'ims' is a tuple of imgs; 
- 'titles' should be in tuple (e.g., a tuple of None, if not available); 
- ncols is calc by total imgs and nrows;
- empty fig and subplots are created by subplots; 
- finally loop through each img, title, ax to show_image

```python
# fastview("show_images")
```

Show all images `ims` as subplots with `rows` using `titles`. `suptitle` provides a way to create a figure title for all images. If you use `suptitle`, `constrained_layout` is used unless you set `constrained_layout` to `False`. 

```python
#|export
# @snoop
@delegates(subplots)
def show_images(ims, nrows=1, ncols=None, titles=None, **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`."
    if ncols is None: ncols = int(math.ceil(len(ims)/nrows))
    if titles is None: titles = [None]*len(ims)
#     pp.deep(lambda: subplots(nrows, ncols, **kwargs)[1].flat)
    axs = subplots(nrows, ncols, **kwargs)[1].flat
    for im,t,ax in zip(ims, titles, axs): show_image(im, ax=ax, title=t)
```

```python
# fdb = Fastdb(show_images)
# fdb.eg = """
# %%snoop
# show_images((im,im3),titles=('number','puppy'),suptitle='Number Puppy',  imsize=3)
# """
# fdb.docsrc(3, "to display multiple images with/wihtout titles in rows and cols; first input 'ims' is a tuple of imgs; 'titles' should be in tuple if not None; ncols is calc by total imgs and nrows;\
# titles are given as None for all images if not available; empty fig and subplots are created by subplots; \
# finally loop through each img, title, ax to show_image; ")
# fdb.print()
```

```python
# fastview("show_images")
```

```python
# %%snoop
show_images((im,im3),titles=('number','puppy'),suptitle='Number Puppy',  imsize=3)
```

### ArrayBase, ArrayImageBase, ArrayImage, ArrayImageBW, ArrayMask
`ArrayImage`, `ArrayImageBW` and `ArrayMask` are subclasses of `ndarray`, through which images can be turned arrays and get displayed.

```python
#|export
class ArrayBase(ndarray):
    "An `ndarray` that can modify casting behavior"
    @classmethod
#     @snoop
    def _before_cast(cls, x): 
#         pp(x, array(x).shape)
#         n = array(x)
        return x if isinstance(x,ndarray) else array(x)
```

```python
#|export
class ArrayImageBase(ArrayBase):
    "Base class for arrays representing images"
    _show_args = {'cmap':'viridis'}
#     @snoop
    def show(self, ctx=None, **kwargs):
        return show_image(self, ctx=ctx, **{**self._show_args, **kwargs})
```

```python
#|export
class ArrayImage(ArrayImageBase):
    "An array representing an image"
    pass
```

```python
#|export
class ArrayImageBW(ArrayImage):
    "An array representing an image"
    _show_args = {'cmap':'Greys'}
```

```python
#|export
class ArrayMask(ArrayImageBase):
    "An array representing an image mask"
    _show_args = {'alpha':0.5, 'cmap':'tab20', 'interpolation':'nearest'}
```

```python
im = Image.open(TEST_IMAGE)
```

```python
# %%snoop
im_t = cast(im, ArrayImage) # how to use cast to turn PIL image object into an ArrayImage object
test_eq(type(im_t), ArrayImage)
```

```python
ax = im_t.show(figsize=(2,2))
```

```python
# fastview(show_image)
# fastnotes("use cast")
```

```python
test_fig_exists(ax)
```

## Basics

### Tensor.`__array_eq__(self:Tensor, b)`
make 0 dim and 1 dim tensor comparison possible: `test_eq(tensor(1), tensor([1]))`

```python
from torch import tensor 
```

```python
# before __array_eq__, we have to choose between torch.equal or ==; after, we don't need to
tensor(100).dim()
assert not torch.equal(tensor(100), tensor([100])) 
assert tensor(100) == tensor([100])
```

```python
if tensor([True]): print("yes") # tensor([True]) is equivalent to True
```

```python
#|export
@patch
# @snoop
def __array_eq__(self:Tensor,b):
#     pp.deep(lambda: torch.equal(self,b) if self.dim() else self==b)
    return torch.equal(self,b) if self.dim() else self==b
```

```python
# from fastcore.test import equals
```

```python
# Tensor.__array_eq__ is actually used inside equals which is from fastcore.test
# test_eq??
# equals??
# test??
```

```python
# %%snoop
test_eq(tensor(1), torch.tensor([1])) 
```

```python
# fastnbs("__array_eq__")
```

### Tensor._array2tensor(x)
wrap around `torch.from_numpy` to turn np.array into torch tensor for special cases (np.uint16=> np.float32; np.int=>np.int64 win32)

```python
# fastnbs("__array_eq__")
# fastlistnbs("src_fastai")
```

```python
#|export
def _array2tensor(x):
    if x.dtype==np.uint16: x = x.astype(np.float32)
    # windows default numpy int dytpe is int32, while torch tensor default int dtype is int64
    # https://github.com/numpy/numpy/issues/9464
    if sys.platform == "win32":
        if x.dtype==np.int: x = x.astype(np.int64)
    return torch.from_numpy(x)
```

```python
test_eq(array([1,2,3]).dtype, 'int64')
test_eq(array([1.,2,3]).dtype, 'float64')
test_eq(torch.from_numpy(array([1,2,3])).dtype, torch.int64)
test_eq(torch.from_numpy(array([1.,2,3])).dtype, torch.float64)
```

### ```tensor(x, *rest, **kwargs)```
- fastai.tensor can convert tuple, list, ndarray, pd.Series, pd.DataFrame and pure free flowing elements into torch.tensor
- it has potential to turn data.__array__ or iterable data into torch.tensor (commented out at the moment)

```python
#|export
@use_kwargs_dict(dtype=None, device=None, requires_grad=False, pin_memory=False)
def tensor(x, *rest, **kwargs):
    "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
    if len(rest): x = (x,)+rest
    # There was a Pytorch bug in dataloader using num_workers>0. Haven't confirmed if fixed
    # if isinstance(x, (tuple,list)) and len(x)==0: return tensor(0)
    res = (x if isinstance(x, Tensor)
           else torch.tensor(x, **kwargs) if isinstance(x, (tuple,list))
           else _array2tensor(x) if isinstance(x, ndarray)
           else as_tensor(x.values, **kwargs) if isinstance(x, (pd.Series, pd.DataFrame))
#            else as_tensor(array(x, **kwargs)) if hasattr(x, '__array__') or is_iter(x)
           else _array2tensor(array(x), **kwargs))
    if res.dtype is torch.float64: return res.float()
    return res
```

```python
# how to make a copy of a tensor properly using pytorch.tensor or fastai.tensor
test_eq(torch.tensor([1,2,3]) is torch.tensor(torch.tensor([1,2,3])), False) # being False, meaning they are copies not reference
# The above way of making a tensor copy will throw a warning, the pytorch recommend the following way
test_eq(torch.tensor([1,2,3]) is torch.tensor([1,2,3]).clone().detach(), False) 
# Fastai has the following intuitive way
test_eq(tensor(torch.tensor([1,2,3])) is torch.tensor([1,2,3]), False)
```

```python
# both fastai.tensor and torch.tensor can turn an array or a list into a tensor like below
test_eq(torch.tensor(array([1,2,3])), tensor(array([1,2,3])))
test_eq(torch.tensor([1,2,3]), tensor([1,2,3]))
```

```python
# fastai.tensor can handle multiple elements directly, but not torch.tensor
test_eq(tensor(1.), torch.tensor(1.))
try:
    torch.tensor(1,2,3)
except: 
    print("torch.tensor can't handle (1,2,3)")
test_eq(tensor(1,2,3), torch.tensor([1,2,3]))
```

```python
test_eq(tensor(torch.tensor([1,2,3])), torch.tensor([1,2,3]))
test_eq(tensor(array([1,2,3])), torch.tensor([1,2,3]))
test_eq(tensor(1,2,3), torch.tensor([1,2,3]))
test_eq_type(tensor(1.0), torch.tensor(1.0))
```

### ```set_seed(s, reproducible=False)```
set random seed for either torch.tensor, ndarray, or just pure python random, and set True or False for ```torch.backends.cudnn.deterministic/benchmark```


```set_seed``` is useful for reproducibility between runs. It is important to remember that certain classes such as ```Dataloaders``` have internal random number generators that is not effected by this function, so this must be run before such objects are created in order to guarantee reproducibility. 

```python
test_eq(22 % 3, 1)
test_eq(22 // 3, 7)
```

```python
#|export
def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

Here is an example of how ```set_seed``` can be used to reset the state of random number generators.

```python
# set_seed(2*33)
set_seed(12)
a1 = np.random.random()
a2 = torch.rand(())
a3 = random.random()
# set_seed(2*33)
set_seed(12)
b1 = np.random.random()
b2 = torch.rand(())
b3 = random.random()
print('a\'s: {0:3.3f} {1:3.3f} {2:3.3f}'.format(a1,a2,a3))
print('b\'s: {0:3.3f} {1:3.3f} {2:3.3f}'.format(b1,b2,b3))
```

```python
test_eq(a1,b1)
test_eq(a2,b2)
test_eq(a3,b3)
```

### ```get_random_states``` and ```set_random_states```
get random states for pure python random, ndarray, torch.tensor... and set values for those random_states

#### ```get_random_states```
return a dict of states for python_random_state, numpy_state, torch_state, torch_cuda_state, torch_deterministic, torch_benchmark

```get_random_states``` and ```set_random_states``` are useful for storing a state so you can go back to it later. 

```python
#|export
def get_random_states():
    "Gets states for `random`, `torch`, and `numpy` random number generators"
    return {'random_state':random.getstate(),
            'numpy_state':np.random.get_state(),
            'torch_state':torch.get_rng_state(),
            'torch_cuda_state':torch.cuda.get_rng_state_all(),
            'torch_deterministic':torch.backends.cudnn.deterministic,
            'torch_benchmark':torch.backends.cudnn.benchmark}
```

#### ```set_random_states```
set values for random_state,numpy_state,torch_state,torch_cuda_state,torch_deterministic,torch_benchmark

```python
#|export
def set_random_states(random_state,numpy_state,torch_state,torch_cuda_state,torch_deterministic,torch_benchmark):
    "Set states for `random`, `torch`, and `numpy` random number generators"
    random.setstate(random_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state_all(torch_cuda_state)
    torch.backends.cudnn.deterministic=torch_deterministic
    torch.backends.cudnn.benchmark=torch_benchmark
```

Below notice that the old values and rewinded values are the same because we were able to return to the previous state. 

```python
old_states = get_random_states()
olds = (random.random(),np.random.random(),torch.rand(()))
news = (random.random(),np.random.random(),torch.rand(()))
set_random_states(**old_states)
rewinds = (random.random(),np.random.random(),torch.rand(()))

print('olds:    {0:3.3f} {1:3.3f} {2:3.3f}'.format(*olds))
print('news:    {0:3.3f} {1:3.3f} {2:3.3f}'.format(*news))
print('rewinds: {0:3.3f} {1:3.3f} {2:3.3f}'.format(*rewinds))
```

```python
test_ne(olds,news)
test_eq(olds,rewinds)
```

### ```no_random(seed=42, reproducible=True)```
```with no_random(): ``` can create a random context


In ```no_random``` we combine the ideas of rewinding state with ```get_random_states``` and ```set_random_states``` with the ability to ```set_seed``` and create a context manager that can allow us to control randomness in a portion of our code. 

Note: Similar to ```torch.random.fork_rng```, but also with ```numpy``` and ```random```

```python
#|export
@contextmanager
def no_random(seed=42,reproducible=True):
    "Stores and retrieves state of random number generators. Sets random seed for `random`, `torch`, and `numpy`."
    states = get_random_states()
    set_seed(seed,reproducible=reproducible)
    try:
        yield #we are managing global variables
    finally:
        set_random_states(**states)
```

Here are some examples on how we can use ```no_random``` to control the randomness within a block of code.  

```python
states=get_random_states()
olds = (random.random(),np.random.random(),torch.rand(()))
set_random_states(**states) #rewinding above random calls

with no_random():
    new1 = (random.random(),np.random.random(),torch.rand(()))
with no_random():
    new2 = (random.random(),np.random.random(),torch.rand(()))
with no_random(seed=100):
    seeded1 = (random.random(),np.random.random(),torch.rand(()))
with no_random(seed=100):
    seeded2 = (random.random(),np.random.random(),torch.rand(()))
        
rewinds = (random.random(),np.random.random(),torch.rand(()))

print('olds:    {0:3.3f} {1:3.3f} {2:3.3f}'.format(*olds))
print('new1:    {0:3.3f} {1:3.3f} {2:3.3f}'.format(*new1))
print('new2:    {0:3.3f} {1:3.3f} {2:3.3f}'.format(*new2))
print('seeded1: {0:3.3f} {1:3.3f} {2:3.3f}'.format(*seeded1))
print('seeded2: {0:3.3f} {1:3.3f} {2:3.3f}'.format(*seeded2))
print('rewinds: {0:3.3f} {1:3.3f} {2:3.3f}'.format(*rewinds))
```

Notice that olds, and rewinds are also both equal to each other. From this  we can see that everything in the ```with``` blocks did not update the state outside of the block. Inside of the block, the state is reset for any particular seed, so for the same seed you should get the same random number generator results.  


Note: It is important to remember that classes like ``` Dataloader``` have internal random number generators, and ```no_random``` will have no effect on those random number generators.

```python
test_ne(olds,new1)
test_eq(new1,new2)
test_ne(new1,seeded1)
test_eq(seeded1,seeded2)
test_eq(olds,rewinds)
```

### ```unsqueeze(x, dim=-1,n=1)```
unsqueeze or expand a tensor another n dimensions, if current x is 1 dim, and n=2, then x will be 3 dim

```python
#|export
# @snoop
def unsqueeze(x, dim=-1, n=1):
    "Same as `torch.unsqueeze` but can add `n` dims"
    for _ in range(n): x = x.unsqueeze(dim)
    return x
```

```python
t = tensor([1])
t2 = unsqueeze(t, n=2)
test_eq(t2,t[:,None,None])
test_eq(t.dim() + 2, t2.dim()) # dim == 3
```

### ```unsqueeze_(x, dim=-1, n=1)```
It can do unsqueeze above but inplace (without making another copy)

```python
#|export
def unsqueeze_(x, dim=-1, n=1):
    "Same as `torch.unsqueeze_` but can add `n` dims"
    for _ in range(n): x.unsqueeze_(dim)
    return x
```

```python
t = tensor([1])
unsqueeze_(t, n=2)
test_eq(t, tensor([1]).view(1,1,1))
```

### ```_fa_rebuild_tensor``` and ```_fa_rebuild_qtensor```
not used in this notebook

```python
#|export
def _fa_rebuild_tensor (cls, *args, **kwargs): return cls(torch._utils._rebuild_tensor_v2(*args, **kwargs))
def _fa_rebuild_qtensor(cls, *args, **kwargs): return cls(torch._utils._rebuild_qtensor  (*args, **kwargs))
```

### ```apply(func, x, *args, **kwargs)```
apply func to x, and x can be a list, tuple, L, dict or a scalar object

```python
# retain_type?
# is_listy?
```

```python
#|export
def apply(func, x, *args, **kwargs):
    "Apply `func` recursively to `x`, passing on args"
    if is_listy(x): return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x,dict):  return {k: apply(func, v, *args, **kwargs) for k,v in x.items()}
    res = func(x, *args, **kwargs)
    return res if x is None else retain_type(res, x)
```

```python
# fastnotes("slice(")
```

```python
type([])([1,2,3]) # is equivalent to list([1,2,3])
type(slice(1,3))(1,2,3) # is equivalent to slice(1,2,3)
[0,1,2,3,4,5,6,7][slice(0,-1,2)] # how to use slice
```

```python
apply(lambda x: x + 1, [1,2,3])
apply(lambda x: x + 1, (1,2,3))
apply(lambda x: x + 1, L(1,2,3))
apply(lambda x: x + 1, {'a':1, 'b':2,'c':3})
```

### ```maybe_gather(x, axis=0)```
Gather copies of `x` on `axis` (if training is distributed). used in ```to_detach```

```python
# num_distrib(): Return the number of processes in distributed training (if applicable). defined below in this notebook
# num_distrib? 
```

```python
test_eq(tensor(0).ndim, 0)
test_eq(tensor(1,2,3).ndim, 1)
test_eq(tensor([[[1],[2],[3]]]).ndim, 3)
test_eq(tensor([[[1],[2],[3]]]).shape, (1,3,1))
```

```python
def num_distrib():
    "Return the number of processes in distributed training (if applicable)."
    return int(os.environ.get('WORLD_SIZE', 0))
```

```python
#|export
@snoop
def maybe_gather(x, axis=0):
    "Gather copies of `x` on `axis` (if training is distributed)"
    if num_distrib()<=1: return x
    ndim = x.ndim
    res = [x.new_zeros(*x.shape if ndim > 0 else (1,)) for _ in range(num_distrib())]
    torch.distributed.all_gather(res, x.contiguous() if ndim > 0 else x[None])
    return torch.cat(res, dim=axis) if ndim > 0 else torch.cat(res, dim=axis).mean()
```

### ```to_detach(b, cpu=True, gather=True)```
Recursively detach lists of tensors in `b ` (from graph and no more gradients); put them on the CPU if `cpu=True`. (non-tensor is dealt too)

```python
# t.detach(): return a new tensor detached from graph so there will be no gradient
# Tensor.detach? 
# t.cpu(): return a new tensor stored in cpu memory
# Tensor.cpu?
```

```python
#|export
@snoop
def to_detach(b, cpu=True, gather=True):
    "Recursively detach lists of tensors in `b `; put them on the CPU if `cpu=True`."
    @snoop
    def _inner(x, cpu=True, gather=True):
        if not isinstance(x,Tensor): return x
        x = x.detach()
        if gather: x = maybe_gather(x)
        return x.cpu() if cpu else x
    return apply(_inner, b, cpu=cpu, gather=gather)
```

`gather` only applies during distributed training and the result tensor will be the one gathered across processes if `gather=True` (as a result, the batch size will be multiplied by the number of processes).

```python
lstold = [tensor(1.,2, requires_grad=True), tensor([3.,4], requires_grad=True)]
test_eq(lstold[0].requires_grad, True)
lstnew = to_detach([tensor(1,2), tensor([3,4])])
test_eq(lstnew[0].requires_grad, False)
```

### ```to_half(b)```
Recursively map lists of tensors in `b ` to FP16, if b is not floating point then return unchanged.

```python
# check(torch.is_floating_point)
# check(Tensor.half)
```

```python
test_eq(torch.is_floating_point(tensor(1.,2)), True)
test_eq(torch.is_floating_point(tensor(1,2)), False)
test_eq(tensor(1.,2.).dtype, torch.float32)
test_eq(tensor(1.,2.).half().dtype, torch.float16)
```

```python
#|export
def to_half(b):
    "Recursively map lists of tensors in `b ` to FP16."
    return apply(lambda x: x.half() if torch.is_floating_point(x) else x, b)
```

```python
test_eq([tensor(1.), tensor([1,2.])][0].dtype, torch.float32)
test_eq(to_half([tensor(1.), tensor([1,2.])])[0].dtype, torch.float16)
```

### ```to_float(b)```
Recursively map lists of int tensors in `b ` to float

```python
# from fastai.torch_core import to_float
```

```python
# to_float??
```

```python
#|export
def to_float(b):
    "Recursively map lists of int tensors in `b ` to float."
    return apply(lambda x: x.float() if torch.is_floating_point(x) else x, b)
```

```python
test_eq(torch.is_floating_point(tensor(1,2)), False)
test_eq(torch.is_floating_point(tensor(1.,2)), True)
test_eq([t.float() for t in [tensor(1,2), tensor([3,4])]], [tensor([1., 2.]), tensor([3., 4.])])
test_eq(to_float([tensor(1,2), tensor([3,4])]), [tensor([1, 2]), tensor([3, 4])]) 
```

### ```_has_mps()```
whether `torch.backends.mps` is available or not. what is [mps](https://huggingface.co/docs/accelerate/usage_guides/mps)?

```python
# check(torch.backends.mps.is_available)
```

```python
#|export
# @snoop
def _has_mps(): return nested_attr(torch, 'backends.mps.is_available', noop)()
```

```python
# nested_attr??
test_eq(torch.backends.mps.is_available(), True)
```

### ```default_device(use_cuda=-1)```
use `default_device(True)` for torch to work in cuda or mps, and `default_device(False)` or `default_device(-1)` to work in cpu

```python
#|export
# None: True if available; True: error if not available; False: use CPU
defaults.use_cuda = None
```

```python
#|export
# @snoop
def default_device(use_cuda=-1):
    "Return or set default device; `use_cuda`: None - CUDA if available; True - error if not available; False - CPU"
    if use_cuda != -1: defaults.use_cuda=use_cuda
    use = defaults.use_cuda or (torch.cuda.is_available() and defaults.use_cuda is None)
    assert torch.cuda.is_available() or not use
    return torch.device(torch.cuda.current_device()) if use else torch.device('cpu')
```

```python
#|export
# @snoop
def default_device(use=-1):
    "Return or set default device; `use_cuda`: -1 - CUDA/mps if available; True - error if not available; False - CPU"
    if use == -1: use = defaults.use_cuda
    else: defaults.use_cuda=use
    if use is None:
        if torch.cuda.is_available(): use = True
    if use:
        if torch.cuda.is_available(): return torch.device(torch.cuda.current_device())
        if _has_mps(): return torch.device('mps')
    return torch.device('cpu')
```

```python
#|cuda
# %%snoop # to work need to remove the line above, which is  #|cuda
if torch.cuda.is_available():
    _td = torch.device(torch.cuda.current_device())
    test_eq(default_device(-1), _td)
    test_eq(default_device(True), _td)
else:
    test_eq(default_device(False), torch.device('cpu'))
    test_eq(default_device(-1), torch.device('cpu'))
    test_eq(default_device(True), torch.device('mps'))    
    test_eq(_has_mps(), True)

```

### ```to_device(b, device=None, non_blocking=False)```
Recursively put `b` (a list of tensors) on `device` (on cpu or cuda or mps).

```python
# check(Tensor.to)
```

```python
#|export
# @snoop
def to_device(b, device=None, non_blocking=False):
    "Recursively put `b` on `device`."
    if defaults.use_cuda==False: device='cpu'
    elif device is None: device=default_device()
    def _inner(o):
        if isinstance(o,Tensor): return o.to(device, non_blocking=non_blocking)
#         if hasattr(o, "to_device"): return o.to_device(device)
        return o
    return apply(_inner, b)
```

```python
defaults.use_cuda
```

```python
t = to_device((3,(tensor(3),tensor(2))))
t1,(t2,t3) = t
test_eq(str(t2.device), 'mps:0')
test_eq(str(t3.device), 'mps:0')
```

```python
#|cuda
if torch.cuda.is_available():
    test_eq_type(t,(3,(tensor(3).cuda(),tensor(2).cuda())))
    test_eq(t2.type(), "torch.cuda.LongTensor")
    test_eq(t3.type(), "torch.cuda.LongTensor")
else: 
    print("cuda is not available")
```

### ```to_cpu(b)```
Recursively map lists of tensors in `b ` to the cpu exclusively.

```python
#|export
def to_cpu(b):
    "Recursively map lists of tensors in `b ` to the cpu."
    return to_device(b,'cpu')
```

```python
test_eq(t3.type(), "torch.mps.LongTensor")
```

```python
t3 = to_cpu(t3)
test_eq(t3.type(), "torch.LongTensor")
test_eq(t3, 2)
test_eq(str(t3.device), 'cpu')
```

### ```to_np(x)```
Convert a tensor to a numpy array recursively.

```python
#|export
def to_np(x):
    "Convert a tensor to a numpy array."
    return apply(lambda o: o.data.cpu().numpy(), x)
```

```python
try: 
    to_np(array(1))
except AttributeError:
    print('array has no attr: cpu')
```

```python
t3 = to_np(t3)
test_eq(type(t3), np.ndarray)
test_eq(t3, 2)
```

```python
to_np([tensor(1), tensor([2,3])])
```

```python
import numpy
```

```python
a, b = map(lambda t: type(t), to_np([tensor(1), tensor([2,3])]))
test_eq(a, numpy.ndarray)
test_eq(b, numpy.ndarray)
```

```python

```

### ```range_of(a, b=None, step=None)```
All indices of collection `a`, if `a` is a collection, otherwise `range`

```python
check(range_of)
```

```python
range_of(tensor([[1,2]]))
range_of(tensor([[1,2],[3,4]]))
```

### sum

```python
check(sum)
```

```python
L(1,2,3)
sum([1,2,3], L())
sum([[1],[2],[3]], L())
sum([[1],[2],[3]], list())
```

### ```to_concat(xs, dim=0)```
Concat the element in `xs` (recursively if they are tuples/lists of tensors)

```python
#|export
# @snoop(depth=2)
# @snoop
def to_concat(xs, dim=0):
    "Concat the element in `xs` (recursively if they are tuples/lists of tensors)"
    if not xs: return xs
    if is_listy(xs[0]): return type(xs[0])([to_concat([x[i] for x in xs], dim=dim) for i in range_of(xs[0])])
    if isinstance(xs[0],dict):  return {k: to_concat([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
    #We may receive xs that are not concatenable (inputs of a text classifier for instance),
    #   in this case we return a big list
    try:    return retain_type(torch.cat(xs, dim=dim), xs[0])
    except: 
        lst = []
        for o_ in xs:
            for i in range_of(o_):
#                 pp.deep(lambda: o_.index_select(dim, tensor(i)).squeeze(dim))
                item = retain_type(o_.index_select(dim, tensor(i)).squeeze(dim), xs[0]) # squeeze out one dim for text classifier
                lst.append(L(item))
        return sum(lst, L()) # bring a list of things into a single L list
#         return sum([L(retain_type(o_.index_select(dim, tensor(i)).squeeze(dim), xs[0])
#                           for i in range_of(o_)) for o_ in xs], L())
```

```python
retain_type
```

```python
test_eq(tensor([1,2]).shape, torch.Size([2]))
test_eq(tensor([1,2]).dim(), 1)
test_eq(tensor([1,2,3,4]).dim(), 1)
test_eq(to_concat([tensor([1,2]), tensor([3,4])]), tensor([1,2,3,4])) # dim = 1 won't work as the tensors have only 0 dim
```

```python
test_eq(to_concat([tensor([[1,2]]), tensor([[3,4]])], dim=0).shape[0], 2) # concat on rows, as it has 2 dims
test_eq(to_concat([tensor([[1,2]]), tensor([[3,4]])], dim=1).shape[1], 4) # concat on cols
```

```python
to_concat([(tensor([1,2]), tensor([3,4])), (tensor([3,4]), tensor([5,6]))]) 
to_concat([(tensor([1,2]), tensor([3,4])), (tensor([3,4]), tensor([5,6]))], dim=-1) # when tensor has 1 dim, then dim=0 or -1 same
to_concat([(tensor([[1,2]]), tensor([[3,4]])), (tensor([[3,4]]), tensor([[5,6]]))]) # when each tensor has 2 dims, dim=1 is working
to_concat([(tensor([[1,2]]), tensor([[3,4]])), (tensor([[3,4]]), tensor([[5,6]]))], dim=1) # or dim=1 or -1
```

```python
# when tensors have different dims, meaning can't concat, then sum(L(...)) get run to squeeze them out, and put into a L list
# tensor([3,4]) is squeezed into tensor(3), tensor(4)
# tensor([[5,6]]) is squeezed into tensor([5,6])
to_concat([tensor([3,4]), tensor([[5,6]])]) 
```

```python
# dim can only be 0 or -1, as tensor([3,4]) and tensor([7,8]) have just 1 dim
# now xs is a list of two lists, it uses a different formula, which concat the first parts of two lists, then 2nd parts of two lists
# as as tensor([3,4]) and tensor([7,8]) have just 1 dim, so they can only concat as a long row
to_concat([[tensor([3,4]), tensor([[5,6]])], [tensor([7,8]), tensor([[9,10]])]], dim=0) 
to_concat([[tensor([3,4]), tensor([[5,6]])], [tensor([7,8]), tensor([[9,10]])]], dim=-1) 
```

```python
to_concat([(tensor([1,2]),), (tensor([3,4]),)])
```

```python
# although xs[0] is not a list, but both tensors in xs share the same dim e.g., 2 here, torch.cat(xs,dim=0) can apply directly
# test_eq(tensor([[1,2]]).dim(), tensor([[3,4], [5,6]]).dim())
# tensor([[1,2]]).shape, tensor([[3,4], [5,6]]).shape
to_concat([tensor([[1,2]]), tensor([[3,4], [5,6]])], dim=0) # concat on rows, ok because both tensors have same num of cols
```

```python
# to concat tensors on cols, require they having same num of rows, as this example below
to_concat([tensor([[1,2],[7,8]]), tensor([[3,4], [5,6]])], dim=1)
```

```python
# to concat tensors on cols, require they having same num of rows, but the tensors below don't match on rows, so 
# we have to squeeze them and put into a long L list
# only the tensor(1) get out of tensor([[1,2]]) is due to range_of(tensor([[1,2]])) returns [0]
to_concat([tensor([[1,2]]), tensor([[3,4], [5,6]])], dim=1)
```

```python
to_concat([tensor([[1,2]]), tensor([[3,4], [5,6]])], dim=0)
```

```python
# 1st part of two lists have different dims, so they are squeezed and put into a long L list as [tensor([1, 2]),tensor(3),tensor(4)]
# 2nd parts of two lists have same dims, so they can concat as above
to_concat([(tensor([[1,2]]), tensor([[3,4]])), (tensor([3,4]), tensor([[5,6]]))])
to_concat([(tensor([[1,2]]), tensor([[3,4]])), (tensor([3,4]), tensor([[5,6]]))], dim=-1)
```

```python
dict(foo=tensor([1,2]), bar=tensor(3,4))
to_concat([dict(foo=tensor([1,2]), bar=tensor(3,4))])
```

```python
test_eq(type(to_concat([dict(foo=tensor([1,2]), bar=tensor(3,4))])), dict)
```

### ```Tensor.index_select(dim, index)```
- select part of a tensor to become a new tensor, ```dim=0``` to select row, ```dim=1``` to select col, 
- so when `dim=1` and `index=0` will select first col
- `dim=0` and `index=1` will select second row

```python
# check(Tensor.index_select)
tensor([[1,2]]).shape
tensor([[1,2]]).dim()
```

```python
tensor([[1,2],[3,4]]).index_select(1, tensor(0))
tensor([[1,2],[3,4]]).index_select(1, tensor(1))
tensor([[1,2],[3,4]]).index_select(0, tensor(0))
tensor([[1,2]]).index_select(1, tensor(0))
tensor([[1,2]]).index_select(1, tensor(1))
```

## Tensor subtypes

### ```t1.set_meta(t2, as_copy=False)```
pass tensor t2's `__dict__` to Tensor t1, either as a copy or not

```python
#|export
@patch
def set_meta(self:Tensor, x, as_copy=False):
    "Set all metadata in `__dict__`"
    if not hasattr(x,'__dict__'): return
    # XXX: change to `deepcopy` once PyTorch 1.7.1 is out, and check nb 23 segmentation fit works
    self.__dict__ = copy(x.__dict__) if as_copy else x.__dict__
```

```python
t2 = tensor(1,2)
t2.__dict__ = {'a': 1}
test_eq(t2.__dict__, {'a': 1})
t1 = tensor(3,4)
t1.set_meta(t2)
test_eq(t1.__dict__, {'a': 1})
```

### pass ```torch.Tensor.as_subclass``` to ```torch.as_subclass```

```python
#|export
if not hasattr(torch,'as_subclass'): torch.as_subclass = torch.Tensor.as_subclass
```

### ```retain_meta(x, res, as_copy=False)```
- copy `x.__dict__` to `res.__dict__`

```python
def retain_meta(x, res, as_copy=False):
    "Call `res.set_meta(x)`, if it exists"
    if hasattr(res,'set_meta'): 
        res.set_meta(x, as_copy=as_copy)
#         if hasattr(x, '__dict__'): print("{} of type {} .__dict__: {}".format(x, type(x), x.__dict__))
#         if hasattr(res, '__dict__'): print("{} of type {} .__dict__: {}".format(res, type(res), res.__dict__))            
    return res
```

### ```t2 = t.as_subclass(_T)```
create another tensor `t2` as an instance of `_T` and copy `__dict__` from `t`

```python
#|export
@patch
# @snoop(depth=2)
def as_subclass(self:Tensor, typ):
    "Cast to `typ` and include `__dict__` and meta"
    return retain_meta(self, torch.as_subclass(self, typ))
```

```python
# check(torch.as_subclass)
class _T(Tensor): pass
t = tensor(1.).requires_grad_()
t.img_size = 1
```

```python
# torch.as_subclass(t, _T) only makes a new tensor instance of the class _T
t1 = torch.as_subclass(t, _T) 
test_eq(isinstance(t1, _T), True)
test_eq(isinstance(t, _T), False)
test_eq(t1.__dict__, {})
test_eq(t.__dict__, {'img_size':1})
```

```python
# fastai's Tensor.as_subclass(self, _T): 
# not only create an new tensor from _T, but also pass self (another tensor)'s __dict__ to the new tensor
t2 = t.as_subclass(_T) 
test_eq(isinstance(t2, _T), True)
test_eq(t2.__dict__, t.__dict__)
```

`Tensor.set_meta` and `Tensor.as_subclass` work together to maintain `__dict__` after casting.

```python

```

```python
class _T(Tensor): pass
t = tensor(1.).requires_grad_()
t.img_size = 1
t2 = t.as_subclass(_T) # create 
test_eq(t.img_size, t2.img_size)
test_eq(t2.img_size, 1)
assert(t2.requires_grad_)
```

### ```_torch_handled(args, opt, func)```
False, if `func` is not in `opt`; True, if func in `opt` and `args` is an instance of `opt[func]`

```python
test_eq(all([True, True, True]), True)
test_eq(all([True, False, True]), False)
```

```python
#|export
def _torch_handled(args, opt, func):
    if func not in opt: return False
    for oks in opt[func]:
        if all(isinstance(arg,ok) for arg,ok in zip(args,oks) if ok): return True
```

### ```_rebuild_from_type(func, type, args, dict)```
`t = _rebuild_from_type(tensor, torch.Tensor, [1,2,3], {'b':1})`, to create an instance using a func, a class, args and a dict

```python
#|export
# from https://github.com/pytorch/pytorch/blob/13c975684a220ec096216ec6468ccd0dc90ff50a/torch/_tensor.py#L34
def _rebuild_from_type(func, type, args, dict):
    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret
```

```python
t = _rebuild_from_type(tensor, torch.Tensor, [1,2,3], {'b':1})
```

```python
test_eq(t.__dict__, {'b':1})
```

### ```_find_args(x)```
`_find_args(["a", "b", tensor(2)])` find the arg with `__dict__` from a list of args

```python
#|export
def _find_args(x):   
    x0 = x[0] if is_listy(x[0]) and x[0] else x
    return [a for a in x0 if hasattr(a,'__dict__')]
```

```python
test_eq(_find_args(["a", "b", tensor(2)]), [tensor(2)])
```

### ```TensorBase(Tensor)```
Many methods inside are overriding those inside Tensor

#### ```TensorBase.__new__(cls, x, **kwargs)```

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res
```

```python
a = TensorBase(1)
test_eq(type(a), TensorBase)
a1 = TensorBase.__new__(TensorBase, 1)
a2 = a.__new__(TensorBase, 1)
test_eq_type(a, a1)
test_eq_type(a2, a1)
```

```python
# `TensorBase` and its subclasses also allow for passing through metadata size as img_size...
a = TensorBase(1,img_size=(128,128))
test_eq(a.img_size,(128,128))
```

### ```TensorBase._before_cast(cls, x)```
just return `tensor(x)` instead

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
```

```python
test_eq(TensorBase._before_cast(1), tensor(1))
test_eq(TensorBase(2)._before_cast(1), tensor(1))
```

### ```TensorBase.__repr__(self)```
represent the tensor in `TensorBase`

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
#         pp(self.__class__, self.__class__.__name__, super().__repr__())
        return re.sub('tensor', self.__class__.__name__, super().__repr__())
```

```python
TensorBase(1).__repr__()
```

### ```TensorBase.__reduce_ex__(self, proto)```
handling pickle.dump and pickle.loads, but no idea how this method get triggered (question)

```python
# pickle.dump??
# pickle.loads??
```

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
#         pp(self.__class__, self.__class__.__name__, super().__repr__())
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

#     @snoop
    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))
```

```python
class _T(TensorBase): pass
t = _T(range(5))
test_eq(type(pickle.loads(pickle.dumps(t))), _T) # trigger __reduce_ex__(self,proto)
```

```python

```

### ```TensorBase.register_func(cls, func, *oks)```
add a list of functions stored in `oks` to `cls._opt[func]`. There is no code example in this notebook

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
#         pp(self.__class__, self.__class__.__name__, super().__repr__())
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

#     @snoop
    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))


    @classmethod
#     @snoop
    def register_func(cls, func, *oks): cls._opt[func].append(oks)
#         pp.deep(lambda: cls._opt[func].append(oks))
```

```python
# to run this example, need to run cells below which define TensorMask etc
# for o in Tensor.__getitem__, Tensor.__ne__,Tensor.__eq__,Tensor.add,Tensor.sub,Tensor.mul,Tensor.div,Tensor.__rsub__,Tensor.__radd__,Tensor.matmul,Tensor.bmm:
#     TensorBase.register_func(o, TensorMask, TensorImageBase)
#     TensorBase.register_func(o, TensorImageBase, TensorMask)
```

### ```TensorBase.__torch_function__```
- performing operation like addition, repr, multiplication etc 
- and take meta info from tensors and give it to the resulting tensor of the operation; 
- set `TensorBase.debug=True` to print out more info; 

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
#         pp(self.__class__, self.__class__.__name__, super().__repr__())
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

#     @snoop
    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @classmethod
#     @snoop
    def register_func(cls, func, *oks): cls._opt[func].append(oks)

    @classmethod
#     @snoop
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
        res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)
        return res

```

```python
# TensorBase.__torch_function__??
```

`TensorBase` hooks into `__torch_function__` to ensure metadata is not lost. To see all functions being called, set `debug`.

```python
a = TensorBase(1)
# a.debug=True
TensorBase.debug=True
# a.debug=False
1/(a+1)
# a + 1
```

#### ```default_collate(listy)```
to prepare a batch of data or tensors

```python
from torch.utils.data._utils.collate import default_collate
```

```python
# default_collate??
```

```python
# Example with a batch of `int`s:
test_eq(default_collate([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
# Example with a batch of `str`s:
test_eq(default_collate(['a', 'b', 'c']), ['a', 'b', 'c'])
# Example with `Map` inside the batch:
default_collate([{'A':0, 'B':1}, {'A': 100, 'B': 100}]) # {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
# Example with `NamedTuple` inside the batch:
Point = namedtuple('Point', ['x', 'y'])
test_eq(default_collate([Point(0, 0), Point(1, 1)]), Point(x=tensor([0, 1]), y=tensor([0, 1])))
# Example with `Tuple` inside the batch:
test_eq(default_collate([(0, 1), (2, 3)]), [tensor([0, 2]), tensor([1, 3])])
# Example with `List` inside the batch:
test_eq(default_collate([[0, 1], [2, 3]]), [tensor([0, 2]), tensor([1, 3])])
```

### ```TensorBase.new_tensor(size, dtype=None, device=None, requires_grad=False)```
- `t.new_tensor(1)`
- first, use `tensor.new_tensor` to create a new tensor with data 1, then use `as_subclass` to make the new tensor of the same class as t

```python
# check out the original pytorch Tensor.new_tensor docs
# Tensor.new_tensor??
```

```python
tensor(1).new_tensor((1))
```

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @classmethod
    def register_func(cls, func, *oks): cls._opt[func].append(oks)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
        res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)
        return res

#     @snoop
    def new_tensor(self, size, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

```

```python
# fastnbs("as_subclass")
```

```python
# TensorBase.new_tensor??
```

```python
class _T(TensorBase): pass
t = _T(range(5))
test_eq(t.new_tensor(1), _T(1))
# test_eq_type(t.new_tensor([1,2]), _T([1,2]))
```

### ```TensorBase.new_ones(self, data, dtype=None, device=None, requires_grad=False)```
- `t.new_ones((2,3))`
- make `t` a subclass of `torch.Tensor` and use `Tensor.new_ones` to create a tensor of value with shape (2,3)
- then make the new tensor a subclass as that of `t`

```python
# Tensor.new_ones?
# torch.as_subclass??
```

```python
t1 = torch.tensor((), dtype=torch.int32)
t1.new_ones((2, 3))
```

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
#         pp(self.__class__, self.__class__.__name__, super().__repr__())
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @classmethod
    def register_func(cls, func, *oks): cls._opt[func].append(oks)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
        res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)
        return res

    def new_tensor(self, size, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

#     @snoop
    def new_ones(self, data, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_ones(data, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

```

```python
class _T(TensorBase): pass
t = _T(range(5))
t
```

```python
t.new_ones(3)
t.new_ones((2,3))
```

```python
test_eq_type(t.new_ones(1), _T([1]))
```

### ```TensorBase.new((self, x=None)```
- use `Tensor.new` to create a new tensor
- use `as_subclass` to make the new tensor to receive `t`'s class but **not** `t.__dict__`

```python
t2 = Tensor([1,2])
t2.__dict__ = {'a':1}
t2.new()
```

```python
t3 = t2.new((2,3))
getattr(t3, "__dict__")
# help(t2.new)
```

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
#         pp(self.__class__, self.__class__.__name__, super().__repr__())
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @classmethod
    def register_func(cls, func, *oks): cls._opt[func].append(oks)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
        res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)
        return res

    def new_tensor(self, size, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)


    def new_ones(self, data, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_ones(data, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

#     @snoop(depth=3)
    def new(self, x=None):
        cls = type(self)
        res = self.as_subclass(Tensor).new() if x is None else self.as_subclass(Tensor).new(x)
        return res.as_subclass(cls)
```

```python
class T(TensorBase): pass
t = T(1)
t.__dict__ = {'a':1}
```

```python
# t.as_subclass??
# retain_meta??
```

```python
t1 = t.new()
t1.__class__
```

```python
# help(tensor)
# tensor??
```

```python
# fastnbs("as_subclass")
```

```python
T(1).new((1,2))
```

```python
T(1.).new((1.,2.))
```

### ```TensorBase.requires_grad_(self, requires_grad=True)```
- set a tensor's `requires_grad` to be True or False

```python
#|export
class TensorBase(Tensor):
    "A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"
    debug,_opt = False,defaultdict(list)
    def __new__(cls, x, **kwargs):
        res = cast(tensor(x), cls)
        for k,v in kwargs.items(): setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x): return tensor(x)
    def __repr__(self): 
#         pp(self.__class__, self.__class__.__name__, super().__repr__())
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

    def __reduce_ex__(self,proto):
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
        if self.is_quantized: args = args + (self.q_scale(), self.q_zero_point())
        args = args + (self.requires_grad, OrderedDict())
        f = torch._utils._rebuild_qtensor if self.is_quantized else  torch._utils._rebuild_tensor_v2
        return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @classmethod
    def register_func(cls, func, *oks): cls._opt[func].append(oks)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
        res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)
        return res

    def new_tensor(self, size, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

    def new_ones(self, data, dtype=None, device=None, requires_grad=False):
        cls = type(self)
        return self.as_subclass(Tensor).new_ones(data, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

    def new(self, x=None):
        cls = type(self)
        res = self.as_subclass(Tensor).new() if x is None else self.as_subclass(Tensor).new(x)
        return res.as_subclass(cls)
    
#     @snoop
    def requires_grad_(self, requires_grad=True):
        # Workaround https://github.com/pytorch/pytorch/issues/50219
        self.requires_grad = requires_grad
        return self
```

```python
#|hide
# test of https://github.com/pytorch/pytorch/issues/50219
x = TensorBase(torch.rand(4,3,16,16))
with torch.no_grad():
    y = x.requires_grad_()
    assert y.requires_grad and x.requires_grad
```

### More demonstration of ```TensorBase```

```python
# fastnbs("cast(x", filter_folder="src")
```

```python
a = TensorBase(1,img_size=(128,128))
test_eq(a.img_size,(128,128))
b = cast(a,TensorBase)
test_eq(b.img_size,(128,128))
test_eq(torch.stack([a,b],0), TensorBase([1, 1]))
test_eq(torch.stack([a,b],0).img_size,(128,128))
```

```python
class _TImage(TensorBase): pass
class _TImage2(_TImage): pass
t1 = _TImage([1.])
t2 = _TImage2([1.])
t2+t1
```

```python
class _T(TensorBase): pass

t = _T(range(5))
test_eq(repr(t), '_T([0, 1, 2, 3, 4])')
test_eq(t[0], 0)
test_eq_type(t+1, _T(range(1,6)))

test_eq_type(t[_T([False,False,True,True,True])], _T([2,3,4]))
test_eq_type(t[_T([2,3,4])], _T([2,3,4]))

```

```python
t = tensor([1,2,3])
m = TensorBase([False,True,True])
test_eq(t[m], tensor([2,3]))
t = tensor([[1,2,3],[1,2,3]])
m = cast(tensor([[False,True,True],
                 [False,True,True]]), TensorBase)
test_eq(t[m], tensor([2,3,2,3]))
```

```python
t = tensor([[1,2,3],[1,2,3]])
t.img_size = 1
t2 = cast(t, TensorBase)
test_eq(t2.img_size, t.img_size)
x = retain_type(tensor([4,5,6]), t2)
test_eq(x.img_size, t.img_size)
t3 = TensorBase([[1,2,3],[1,2,3]], img_size=1)
test_eq(t3.img_size, t.img_size)
t4 = t2+1
t4.img_size = 2
test_eq(t2.img_size, 1)
test_eq(t4.img_size, 2)
# this will fail with `Tensor` but works with `TensorBase`
test_eq(pickle.loads(pickle.dumps(t2)).img_size, t2.img_size)
```

```python
#|hide
# test of https://github.com/pytorch/pytorch/issues/47186
class _T(TensorBase): ...
t = _T([1.])
test_eq_type(t.new([1,2]), _T([1.,2.]))
test_eq_type(t.new(), _T([]))
```

### ```TensorImageBase(TensorBase)```
- add `ArrayImageBase._show_args` to ```TensorImageBase._show_args```
- add method `show(self, ctx=None, **kwargs)` to enable ```show_image``` here

```python
#|export
class TensorImageBase(TensorBase):
    _show_args = ArrayImageBase._show_args
    def show(self, ctx=None, **kwargs):
        return show_image(self, ctx=ctx, **{**self._show_args, **kwargs})
```

### ```TensorImage(TensorImageBase)```
- empty subclass of ```TensorImageBase```

```python
#|export
class TensorImage(TensorImageBase): pass
```

```python
im = Image.open(TEST_IMAGE)
im_t = cast(array(im), TensorImage)
test_eq(type(im_t), TensorImage)
```

```python
ax = im_t.show(figsize=(2,2))
```

```python
test_fig_exists(ax)
```

### ```TensorImageBW(TensorImage)```
- a subclass of ```TensorImage```
- add ```ArrayImageBW._show_args``` to ```TensorImageBW._show_args```

```python
#|export
class TensorImageBW(TensorImage): _show_args = ArrayImageBW._show_args
```

### ```TensorMask(TensorImageBase)```
- a subclass of TensorImageBase
- use ```ArrayMask._show_args``` as its own `_show_args`
- ```TensorMask.show``` is a wrapper around ```TensorImageBase.show``` with attr `codes` setting up the `vmin` and `vmax`

```python
#|export
class TensorMask(TensorImageBase):
    _show_args = ArrayMask._show_args

    def show(self, ctx=None, **kwargs):
        codes = getattr(self, 'codes', None)
        if codes is not None: kwargs = merge({'vmin': 0, 'vmax': len(codes)}, kwargs)
        return super().show(ctx=ctx, **kwargs)
```

```python
im_t2 = cast(tensor(1), TensorMask)
test_eq(type(im_t2), TensorMask)
test_eq(im_t2, tensor(1))
```

### register many subclasses into ```TensorBase``` and ```TensorMask```

```python
#|export
for o in Tensor.__getitem__, Tensor.__ne__,Tensor.__eq__,Tensor.add,Tensor.sub,Tensor.mul,Tensor.div,Tensor.__rsub__,Tensor.__radd__,Tensor.matmul,Tensor.bmm:
    TensorBase.register_func(o, TensorMask, TensorImageBase)
    TensorBase.register_func(o, TensorImageBase, TensorMask)

TensorMask.register_func(torch.einsum, str, TensorImageBase, TensorMask)
TensorMask.register_func(torch.einsum, str, TensorMask, TensorImageBase)
```

### when ```TensorMask``` operate with ```TensorImageBase```


Operations between `TensorMask` and `TensorImageBase` objects return the type of the `TensorImageBase` object:

```python
a = TensorMask([1,2])
test_eq_type(TensorImage(1)+a, TensorImage([2,3]))
test_eq_type(1-a, TensorMask([0,-1]))
```

```python
#|hide (last test of to_concat)
test_eq_type(to_concat([TensorImage([1,2]), TensorImage([3,4])]), TensorImage([1,2,3,4]))
```

### ```TensorFlowField(TensorBase)```
- a subclass of ```TensorBase```
- store ```TensorImageBase``` and ```TensorFlowField``` under ```TensorImage._opt[F.grid_sample]```

```python
#|export
class TensorFlowField(TensorBase): pass
TensorImage.register_func(F.grid_sample, TensorImageBase, TensorFlowField)
```

```python
TensorImage._opt[F.grid_sample]
```

```python
t1 = TensorImage([1.]).view(1,1,1,1)
test_eq(t1.shape, torch.Size([1, 1, 1, 1]))
t2 = TensorFlowField([1.,1.]).view(1,1,1,2)
test_eq(t2.shape, torch.Size([1, 1, 1, 2]))
```

```python
test_eq(F.grid_sample(t1, t2), TensorImage([[[[0.2500]]]]))
test_eq(F.grid_sample(t1, t2).shape, torch.Size([1, 1, 1, 1]))
```

```python
test_eq_type(F.grid_sample(t1, t2), TensorImage([[[[0.25]]]]))
```

```python
# TensorImage.register_func??
# F.grid_sample?
```

### ```TensorCategory(TensorBase)```
- a subclass of ```TensorBase```
- register ```TensorImageBase, TensorCategory``` under ```TensorBase._opt[Tensor.__getitem__]```
- ```TensorMask, TensorImage``` can use ```TensorCategory``` as index

```python
#|export 
class TensorCategory(TensorBase): pass
TensorBase.register_func(Tensor.__getitem__, TensorImageBase, TensorCategory)
```

```python
tc = TensorCategory([1,2,3])
mask_t = TensorMask([0,2,4,5])
im_t = TensorImage([0,2,4,5])
test_eq(mask_t[tc], tensor([2,4,5]))
test_eq(im_t[tc], tensor([2,4,5]))
```

### ```TensorMultiCategory(TensorCategory)```

```python
#|export 
class TensorMultiCategory(TensorCategory): pass
```

### ```TitledTensorScalar(TensorBase)```
- ```show_title``` not defined (question)

```python
#|export
class TitledTensorScalar(TensorBase):
    "A tensor containing a scalar that has a `show` method"
    def show(self, **kwargs): show_title(self.item(), **kwargs)
```

```python
# TitledTensorScalar(1).show()
```

## L -

### ```tensored(self:L)```
- turn a L list of lists into a list of tensors

```python
#|export
@patch
# @snoop
def tensored(self:L):
    "`mapped(tensor)`"
    ### original source
#     return self.map(tensor)

    ### fastdebug source
    sp1 = tensor
    sp2 = self.map(sp1)
    return sp2
```

```python
t = L(([1,2],[3,4]))
test_eq(t, [[1, 2],[3, 4]])
test_eq(t.tensored(), [tensor(1,2),tensor(3,4)])
```

### ```stack(self:L, dim=0)```
- run `L.tensored` to turn a L list of lists into a list of tensors
- stack a list of tenors on top of each other (on the same cols extending rows) into a large tensor

```python
@patch
# @snoop
def stack(self:L, dim=0):
    "Same as `torch.stack`"
    ### original source
#     return torch.stack(list(self.tensored()), dim=dim)

    ### fastdebug source
    sp1 = self.tensored()
    sp2 = list(sp1)
    sp3 = torch.stack(sp2, dim=dim)
    return sp3
```

```python
test_eq(t.stack(), tensor([[1,2],[3,4]]))
```

```python
# torch.stack??
```

### ```cat  (self:L, dim=0)```
- - run `L.tensored` to turn a L list of lists into a list of tensors
- stack a list of tenors on the same rows (extend cols) into a large tensor

```python
@patch
# @snoop
def cat  (self:L, dim=0):
    "Same as `torch.cat`"
    ### original source
#     return torch.cat  (list(self.tensored()), dim=dim)
    
    ### fastdebug
    sp1 = self.tensored()
    sp2 = list(sp1) # convert a L into a list
    sp3 = torch.cat(sp2, dim=dim)
    return sp3
```

```python
test_eq(t.cat(), tensor([1,2,3,4]))
```

```python
# torch.cat?? vs torch.stack??
```

```python
# show_doc(L.tensored)
```

```python

```

## Chunks

### ```concat(*ls)```
- merge a num of lists, tuples, arrays, tensors, TensorBases into a longer ones
- the output type is the first item's type (different types allowed)
- if the first item is not those types above, then make it a L type

```python
#|export
# @snoop
def concat(*ls):
    "Concatenate tensors, arrays, lists, or tuples"
    ### original source
#     if not len(ls): return []
#     it = ls[0]
#     if isinstance(it,torch.Tensor): res = torch.cat(ls)
#     elif isinstance(it,ndarray): res = np.concatenate(ls)
#     else:
#         res = itertools.chain.from_iterable(map(L,ls))
#         if isinstance(it,(tuple,list)): res = type(it)(res)
#         else: res = L(res)
#     return retain_type(res, it)

    ### fastdebug source
    if not len(ls): 
        return []
    it = ls[0]
    if isinstance(it,torch.Tensor): 
        res = torch.cat(ls)
    elif isinstance(it,ndarray): 
        res = np.concatenate(ls)
    else:
        sp1 = map(L,ls) # make a tuple of lists into a map object
        res = itertools.chain.from_iterable(sp1) # make a map object into a itertool chain object
        if isinstance(it,(tuple,list)): 
            res = type(it)(res) # convert a itertool chain object into a longer/merged list
        else: 
            res = L(res)
    return retain_type(res, it)
```

```python
a,b,c = [1],[1,2],[1,1,2]
test_eq(concat(a,b), c) # concat 2 lists into a longer list
```

```python
test_eq_type(concat(tuple (a),tuple (b)), tuple (c)) # concat 2 tuples into a long tuple
```

```python
test_eq_type(concat(array (a),array (b)), array (c)) # concat 2 arrays into a long array
```

```python
test_eq_type(concat(tensor(a),tensor(b)), tensor(c)) # concat 2 tensors into a long tensor
```

```python
test_eq_type(concat(TensorBase(a),TensorBase(b)), TensorBase(c)) # concat 2 TensorBase into a long TensorBase
```

```python
test_eq_type(concat([1,1],1), [1,1,1]) # concat a list and an int into a long list
```

```python
test_eq_type(concat(1,1,1), L(1,1,1)) # concat 3 int into a long L
```

```python
test_eq_type(concat(L(1,2),1), L(1,2,1)) # concat a L and an int into a long L
```

## ```Chunks(docs)```
- docs is a L list of lists
- merge a list of lists into an iterable and indexable (like a long list)
- merge a list of tensors or TensorBase(of different len) into a long list of single number tensors (iterable and indexable)

### ```Chunks.__init__(self, chunks, lens=None)```

```python
#|export
class Chunks:
    "Slice and int indexing into a list of lists"
#     @snoop
    def __init__(self, chunks, lens=None):
        ### original source
#         self.chunks = chunks
#         self.lens = L(map(len,self.chunks) if lens is None else lens)
#         self.cumlens = np.cumsum(0+self.lens)
#         self.totlen = self.cumlens[-1]
        
        ### fastdebug
        self.chunks = chunks
#         pp.deep(lambda: L(map(len,self.chunks) if lens is None else lens))
        self.lens = L(map(len,self.chunks) if lens is None else lens)
#         pp.deep(lambda: np.cumsum(0+self.lens))
        self.cumlens = np.cumsum(0+self.lens)
        self.totlen = self.cumlens[-1]        
```

```python
### original example
docs = L(list(string.ascii_lowercase[a:b]) for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))
docs
### fastdebug example
# docs = pp.deep(lambda: L(list(string.ascii_lowercase[a:b]) for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26))))
# docs
```

```python
b = Chunks(docs) # merge a list of lists into an iterable (like a long list)
```

```python
test_eq(type(b), Chunks)
```

```python

```

### ```np.searchsorted(a, v, side='left', sorter=None)```

```python
np.searchsorted??
```

```python
np.searchsorted([1,2,3,4,5],-10)
```

```python
np.searchsorted([1,2,3,4,5], [-10, 10, 12, 13])
```

```python

```

### ```Chunks.__getitem__(self, i), getslice(self, i), doc_idx(self, i)```

```python
#|export
class Chunks:
    "Slice and int indexing into a list of lists"
    def __init__(self, chunks, lens=None):
        self.chunks = chunks
        self.lens = L(map(len,self.chunks) if lens is None else lens)
        self.cumlens = np.cumsum(0+self.lens)
        self.totlen = self.cumlens[-1]

#     @snoop(depth=1)
    def __getitem__(self,i):
        ### original source
#         if isinstance(i,slice): return retain_type(self.getslice(i), old=self.chunks[0])
#         di,idx = self.doc_idx(i)
#         return retain_type(self.chunks[di][idx], old=self.chunks[0])
    
        ### fastdebug source
        if isinstance(i,slice): 
            return retain_type(self.getslice(i), old=self.chunks[0])
        di,idx = self.doc_idx(i)
        ret = retain_type(self.chunks[di][idx], old=self.chunks[0])
        return ret

#     @snoop
    def getslice(self, i):
        "get the elements of the selected lists concatenated into a longer list. Not sure about the logic of the source"
        ### original source
#         st_d,st_i = self.doc_idx(ifnone(i.start,0))
#         en_d,en_i = self.doc_idx(ifnone(i.stop,self.totlen+1))
#         res = [self.chunks[st_d][st_i:(en_i if st_d==en_d else sys.maxsize)]]
#         for b in range(st_d+1,en_d): res.append(self.chunks[b])
#         if st_d!=en_d and en_d<len(self.chunks): res.append(self.chunks[en_d][:en_i])
#         return concat(*res)

        ### fastdebug source
        st_d,st_i = self.doc_idx(ifnone(i.start,0))
        en_d,en_i = self.doc_idx(ifnone(i.stop,self.totlen+1))
#         pp.deep(lambda: [self.chunks[st_d][st_i:(en_i if st_d==en_d else sys.maxsize)]])
        res = [self.chunks[st_d][st_i:(en_i if st_d==en_d else sys.maxsize)]]
        for b in range(st_d+1,en_d): 
            res.append(self.chunks[b])
        if st_d!=en_d and en_d<len(self.chunks): 
            res.append(self.chunks[en_d][:en_i])
        return concat(*res)
    
#     @snoop
    def doc_idx(self, i):
        "get index for outer and inner lists. Still not clear about the logic of the source "
        ### original source
#         if i<0: i=self.totlen+i # count from end
#         docidx = np.searchsorted(self.cumlens, i+1)-1
#         cl = self.cumlens[docidx]
#         return docidx,i-cl

        ### fastdebug source
        if i<0: 
            i = self.totlen+i # count from end
#         pp.deep(lambda: np.searchsorted(self.cumlens, i+1)-1)
        docidx = np.searchsorted(self.cumlens, i+1)-1
#         pp.deep(lambda: self.cumlens[docidx])
        cl = self.cumlens[docidx]
        return docidx,i-cl     
```

```python
# depth=1

docs = L(list(string.ascii_lowercase[a:b]) for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))
docs
b = Chunks(docs)
test_eq([b[ o] for o in range(0,1)], ['a'])
```

```python
# depth=2
docs = L(list(string.ascii_lowercase[a:b]) for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))
docs
b = Chunks(docs)
test_eq([b[ o] for o in range(0,1)], ['a'])
```

```python
test_eq([b[ o] for o in range(0,2)], ['a', 'b'])
```

```python
test_eq([b[ o] for o in range(0,5)], ['a','b','c','d','e'])
```

```python
test_eq([b[-o] for o in range(1,6)], ['z','y','x','w','v'])
```

```python
test_eq(b[6:13], 'g,h,i,j,k,l,m'.split(','))
```

```python
test_eq(b[20:77], 'u,v,w,x,y,z'.split(','))
test_eq(b[:5], 'a,b,c,d,e'.split(','))
test_eq(b[:2], 'a,b'.split(','))
```

```python
t = torch.arange(26)
test_eq(t, tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25]))
docs = L(t[a:b] for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))
test_eq(docs, [tensor([0, 1, 2]),tensor([3, 4, 5, 6]),tensor([7]),\
               tensor([ 8,  9, 10, 11, 12, 13, 14, 15]),tensor([16, 17, 18, 19, 20, 21, 22, 23]),tensor([24, 25])])
```

```python
b = Chunks(docs) # merge a list of tensors (of different len) into a long list of single number tensors (iterable and indexable)
test_eq(list(b), list(t))
```

```python
[b[-o] for o in range(1,6)]
```

```python
test_eq([b[ o] for o in range(0,5)], [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4)])
test_eq([b[ o] for o in range(0,5)], range(0,5))
test_eq([b[-o] for o in range(1,6)], [tensor(25), tensor(24), tensor(23), tensor(22), tensor(21)])
test_eq([b[-o] for o in range(1,6)], [25,24,23,22,21])
```

```python
test_eq(b[6:13], torch.arange(6,13))
test_eq(b[20:77], torch.arange(20,26))
test_eq(b[:5], torch.arange(5))
test_eq(b[:2], torch.arange(2))
```

```python
docs = L(TensorBase(t[a:b]) for a,b in ((0,3),(3,7),(7,8),(8,16),(16,24),(24,26)))
docs
```

```python
b = Chunks(docs)
# list(b)
```

```python
test_eq_type(b[:2], TensorBase(range(2)))
test_eq_type(b[:5], TensorBase(range(5)))
test_eq_type(b[9:13], TensorBase(range(9,13)))
```

#### more demos of ```Chunks```

```python
ck = Chunks([[1,3], [2,4]])
test_eq(ck[2], 2)
```

## Simple types

### ```show_title(o, ax=None, ctx=None, label=None, color='black', **kwargs)```
- why do we need `show_title`?
- add title to pd.series

```python
#|export
# @snoop
def show_title(o, ax=None, ctx=None, label=None, color='black', **kwargs):
    "Set title of `ax` to `o`, or print `o` if `ax` is `None`"
    ax = ifnone(ax,ctx)
    if ax is None: print(o)
    elif hasattr(ax, 'set_title'):
        t = ax.title.get_text()
        if len(t) > 0: 
            o = t+'\n'+str(o)
        ax.set_title(o, color=color)
    elif isinstance(ax, pd.Series):
        while label in ax: 
            label += '_'
        ax = pd.concat([ax,pd.Series({label: o})])
    return ax
```

```python
show_title("title")
```

```python
test_stdout(lambda: show_title("title"), "title")
```

```python
pd.Series(dict(a=1))
```

```python
show_title("title", ctx=pd.Series(dict(a=1)), label='a')
```

```python
pd.Series(dict(a=1,a_='title'))
```

```python
# ensure that col names are unique when showing to a pandas series
assert show_title("title", ctx=pd.Series(dict(a=1)), label='a').equals(pd.Series(dict(a=1,a_='title')))
```

### ```ShowTitle as superclass to TitledInt, TitledFloat, TitledStr, TitledTuple```
- why having all these classes?
- to give int, float, str, tuple a title and print them out

```python
#|export
class ShowTitle:
    "Base class that adds a simple `show`"
    _show_args = {'label': 'text'}
    @delegates(show_title, keep=True)    
    def show(self, ctx=None, **kwargs):
        "Show self"
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledInt(Int, ShowTitle):
    _show_args = {'label': 'text'}
    @delegates(show_title, keep=True)    
    def show(self, ctx=None, **kwargs):
        "Show self"
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledFloat(Float, ShowTitle):
    _show_args = {'label': 'text'}
    @delegates(show_title, keep=True)
    def show(self, ctx=None, **kwargs):
        "Show self"
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledStr(Str, ShowTitle):
    _show_args = {'label': 'text'}
    @delegates(show_title, keep=True)    
    def show(self, ctx=None, **kwargs):
        "Show self"
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledTuple(fastuple, ShowTitle):
    _show_args = {'label': 'text'}
    @delegates(show_title, keep=True)    
    def show(self, ctx=None, **kwargs):
        "Show self"
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

add_docs(TitledInt, "An `int` with `show`"); add_docs(TitledStr, "An `str` with `show`");
add_docs(TitledFloat, "A `float` with `show`"); add_docs(TitledTuple, "A `fastuple` with `show`")
```

```python
test_stdout(lambda: TitledStr('s').show(), 's')
test_stdout(lambda: TitledInt(1).show(), '1')
test_stdout(lambda: TitledFloat(1.).show(), '1.0')
```

```python
#|hide
df = pd.DataFrame(index = range(2,5))
df
```

```python
TitledStr('s').show()
TitledStr('s').show("title")
TitledInt(1).show()
TitledFloat(1.).show()
```

```python
row1 = df.iloc[0]
row2 = df.iloc[1]
```

```python
x1 = TitledFloat(2.56)
x2 = TitledInt(3)
```

```python
TitledFloat(1.)
TitledFloat(1.).show()
```

```python
# show_doc(TitledTuple, title_level=3)
```

```python
x1show = x1.show(ctx=row1, label='lbl')
x2show = x2.show(ctx=row2, label='lbl')
```

```python
x1show
x2show
```

```python
x1show, type(x1show), x1show.lbl
```

### ```truncate(self:TitledStr, n)```
- why need ```TitledStr.truncate(n)```?
- to cut the first n words from the string and print them out together as a string

```python
#|export
@patch
def truncate(self:TitledStr, n):
    "Truncate self to `n`"
    words = self.split(' ')[:n]
    return TitledStr(' '.join(words))
```

```python
TitledStr("this is me")
TitledStr("this is me").truncate(1)
```

```python

```

## Other functions

```python
#|export
if not hasattr(pd.DataFrame,'_old_init'): pd.DataFrame._old_init = pd.DataFrame.__init__
```

### ```pd.DataFrame.__init__(self:pd.DataFrame, data=None, index=None, columns=None, dtype=None, copy=None)```
- why do we need a new ```__init__``` for ```pd.DataFrame```? because
- 1. to first turn a tensor to a np.ndarray first using ```to_np```
- 2. and then use original ```pd.DataFrame.__init__``` above to turn array into a DataFrame

```python
#|export
@patch
def __init__(self:pd.DataFrame, data=None, index=None, columns=None, dtype=None, copy=None):
    if data is not None and isinstance(data, Tensor): data = to_np(data)
    self._old_init(data, index=index, columns=columns, dtype=dtype, copy=copy)
```

```python
pd.DataFrame([1,2])
to_np(tensor(1,2))
pd.DataFrame(tensor(1,2))
```

### ```get_empty_df(n)```
- Return `n` empty rows of a dataframe

```python
#|export
def get_empty_df(n):
    "Return `n` empty rows of a dataframe"
    df = pd.DataFrame(index = range(n))
    return [df.iloc[i] for i in range(n)]
```

```python
get_empty_df(3)
```

### ```display_df(df)```
- print out in HTML of a dataframe or just print

```python
#|export
def display_df(df):
    "Display `df` in a notebook or defaults to print"
    try: from IPython.display import display, HTML
    except: return print(df)
    display(HTML(df.to_html()))
```

```python
display_df(pd.DataFrame(tensor(1,2)))
```

### ```get_first(c)```
- Get the first element of c (anything listy), even if c is a dataframe

```python
#|export
def get_first(c):
    "Get the first element of c, even if c is a dataframe"
    return getattr(c, 'iloc', c)[0]
```

```python
get_first([1,2,3])
get_first((1,2,3))
get_first(range(5))
get_first(array([1,2,3]))
get_first(tensor([1,2,3]))
get_first(pd.DataFrame(tensor(1,2)))
```

### ```one_param(m)```
- get the first parameter of a model object

```python
#|export
def one_param(m):
    "First parameter in `m`"
    return first(m.parameters())
```

### ```item_find(x, idx=0)```
- Recursively takes the `idx`-th element of `x`, and take the 1st element of that
- and `x` can be a list of lists or dict of dict or list

```python
#|export
# @snoop
def item_find(x, idx=0):
    "Recursively takes the `idx`-th element of `x`"
    if is_listy(x): return item_find(x[idx])
#     if is_listy(x): return item_find(x[idx], idx) # another way of interpreting recursively take the idx-th element   
    if isinstance(x,dict):
        key = list(x.keys())[idx] if isinstance(idx, int) else idx
        return item_find(x[key])
    return x
```

```python
item_find({'a':[1,2,9], 'b':[3,4,8], 'c':[5,6,7]}, idx=2)
```

```python
item_find([[1,2,9], [3,4,8],[5,6,7]], idx=1)
```

### ```find_device(b)```
Recursively search the device of `b`. the first of the first device

```python
#|export
# @snoop(depth=2)
def find_device(b):
    "Recursively search the device of `b`. the first of the first device"
    return item_find(b).device
```

```python
t1 = to_device(tensor(1))
t2 = to_device(tensor(0))
dev = default_device()
t1
t2
dev
```

```python
find_device(t2)
find_device([t1,t2])
find_device({'a':t1,'b':t2})
```

### ```finds_bs(b)```
- Recursively search the batch size of `b`.
- b can be a list (single array) or a 2d matrix or a tuple of 2 matricies, or a dict of many matricies

```python
#|export
# @snoop
def find_bs(b):
    "Recursively search the batch size of `b`."
    res = item_find(b)
    if not hasattr(res, "shape"): return len(b)
    return res.shape[0]
```

```python
x1 = [1,2,3]
test_eq(find_bs(x1), 3)
```

```python
x2 = array([1,2,3])
test_eq(find_bs(x2), 3)
```

```python
x = torch.randn(4,5)
test_eq(find_bs(x), 4)
```

```python
test_eq(find_bs((x,x)), 4)
```

```python
test_eq(find_bs([x, x]), 4)
test_eq(find_bs({'a':x,'b':x}), 4)
test_eq(find_bs({'a':[[x],[x]],'b':x}), 4)
```

### ```np_func(f)```
- Convert a function taking and returning numpy arrays to one taking and returning tensors

```python
#|export
def np_func(f):
    "Convert a function taking and returning numpy arrays to one taking and returning tensors"
    def _inner(*args, **kwargs):
        nargs = [to_np(arg) if isinstance(arg,Tensor) else arg for arg in args]
        return tensor(f(*nargs, **kwargs))
    functools.update_wrapper(_inner, f)
    return _inner
```

This decorator is particularly useful for using numpy functions as fastai metrics, for instance:

```python
from sklearn.metrics import f1_score
```

```python
@np_func
def f1(inp,targ): return f1_score(targ, inp)

a1,a2 = array([0,1,1]),array([1,0,1])
t = f1(tensor(a1),tensor(a2))
test_eq(f1_score(a1,a2), t)
assert isinstance(t,Tensor)
```

### ```Module(nn.Module, metaclass=PrePostInitMeta)```
- Same as `nn.Module`, but no need for subclasses to call `super().__init__`

```python
#|export
class Module(nn.Module, metaclass=PrePostInitMeta):
    "Same as `nn.Module`, but no need for subclasses to call `super().__init__`"
    def __pre_init__(self, *args, **kwargs): super().__init__()
    def __init__(self): pass
```

```python
# show_doc(Module, title_level=3)
```

```python
class _T(Module):
    def __init__(self): self.f = nn.Linear(1,1)
    def forward(self,x): return self.f(x)
```

```python
t = _T()
t(tensor([1.]))
```

### ```get_model(model)```
Return the model even when the model is wrapped inside `model` through `model.module` and `model` is an instance of ```DistributedDataParallel, nn.DataParallel```

```python
#|export
from torch.nn.parallel import DistributedDataParallel
```

```python
#|export
def get_model(model):
    "Return the model maybe wrapped inside `model`."
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model
```

### ```one_hot(x, c)```
- why need one_hot(x, c)? 
- one_hot(3, 5) can give us the 3rd class out of 5 classes in one hot encoding

```python
#|export
def one_hot(x, c):
    "One-hot encode `x` with `c` classes."
    res = torch.zeros(c, dtype=torch.uint8)
    if isinstance(x, Tensor) and x.numel()>0: 
        res[x] = 1.
    else: res[list(L(x, use_list=None))] = 1.
    return res
```

```python
# help(L)
# L([1,2], use_list=False)
# L((1,2), use_list=True)
```

```python
tensor(3).numel()
tensor(2,3).numel() # number of elements in the tensor?
tensor([[2,3],[3,4]]).numel()
```

```python
one_hot([1,2], 5)
one_hot([1,4,3,0,0,0], 6)
one_hot([0], 5)
one_hot(tensor(2,3,4), 5)
```

```python
tensor(0,1,0,0,1).byte()
```

```python
test_eq(one_hot([1,4], 5), tensor(0,1,0,0,1).byte())
test_eq(one_hot(torch.tensor([]), 5), tensor(0,0,0,0,0).byte())
test_eq(one_hot(2, 5), tensor(0,0,1,0,0).byte())
```

### ```one_hot_decode(x, vocab=None)```
- reverse from one_hot to classes

```python
#|export
def one_hot_decode(x, vocab=None):
    return L(vocab[i] if vocab else i for i,x_ in enumerate(x) if x_==1)
```

```python
test_eq(one_hot_decode(tensor(0,1,0,0,1)), [1,4])
test_eq(one_hot_decode(tensor(0,0,0,0,0)), [   ])
test_eq(one_hot_decode(tensor(0,0,1,0,0)), [2  ])
```

```python
one_hot_decode(tensor(0,1,0,0,1), vocab=['a', 'b', 'c', 'd', 'e'])
```

### ```params(m)```
- return a list of parameters in a model `m`

```python
#|export
def params(m):
    "Return all parameters of `m`"
    return [p for p in m.parameters()]
```

### ```trainable_params(m)```
- Return all trainable parameters of `m` into a list, meaning their `requires_grad` is True

```python
#|export
def trainable_params(m):
    "Return all trainable parameters of `m`"
    return [p for p in m.parameters() if p.requires_grad]
```

```python
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
```

```python
m = nn.Linear(4,5)
test_eq(trainable_params(m), [m.weight, m.bias])
m.weight.requires_grad_(False)
test_eq(trainable_params(m), [m.bias])
```

### ```norm_types``` 
- have all normalization functions stored inside a tuple

```python
#|export
norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm)
```

### ```norm_bias_params(m, with_bias=True)```
- Return all bias and BatchNorm parameters, not other weights parameters

```python
#|export
def norm_bias_params(m, with_bias=True):
    "Return all bias and BatchNorm parameters"
    if isinstance(m, norm_types): return L(m.parameters())
    res = L(m.children()).map(norm_bias_params, with_bias=with_bias).concat()
    if with_bias and getattr(m, 'bias', None) is not None: res.append(m.bias)
    return res
```

```python
m = nn.Linear(10, 20)
m.weight.shape
m.bias.shape
# m.weight.grad?
```

```python
model = nn.Sequential(nn.Linear(10,20), nn.BatchNorm1d(20), nn.Conv1d(3,4, 3))
model
# model[0].weight.shape
model[0].bias.shape
model[1].weight.shape
model[1].bias.shape
# model[2].weight.shape
model[2].bias.shape
```

```python
[i.shape for i in norm_bias_params(model)]
```

```python
for norm_func in [nn.BatchNorm1d, partial(nn.InstanceNorm1d, affine=True)]:
    model = nn.Sequential(nn.Linear(10,20), norm_func(20), nn.Conv1d(3,4, 3))
    test_eq(norm_bias_params(model), [model[0].bias, model[1].weight, model[1].bias, model[2].bias])
    model = nn.ModuleList([nn.Linear(10,20, bias=False), nn.Sequential(norm_func(20), nn.Conv1d(3,4,3))])
    test_eq(norm_bias_params(model), [model[1][0].weight, model[1][0].bias, model[1][1].bias])
    model = nn.ModuleList([nn.Linear(10,20), nn.Sequential(norm_func(20), nn.Conv1d(3,4,3))])
    test_eq(norm_bias_params(model, with_bias=False), [model[1][0].weight, model[1][0].bias])
```

### ```batch_to_samples(b, max_n=10)```
- 'Transposes' a batch to (at most `max_n`) samples
- it's like to slice a batch of data by n cols for a smaller sample data, and transpose it

```python
#|export
# @snoop
def batch_to_samples(b, max_n=10):
    "'Transposes' a batch to (at most `max_n`) samples"
    if isinstance(b, Tensor): 
        return retain_types(list(b[:max_n]), [b])
    else:
#         pp.deep(lambda: L(b).map(partial(batch_to_samples,max_n=max_n)))
        res = L(b).map(partial(batch_to_samples,max_n=max_n))
        return retain_types(res.zip(), [b])
```

```python
batch_to_samples(tensor(1,2,3), 2)
```

```python
t = tensor([1,2,3])
[t,t+1]
batch_to_samples([t,t+1], max_n=2)
```

```python
batch_to_samples([tensor([1,2,3]), tensor([4,5,6])], 10)
```

```python
test_eq(batch_to_samples([t,t+1], max_n=2), ([1,2],[2,3]))
test_eq(batch_to_samples(tensor([1,2,3]), 10), [1, 2, 3])
test_eq(batch_to_samples([tensor([1,2,3]), tensor([4,5,6])], 10), [(1, 4), (2, 5), (3, 6)])
test_eq(batch_to_samples([tensor([1,2,3]), tensor([4,5,6])], 2), [(1, 4), (2, 5)])
test_eq(batch_to_samples([tensor([1,2,3]), [tensor([4,5,6]),tensor([7,8,9])]], 10), 
        [(1, (4, 7)), (2, (5, 8)), (3, (6, 9))])
test_eq(batch_to_samples([tensor([1,2,3]), [tensor([4,5,6]),tensor([7,8,9])]], 2), [(1, (4, 7)), (2, (5, 8))])
```

```python
# fastnbs("fastuple")
```

```python
t = fastuple(tensor([1,2,3]),TensorBase([2,3,4]))
t
```

```python
test_eq_type(batch_to_samples(t)[0][1], TensorBase(2))
test_eq(batch_to_samples(t).map(type), [fastuple]*3)
```

### ```interp_1d(x:Tensor, xp, fp)```
- Same as `np.interp`, run `np.interp??`
- a quick [video](https://youtu.be/nGwg5MrbZxo?t=82) about what 1d linear interpolation mean and for 

```python

```

```python
#|export
@patch
def interp_1d(x:Tensor, xp, fp):
    "Same as `np.interp`"
    slopes = (fp[1:]-fp[:-1])/(xp[1:]-xp[:-1])
    incx = fp[:-1] - (slopes*xp[:-1])
    locs = (x[:,None]>=xp[None,:]).long().sum(1)-1
    locs = locs.clamp(0,len(slopes)-1)
    return slopes[locs]*x + incx[locs]
```

```python

```

```python
# %%snoop
brks = tensor(0,1,2,4,8,64).float()
ys = tensor(range_of(brks)).float()
ys /= ys[-1].item()
pts = tensor(0.2,0.5,0.8,3,5,63)
```

```python
# %%snoop
preds = pts.interp_1d(brks, ys)
preds
test_close(preds.numpy(), np.interp(pts.numpy(), brks.numpy(), ys.numpy()))
```

```python
plt.scatter(brks,ys)
plt.scatter(pts,preds)
plt.legend(['breaks','preds']);
```

### ```pca(x:Tensor, k=2)```

```python
#|export
@patch
def pca(x:Tensor, k=2):
    "Compute PCA of `x` with `k` dimensions."
    x = x-torch.mean(x,0)
    U,S,V = torch.svd(x.t())
    return torch.mm(x,U[:,:k])
```

### ```logit(x)```
- Logit of `x`, clamped to avoid inf.

```python
#|export
def logit(x):
    "Logit of `x`, clamped to avoid inf."
    x = x.clamp(1e-7, 1-1e-7)
    return -(1/x-1).log()
```

```python
logit(tensor(1e-18))
logit(tensor(1-1e-8))
```

### ```num_distrib()```
- Return the number of processes in distributed training (if applicable).

```python
#|export
def num_distrib():
    "Return the number of processes in distributed training (if applicable)."
    return int(os.environ.get('WORLD_SIZE', 0))
```

```python
num_distrib()
```

### ```rank_distrib()```
Return the distributed rank of this process (if applicable).

```python
#|export
def rank_distrib():
    "Return the distributed rank of this process (if applicable)."
    return int(os.environ.get('RANK', 0))
```

```python
rank_distrib()
```

### ```distrib_barrier()```
Place a synchronization barrier in distributed training

```python
#|export
def distrib_barrier():
    "Place a synchronization barrier in distributed training"
    if num_distrib() > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()
```

After calling this, ALL sub-processes in the pytorch process group must arrive here before proceeding.

```python
#|export
# Saving arrays requires pytables - optional dependency
try: import tables
except: pass
```

```python
#|export
def _comp_filter(lib='lz4',lvl=3): return tables.Filters(complib=f'blosc:{lib}', complevel=lvl)
```

### ```save_array(p:Path, o, complib='lz4', lvl=3)```
- turn a tensor into an array
- save/write the array in a pytable file under the Path

```python
#|export
@patch
def save_array(p:Path, o, complib='lz4', lvl=3):
    "Save numpy array to a compressed `pytables` file, using compression level `lvl`"
    if isinstance(o,Tensor): o = to_np(o)
    with tables.open_file(p, mode='w', filters=_comp_filter(lib=complib,lvl=lvl)) as f: f.create_carray('/', 'data', obj=o)
```

Compression lib can be any of: blosclz, lz4, lz4hc, snappy, zlib or zstd.

### ```load_array(p:Path)```
- add a method ```load_array``` to Path
- to open and read from a pytable file 

```python
#|export
@patch
def load_array(p:Path):
    "Save numpy array to a `pytables` file"
    with tables.open_file(p, 'r') as f: return f.root.data.read()
```

### ```base_doc(elt), doc(elt)```
- how to use ```doc``` and what goes on behind the scene

```python
#|export
def base_doc(elt):
    "Print a base documentation of `elt`"
    name = getattr(elt, '__qualname__', getattr(elt, '__name__', ''))
    print(f'{name}{inspect.signature(elt)}\n{inspect.getdoc(elt)}\n')
    print('To get a prettier result with hyperlinks to source code and documentation, install nbdev: pip install nbdev')
```

```python
#|export
def doc(elt):
    "Try to use doc form nbdev and fall back to `base_doc`"
    try:
        from nbdev.showdoc import doc
        doc(elt)
    except: base_doc(elt)
```

### ```nested_reorder(t, idxs)```
- Reorder all tensors in `t` using `idxs`
- all the tensors reordered will be put back to list or tuple where they were from

```python
#|export
def nested_reorder(t, idxs):
    "Reorder all tensors in `t` using `idxs`"
    if isinstance(t, (Tensor,L)): return t[idxs]
    elif is_listy(t): return type(t)(nested_reorder(t_, idxs) for t_ in t)
    if t is None: return t
    raise TypeError(f"Expected tensor, tuple, list or L but got {type(t)}")
```

```python
x = tensor([0,1,2,3,4,5])
idxs = tensor([2,5,1,0,3,4])
nested_reorder(x, idxs)
```

```python
nested_reorder([[x], x], idxs)
test_eq_type(nested_reorder(([x], x), idxs), ([idxs], idxs))
```

```python
tensor(1).item()
```

```python
y = L(0,1,2,3,4,5)
# pp.deep(lambda: L(i.item() for i in idxs))
z = L(i.item() for i in idxs)
nested_reorder((y, x), idxs)
test_eq_type(nested_reorder((y, x), idxs), (z,idxs))
```

### ```flatten_check(inp, targ)```
- Check that `inp` and `targ` have the same number of elements and flatten them into a single row or 1d data

```python
#|export
def flatten_check(inp, targ):
    "Check that `inp` and `targ` have the same number of elements and flatten them."
    inp,targ = TensorBase(inp.contiguous()).view(-1),TensorBase(targ.contiguous()).view(-1)
    test_eq(len(inp), len(targ))
    return inp,targ
```

```python
x1,x2 = torch.randn(5,4),torch.randn(20)
x1,x2 = flatten_check(x1,x2)
test_eq(x1.shape, [20])
test_eq(x2.shape, [20])
```

```python
x1,x2 = torch.randn(5,4),torch.randn(21)
test_fail(lambda: flatten_check(x1,x2))
```

## Image helpers

### ```make_cross_image(bw=True)```
Create a tensor containing a cross image, either `bw` (True) or color

```python
#|export
def make_cross_image(bw=True):
    "Create a tensor containing a cross image, either `bw` (True) or color"
    if bw:
        im = torch.zeros(5,5)
        im[2,:] = 1.
        im[:,2] = 1.
    else:
        im = torch.zeros(3,5,5)
        im[0,2,:] = 1.
        im[1,:,2] = 1.
    return im
```

```python
plt.imshow(make_cross_image()) # imshow's cmap default is with color
```

```python
plt.imshow(make_cross_image(), cmap="Greys");
```

```python
plt.imshow(make_cross_image(False).permute(1,2,0));
```

### ```show_image_batch(b, show=show_titled_image, items=9, cols=3, figsize=None, **kwargs)```
- Display batch `b` in a grid of size `items` with `cols` width
- images with titles

```python
#|export
# @snoop
def show_image_batch(b, show=show_titled_image, items=9, cols=3, figsize=None, **kwargs):
    "Display batch `b` in a grid of size `items` with `cols` width"
    if items<cols: cols=items
    rows = (items+cols-1) // cols
    if figsize is None: figsize = (cols*3, rows*3)
    fig,axs = plt.subplots(rows, cols, figsize=figsize)
    for *o,ax in zip(*to_cpu(b), axs.flatten()): 
        show(o, ax=ax, **kwargs)
```

```python
# fastnbs("show_titled_image")
```

```python
to_cpu(([Image.open(TEST_IMAGE_BW),Image.open(TEST_IMAGE)],['bw','color']))
```

```python
show_image_batch(([Image.open(TEST_IMAGE_BW),Image.open(TEST_IMAGE)],['bw','color']), items=2)
```

## Model init

### ```requires_grad(m)```

- Check if the first parameter of `m` requires grad or not


```python
#|export
def requires_grad(m):
    "Check if the first parameter of `m` requires grad or not"
    ps = list(m.parameters())
    return ps[0].requires_grad if len(ps)>0 else False
```

```python
tst = nn.Linear(4,5)
requires_grad(tst)
assert requires_grad(tst)
```

```python
# fastnbs("requires_grad_")
```

```python
for p in tst.parameters(): p.requires_grad_(False)
assert not requires_grad(tst)
```

### ```init_default(m, func=nn.init.kaiming_normal_)```
- Initialize `m` weights with `func` and set `bias` to 0.

```python
#|export
def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m
```

```python
tst = nn.Linear(4,5)
tst.weight.data
tst.weight.data.uniform_(-1,1)
```

```python
tst.bias.data
tst.bias.data.uniform_(-1,1)
```

```python
tst = init_default(tst, func = lambda x: x.data.fill_(1.))
tst.weight.data
tst.bias.data
```

```python
test_eq(tst.weight, torch.ones(5,4))
test_eq(tst.bias, torch.zeros(5))
```

### ```cond_init(m, func)```
- Apply `init_default` to `m` as long as it is not a batchnorm module and first parameters require grads"

```python
#|export
def cond_init(m, func):
    "Apply `init_default` to `m` unless it's a batchnorm module"
    if (not isinstance(m, norm_types)) and requires_grad(m): init_default(m, func)
```

```python
norm_types
# requires_grad?
```

```python
tst = nn.Linear(4,5)
tst.weight.data.uniform_(-1,1)
tst.bias.data.uniform_(-1,1)
cond_init(tst, func = lambda x: x.data.fill_(1.))
test_eq(tst.weight, torch.ones(5,4))
test_eq(tst.bias, torch.zeros(5))
```

```python
tst = nn.BatchNorm2d(5)
init = [tst.weight.clone(), tst.bias.clone()]
cond_init(tst, func = lambda x: x.data.fill_(1.))
test_eq(tst.weight, init[0])
test_eq(tst.bias, init[1])
```

### ```apply_leaf(m, f)```
- Apply `f` to children of `m` (every layers of a model)

```python
#|export
def apply_leaf(m, f):
    "Apply `f` to children of `m`."
    c = m.children()
    if isinstance(m, nn.Module): f(m)
    for l in c: apply_leaf(l,f)
```

```python
tst = nn.Sequential(nn.Linear(4,5), nn.Sequential(nn.Linear(4,5), nn.Linear(4,5)))
tst[0].weight.shape
```

```python
list(tst.children())
```

```python
apply_leaf(tst, partial(init_default, func=lambda x: x.data.fill_(1.)))
for l in [tst[0], *tst[1]]: 
    print(f"l.weight.shape: {l.weight.shape}")
    test_eq(l.weight, torch.ones(5,4))
for l in [tst[0], *tst[1]]: test_eq(l.bias,   torch.zeros(5))
```

### ```apply_init(m, func=nn.init.kaiming_normal_)```
- Initialize all non-batchnorm layers of `m` with `func`.

```python
#|export
def apply_init(m, func=nn.init.kaiming_normal_):
    "Initialize all non-batchnorm layers of `m` with `func`."
    apply_leaf(m, partial(cond_init, func=func))
```

```python
tst = nn.Sequential(nn.Linear(4,5), nn.Sequential(nn.Linear(4,5), nn.BatchNorm1d(5)))
init = [tst[1][1].weight.clone(), tst[1][1].bias.clone()]
apply_init(tst, func=lambda x: x.data.fill_(1.))
for l in [tst[0], tst[1][0]]: test_eq(l.weight, torch.ones(5,4))
for l in [tst[0], tst[1][0]]: test_eq(l.bias,   torch.zeros(5))
test_eq(tst[1][1].weight, init[0])
test_eq(tst[1][1].bias,   init[1])
```

## autograd jit functions

### ```script_use_ctx(f), script_save_ctx(static, *argidx), script_fwd(*argidx), script_bwd(f), grad_module(cls)```
- funcs to automate autograd 

```python
#|export
def script_use_ctx(f):
    "Decorator: create jit script and pass everything in `ctx.saved_variables to `f`, after `*args`"
    sf = torch.jit.script(f)
    def _f(ctx, *args, **kwargs): return sf(*args, *ctx.saved_variables, **kwargs)
    return update_wrapper(_f,f)
```

```python
#|export
def script_save_ctx(static, *argidx):
    "Decorator: create jit script and save args with indices `argidx` using `ctx.save_for_backward`"
    def _dec(f):
        sf = torch.jit.script(f)
        def _f(ctx, *args, **kwargs):
            if argidx:
                save = [args[o] for o in argidx]
                ctx.save_for_backward(*save)
            if not argidx: args = [ctx]+args
            return sf(*args, **kwargs)
        if static: _f = staticmethod(_f)
        return update_wrapper(_f,f)
    return _dec
```

```python
#|export
def script_fwd(*argidx):
    "Decorator: create static jit script and save args with indices `argidx` using `ctx.save_for_backward`"
    return script_save_ctx(True, *argidx)
```

```python
#|export
def script_bwd(f):
    "Decorator: create static jit script and pass everything in `ctx.saved_variables to `f`, after `*args`"
    return staticmethod(script_use_ctx(f))
```

```python
#|export
def grad_module(cls):
    "Decorator: convert `cls` into an autograd function"
    class _c(nn.Module):
        def forward(self, *args, **kwargs): return cls.apply(*args, **kwargs)
    return _c
```

## Torch Version Checks -

### ```ismin_torch(min_version), notmax_torch(max_version)```
- check whether the torch version greater than the minimum version; check whether is less than the maximum version

```python
#|export
def ismin_torch(min_version):
    "Check if `torch.__version__` >= `min_version` using packaging.version"
    return parse(torch.__version__) >= parse(min_version)
```

```python
#|export
def notmax_torch(max_version):
    "Check if `torch.__version__` < `max_version` using packaging.version"
    return parse(torch.__version__) < parse(max_version)
```

# Export -

```python
#|hide
import nbdev; nbdev.nbdev_export()
```

```python

```
