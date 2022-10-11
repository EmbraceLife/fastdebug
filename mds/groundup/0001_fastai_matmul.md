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

# 0001_get_data

```python
#| default_exp access_data
```

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

```python
from fastdebug.utils import *
```

# Matrix multiplication from scratch


The *foundations* we'll assume throughout this course are:

- Python
- Python modules (non-DL)
- pytorch indexable tensor, and tensor creation (including RNGs - random number generators)
- fastai.datasets


## autoreload and matplotlib inline

```python
from fastdebug.utils import *
```

```python
# autoreload??
```

```python
fastnbs("autoreload plus matplotlib")
```

## operator module and test_eq


### import operator

```python
import operator
from fastdebug.test import *
```

### operatore: About

```python
import inspect
print(inspect.getdoc(operator))
```

```python
whatinside(operator)
```

## test and test_eq

```python
TEST
```

```python
#| export
def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"
```

```python
#| export
def test_eq(a,b): test(a,b,operator.eq,'==')
```

```python
test_eq(TEST,'test')
```

## Get data


### fastai.data.external/transforms

```python
import fastai.data.external as fde
import fastai.data.transforms as fdt
```

```python
whatinside(fde, dun=True)
```

### untar_data and URLs
Download and extract dataset 

```python
from fastai.data.external import untar_data,URLs
from fastai.data.transforms import get_image_files
```

```python
URLs.MNIST_VAR_SIZE_TINY
```

```python
inspect.getdoc(get_image_files)
```

```python
path = untar_data(URLs.MNIST_VAR_SIZE_TINY)
```

```python
path
```

```python
%cd /Users/Natsume/.fastai/data/mnist_tiny
%ls
```

```python
files_train = get_image_files(path/"train")
files_valid = get_image_files(path/"valid")
files_test = get_image_files(path/"test")
len(files_train), len(files_valid), len(files_test)
```

### get_labels
extract label y

```python
import pandas as pd
```

```python
df = pd.read_csv(path/"labels.csv")
```

```python
df[:5]
files_train[:5]
```

```python
str(files_train[0])
```

```python
df.index
```

```python
label_train = []
for f in files_train:
    for i in df.index:
        if df["name"][i] in str(f):
            label_train.append(df["label"][i])
```

```python
len(label_train)
label_train[:5]
```

```python
import random
```

```python
rand = [random.randint(0,len(files_train)) for i in range(5)]
rand
files_train[rand]
type(files_train)
```

```python
files_train[1,3,4]
```

```python
#| export
def get_labels(img_files):
    df = pd.read_csv(path/"labels.csv")
    labels = []
    for f in img_files:
        for i in df.index:
            if df["name"][i] in str(f):
                labels.append(df["label"][i])
    import random
    from fastcore.foundation import L
    rand = [random.randint(0,len(labels)) for i in range(5)]
    print(f"len: {len(labels)}, random view: {L(labels)[rand]}")
    print(img_files[rand])
    return labels
```

```python
y_train = get_labels(files_train)
y_valid = get_labels(files_valid)
```

```python
len(y_train)
```

### idx_line and check
I have used these funcs more than once, so turned into funcs

```python
#| export utils
def idx_line(lst):
    "return zip(range(len(lst)), lst)"
    return zip(range(len(lst)), lst)
```

```python
#| export utils
def check(f, # function name, like PIL.Image.open
        n=10 # num of lines to print, if n = -1 then print the entire __doc__
       ): 
    "check any object on its signature, class, __repr__, docs, __dict__, and other checks borrowed from utils"
    if callable(f) and not inspect.isbuiltin(f) and not inspect.ismethoddescriptor(f) or hasattr(f, '__signature__'):
        print(f"signature: {inspect.signature(f)}")
    else: print("signature: None")
        
    print(f"__class__: {getattr(f, '__class__', None)}")
    print(f'__repr__: {f}\n')
    
    if bool(getattr(f, '__doc__', None)): # make sure not None
        doclst = inspect.getdoc(f).split("\n")
        print(f'__doc__:')
        for idx, l in idx_line(doclst):
            print(l)
            if n > 0 and idx >= n: break
    else: print("__doc__: not exist\n")

    from pprint import pprint
    if hasattr(f, '__dict__') and f.__dict__ != None:
        print(f"__dict__: ")
        pprint(f.__dict__)
    elif hasattr(f, '__dict__') and f.__dict__ == None: print(f"__dict__: None")
    else: print(f'__dict__: not exist \n')
        
    from fastdebug.utils import ismetaclass, isdecorator
    print(f"metaclass: {ismetaclass(f)}")
    print(f"class: {inspect.isclass(f)}")
    print(f"decorator: {isdecorator(f)}")
    print(f"function: {inspect.isfunction(f)}")
    print(f"method: {inspect.ismethod(f)}")
```

### PIL and PIL.Image.open
to view an image from the image file

```python
import PIL
```

```python
whatinside(PIL)
```

```python
check(PIL.Image.open, n=100)
```

```python
PIL.Image.open(files_train[0])
PIL.Image.open(files_valid[-1])
```

### img, img.convert, img.resize
convert img color mode to L or RGB, and resize img to shape (28, 28)

```python
import torch
import numpy as np
```

```python
img = PIL.Image.open(files_train[0])
```

```python
print(inspect.signature(img.convert)) # add this line to the doc func defined above
```

```python
img.__class__
type(img) # add this to doc func above
```

```python
# how to quickly know about an object, make doc to solve this problem
type(img)
```

```python
img.__doc__ == None # add this to func doc
```

```python
img.__dict__ # add this into doc
```

```python
check(img)
```

```python
print(img)
```

```python
print(img.convert)
```

```python
check(img.convert)
```

```python
img.convert("RGB")
img.size
img.convert("L")
img.size
```

```python
check(img.resize)
```

```python
img = img.resize((28, 28))
img.size
# img = img.resize((1, 28, 28)) # this is not allowed
```

### torch.Tensor, torch.stack, imgs2tensor
convert from img type to np.array to pytorch tensor

```python
img
```

```python
print(img)
```

```python
np.array(img).shape
```

```python
np.array(img)[10:15, 10:15]
```

```python
from torch import tensor
```

```python
tensor(np.array(img)).shape
```

```python
t = torch.Tensor(np.array(img))
```

```python
files_train[1]
img1 = PIL.Image.open(files_train[1])
img1 = img1.resize((28,28))
t1 = torch.Tensor(np.array(img1))
# t1
torch.stack([t, t1], dim=0).shape
torch.stack([t, t1], dim=1).shape
torch.stack([t, t1], dim=-1).shape
```

```python
lst_t = []
for f in files_train[:5]:
    img = PIL.Image.open(f).resize((28,28))
    t = torch.Tensor(np.array(img))
    lst_t.append(t)
torch.stack(lst_t, dim=0).shape
```

```python

def imgs2tensor(img_folder, n=-1, size=28):
    "convert image folders into a tensor in which images stack on each other"
    lst_t = []
    if n > 0: selected = img_folder[:n]
    else: selected = img_folder
    for f in selected:
        img = PIL.Image.open(f).resize((size,size))
        t = torch.Tensor(np.array(img))
        lst_t.append(t)
    res = torch.stack(lst_t, dim=0)
    print(res.shape)
    return res
```

```python
x_train = imgs2tensor(files_train)
```

### torch.permute, torch.float

```python
inspect.isbuiltin(x_train.permute)
```

```python
x_train.permute.__doc__
```

```python
check(x_train.permute)
```

```python
check(torch.permute,n=-1)
```

```python
check(x_train.float)
```

```python
check(torch.Tensor.float, n=-1)
```

```python
check(inspect.ismethoddescriptor)
```

```python
inspect.ismethoddescriptor(torch.Tensor.float) # to improve on func check
```

```python
check(x_train.is_floating_point)
```

### mean_std, normalize, imgs2tensor

```python
x_train.max(), x_train.min(), x_train.median(), x_train.mean()

x_train.mean(), x_train.std()
```

```python
#| export
def mean_std(t):
    "check mean and std of a tensor"
    print(f'mean: {t.mean()}, std: {t.std()}')
```

```python
x_train = x_train/x_train.max()
```

```python
#| export 
def normalize(t):
    "to normalize a tensor by dividing its maximum value"
    return t/t.max()
```

```python
x_train = normalize(x_train)
```

```python
mean_std(x_train)
```

```python
#| export
def imgs2tensor(img_folder:list, # a list of image files path in string
                n=-1, # n == -1 to process all files in the list, otherwise just [:n] files
                size=28 # images to be resized to (size, size)
               ): 
    "convert image folders into a tensor in which images stack on each other, and normalize it"
    lst_t = []
    if n > 0: selected = img_folder[:n]
    else: selected = img_folder
    for f in selected:
        img = PIL.Image.open(f).resize((size,size))
        t = torch.Tensor(np.array(img))
        lst_t.append(t)
    res = torch.stack(lst_t, dim=0)
    res = normalize(res)
    print(res.shape)
    mean_std(res)
    return res
```

```python
x_train = imgs2tensor(files_train)
x_valid = imgs2tensor(files_valid)
x_test = imgs2tensor(files_test)
```

```python

```

### %whos 
https://www.wrighters.io/how-to-view-all-your-variables-in-a-jupyter-notebook/

```python
[d for d in dir() if not "__" in d and not "_" in d]
```

```python
%whos L list
```

## export

```python

```

```python
# |hide
import nbdev
```

```python
nbdev.nbdev_export()
```

```python

```
