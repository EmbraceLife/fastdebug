# 0001_get_data


```
#| default_exp access_data
```
---
skip_exec: true
---

```
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>


# Matrix multiplication from scratch

The *foundations* we'll assume throughout this course are:

- Python
- Python modules (non-DL)
- pytorch indexable tensor, and tensor creation (including RNGs - random number generators)
- fastai.datasets

## autoreload and matplotlib inline


```
from fastdebug.utils import *
```


```
# autoreload??
```


```
fastnbs("autoreload plus matplotlib")
```


## <mark style="background-color: #ffff00">autoreload</mark>  <mark style="background-color: #ffff00">plus</mark>  <mark style="background-color: #FFFF00">matplotlib</mark>  inline for every notebook




This section contains only the current heading 2 and its subheadings
<!-- #region -->
As mentioned above, you need the autoreload extension. If you want it to automatically start every time you launch ipython, you need to add it to the ipython_config.py startup file:

It may be necessary to generate one first:
```python
ipython profile create
```
Then include these lines in ~/.ipython/profile_default/ipython_config.py:

```python
c.InteractiveShellApp.exec_lines = []
c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
c.InteractiveShellApp.exec_lines.append('%autoreload 2')
c.InteractiveShellApp.exec_lines.append('%matplotlib inline')
```

As well as an optional warning in case you need to take advantage of compiled Python code in .pyc files:
```python
c.InteractiveShellApp.exec_lines.append('print("Warning: disable autoreload in ipython_config.py to improve performance.")')
```
<!-- #endregion -->

### If individual notebook, I can just run the function below to setup autoreload

```python
#| export
def autoreload():
    from IPython.core.interactiveshell import InteractiveShell
    get_ipython().run_line_magic(magic_name="load_ext", line = "autoreload")
    get_ipython().run_line_magic(magic_name="autoreload", line = "2")
    get_ipython().run_line_magic(magic_name="matplotlib", line = "inline")
```

```python

```

start of another heading 2
## Expand cells



[Open `utils` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/utils.ipynb)


## operator module and test_eq

### import operator


```
import operator
from fastdebug.test import *
```

### operatore: About


```
import inspect
print(inspect.getdoc(operator))
```

    Operator interface.
    
    This module exports a set of functions implemented in C corresponding
    to the intrinsic operators of Python.  For example, operator.add(x, y)
    is equivalent to the expression x+y.  The function names are those
    used for special methods; variants without leading and trailing
    '__' are also provided for convenience.



```
whatinside(operator)
```

    operator has: 
    54 items in its __all__, and 
    0 user defined functions, 
    3 classes or class objects, 
    97 builtin funcs and methods, and
    100 callables.
    
    Operator interface.
    
    This module exports a set of functions implemented in C corresponding
    to the intrinsic operators of Python.  For example, operator.add(x, y)
    is equivalent to the expression x+y.  The function names are those
    used for special methods; variants without leading and trailing
    '__' are also provided for convenience.


## test and test_eq


```
TEST
```




    'test'




```
#| export
def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"
```


```
#| export
def test_eq(a,b): test(a,b,operator.eq,'==')
```


```
test_eq(TEST,'test')
```

## Get data

### fastai.data.external/transforms


```
import fastai.data.external as fde
import fastai.data.transforms as fdt
```


```
whatinside(fde, dun=True)
```

    fastai.data.external has: 
    4 items in its __all__, and 
    337 user defined functions, 
    180 classes or class objects, 
    4 builtin funcs and methods, and
    541 callables.
    
    None
    fastai_cfg:_lru_cache_wrapper
    fastai_path:     function    Local path to `folder` in `Config`
    URLs:            class, type    Global constants for dataset and model URLs.
    untar_data:      function    Download `url` using `FastDownload.get`


### untar_data and URLs
Download and extract dataset 


```
from fastai.data.external import untar_data,URLs
from fastai.data.transforms import get_image_files
```


```
URLs.MNIST_VAR_SIZE_TINY
```




    'https://s3.amazonaws.com/fast-ai-imageclas/mnist_var_size_tiny.tgz'




```
inspect.getdoc(get_image_files)
```




    'Get image files in `path` recursively, only in `folders`, if specified.'




```
path = untar_data(URLs.MNIST_VAR_SIZE_TINY)
```


```
path
```




    Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny')




```
%cd /Users/Natsume/.fastai/data/mnist_tiny
%ls
```

    /Users/Natsume/.fastai/data/mnist_tiny
    labels.csv  [34mmnist_tiny[m[m/ [34mmodels[m[m/     [34mtest[m[m/       [34mtrain[m[m/      [34mvalid[m[m/



```
files_train = get_image_files(path/"train")
files_valid = get_image_files(path/"valid")
files_test = get_image_files(path/"test")
len(files_train), len(files_valid), len(files_test)
```




    (709, 699, 20)



### get_labels
extract label y


```
import pandas as pd
```


```
df = pd.read_csv(path/"labels.csv")
```


```
df[:5]
files_train[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/3/7463.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train/3/9829.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/3/7881.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/3/8065.png</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/3/7046.png</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>






    (#5) [Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9519.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/7534.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9082.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/8377.png')]




```
str(files_train[0])
```




    '/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png'




```
df.index
```




    RangeIndex(start=0, stop=1408, step=1)




```
label_train = []
for f in files_train:
    for i in df.index:
        if df["name"][i] in str(f):
            label_train.append(df["label"][i])
```


```
len(label_train)
label_train[:5]
```




    709






    [7, 7, 7, 7, 7]




```
import random
```


```
rand = [random.randint(0,len(files_train)) for i in range(5)]
rand
files_train[rand]
type(files_train)
```




    [505, 35, 83, 367, 213]






    (#5) [Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/3/8882.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/954.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/7527.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/3/7483.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/7616.png')]






    fastcore.foundation.L




```
files_train[1,3,4]
```




    (#3) [Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9519.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9082.png'),Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/8377.png')]




```
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


```
y_train = get_labels(files_train)
y_valid = get_labels(files_valid)
```

    len: 709, random view: [7, 7, 3, 3, 7]
    [Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/7525.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9434.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/3/8014.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/3/7778.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9988.png')]
    len: 699, random view: [7, 7, 7, 7, 3]
    [Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/valid/7/7950.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/valid/7/9957.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/valid/7/7701.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/valid/7/960.png'), Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/valid/3/8777.png')]



```
len(y_train)
```


<style>.container { width:100% !important; }</style>





    709



### idx_line and check
I have used these funcs more than once, so turned into funcs


```
#| export utils
def idx_line(lst):
    "return zip(range(len(lst)), lst)"
    return zip(range(len(lst)), lst)
```


```
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


```
import PIL
```


```
whatinside(PIL)
```

    PIL has: 
    0 items in its __all__, and 
    0 user defined functions, 
    1 classes or class objects, 
    0 builtin funcs and methods, and
    1 callables.
    
    Pillow (Fork of the Python Imaging Library)
    
    Pillow is the friendly PIL fork by Alex Clark and Contributors.
        https://github.com/python-pillow/Pillow/
    
    Pillow is forked from PIL 1.1.7.
    
    PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
    Copyright (c) 1999 by Secret Labs AB.
    
    Use PIL.__version__ for this Pillow version.
    
    ;-)



```
check(PIL.Image.open, n=100)
```

    signature: (fp, mode='r', formats=None)
    __class__: <class 'function'>
    __repr__: <function open>
    
    __doc__:
    Opens and identifies the given image file.
    
    This is a lazy operation; this function identifies the file, but
    the file remains open and the actual image data is not read from
    the file until you try to process the data (or call the
    :py:meth:`~PIL.Image.Image.load` method).  See
    :py:func:`~PIL.Image.new`. See :ref:`file-handling`.
    
    :param fp: A filename (string), pathlib.Path object or a file object.
       The file object must implement ``file.read``,
       ``file.seek``, and ``file.tell`` methods,
       and be opened in binary mode.
    :param mode: The mode.  If given, this argument must be "r".
    :param formats: A list or tuple of formats to attempt to load the file in.
       This can be used to restrict the set of formats checked.
       Pass ``None`` to try all supported formats. You can print the set of
       available formats by running ``python3 -m PIL`` or using
       the :py:func:`PIL.features.pilinfo` function.
    :returns: An :py:class:`~PIL.Image.Image` object.
    :exception FileNotFoundError: If the file cannot be found.
    :exception PIL.UnidentifiedImageError: If the image cannot be opened and
       identified.
    :exception ValueError: If the ``mode`` is not "r", or if a ``StringIO``
       instance is used for ``fp``.
    :exception TypeError: If ``formats`` is not ``None``, a list or a tuple.
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: True
    method: False



```
PIL.Image.open(files_train[0])
PIL.Image.open(files_valid[-1])
```




    
![png](0001_fastai_matmul_files/0001_fastai_matmul_54_0.png)
    






    
![png](0001_fastai_matmul_files/0001_fastai_matmul_54_1.png)
    



### img, img.convert, img.resize
convert img color mode to L or RGB, and resize img to shape (28, 28)


```
import torch
import numpy as np
```


```
img = PIL.Image.open(files_train[0])
```


```
print(inspect.signature(img.convert)) # add this line to the doc func defined above
```

    (mode=None, matrix=None, dither=None, palette=<Palette.WEB: 0>, colors=256)



```
img.__class__
type(img) # add this to doc func above
```




    PIL.PngImagePlugin.PngImageFile






    PIL.PngImagePlugin.PngImageFile




```
# how to quickly know about an object, make doc to solve this problem
type(img)
```




    PIL.PngImagePlugin.PngImageFile




```
img.__doc__ == None # add this to func doc
```




    True




```
img.__dict__ # add this into doc
```




    {'im': None,
     'mode': 'L',
     '_size': (33, 41),
     'palette': None,
     'info': {'gamma': 0.45455,
      'chromaticity': (0.3127, 0.329, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06)},
     '_category': 0,
     'readonly': 1,
     'pyaccess': None,
     '_exif': None,
     '_min_frame': 0,
     'custom_mimetype': None,
     'tile': [('zip', (0, 0, 33, 41), 134, 'L')],
     'decoderconfig': (),
     'decodermaxblock': 65536,
     'fp': <_io.BufferedReader name='/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png'>,
     'filename': '/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png',
     '_exclusive_fp': True,
     '_fp': <_io.BufferedReader name='/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png'>,
     '_PngImageFile__frame': 0,
     'private_chunks': [],
     'png': <PIL.PngImagePlugin.PngStream>,
     '_text': None,
     'n_frames': 1,
     'default_image': False,
     '_PngImageFile__prepare_idat': 312,
     'is_animated': False}




```
check(img)
```

    signature: None
    __class__: <class 'PIL.PngImagePlugin.PngImageFile'>
    __repr__: <PIL.PngImagePlugin.PngImageFile image mode=L size=33x41>
    
    __doc__: not exist
    
    __dict__: 
    {'_PngImageFile__frame': 0,
     '_PngImageFile__prepare_idat': 312,
     '_category': 0,
     '_exclusive_fp': True,
     '_exif': None,
     '_fp': <_io.BufferedReader name='/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png'>,
     '_min_frame': 0,
     '_size': (33, 41),
     '_text': None,
     'custom_mimetype': None,
     'decoderconfig': (),
     'decodermaxblock': 65536,
     'default_image': False,
     'filename': '/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png',
     'fp': <_io.BufferedReader name='/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9243.png'>,
     'im': None,
     'info': {'chromaticity': (0.3127, 0.329, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06),
              'gamma': 0.45455},
     'is_animated': False,
     'mode': 'L',
     'n_frames': 1,
     'palette': None,
     'png': <PIL.PngImagePlugin.PngStream object>,
     'private_chunks': [],
     'pyaccess': None,
     'readonly': 1,
     'tile': [('zip', (0, 0, 33, 41), 134, 'L')]}
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```
print(img)
```

    <PIL.PngImagePlugin.PngImageFile image mode=L size=33x41>



```
print(img.convert)
```

    <bound method Image.convert of <PIL.PngImagePlugin.PngImageFile image mode=L size=33x41>>



```
check(img.convert)
```

    signature: (mode=None, matrix=None, dither=None, palette=<Palette.WEB: 0>, colors=256)
    __class__: <class 'method'>
    __repr__: <bound method Image.convert of <PIL.PngImagePlugin.PngImageFile image mode=L size=33x41>>
    
    __doc__:
    Returns a converted copy of this image. For the "P" mode, this
    method translates pixels through the palette.  If mode is
    omitted, a mode is chosen so that all information in the image
    and the palette can be represented without a palette.
    
    The current version supports all possible conversions between
    "L", "RGB" and "CMYK." The ``matrix`` argument only supports "L"
    and "RGB".
    
    When translating a color image to greyscale (mode "L"),
    the library uses the ITU-R 601-2 luma transform::
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: False
    method: True



```
img.convert("RGB")
img.size
img.convert("L")
img.size
```




    
![png](0001_fastai_matmul_files/0001_fastai_matmul_67_0.png)
    






    (33, 41)






    
![png](0001_fastai_matmul_files/0001_fastai_matmul_67_2.png)
    






    (33, 41)




```
check(img.resize)
```

    signature: (size, resample=None, box=None, reducing_gap=None)
    __class__: <class 'method'>
    __repr__: <bound method Image.resize of <PIL.PngImagePlugin.PngImageFile image mode=L size=33x41>>
    
    __doc__:
    Returns a resized copy of this image.
    
    :param size: The requested size in pixels, as a 2-tuple:
       (width, height).
    :param resample: An optional resampling filter.  This can be
       one of :py:data:`PIL.Image.Resampling.NEAREST`,
       :py:data:`PIL.Image.Resampling.BOX`,
       :py:data:`PIL.Image.Resampling.BILINEAR`,
       :py:data:`PIL.Image.Resampling.HAMMING`,
       :py:data:`PIL.Image.Resampling.BICUBIC` or
       :py:data:`PIL.Image.Resampling.LANCZOS`.
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: False
    method: True



```
img = img.resize((28, 28))
img.size
# img = img.resize((1, 28, 28)) # this is not allowed
```




    (28, 28)



### torch.Tensor, torch.stack, imgs2tensor
convert from img type to np.array to pytorch tensor


```
img
```




    
![png](0001_fastai_matmul_files/0001_fastai_matmul_71_0.png)
    




```
print(img)
```

    <PIL.Image.Image image mode=L size=28x28>



```
np.array(img).shape
```




    (28, 28)




```
np.array(img)[10:15, 10:15]
```




    array([[184, 222,  70,  17,  14],
           [223, 140,  16,   0,   0],
           [182,  52,   3,   0,   0],
           [ 34,   6,   0,   0,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)




```
from torch import tensor
```


```
tensor(np.array(img)).shape
```




    torch.Size([28, 28])




```
t = torch.Tensor(np.array(img))
```


```
files_train[1]
img1 = PIL.Image.open(files_train[1])
img1 = img1.resize((28,28))
t1 = torch.Tensor(np.array(img1))
# t1
torch.stack([t, t1], dim=0).shape
torch.stack([t, t1], dim=1).shape
torch.stack([t, t1], dim=-1).shape
```




    Path('/Users/Natsume/.fastai/data/mnist_var_size_tiny/train/7/9519.png')






    torch.Size([2, 28, 28])






    torch.Size([28, 2, 28])






    torch.Size([28, 28, 2])




```
lst_t = []
for f in files_train[:5]:
    img = PIL.Image.open(f).resize((28,28))
    t = torch.Tensor(np.array(img))
    lst_t.append(t)
torch.stack(lst_t, dim=0).shape
```




    torch.Size([5, 28, 28])




```

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


```
x_train = imgs2tensor(files_train)
```

    torch.Size([709, 28, 28])


### torch.permute, torch.float


```
inspect.isbuiltin(x_train.permute)
```




    True




```
x_train.permute.__doc__
```




    '\npermute(*dims) -> Tensor\n\nSee :func:`torch.permute`\n'




```
check(x_train.permute)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method permute of Tensor object>
    
    __doc__:
    permute(*dims) -> Tensor
    
    See :func:`torch.permute`
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```
check(torch.permute,n=-1)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method permute of type object>
    
    __doc__:
    permute(input, dims) -> Tensor
    
    Returns a view of the original tensor :attr:`input` with its dimensions permuted.
    
    Args:
        input (Tensor): the input tensor.
        dims (tuple of ints): The desired ordering of dimensions
    
    Example:
        >>> x = torch.randn(2, 3, 5)
        >>> x.size()
        torch.Size([2, 3, 5])
        >>> torch.permute(x, (2, 0, 1)).size()
        torch.Size([5, 2, 3])
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```
check(x_train.float)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method float of Tensor object>
    
    __doc__:
    float(memory_format=torch.preserve_format) -> Tensor
    
    ``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.
    
    Args:
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.preserve_format``.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```
check(torch.Tensor.float, n=-1)
```

    signature: None
    __class__: <class 'method_descriptor'>
    __repr__: <method 'float' of 'torch._C._TensorBase' objects>
    
    __doc__:
    float(memory_format=torch.preserve_format) -> Tensor
    
    ``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.
    
    Args:
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.preserve_format``.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False



```
check(inspect.ismethoddescriptor)
```

    signature: (object)
    __class__: <class 'function'>
    __repr__: <function ismethoddescriptor>
    
    __doc__:
    Return true if the object is a method descriptor.
    
    But not if ismethod() or isclass() or isfunction() are true.
    
    This is new in Python 2.2, and, for example, is true of int.__add__.
    An object passing this test has a __get__ attribute but not a __set__
    attribute, but beyond that the set of attributes varies.  __name__ is
    usually sensible, and __doc__ often is.
    
    Methods implemented via descriptors that also pass one of the other
    tests return false from the ismethoddescriptor() test, simply because
    __dict__: 
    {}
    metaclass: False
    class: False
    decorator: False
    function: True
    method: False



```
inspect.ismethoddescriptor(torch.Tensor.float) # to improve on func check
```




    True




```
check(x_train.is_floating_point)
```

    signature: None
    __class__: <class 'builtin_function_or_method'>
    __repr__: <built-in method is_floating_point of Tensor object>
    
    __doc__:
    is_floating_point() -> bool
    
    Returns True if the data type of :attr:`self` is a floating point data type.
    __dict__: not exist 
    
    metaclass: False
    class: False
    decorator: False
    function: False
    method: False


### mean_std, normalize, imgs2tensor


```
x_train.max(), x_train.min(), x_train.median(), x_train.mean()

x_train.mean(), x_train.std()
```




    (tensor(255.), tensor(0.), tensor(0.), tensor(28.4574))






    (tensor(28.4574), tensor(68.6432))




```
#| export
def mean_std(t):
    "check mean and std of a tensor"
    print(f'mean: {t.mean()}, std: {t.std()}')
```


```
x_train = x_train/x_train.max()
```


```
#| export 
def normalize(t):
    "to normalize a tensor by dividing its maximum value"
    return t/t.max()
```


```
x_train = normalize(x_train)
```


```
mean_std(x_train)
```

    mean: 0.11159760504961014, std: 0.2691890597343445



```
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


```
x_train = imgs2tensor(files_train)
x_valid = imgs2tensor(files_valid)
x_test = imgs2tensor(files_test)
```

    torch.Size([709, 28, 28])
    mean: 0.11159760504961014, std: 0.2691890597343445
    torch.Size([699, 28, 28])
    mean: 0.12307247519493103, std: 0.28430673480033875
    torch.Size([20, 28, 28])
    mean: 0.11439651250839233, std: 0.27472028136253357



```

```

### %whos 
https://www.wrighters.io/how-to-view-all-your-variables-in-a-jupyter-notebook/


```
[d for d in dir() if not "__" in d and not "_" in d]
```




    ['FunctionType',
     'In',
     'MethodType',
     'Out',
     'PIL',
     'TEST',
     'URLs',
     'check',
     'df',
     'exit',
     'f',
     'fastcodes',
     'fastlistnbs',
     'fastlistsrcs',
     'fastnbs',
     'fastnotes',
     'fastsrcs',
     'fastview',
     'fde',
     'fdt',
     'i',
     'img',
     'img1',
     'imgs2tensor',
     'inspect',
     'ipy2md',
     'isdecorator',
     'ismetaclass',
     'kagglenbs',
     'normalize',
     'np',
     'openNB',
     'openNBKaggle',
     'operator',
     'path',
     'pd',
     'quit',
     'rand',
     'random',
     't',
     't1',
     'tensor',
     'test',
     'torch',
     'whatinside',
     'whichversion']




```
%whos L list
```

    Variable      Type    Data/Info
    -------------------------------
    files_test    L       [Path('/Users/Natsume/.fa<...>ize_tiny/test/5071.png')]
    files_train   L       [Path('/Users/Natsume/.fa<...>_tiny/train/3/7288.png')]
    files_valid   L       [Path('/Users/Natsume/.fa<...>_tiny/valid/3/8811.png')]
    kagglenbs     list    n=27
    label_train   list    n=709
    lst_t         list    n=5
    rand          list    n=5
    y_train       list    n=709
    y_valid       list    n=699


## export


```

```


```
# |hide
import nbdev
```


```
nbdev.nbdev_export()
```


```

```
