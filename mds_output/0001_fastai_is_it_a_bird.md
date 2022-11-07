# 0001_fastai_Is it a bird? Creating a model from your own data
---
skip_exec: true
---

```
#| default_exp delete_bird
```

## Useful Course sites

**Official course site**:  for lesson [1](https://course.fast.ai/Lessons/lesson1.html)    

**Official notebooks** [repo](https://github.com/fastai/course22), on [nbviewer](https://nbviewer.org/github/fastai/course22/tree/master/)

Official **Is it a bird** [notebook](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data) on kaggle     



```
from fastdebug.utils import *
from __future__ import annotations
from fastai.vision.all import *
```


<style>.container { width:100% !important; }</style>



```
DataLoaders??
```

## mamba update -q -y fastai; pip install -Uqq


```
#| eval: false
# !mamba update -q -y fastai
```


```
#| eval: false
# !pip install -Uqq duckduckgo_search
```

## fastai, duckduckgo_search


```
from fastdebug.utils import *
from fastdebug.core import *
```


```
import fastai
```


```
whatinside(fastai, lib=True)
```

    The library has 49 modules
    ['10_tutorial',
     '10b_tutorial',
     '22_tutorial',
     '23_tutorial',
     '24_tutorial',
     '35_tutorial',
     '38_tutorial',
     '39_tutorial',
     '44_tutorial',
     '46_tutorial',
     '50_tutorial',
     '61_tutorial',
     '_modidx',
     '_nbdev',
     '_pytorch_doc',
     'app_examples',
     'basics',
     'callback',
     'camvid',
     'collab',
     'data',
     'dev-setup',
     'distributed',
     'fp16_utils',
     'imports',
     'index',
     'interpret',
     'launch',
     'layers',
     'learner',
     'losses',
     'medical',
     'metrics',
     'migrating_catalyst',
     'migrating_ignite',
     'migrating_lightning',
     'migrating_pytorch',
     'migrating_pytorch_verbose',
     'optimizer',
     'quick_start',
     'tabular',
     'test_utils',
     'text',
     'torch_basics',
     'torch_core',
     'torch_imports',
     'tutorial',
     'ulmfit',
     'vision']



```
import duckduckgo_search
```


```
whichversion("duckduckgo_search")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [11], in <cell line: 1>()
    ----> 1 whichversion("duckduckgo_search")


    NameError: name 'whichversion' is not defined



```
whatinside(duckduckgo_search)
```


```
whatinside(duckduckgo_search, func=True)
```


```

```


```

```


```

```

## Experiment with images using duckduckgo_search library
Trying out duckduckgo_search lib to extract image urls and use fastdownload to download image files and display them.


```
from duckduckgo_search import ddg_images
from fastcore.all import *
```

### src: itemgot(self:L, *idxs)
apply itemgetter(idx) to every item of self or L


```
@patch
# @snoop
def itemgot(self:L, *idxs):
    "Create new `L` with item `idx` of all `items`"
        # itemgetter(idx) is a func, and is applied to every item of x, so I have itemgetter('image')(x)
        # but itemgetter('image')(x) is in fact x['image'], according to itemgetter.__init__ below    
### the offiical version: can take only one idx such as `image` or `url`
#     x = self
#     for idx in idxs: x = x.map(itemgetter(idx))  
#     return x

### my version can take on both or more than `image` and `url`
    res = []
    for idx in idxs: 
        res.append(self.map(itemgetter(idx)))
    res = res if len(res) > 1 else res[0] if len(res) == 1 else None
    return res


# ~/mambaforge/lib/python3.9/site-packages/fastcore/foundation.py
```

### src: itemgetter
After f = itemgetter(2), the call f(r) returns r[2].

After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3])


```
class itemgetter:
    """
    Return a callable object that fetches the given item(s) from its operand. The following examples are very illuminating.
    After f = itemgetter(2), the call f(r) returns r[2].
    After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3])
    """
    __slots__ = ('_items', '_call')
#     @snoop
    def __init__(self, item, *items):
        if not items:
            self._items = (item,)
#             @snoop            
            def func(obj):
                return obj[item]
            self._call = func
        else:
            self._items = items = (item,) + items
#             @snoop
            def func(obj):
                return tuple(obj[i] for i in items)
            self._call = func
#     @snoop
    def __call__(self, obj):
        return self._call(obj)

    def __repr__(self):
        return '%s.%s(%s)' % (self.__class__.__module__,
                              self.__class__.__name__,
                              ', '.join(map(repr, self._items)))

    def __reduce__(self):
        return self.__class__, self._items
# ~/mambaforge/lib/python3.9/operator.py    
```

### ```search_images(term, max_images=30)```
use `ddg_images` and `L.itemtogt` and `itemgetter` to extract image download urls into a list


```
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
#     return pp(L(ddg_images(term, max_results=max_images))).itemgot('image')
```


```
# help(ddg_images)
# ddg_images??
# L.itemgot??
# itemgetter??
```


```
#|eval: false
#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).

urls = search_images('bird photos', max_images=3)
urls
urls[0]
```

    Searching for 'bird photos'



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [15], in <cell line: 5>()
          1 #|eval: false
          2 #NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
          3 #    If you get a JSON error, just try running it again (it may take a couple of tries).
    ----> 5 urls = search_images('bird photos', max_images=3)
          6 urls
          7 urls[0]


    Input In [13], in search_images(term, max_images)
          1 def search_images(term, max_images=30):
          2     print(f"Searching for '{term}'")
    ----> 3     return L(ddg_images(term, max_results=max_images)).itemgot('image')


    NameError: name 'ddg_images' is not defined



```
# fastnbs("search_images")
```

### ```download_url(urls[0], dest, show_progress=False)```
download a single image from a single url using fastdownload lib


```
#|eval: false
from fastdownload import download_url
dest = 'bird.jpg' # dest is a filename
download_url(urls[0], dest, show_progress=True)
```

### ```Image.open(filename)```


```
from fastai.vision.all import *
```


```
im = Image.open(dest)
im.width
im.height
```

### ```Image.Image.to_thumb```
Same as `thumbnail` to display image in a square with specified h and w, but uses a copy


```
@patch
@snoop
def to_thumb(self:Image.Image, h, w=None):
    "Same as `thumbnail`, but uses a copy"
    if w is None: 
        w=h
    im = self.copy()
    im.thumbnail((w,h))
    return im
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/core.py
# Type:      method
```


```
im.to_thumb(256,256)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [17], in <cell line: 1>()
    ----> 1 im.to_thumb(256,256)


    NameError: name 'im' is not defined



```
@patch
def to_thumb(self:Image.Image, h, w=None):
    "Same as `thumbnail`, but uses a copy"
    if w is None: w=h
    im = self.copy()
    im.thumbnail((w,h))
    return im
```


```
#|eval: false
download_url(search_images('Spinosaurus aegyptiacus photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```


```

```

### PILImage.create or PILBase.create(fn, **kwargs)
Open an `Image` from `fn`, which can be path or str, Tensor, numpy, ndarray, bytes


```
doc(PILImage.create)
```


```
class PILBase(Image.Image, metaclass=BypassNewMeta):
    _bypass_type=Image.Image
    _show_args = {'cmap':'viridis'}
    _open_args = {'mode': 'RGB'}
    @classmethod
    @snoop
    def create(cls, fn:Path|str|Tensor|ndarray|bytes, **kwargs)->None:
        "Open an `Image` from path `fn`"
        if isinstance(fn,TensorImage): 
            fn = fn.permute(1,2,0).type(torch.uint8)
        if isinstance(fn, TensorMask): 
            fn = fn.type(torch.uint8)
        if isinstance(fn,Tensor): 
            fn = fn.numpy()
        if isinstance(fn,ndarray): 
            return cls(Image.fromarray(fn))
        if isinstance(fn,bytes): 
            fn = io.BytesIO(fn)
        pp(cls._open_args, kwargs, merge(cls._open_args, kwargs))
        res = cls(load_image(fn, **merge(cls._open_args, kwargs)))
        return res

    def show(self, ctx=None, **kwargs):
        "Show image using `merge(self._show_args, kwargs)`"
        return show_image(self, ctx=ctx, **merge(self._show_args, kwargs))

    def __repr__(self): return f'{self.__class__.__name__} mode={self.mode} size={"x".join([str(d) for d in self.size])}'

```


```
# res = PILImage.create('forest.jpg')
res = PILBase.create('forest.jpg')
res
```


```
@classmethod
@snoop
def create(cls:PILBase, fn:Path|str|Tensor|ndarray|bytes, **kwargs)->None:
    "Open an `Image` from path `fn`"
    if isinstance(fn,TensorImage): fn = fn.permute(1,2,0).type(torch.uint8)
    if isinstance(fn, TensorMask): fn = fn.type(torch.uint8)
    if isinstance(fn,Tensor): fn = fn.numpy()
    if isinstance(fn,ndarray): return cls(Image.fromarray(fn))
    if isinstance(fn,bytes): fn = io.BytesIO(fn)
    return cls(load_image(fn, **merge(cls._open_args, kwargs)))
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/core.py
# Type:      method
```


```

```

### ```Path.parent```


```
#|eval: false
searches = 'forest','bird'
path = Path('bird_or_not')
(path/"forest").parent
```




    Path('bird_or_not')



### ```mkdir(self:Path, mode=0o777, parents=False, exist_ok=False)```
Create path including parent path, if exist, don't create

Our searches seem to be giving reasonable results, so let's grab 200 examples of each of "bird" and "forest" photos, and save each group of photos to a different folder:


```
@patch
# @snoop
def mkdir(self:Path, mode=0o777, parents=False, exist_ok=False):
        """
        Create a new directory at this given path.
        """
        try:
            self._accessor.mkdir(self, mode)
        except FileNotFoundError:
            if not parents or self.parent == self:
                raise
            self.parent.mkdir(parents=True, exist_ok=True)
            self.mkdir(mode, parents=False, exist_ok=exist_ok)
        except OSError:
            # Cannot rely on checking for EEXIST, since the operating system
            # could give priority to other errors like EACCES or EROFS
            if not exist_ok or not self.is_dir():
                raise
# ~/mambaforge/lib/python3.9/pathlib.py                
```


```
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
```

### Path('bird_or_not')/'bird' and Path.ls()


```
birds = (Path('bird_or_not')/'bird').ls()
forests = (Path('bird_or_not')/'forest').ls()
```


```
len(birds)
len(forests)
```




    1






    1




```
# fastnbs("path", "fastai", True)
```

## Search and Download images for your model

### resize_image(file, dest, src='.', max_size=None, n_channels=3, ext=None, ...)
Resize an image file to max_size, from src folder and saved a copy into dest folder

If the same filename is already inside dest folder, then do nothing

If the file to be resized is broken, then do nothing

If max_size is none, then make a copy and saved into dest/file


```
@snoop
def resize_image(file, # str for image filename
                 dest, # str for image destination folder
                 src='.', # str for image source folder
                 max_size=None, # int for image maximum size to be changed
                 n_channels=3, ext=None,
                 img_format=None, resample=BILINEAR, resume=False, **kwargs ):
    "Resize file to dest to max_size"
    dest = Path(dest)
    
    dest_fname = dest/file
    dest_fname.parent.mkdir(exist_ok=True, parents=True)
    file = Path(src)/file
    if resume and dest_fname.exists(): return
    if not verify_image(file): return

    img = Image.open(file)
    imgarr = np.array(img)
    img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
    if ext is not None: dest_fname=dest_fname.with_suffix(ext) # specify file extensions
    if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
        if max_size is not None:
            pp(doc_sig(resize_to))
            pp(img.height, img.width)
            new_sz = resize_to(img, max_size) # keep the ratio
            pp(doc_sig(img.resize))
            img = img.resize(new_sz, resample=resample)
        if n_channels == 3: 
            img = img.convert("RGB")
        pp(doc_sig(img.save))
        img.save(dest_fname, img_format, **kwargs)
    elif file != dest_fname : 
        shutil.copy2(file, dest_fname)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```


```
# doc(resize_image)
```


```
file = 'bird.jpg'
src = Path('.')
dest = src/"resized"
resize_image(file, dest, src=src, max_size=400)
im = Image.open(dest/file)
test_eq(im.shape[1],400)
# (dest/file).unlink()
```

    20:17:36.75 >>> Call to resize_image in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/4048456158.py", line 2
    20:17:36.75 ...... file = 'bird.jpg'
    20:17:36.75 ...... dest = Path('resized')
    20:17:36.75 ...... src = Path('.')
    20:17:36.75 ...... max_size = 400
    20:17:36.75 ...... n_channels = 3
    20:17:36.75 ...... ext = None
    20:17:36.75 ...... img_format = None
    20:17:36.75 ...... resample = <Resampling.BILINEAR: 2>
    20:17:36.75 ...... resume = False
    20:17:36.75 ...... kwargs = {}
    20:17:36.75    2 | def resize_image(file, # str for image filename
    20:17:36.75    9 |     dest = Path(dest)
    20:17:36.75   11 |     dest_fname = dest/file
    20:17:36.75 .......... dest_fname = Path('resized/bird.jpg')
    20:17:36.75   12 |     dest_fname.parent.mkdir(exist_ok=True, parents=True)
    20:17:36.75   13 |     file = Path(src)/file
    20:17:36.75 .......... file = Path('bird.jpg')
    20:17:36.75   14 |     if resume and dest_fname.exists(): return
    20:17:36.75   15 |     if not verify_image(file): return
    20:17:36.77   17 |     img = Image.open(file)
    20:17:36.77 .......... img = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1600x1200>
    20:17:36.77   18 |     imgarr = np.array(img)
    20:17:36.79 .......... imgarr = array([[[98, 52, 26],
    20:17:36.79                             [94, 48, 22],
    20:17:36.79                             ...,
    20:17:36.79                             [13, 14,  0],
    20:17:36.79                             [14, 15,  1]],
    20:17:36.79                     
    20:17:36.79                            [[95, 49, 23],
    20:17:36.79                             [94, 48, 22],
    20:17:36.79                             ...,
    20:17:36.79                             [14, 15,  1],
    20:17:36.79                             [15, 16,  2]],
    20:17:36.79                     
    20:17:36.79                            ...,
    20:17:36.79                     
    20:17:36.79                            [[94, 64, 38],
    20:17:36.79                             [91, 61, 35],
    20:17:36.79                             ...,
    20:17:36.79                             [74, 95,  0],
    20:17:36.79                             [74, 95,  0]],
    20:17:36.79                     
    20:17:36.79                            [[91, 61, 35],
    20:17:36.79                             [90, 60, 34],
    20:17:36.79                             ...,
    20:17:36.79                             [76, 97,  2],
    20:17:36.79                             [75, 96,  1]]], dtype=uint8)
    20:17:36.79   19 |     img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
    20:17:36.79 .......... img_channels = 3
    20:17:36.79   20 |     if ext is not None: dest_fname=dest_fname.with_suffix(ext) # specify file extensions
    20:17:36.79   21 |     if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
    20:17:36.79   22 |         if max_size is not None:
    20:17:36.79   23 |             pp(doc_sig(resize_to))
    20:17:36.79 LOG:
    20:17:36.81 .... doc_sig(resize_to) = ('no mro',
    20:17:36.81                            'Size to resize to, to hit `targ_sz` at same aspect ratio, in PIL coords (i.e '
    20:17:36.81                            'w*h)',
    20:17:36.81                            <Signature (img, targ_sz, use_min=False)>)
    20:17:36.81   24 |             pp(img.height, img.width)
    20:17:36.81 LOG:
    20:17:36.82 .... img.height = 1200
    20:17:36.82 .... img.width = 1600
    20:17:36.82   25 |             new_sz = resize_to(img, max_size) # keep the ratio
    20:17:36.82 .................. new_sz = (400, 300)
    20:17:36.82   26 |             pp(doc_sig(img.resize))
    20:17:36.82 LOG:
    20:17:36.83 .... doc_sig(img.resize) = ('no mro',
    20:17:36.83                             'Returns a resized copy of this image.\n'
    20:17:36.83                             '\n'
    20:17:36.83                             ':param size: The requested size in pixels, as a 2-tuple:\n'
    20:17:36.83                             '   (width, height).\n'
    20:17:36.83                             ':param resample: An optional resampling filter.  This can be\n'
    20:17:36.83                             '   one of :py:data:`PIL.Image.Resampling.NEAREST`,\n'
    20:17:36.83                             '   :py:data:`PIL.Image.Resampling.BOX`,\n'
    20:17:36.83                             '   :py:data:`PIL.Image.Resampling.BILINEAR`,\n'
    20:17:36.83                             '   :py:data:`PIL.Image.Resampling.HAMMING`,\n'
    20:17:36.83                             '   :py:data:`PIL.Image.Resampling.BICUBIC` or\n'
    20:17:36.83                             '   :py:data:`PIL.Image.Resampling.LANCZOS`.\n'
    20:17:36.83                             '   If the image has mode "1" or "P", it is always set to\n'
    20:17:36.83                             '   :py:data:`PIL.Image.Resampling.NEAREST`.\n'
    20:17:36.83                             '   If the image mode specifies a number of bits, such as "I;16", then the\n'
    20:17:36.83                             '   default filter is :py:data:`PIL.Image.Resampling.NEAREST`.\n'
    20:17:36.83                             '   Otherwise, the default filter is\n'
    20:17:36.83                             '   :py:data:`PIL.Image.Resampling.BICUBIC`. See: :ref:`concept-filters`.\n'
    20:17:36.83                             ':param box: An optional 4-tuple of floats providing\n'
    20:17:36.83                             '   the source image region to be scaled.\n'
    20:17:36.83                             '   The values must be within (0, 0, width, height) rectangle.\n'
    20:17:36.83                             '   If omitted or None, the entire source is used.\n'
    20:17:36.83                             ':param reducing_gap: Apply optimization by resizing the image\n'
    20:17:36.83                             '   in two steps. First, reducing the image by integer times\n'
    20:17:36.83                             '   using :py:meth:`~PIL.Image.Image.reduce`.\n'
    20:17:36.83                             '   Second, resizing using regular resampling. The last step\n'
    20:17:36.83                             '   changes size no less than by ``reducing_gap`` times.\n'
    20:17:36.83                             '   ``reducing_gap`` may be None (no first step is performed)\n'
    20:17:36.83                             '   or should be greater than 1.0. The bigger ``reducing_gap``,\n'
    20:17:36.83                             '   the closer the result to the fair resampling.\n'
    20:17:36.83                             '   The smaller ``reducing_gap``, the faster resizing.\n'
    20:17:36.83                             '   With ``reducing_gap`` greater or equal to 3.0, the result is\n'
    20:17:36.83                             '   indistinguishable from fair resampling in most cases.\n'
    20:17:36.83                             '   The default value is None (no optimization).\n'
    20:17:36.83                             ':returns: An :py:class:`~PIL.Image.Image` object.',
    20:17:36.83                             <Signature (size, resample=None, box=None, reducing_gap=None)>)
    20:17:36.83   27 |             img = img.resize(new_sz, resample=resample)
    20:17:36.84 .................. img = <PIL.Image.Image image mode=RGB size=400x300>
    20:17:36.84   28 |         if n_channels == 3: 
    20:17:36.84   29 |             img = img.convert("RGB")
    20:17:36.84   30 |         pp(doc_sig(img.save))
    20:17:36.84 LOG:
    20:17:36.85 .... doc_sig(img.save) = ('no mro',
    20:17:36.85                           'Saves this image under the given filename.  If no format is\n'
    20:17:36.85                           'specified, the format to use is determined from the filename\n'
    20:17:36.85                           'extension, if possible.\n'
    20:17:36.85                           '\n'
    20:17:36.85                           'Keyword options can be used to provide additional instructions\n'
    20:17:36.85                           "to the writer. If a writer doesn't recognise an option, it is\n"
    20:17:36.85                           'silently ignored. The available options are described in the\n'
    20:17:36.85                           ':doc:`image format documentation\n'
    20:17:36.85                           '<../handbook/image-file-formats>` for each writer.\n'
    20:17:36.85                           '\n'
    20:17:36.85                           'You can use a file object instead of a filename. In this case,\n'
    20:17:36.85                           'you must always specify the format. The file object must\n'
    20:17:36.85                           'implement the ``seek``, ``tell``, and ``write``\n'
    20:17:36.85                           'methods, and be opened in binary mode.\n'
    20:17:36.85                           '\n'
    20:17:36.85                           ':param fp: A filename (string), pathlib.Path object or file object.\n'
    20:17:36.85                           ':param format: Optional format override.  If omitted, the\n'
    20:17:36.85                           '   format to use is determined from the filename extension.\n'
    20:17:36.85                           '   If a file object was used instead of a filename, this\n'
    20:17:36.85                           '   parameter should always be used.\n'
    20:17:36.85                           ':param params: Extra parameters to the image writer.\n'
    20:17:36.85                           ':returns: None\n'
    20:17:36.85                           ':exception ValueError: If the output format could not be determined\n'
    20:17:36.85                           '   from the file name.  Use the format option to solve this.\n'
    20:17:36.85                           ':exception OSError: If the file could not be written.  The file\n'
    20:17:36.85                           '   may have been created, and may contain partial data.',
    20:17:36.85                           <Signature (fp, format=None, **params)>)
    20:17:36.85   31 |         img.save(dest_fname, img_format, **kwargs)
    20:17:36.85 <<< Return value from resize_image: None



```
file = 'bird.jpg'
src = Path('.')
dest = src/"resized"
resize_image(file, dest, src=src, max_size=None) # just copy not size changed
im = Image.open(dest/file)
# test_eq(im.shape[1],1920)
```

    20:17:36.87 >>> Call to resize_image in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/4048456158.py", line 2
    20:17:36.87 ...... file = 'bird.jpg'
    20:17:36.87 ...... dest = Path('resized')
    20:17:36.87 ...... src = Path('.')
    20:17:36.87 ...... max_size = None
    20:17:36.87 ...... n_channels = 3
    20:17:36.87 ...... ext = None
    20:17:36.87 ...... img_format = None
    20:17:36.87 ...... resample = <Resampling.BILINEAR: 2>
    20:17:36.87 ...... resume = False
    20:17:36.87 ...... kwargs = {}
    20:17:36.87    2 | def resize_image(file, # str for image filename
    20:17:36.87    9 |     dest = Path(dest)
    20:17:36.87   11 |     dest_fname = dest/file
    20:17:36.87 .......... dest_fname = Path('resized/bird.jpg')
    20:17:36.87   12 |     dest_fname.parent.mkdir(exist_ok=True, parents=True)
    20:17:36.88   13 |     file = Path(src)/file
    20:17:36.88 .......... file = Path('bird.jpg')
    20:17:36.88   14 |     if resume and dest_fname.exists(): return
    20:17:36.88   15 |     if not verify_image(file): return
    20:17:36.88   17 |     img = Image.open(file)
    20:17:36.88 .......... img = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1600x1200>
    20:17:36.88   18 |     imgarr = np.array(img)
    20:17:36.90 .......... imgarr = array([[[98, 52, 26],
    20:17:36.90                             [94, 48, 22],
    20:17:36.90                             ...,
    20:17:36.90                             [13, 14,  0],
    20:17:36.90                             [14, 15,  1]],
    20:17:36.90                     
    20:17:36.90                            [[95, 49, 23],
    20:17:36.90                             [94, 48, 22],
    20:17:36.90                             ...,
    20:17:36.90                             [14, 15,  1],
    20:17:36.90                             [15, 16,  2]],
    20:17:36.90                     
    20:17:36.90                            ...,
    20:17:36.90                     
    20:17:36.90                            [[94, 64, 38],
    20:17:36.90                             [91, 61, 35],
    20:17:36.90                             ...,
    20:17:36.90                             [74, 95,  0],
    20:17:36.90                             [74, 95,  0]],
    20:17:36.90                     
    20:17:36.90                            [[91, 61, 35],
    20:17:36.90                             [90, 60, 34],
    20:17:36.90                             ...,
    20:17:36.90                             [76, 97,  2],
    20:17:36.90                             [75, 96,  1]]], dtype=uint8)
    20:17:36.90   19 |     img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
    20:17:36.90 .......... img_channels = 3
    20:17:36.90   20 |     if ext is not None: dest_fname=dest_fname.with_suffix(ext) # specify file extensions
    20:17:36.90   21 |     if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
    20:17:36.90   32 |     elif file != dest_fname : 
    20:17:36.90   33 |         shutil.copy2(file, dest_fname)
    20:17:36.90 <<< Return value from resize_image: None



```
def resize_image(file, dest, src='.', max_size=None, n_channels=3, ext=None,
                 img_format=None, resample=BILINEAR, resume=False, **kwargs ):
    "Resize file to dest to max_size"
    dest = Path(dest)
    
    dest_fname = dest/file
    dest_fname.parent.mkdir(exist_ok=True, parents=True)
    file = Path(src)/file
    if resume and dest_fname.exists(): return
    if not verify_image(file): return

    img = Image.open(file)
    imgarr = np.array(img)
    img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
    if ext is not None: dest_fname=dest_fname.with_suffix(ext)
    if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
        if max_size is not None:
            new_sz = resize_to(img, max_size)
            img = img.resize(new_sz, resample=resample)
        if n_channels == 3: img = img.convert("RGB")
        img.save(dest_fname, img_format, **kwargs)
    elif file != dest_fname : shutil.copy2(file, dest_fname)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```

### resize_images(path, max_workers=defaults.cpus, max_size=None, recurse=False,..)
Resize files on path recursively to dest to max_size

'recursively' means if images in subfolders will be resized too, and the subfolders will be created if dest and src are not the same folder


```
@snoop
def resize_images(path, max_workers=defaults.cpus, max_size=None, recurse=False,
                  dest=Path('.'), n_channels=3, ext=None, img_format=None, resample=BILINEAR,
                  resume=None, **kwargs):
    "Resize files on path recursively to dest to max_size"
    path = Path(path)
    if resume is None and dest != Path('.'): 
        resume=False
    os.makedirs(dest, exist_ok=True)
    files = get_image_files(path, recurse=recurse)
    files = [o.relative_to(path) for o in files]
    parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
                   img_format=img_format, resample=resample, resume=resume, **kwargs)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```


```
dest = Path(".")/'try_resize_images'
resize_images('bird_or_not', max_size=100, dest=dest, max_workers=0, recurse=True) # try recurse=True to check the difference
```

    20:17:36.97 >>> Call to resize_images in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1408674014.py", line 2
    20:17:36.97 ...... path = 'bird_or_not'
    20:17:36.97 ...... max_workers = 0
    20:17:36.97 ...... max_size = 100
    20:17:36.97 ...... recurse = True
    20:17:36.97 ...... dest = Path('try_resize_images')
    20:17:36.97 ...... n_channels = 3
    20:17:36.97 ...... ext = None
    20:17:36.97 ...... img_format = None
    20:17:36.97 ...... resample = <Resampling.BILINEAR: 2>
    20:17:36.97 ...... resume = None
    20:17:36.97 ...... kwargs = {}
    20:17:36.97    2 | def resize_images(path, max_workers=defaults.cpus, max_size=None, recurse=False,
    20:17:36.97    6 |     path = Path(path)
    20:17:36.97 .......... path = Path('bird_or_not')
    20:17:36.97    7 |     if resume is None and dest != Path('.'): 
    20:17:36.97    8 |         resume=False
    20:17:36.97 .............. resume = False
    20:17:36.97    9 |     os.makedirs(dest, exist_ok=True)
    20:17:36.97   10 |     files = get_image_files(path, recurse=recurse)
    20:17:36.97 .......... files = [Path('bird_or_not/0b8fcba5-91a5-4689-999c-008e1.../bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')]
    20:17:36.97   11 |     files = [o.relative_to(path) for o in files]
        20:17:36.97 List comprehension:
        20:17:36.97   11 |     files = [o.relative_to(path) for o in files]
        20:17:36.98 .......... Iterating over <list_iterator object>
        20:17:36.98 .......... Values of path: Path('bird_or_not')
        20:17:36.98 .......... Values of o: Path('bird_or_not/0b8fcba5-91a5-4689-999c-008e108828f1.jpg'), Path('bird_or_not/037e9e61-3731-4876-9745-98758ae21be3.jpg'), Path('bird_or_not/forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg'), Path('bird_or_not/bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')
        20:17:36.98 Result: [Path('0b8fcba5-91a5-4689-999c-008e108828f1.jpg'), Path('037e9e61-3731-4876-9745-98758ae21be3.jpg'), Path('forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg'), Path('bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')]
    20:17:36.98   11 |     files = [o.relative_to(path) for o in files]
    20:17:36.98 .......... files = [Path('0b8fcba5-91a5-4689-999c-008e108828f1.jpg'), Path('037e9e61-3731-4876-9745-98758ae21be3.jpg'), Path('forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg'), Path('bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')]
    20:17:36.98   12 |     parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
    20:17:36.98   13 |                    img_format=img_format, resample=resample, resume=resume, **kwargs)
    20:17:36.98   12 |     parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
    20:17:36.98   13 |                    img_format=img_format, resample=resample, resume=resume, **kwargs)
    20:17:36.98   12 |     parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
    20:17:36.99 <<< Return value from resize_images: None



```
dest.ls()
```




    (#4) [Path('try_resize_images/forest'),Path('try_resize_images/0b8fcba5-91a5-4689-999c-008e108828f1.jpg'),Path('try_resize_images/037e9e61-3731-4876-9745-98758ae21be3.jpg'),Path('try_resize_images/bird')]




```
def resize_images(path, max_workers=defaults.cpus, max_size=None, recurse=False,
                  dest=Path('.'), n_channels=3, ext=None, img_format=None, resample=BILINEAR,
                  resume=None, **kwargs):
    "Resize files on path recursively to dest to max_size"
    path = Path(path)
    if resume is None and dest != Path('.'): resume=False
    os.makedirs(dest, exist_ok=True)
    files = get_image_files(path, recurse=recurse)
    files = [o.relative_to(path) for o in files]
    parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
                   img_format=img_format, resample=resample, resume=resume, **kwargs)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```

### prepare_images_dataset_binary(*things, key1="sun", key2="shade")
search and download images use query words using duckduckgo_search and fastdownload

two type of things (images) will be downloaded into two folders "bird" and "forest" under the parent folder "bird_or_forest"


```
sleep
```




    <function time.sleep>




```
# @snoop
def prepare_images_dataset_binary(*things, key1="sun", key2="shade"):
# searches = 'forest','bird'
# path = Path('forest_or_bird')
    folder_name=f"{things[0]}_or_{things[1]}"
    searches = things
    path = pp(Path(folder_name))
    from time import sleep

    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True) # don't recreate if already exist
        if len(dest.ls()) < 10 : # don't download and resize if already have them
            download_images(dest, urls=search_images(f'{o} photo'))
            sleep(10)  # Pause between searches to avoid over-loading server
            download_images(dest, urls=search_images(f'{o} {key1} photo'))
            sleep(10)
            download_images(dest, urls=search_images(f'{o} {key2} photo'))
            sleep(10)
            resize_images(path/o, max_size=400, dest=path/o) # since this is the lowest level of folder for resizing images

        print(dest)
        print(len(dest.ls()))
    
    return path

```


```
cry_dino = prepare_images_dataset_binary("T-rex", "Brachiosaurus", key1="crying cartoon", key2="sad doodle")
```

    20:17:37.10 LOG:
    20:17:37.11 .... Path(folder_name) = Path('T-rex_or_Brachiosaurus')


    T-rex_or_Brachiosaurus/T-rex
    167
    T-rex_or_Brachiosaurus/Brachiosaurus
    109



```
bird = prepare_images_dataset_binary("forest", "bird")
```

    20:17:37.13 LOG:
    20:17:37.13 .... Path(folder_name) = Path('forest_or_bird')


    forest_or_bird/forest
    175
    forest_or_bird/bird
    82



```
dino = prepare_images_dataset_binary("T-rex", "Spinosaurus aegyptiacus")
```

    20:17:37.15 LOG:
    20:17:37.16 .... Path(folder_name) = Path('T-rex_or_Spinosaurus aegyptiacus')


    T-rex_or_Spinosaurus aegyptiacus/T-rex
    84
    T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus
    82


### ```randomdisplay(path)```
to randomly display images in their path


```
#| export utils
from fastai.vision.core import *
```


```
fastnbs("src: randomdisplay")
```


### <mark style="background-color: #ffff00">src:</mark>  <mark style="background-color: #FFFF00">randomdisplay</mark> (path, size, db=false)




heading 3.



display a random images from a L list (eg., test_files, train_files) of image files or from a path/folder of images.\
    the image filename is printed as well

```python
import pathlib
type(path) == pathlib.PosixPath
type(train_files) == L
```

```python
snoopon()
```

```python
#| export utils
# @snoop
def randomdisplay(path, size=128, db=False):
    "display a random images from a L list (eg., test_files, train_files) of image files or from a path/folder of images.\
    the image filename is printed as well"
# https://www.geeksforgeeks.org/python-random-module/
    import random
    import pathlib
    from fastai.vision.all import PILImage
    if type(path) == pathlib.PosixPath:
        rand = random.randint(0,len(path.ls())-1) 
        file = path.ls()[rand]
    elif type(path) == L:
        rand = random.randint(0,len(path)-1) 
        file = path[rand]
    im = PILImage.create(file)
    if db: pp(im.width, im.height, file)
    pp(file)
    return im.to_thumb(size)
```

```python
randomdisplay(test_files, 128)
randomdisplay(train_files, 200)
randomdisplay(path/"train_images/dead_heart", 128)
```

```python
snoopoff()
```

Next, heading 3
### ht: data_prep - remove images that fail to open with `remove_failed(path)`



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb#src:-randomdisplay(path,-size,-db=False)
)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)



```
dino.ls()
bird.ls()
```




    (#5) [Path('T-rex_or_Spinosaurus aegyptiacus/.DS_Store'),Path('T-rex_or_Spinosaurus aegyptiacus/crying'),Path('T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex'),Path('T-rex_or_Spinosaurus aegyptiacus/fierce')]






    (#3) [Path('forest_or_bird/forest'),Path('forest_or_bird/.DS_Store'),Path('forest_or_bird/bird')]




```
randomdisplay(bird/'bird')
randomdisplay(bird/'forest')
```

    20:17:37.43 LOG:
    20:17:37.52 .... file = Path('forest_or_bird/bird/ebd12618-7183-42a5-be8c-d82eea6cb7f1.jpg')
    20:17:37.52 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1866859134.py", line 3
    20:17:37.52 ...... self = PILImage mode=RGB size=400x265
    20:17:37.52 ...... h = 128
    20:17:37.52 ...... w = None
    20:17:37.52    3 | def to_thumb(self:Image.Image, h, w=None):
    20:17:37.52    5 |     if w is None: 
    20:17:37.52    6 |         w=h
    20:17:37.52 .............. w = 128
    20:17:37.52    7 |     im = self.copy()
    20:17:37.52 .......... im = <PIL.Image.Image image mode=RGB size=400x265>
    20:17:37.52    8 |     im.thumbnail((w,h))
    20:17:37.52 .......... im = <PIL.Image.Image image mode=RGB size=128x85>
    20:17:37.52    9 |     return im
    20:17:37.52 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=128x85>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_81_1.png)
    



    20:17:37.53 LOG:
    20:17:37.53 .... file = Path('forest_or_bird/forest/a6d38e4e-43e7-4388-8466-e8667bf703f8.jpg')
    20:17:37.53 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1866859134.py", line 3
    20:17:37.53 ...... self = PILImage mode=RGB size=400x266
    20:17:37.53 ...... h = 128
    20:17:37.53 ...... w = None
    20:17:37.53    3 | def to_thumb(self:Image.Image, h, w=None):
    20:17:37.53    5 |     if w is None: 
    20:17:37.53    6 |         w=h
    20:17:37.53 .............. w = 128
    20:17:37.53    7 |     im = self.copy()
    20:17:37.53 .......... im = <PIL.Image.Image image mode=RGB size=400x266>
    20:17:37.53    8 |     im.thumbnail((w,h))
    20:17:37.53 .......... im = <PIL.Image.Image image mode=RGB size=128x85>
    20:17:37.53    9 |     return im
    20:17:37.53 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=128x85>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_81_3.png)
    




```
randomdisplay(dino/"Spinosaurus aegyptiacus")
```

    20:17:37.56 LOG:
    20:17:37.56 .... file = Path('T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus/c8bb46e0-ccca-4290-b2a0-1795f4b8f227.jpg')
    20:17:37.56 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1866859134.py", line 3
    20:17:37.56 ...... self = PILImage mode=RGB size=400x140
    20:17:37.56 ...... h = 128
    20:17:37.56 ...... w = None
    20:17:37.56    3 | def to_thumb(self:Image.Image, h, w=None):
    20:17:37.56    5 |     if w is None: 
    20:17:37.56    6 |         w=h
    20:17:37.56 .............. w = 128
    20:17:37.56    7 |     im = self.copy()
    20:17:37.56 .......... im = <PIL.Image.Image image mode=RGB size=400x140>
    20:17:37.56    8 |     im.thumbnail((w,h))
    20:17:37.56 .......... im = <PIL.Image.Image image mode=RGB size=128x45>
    20:17:37.56    9 |     return im
    20:17:37.57 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=128x45>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_82_1.png)
    




```
randomdisplay(dino/"T-rex")
```

    20:17:37.60 LOG:
    20:17:37.60 .... file = Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/348b2564-d1c0-4292-82ae-80b38bdab3ed.png')
    20:17:37.60 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1866859134.py", line 3
    20:17:37.60 ...... self = PILImage mode=RGB size=400x225
    20:17:37.60 ...... h = 128
    20:17:37.60 ...... w = None
    20:17:37.60    3 | def to_thumb(self:Image.Image, h, w=None):
    20:17:37.60    5 |     if w is None: 
    20:17:37.60    6 |         w=h
    20:17:37.60 .............. w = 128
    20:17:37.60    7 |     im = self.copy()
    20:17:37.60 .......... im = <PIL.Image.Image image mode=RGB size=400x225>
    20:17:37.60    8 |     im.thumbnail((w,h))
    20:17:37.60 .......... im = <PIL.Image.Image image mode=RGB size=128x72>
    20:17:37.60    9 |     return im
    20:17:37.60 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=128x72>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_83_1.png)
    




```
randomdisplay(cry_dino/"T-rex")
```

    20:17:37.63 LOG:
    20:17:37.63 .... file = Path('T-rex_or_Brachiosaurus/T-rex/929bc1df-7e7e-4300-9a49-4a048b9aabfc.jpg')
    20:17:37.63 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1866859134.py", line 3
    20:17:37.63 ...... self = PILImage mode=RGB size=400x206
    20:17:37.63 ...... h = 128
    20:17:37.63 ...... w = None
    20:17:37.63    3 | def to_thumb(self:Image.Image, h, w=None):
    20:17:37.63    5 |     if w is None: 
    20:17:37.63    6 |         w=h
    20:17:37.63 .............. w = 128
    20:17:37.63    7 |     im = self.copy()
    20:17:37.63 .......... im = <PIL.Image.Image image mode=RGB size=400x206>
    20:17:37.63    8 |     im.thumbnail((w,h))
    20:17:37.63 .......... im = <PIL.Image.Image image mode=RGB size=128x66>
    20:17:37.63    9 |     return im
    20:17:37.63 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=128x66>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_84_1.png)
    



### file_type, file_exts, and n_max in Path.ls
to list all the files inside the path

use file_type = 'binary' to find folders, images, videos; use file_type = 'text' to find `.py` and `.ipynb` files

use n_max=10 if you just want to list out 10 files or items


```
dino.ls(file_type="text")
dino.ls(file_type="binary")
```




    (#0) []






    (#5) [Path('T-rex_or_Spinosaurus aegyptiacus/.DS_Store'),Path('T-rex_or_Spinosaurus aegyptiacus/crying'),Path('T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex'),Path('T-rex_or_Spinosaurus aegyptiacus/fierce')]




```
(dino/"T-rex").ls(file_type='text')
(dino/"T-rex").ls(file_type='binary')
```




    (#0) []






    (#84) [Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/18db90d6-0eb4-4984-ac8a-5aa0b6aeacaa.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/d3439a0d-ce83-44cd-84f6-b4ae6fa77325.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/4893685a-1bb9-4682-aea1-8edcf29469ed.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/829089df-e243-4040-a737-1adb5bf97038.JPG'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/714950c2-022b-4491-9503-22e4310891fc.png'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/01fabe7c-fe23-4a69-b7e3-63ac58e4e9f7.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/04d675f3-addf-49ae-b4fe-894d35df12bb.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/fdafea87-586a-4c6f-a3af-e426f2aae178.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/8b6c67c9-8d82-48df-9ad7-59f8d14578f4.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/b0bdc9ee-0d95-4495-93db-80429f2ef388.jpg')...]




```

```


```

```

### check_subfolders_img(path)
check all subfolders for images and print out the number of images they have recursively


```

```


```
check_subfolders_img(dino)
check_subfolders_img(bird)
check_subfolders_img(cry_dino)
```

    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  crying
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  Spinosaurus aegyptiacus
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  fierce
    addup num: 330
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 170  forest
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 79  bird
    addup num: 249
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 167  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 109  Brachiosaurus
    addup num: 276



```
# list(dino.parent.absolute().ls())
dino.parent.absolute().parent
check_subfolders_img(dino.parent.absolute().parent)
```




    Path('/Users/Natsume/Documents/fastdebug/nbs')



    /Users/Natsume/Documents/fastdebug/nbs/_quarto.yml
    /Users/Natsume/Documents/fastdebug/nbs/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/sidebar.yml
    /Users/Natsume/Documents/fastdebug/nbs/styles.css
    /Users/Natsume/Documents/fastdebug/nbs/.last_checked
    /Users/Natsume/Documents/fastdebug/nbs/nbdev.yml
    /Users/Natsume/Documents/fastdebug/nbs/index.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/Interesting_fastai/0001_The_origin_of_APL .ipynb
    /Users/Natsume/Documents/fastdebug/nbs/Interesting_fastai/Interesting_things_fastai.ipynb
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/Math/math_0002_calculus.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/Math/math_0001_highschool.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/Math/sympy1.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/Math/line.png
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/lib/00_core.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/lib/_quarto.yml
    /Users/Natsume/Documents/fastdebug/nbs/lib/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/lib/sidebar.yml
    /Users/Natsume/Documents/fastdebug/nbs/lib/01_utils.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/lib/nbdev.yml
    /Users/Natsume/Documents/fastdebug/nbs/lib: 16  images
    /Users/Natsume/Documents/fastdebug/nbs/lib/data/mnist.pkl.gz
    addup num: 0
    addup num: 16
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0008_fastai_paddy_001.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0009_fastai_small_models_road_to_the_top_part_2.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/001_fastai_newletter_Radek.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0007_fastai_how_random_forests_really_work.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0001_fastai_is_it_a_bird.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0005_fastai_linear_neuralnet_scratch.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0003_fastai_which_image_model_best.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest.jpg
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0014_iterate_like_grandmaster.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/bird.jpg
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/kernel-metadata.json
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0006_fastai_why_should_use_framework.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/regex.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/00_fastai_Meta_learning_Radek.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/model.pkl
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0013_best_vision_models_for_fine_tuning.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0015_getting_started_with_nlp_for_absolute_beginner.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0017_fastai_pt2_2019_matmul.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0019_fastai_pt2_2019_lecture1_intro.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/fastai_links_forums_kaggle_github.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0000_fastai_kaggle_notebook.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/kaggle_notebook.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0012_fastai_using_nbdev_export_in_kaggle_notebook.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/00_fastai_how_to_follow_Radek.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0021_fastai_pt2_2019_fully_connected.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0010_fastai_scaling_up_road_to_top_part_3.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/001_newletter_Radek.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0022_fastai_pt2_2019_why_sqrt5.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0018_fastai_pt2_2019_exports.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0016_collaborative_filtering_deep_dive.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0011_fastai_multi_target_road_to_top_part_4.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0020_fastai_pt2_2019_source_explained.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0004_fastai_how_neuralnet_work.ipynb
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/resized/bird.jpg
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0005_fastai_linear_neuralnet_scratch_files/0005_fastai_linear_neuralnet_scratch_39_0.png
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0005_fastai_linear_neuralnet_scratch_files/0005_fastai_linear_neuralnet_scratch_141_0.png
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0005_fastai_linear_neuralnet_scratch_files/0005_fastai_linear_neuralnet_scratch_34_0.png
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/sample_submission.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification: 3469  test_images
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 1442  dead_heart
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 337  bacterial_panicle_blight
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 479  bacterial_leaf_blight
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 965  brown_spot
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 1594  hispa
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 620  downy_mildew
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 1738  blast
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 1764  normal
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 380  bacterial_leaf_streak
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train_images: 1088  tungro
    addup num: 10407
    addup num: 3469
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/distributed_train.sh
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/mkdocs.yml
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/benchmark.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/LICENSE
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/requirements.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/avg_checkpoints.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/requirements-docs.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/MANIFEST.in
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/README.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/validate.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/setup.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/.gitignore
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/.gitattributes
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/train.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/bulk_runner.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/inference.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/setup.cfg
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hubconf.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/model-index.yml
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/requirements-modelindex.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/clean_checkpoint.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/tests/test_utils.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/tests/test_layers.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/tests/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/tests/test_optim.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/tests/test_models.py
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/scripts.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/results.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/index.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/archived_changes.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/changes.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/training_hparam_examples.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/feature_extraction.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/vision-transformer.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/ssl-resnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/ssl-resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/gloun-inception-v3.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/fbnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/ensemble-adversarial.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/advprop.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/gloun-resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/tf-mixnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/noisy-student.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/swsl-resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/resnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/inception-resnet-v2.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/legacy-se-resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/.pages
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/ecaresnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/skresnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/tf-efficientnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/pnasnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/xception.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/nasnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/ese-vovnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/tf-efficientnet-lite.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/wide-resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/regnetx.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/gloun-seresnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/resnet-d.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/ig-resnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/inception-v3.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/se-resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/regnety.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/gloun-senet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/skresnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/tf-mobilenet-v3.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/dpn.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/gloun-xception.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/tf-efficientnet-condconv.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/tf-inception-v3.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/swsl-resnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/inception-v4.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/tresnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/mobilenet-v3.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/spnasnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/efficientnet-pruned.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/mobilenet-v2.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/seresnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/hrnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/gloun-resnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/mnasnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/adversarial-inception-v3.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/rexnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/selecsls.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/csp-darknet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/res2net.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/legacy-senet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/big-transfer.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/csp-resnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/dla.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/efficientnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/legacy-se-resnext.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/res2next.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/csp-resnet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/densenet.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/resnest.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/models/mixnet.md
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/docs/javascripts/tables.js
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-infer-amp-nchw-pt111-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-train-amp-nhwc-pt111-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/generate_csv_results.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-imagenet-real.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-imagenet-r.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/imagenet_r_synsets.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-imagenet-a-clean.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/imagenet21k_goog_synsets.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-imagenet-a.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/imagenet_synsets.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-infer-amp-nchw-pt112-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/imagenet_a_synsets.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-imagenet-r-clean.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-train-amp-nhwc-pt112-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/README.md
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-imagenetv2-matched-frequency.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/model_metadata-in1k.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/imagenet_r_indices.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-train-amp-nchw-pt111-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-infer-amp-nhwc-pt111-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/imagenet_a_indices.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/benchmark-infer-amp-nhwc-pt112-cu113-rtx3090.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/imagenet_real_labels.json
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-imagenet.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/results/results-sketch.csv
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/version.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/loss/cross_entropy.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/loss/asymmetric_loss.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/loss/jsd.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/loss/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/loss/binary_cross_entropy.py
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/tanh_lr.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/plateau_lr.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/poly_lr.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/step_lr.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/scheduler.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/cosine_lr.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/multistep_lr.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/scheduler/scheduler_factory.py
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/lamb.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/adan.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/adafactor.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/lars.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/nvnovograd.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/adamp.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/adabelief.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/adahessian.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/adamw.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/radam.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/nadam.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/rmsprop_tf.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/lookahead.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/sgdp.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/madgrad.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/optim/optim_factory.py
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/metrics.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/misc.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/jit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/log.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/model_ema.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/random.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/model.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/summary.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/distributed.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/checkpoint_saver.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/cuda.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/agc.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/decay_batch.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/utils/clip_grad.py
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/deit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/ghostnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/volo.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/selecsls.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/rexnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/efficientnet_blocks.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/inception_resnet_v2.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/edgenext.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/hrnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/vgg.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/byoanet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/convmixer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/nest.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/cspnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/hardcorenas.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/beit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/efficientnet_builder.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/vision_transformer_relpos.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/xcit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/regnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/sequencer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/mvitv2.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/resnetv2.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/vision_transformer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/xception_aligned.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/tresnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/gluon_resnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/senet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/registry.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/gluon_xception.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/efficientformer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/twins.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/byobnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/resnest.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/densenet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/levit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/mobilevit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/vovnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/dla.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/inception_v4.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/maxxvit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/tnt.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/features.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/res2net.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/poolformer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/factory.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/hub.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/efficientnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pnasnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/inception_v3.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/resnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/swin_transformer_v2.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/vision_transformer_hybrid.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/sknet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pvt_v2.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/mobilenetv3.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/fx_features.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/visformer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/gcvit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/swin_transformer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/dpn.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/crossvit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/helpers.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/swin_transformer_v2_cr.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/xception.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/nasnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/cait.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/nfnet.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/convnext.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/coat.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/mlp_mixer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/convit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/ml_decoder.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/std_conv.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/squeeze_excite.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/conv2d_same.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/classifier.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/config.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/mixed_conv2d.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/attention_pool2d.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/space_to_depth.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/create_act.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/activations_me.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/activations.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/create_norm.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/linear.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/separable_conv.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/trace_utils.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/norm.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/gather_excite.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/create_conv2d.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/non_local_attn.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/pool2d_same.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/bottleneck_attn.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/lambda_layer.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/fast_norm.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/mlp.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/adaptive_avgmax_pool.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/inplace_abn.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/pos_embed.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/conv_bn_act.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/halo_attn.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/evo_norm.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/split_batchnorm.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/create_norm_act.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/filter_response_norm.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/patch_embed.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/blur_pool.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/eca.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/global_context.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/median_pool.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/split_attn.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/selective_kernel.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/helpers.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/activations_jit.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/create_attn.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/padding.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/cbam.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/cond_conv2d.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/drop.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/weight_init.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/test_time_pool.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/layers/norm_act.py
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pruned/efficientnet_b3_pruned.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pruned/efficientnet_b1_pruned.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pruned/ecaresnet101d_pruned.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pruned/efficientnet_b2_pruned.txt
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/models/pruned/ecaresnet50d_pruned.txt
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/auto_augment.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/transforms.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/real_labels.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/config.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/mixup.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/transforms_factory.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/constants.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/tf_preprocessing.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/dataset_factory.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/dataset.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/loader.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/random_erasing.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/distributed_sampler.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader_image_tar.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/img_extensions.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/__init__.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader_wds.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader_tfds.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader_image_in_tar.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/shared_count.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader_factory.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/class_map.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader_hfds.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/timm/data/readers/reader_image_folder.py
    addup num: 0
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/archived_changes.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/scripts.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/results.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/index.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/_toctree.yml
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/model_pages.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/feature_extraction.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/changes.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/training_hparam_examples.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/swsl-resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/wide-resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/gloun-inception-v3.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/csp-resnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/gloun-senet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/swsl-resnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/hrnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/mobilenet-v3.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/resnest.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/mobilenet-v2.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/legacy-se-resnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/gloun-resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/tf-efficientnet-condconv.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/ensemble-adversarial.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/csp-resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/tf-efficientnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/mnasnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/resnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/densenet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/xception.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/inception-v4.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/legacy-se-resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/adversarial-inception-v3.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/tresnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/seresnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/spnasnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/res2net.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/rexnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/gloun-resnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/inception-resnet-v2.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/inception-v3.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/ssl-resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/regnety.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/tf-mixnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/skresnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/se-resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/noisy-student.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/regnetx.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/gloun-xception.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/efficientnet-pruned.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/gloun-seresnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/dla.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/big-transfer.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/tf-efficientnet-lite.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/resnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/tf-mobilenet-v3.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/ecaresnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/mixnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/pnasnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/csp-darknet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/selecsls.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/advprop.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/fbnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/ese-vovnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/efficientnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/res2next.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/tf-inception-v3.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/nasnet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/legacy-senet.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/resnet-d.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/ig-resnext.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/dpn.mdx
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/hfdocs/source/models/skresnet.mdx
    addup num: 0
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/convert/convert_nest_flax.py
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/pytorch-image-models/convert/convert_from_mxnet.py
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/models/paddy_10pct_resnet26d_10epochs.pkl
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/models/tmpjvn9jd5h/_tmp.pth
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  crying
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  Spinosaurus aegyptiacus
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  fierce
    addup num: 330
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/try_resize_images/0b8fcba5-91a5-4689-999c-008e108828f1.jpg
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/try_resize_images/037e9e61-3731-4876-9745-98758ae21be3.jpg
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/try_resize_images/forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/try_resize_images/bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 167  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 109  Brachiosaurus
    addup num: 276
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/bird_or_not/0b8fcba5-91a5-4689-999c-008e108828f1.jpg
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/bird_or_not/037e9e61-3731-4876-9745-98758ae21be3.jpg
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/bird_or_not/forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/bird_or_not/bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg
    addup num: 0
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 170  forest
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 79  bird
    addup num: 249
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_35_3.png
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_28_0.png
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_22_1.png
    addup num: 0
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks: 6  0002_fastai_saving_a_basic_fastai_model_files
    addup num: 6
    addup num: 0


### verify_image(fn)
Confirm that `fn` can be opened


```
# @snoop
def verify_image(fn):
    "Confirm that `fn` can be opened"
    try: # if any of the following three lines cause error, we consider it as can't be opened
        im = Image.open(fn)
        im.draft(im.mode, (32,32))
        im.load()
        return True
    except: return False
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```


```
def verify_image(fn):
    "Confirm that `fn` can be opened"
    try:
        im = Image.open(fn)
        im.draft(im.mode, (32,32))
        im.load()
        return True
    except: return False
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```


```
L(1,2,3), L([1,2,3])
```




    ((#3) [1,2,3], (#3) [1,2,3])



### verify_images(fns)
Find images in `fns` that can't be opened. using parallel, to applies `verify_image` in parallel to `fns`, using `n_workers=8`"


```
#| export utils
# @snoop
def verify_images(fns):
    "Find images in `fns` that can't be opened. using parallel, to applies `verify_image` in parallel to `fns`, using `n_workers=8`"
#     return L(fns[i] for i,o in enumerate(parallel(verify_image, fns)) if not o)
    lst = []
    for i,o in enumerate(parallel(verify_image, fns)):
        if not o:
            lst.append(fns[i])
    return L(lst)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```


```
#| export utils
def verify_images(fns):
    "Find images in `fns` that can't be opened"
    return L(fns[i] for i,o in enumerate(parallel(verify_image, fns)) if not o)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/utils.py
# Type:      function
```


```
doc(verify_images)
```


<hr/>
<h3>verify_images</h3>
<blockquote><pre><code>verify_images(fns)</code></pre></blockquote><p>Find images in `fns` that can't be opened</p>
<p><a href="https://EmbraceLife.github.io/fastdebug/fastai_notebooks/fastai_is_it_a_bird.html#verify_images" target="_blank" rel="noreferrer noopener">Show in docs</a></p>


### remove_failed(path)
find all images inside a path which can't be opened and unlink them

Some photos might not download correctly which could cause our model training to fail, so we'll remove them:


```
# #| export utils 
# from fastai.vision.all import *
```


```
# #| export utils
# def remove_failed(path):
# #     from fastai.vision.all import get_image_files, parallel
#     print("before running remove_failed:")
#     check_subfolders_img(path)
#     failed = verify_images(get_image_files(path))
#     print(f"total num: {len(get_image_files(path))}")
#     print(f"num offailed: {len(failed)}")
#     failed.map(Path.unlink)
#     print()
#     print("after running remove_failed:")
#     check_subfolders_img(path)
```


```
remove_failed(dino)
```

    before running remove_failed:
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  crying
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  Spinosaurus aegyptiacus
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  fierce
    addup num: 330
    total num: 331
    num offailed: 0
    
    after running remove_failed:
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  crying
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  Spinosaurus aegyptiacus
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  fierce
    addup num: 330



```
remove_failed(bird)
remove_failed(cry_dino)
```

    before running remove_failed:
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 170  forest
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 79  bird
    addup num: 249
    total num: 251
    num offailed: 0
    
    after running remove_failed:
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 170  forest
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 79  bird
    addup num: 249
    before running remove_failed:
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 167  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 109  Brachiosaurus
    addup num: 276
    total num: 276
    num offailed: 0
    
    after running remove_failed:
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 167  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Brachiosaurus: 109  Brachiosaurus
    addup num: 276



```
# dino.name
# dino.root
# dino.home()
# dino.is_dir()
# dino.is_file()
# dino.exists() 
# dino.is_absolute()
# dino.absolute()
# # dino.BASE_PATH
```


```
# fastnbs("path.ls file_type")
```


```
cry_dino.ls()[0]
```




    Path('T-rex_or_Brachiosaurus/.DS_Store')




```
randomdisplay(cry_dino.ls()[1])
```

    20:17:42.81 LOG:
    20:17:42.82 .... file = Path('T-rex_or_Brachiosaurus/T-rex/08972b48-e805-426c-bea4-abb58647f2f2.png')
    20:17:42.82 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1866859134.py", line 3
    20:17:42.82 ...... self = PILImage mode=RGB size=400x278
    20:17:42.82 ...... h = 128
    20:17:42.82 ...... w = None
    20:17:42.82    3 | def to_thumb(self:Image.Image, h, w=None):
    20:17:42.82    5 |     if w is None: 
    20:17:42.82    6 |         w=h
    20:17:42.82 .............. w = 128
    20:17:42.82    7 |     im = self.copy()
    20:17:42.82 .......... im = <PIL.Image.Image image mode=RGB size=400x278>
    20:17:42.82    8 |     im.thumbnail((w,h))
    20:17:42.82 .......... im = <PIL.Image.Image image mode=RGB size=128x89>
    20:17:42.82    9 |     return im
    20:17:42.82 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=128x89>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_111_1.png)
    




```
randomdisplay(cry_dino.ls()[2])
```

    20:17:42.84 LOG:
    20:17:42.85 .... file = Path('T-rex_or_Brachiosaurus/Brachiosaurus/bddb0096-aae3-4771-9e8f-acca907b06f2.jpg')
    20:17:42.85 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1866859134.py", line 3
    20:17:42.85 ...... self = PILImage mode=RGB size=400x400
    20:17:42.85 ...... h = 128
    20:17:42.85 ...... w = None
    20:17:42.85    3 | def to_thumb(self:Image.Image, h, w=None):
    20:17:42.85    5 |     if w is None: 
    20:17:42.85    6 |         w=h
    20:17:42.85 .............. w = 128
    20:17:42.85    7 |     im = self.copy()
    20:17:42.85 .......... im = <PIL.Image.Image image mode=RGB size=400x400>
    20:17:42.85    8 |     im.thumbnail((w,h))
    20:17:42.85 .......... im = <PIL.Image.Image image mode=RGB size=128x128>
    20:17:42.85    9 |     return im
    20:17:42.85 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=128x128>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_112_1.png)
    




```

```

### Path.unlink(self, missing_ok=False)
Remove this file or link.

If the path is a directory, use rmdir() instead.

If a file is not found, error will be raised


## How to create a DataLoaders with DataBlock and display a batch
To train a model, we'll need DataLoaders:     

1) a training set (the images used to create a model) and 

2) a validation set (the images used to check the accuracy of a model -- not used during training). 

We can view sample images from it:

### get_image_files, get_files, image_extensions
to extract all image files recursively from all subfolders of a parent path


```
from fastai.data.transforms import _get_files
```


```
not None
```




    True




```
def _get_files(p, # path
               fs, # list of filenames
               extensions=None):
    "get the fullnames for the list of filenames of a path"
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```


```
def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders=L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```


```
from fastai.vision.all import *
```


```
len(image_extensions)
".aspx" in image_extensions
```




    65






    False




```
def get_image_files(path, recurse=True, folders=None):
    "Get image files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=image_extensions, recurse=recurse, folders=folders)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```


```
check_subfolders_img(dino)
get_image_files(dino)
check_subfolders_img(bird)
get_image_files(bird)
```

    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  crying
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  Spinosaurus aegyptiacus
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 83  T-rex
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus: 82  fierce
    addup num: 330





    (#331) [Path('T-rex_or_Spinosaurus aegyptiacus/crying/c4016dd8-bde7-4dd4-b2b0-a9a4514ac834.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/20591e18-54dd-49e7-b00c-722dbd8872ab.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/553f845b-c5d9-4d10-94ac-9230da42b866.png'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/6c8a2034-8d0c-4935-a5af-7cb73dd75b63.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/481399dc-7dc5-4f20-b651-bf29e1085611.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/d19616c2-bf32-448a-879a-b409732b1cdb.png'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/2a4e59cd-511f-4a70-a5e4-26998160031a.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/9d2c2803-1403-4e0f-ab37-b16015d506db.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/c176c0f3-c4b5-42e6-ab52-5a4fc8b4d766.png'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/b806226d-287c-46ec-9a0c-882ea3be438c.jpg')...]



    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird/.DS_Store
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 170  forest
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird: 79  bird
    addup num: 249





    (#251) [Path('forest_or_bird/forest/82e9179d-2dd6-4144-8f66-891533ec6467.jpg'),Path('forest_or_bird/forest/6e84b01f-c9a3-466b-b0eb-527ea360d517.jpg'),Path('forest_or_bird/forest/c929266f-7495-434f-bd07-0b1a1d4a9051.jpg'),Path('forest_or_bird/forest/3601c5f2-dc4a-4e56-ada4-58943c0190b3.jpg'),Path('forest_or_bird/forest/eec11827-0f51-429d-84f0-84b3c24c2e1b.JPG'),Path('forest_or_bird/forest/f1742fe6-770d-45b7-a14f-d35ec8514704.jpg'),Path('forest_or_bird/forest/4dd1840a-1a62-43b0-acdc-de476c881e4a.jpg'),Path('forest_or_bird/forest/beb09236-e6d9-49eb-a522-f8c2e9d3ffb3.jpeg'),Path('forest_or_bird/forest/ea756b81-0e41-4554-830b-858fef1e2275.jpg'),Path('forest_or_bird/forest/16ffe691-aa2d-493e-8f1c-1dea9db419e4.jpg')...]



### check the sizes of all images


```
check_sizes_img(get_image_files(dino))
```

    20:17:43.37 LOG:
    20:17:43.40 .... pd.Series(sizes).value_counts() = (400, 400)    69
    20:17:43.40                                        (400, 225)    35
    20:17:43.40                                        (400, 266)    23
    20:17:43.40                                        (400, 300)    19
    20:17:43.40                                        (266, 400)     6
    20:17:43.40                                                      ..
    20:17:43.40                                        (400, 239)     1
    20:17:43.40                                        (400, 156)     1
    20:17:43.40                                        (400, 229)     1
    20:17:43.40                                        (400, 258)     1
    20:17:43.40                                        (400, 271)     1
    20:17:43.40                                        Length: 136, dtype: int64
    20:17:43.40 LOG:
    20:17:43.41 .... imgs = []





    []




```
Image.open(Path('forest_or_bird/bird/3b7a4112-9d77-4d8f-8b4c-a01e2ca1ecea.aspx'))
```




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_127_0.png)
    



### parent_label(o)
extract the label from the filename's parent folder name


```
def parent_label(o):
    "Label `item` with the parent folder name."
    return Path(o).parent.name
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```


```
# dino.ls()[0].ls()[0]
```


```
parent_label(dino.ls()[1].ls()[0])
parent_label(dino.ls()[2].ls()[0])
```




    'crying'






    'Spinosaurus aegyptiacus'



### RandomSplitter(valid_pct=0.2, seed=None), torch.linspace(0,1,100)
Create function that splits `items` between train/val with `valid_pct` randomly.

with seed=42, the random splits can be reproduced


```
def RandomSplitter(valid_pct=0.2, seed=None):
    "Create function that splits `items` between train/val with `valid_pct` randomly."
    def _inner(o):
        if seed is not None: torch.manual_seed(seed)
        rand_idx = L(list(torch.randperm(len(o)).numpy()))
        cut = int(valid_pct * len(o))
        return rand_idx[cut:],rand_idx[:cut]
    return _inner
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```


```
x = torch.linspace(0, 1, 100)
```


```
rs = RandomSplitter()
rs(x)
```




    ((#80) [27,81,90,92,13,98,79,65,44,22...],
     (#20) [91,82,86,60,34,14,1,47,97,69...])




```
rs = RandomSplitter(seed=42)
rs(x)
```




    ((#80) [89,86,18,40,5,38,9,82,83,43...],
     (#20) [42,96,62,98,46,95,60,24,78,16...])



### Resize(RandTransform)
Resize image to `size` using `method` such as 'crop', 'squish' and 'pad'

What does `crop`, `squish` and `pad` resize effects look like

`rsz = Resize(256, method='crop')` returns a func and use it to actually resize image as below

`rsz(img, split_idx=0)`


```
doc(Resize)
```


<hr/>
<h3>Resize</h3>
<blockquote><pre><code>Resize(size:int|tuple, method:ResizeMethod='crop', pad_mode:PadMode='reflection', resamples=(<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), **kwargs)</code></pre></blockquote><p>A transform that before_call its state at each `__call__`</p>
<p><a href="https://docs.fast.ai/vision.augment.html#resize" target="_blank" rel="noreferrer noopener">Show in docs</a></p>



```
ResizeMethod.Crop
ResizeMethod.Squish
ResizeMethod.Pad
```




    'crop'






    'squish'






    'pad'




```
Resize(192, method='squish')
```




    Resize -- {'size': (192, 192), 'method': 'squish', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0}:
    encodes: (Image,object) -> encodes
    (TensorBBox,object) -> encodes
    (TensorPoint,object) -> encodes
    decodes: 




```
Resize(220, method="crop")
```




    Resize -- {'size': (220, 220), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0}:
    encodes: (Image,object) -> encodes
    (TensorBBox,object) -> encodes
    (TensorPoint,object) -> encodes
    decodes: 




```
dino_img = Path('T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus/07b15395-ce8d-4250-a536-920159dfe57d.png')
img = PILImage(PILImage.create(dino_img).resize((600,400)))
show_image(img)
```




    <AxesSubplot:>




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_142_1.png)
    



```
rsz = Resize(256, method='crop')
rsz(img, split_idx=0)
```




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_143_0.png)
    




```
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):
    rsz = Resize(256, method=method)
    show_image(rsz(img, split_idx=0), ctx=ax, title=method);
```




    <AxesSubplot:title={'center':'squish'}>






    <AxesSubplot:title={'center':'pad'}>






    <AxesSubplot:title={'center':'crop'}>




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_144_3.png)
    



```
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):
    rsz = Resize(256, method=method)
    show_image(rsz(img, split_idx=1), ctx=ax, title=method);
```




    <AxesSubplot:title={'center':'squish'}>






    <AxesSubplot:title={'center':'pad'}>






    <AxesSubplot:title={'center':'crop'}>




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_145_3.png)
    


### Resize inherited from RandTransform...Transform
Read the comments I made in the 3 sources to get a basic understanding of what they do

`Resize` class prepares args, `before_call` func, `encodes` func which is to crop and pad images

`RandomTransform` class prepares args, define `before_call` func for the first time in history, override `__call__` so that `before_call` is called first before `Transform.__call__` is called

`Transform` class prepares all args (most original), defines `decodes` func for the first time and override `__call__` inherit from `TfmMeta`


```
from __future__ import annotations
from fastai.vision.augment import _process_sz, _get_sz
```


```
from fastcore.transform import _TfmMeta, _tfm_methods, _is_tfm_method, _get_name, _TfmDict
```


```
Resize.mro()
```




    [fastai.vision.augment.Resize,
     fastai.vision.augment.RandTransform,
     fastcore.transform.DisplayedTransform,
     fastcore.transform.Transform,
     object]




```
@delegates()
class Resize(RandTransform):
    split_idx,mode,mode_mask,order = None,BILINEAR,NEAREST,1
    "Resize image to `size` using `method`"
#     @snoop
    def __init__(self, 
        size:int|tuple, # Size to resize to, duplicated if one value is specified
        method:ResizeMethod=ResizeMethod.Crop, # A `ResizeMethod`
        pad_mode:PadMode=PadMode.Reflection, # A `PadMode`
        resamples=(BILINEAR, NEAREST), # Pillow `Image` resamples mode, resamples[1] for mask
        **kwargs
    ):
        "set up size into a tuple, put all attrs into self, run initialization from RandTransform, set up mode and mode_mask"
        size = _process_sz(size)
        store_attr()
#         pp(self)
        super().__init__(**kwargs)
        self.mode,self.mode_mask = resamples

    def before_call(self, 
        b, 
        split_idx:int # Index of the train/valid dataset
    ):
        "override before_call from RandomTransform.before_call"
        if self.method==ResizeMethod.Squish: return
        self.pcts = (0.5,0.5) if split_idx else (random.random(),random.random())

#     @snoop
    def encodes(self, x:Image.Image|TensorBBox|TensorPoint):
        "Preparing all the args for running x.crop_pad function"
        orig_sz = _get_sz(x)
        if self.method==ResizeMethod.Squish:
            return x.crop_pad(orig_sz, fastuple(0,0), orig_sz=orig_sz, pad_mode=self.pad_mode,
                   resize_mode=self.mode_mask if isinstance(x,PILMask) else self.mode, resize_to=self.size)

        w,h = orig_sz
        op = (operator.lt,operator.gt)[self.method==ResizeMethod.Pad]
        m = w/self.size[0] if op(w/self.size[0],h/self.size[1]) else h/self.size[1]
        cp_sz = (int(m*self.size[0]),int(m*self.size[1]))
        tl = fastuple(int(self.pcts[0]*(w-cp_sz[0])), int(self.pcts[1]*(h-cp_sz[1])))
        return x.crop_pad(cp_sz, tl, orig_sz=orig_sz, pad_mode=self.pad_mode,
                   resize_mode=self.mode_mask if isinstance(x,PILMask) else self.mode, resize_to=self.size)

### what inside x.crop_pad from the last line of code above?
# im._do_crop_pad?? # use x.crop and tvpad on x, finally run x.resize
# img.crop_pad?? # prepare all args for running im._do_crop_pad

# File:           ~/mambaforge/lib/python3.9/site-packages/fastai/vision/augment.py
```


```
class RandTransform(DisplayedTransform):
    "A transform that before_call its state at each `__call__`"
    do,nm,supports,split_idx = True,None,[],0
    def __init__(self, 
        p:float=1., # Probability of applying Transform
        nm:str=None,
        before_call:callable=None, # Optional batchwise preprocessing function
        **kwargs
    ):
        "prepare attr p, run Transform.__init__ to prepare other args, and prepare `before_call` func"
        store_attr('p')
        super().__init__(**kwargs)
        self.before_call = ifnone(before_call,self.before_call)

    def before_call(self, 
        b, 
        split_idx:int, # Index of the train/valid dataset
    ):
        "Define the before_call function. This function can be overridden. Set `self.do` based on `self.p`"
        self.do = self.p==1. or random.random() < self.p

    def __call__(self, 
        b, 
        split_idx:int=None, # Index of the train/valid dataset
        **kwargs
    ):
        "override Transform.__call__ by calling self.before_call before Transform.__call__"
        self.before_call(b, split_idx=split_idx)
        return super().__call__(b, split_idx=split_idx, **kwargs) if self.do else b
# File:           ~/mambaforge/lib/python3.9/site-packages/fastai/vision/augment.py
# Type:           _TfmMeta
# Subclasses:     FlipItem, DihedralItem, RandomCrop, Resize, RandomResizedCrop, AffineCoordTfm, RandomResizedCropGPU, SpaceTfm, RandomErasing, Resize, ...

```


```
class Transform(metaclass=_TfmMeta):
    "Delegates (`__call__`,`decode`,`setup`) to (<code>encodes</code>,<code>decodes</code>,<code>setups</code>) if `split_idx` matches"
    split_idx,init_enc,order,train_setup = None,None,0,None
#     @snoop
    def __init__(self, enc=None, dec=None, split_idx=None, order=None):
        "prepares self.split_idx, self.order, self.init_enc, self.encodes, self.decodes"
        self.split_idx = ifnone(split_idx, self.split_idx)
        if order is not None: self.order=order
        self.init_enc = enc or dec
        if not self.init_enc: return

        self.encodes,self.decodes,self.setups = TypeDispatch(),TypeDispatch(),TypeDispatch()
        if enc:
            self.encodes.add(enc)
            self.order = getattr(enc,'order',self.order)
            if len(type_hints(enc)) > 0: self.input_types = union2tuple(first(type_hints(enc).values()))
            self._name = _get_name(enc)
        if dec: self.decodes.add(dec)

    @property
    def name(self): return getattr(self, '_name', _get_name(self))
    def __call__(self, x, **kwargs): 
        "run self.encodes(x)"
        return self._call('encodes', x, **kwargs)
    def decode  (self, x, **kwargs): 
        "run self.decodes(x)"
        return self._call('decodes', x, **kwargs)
    def __repr__(self): return f'{self.name}:\nencodes: {self.encodes}decodes: {self.decodes}'

    def setup(self, items=None, train_setup=False):
        train_setup = train_setup if self.train_setup is None else self.train_setup
        return self.setups(getattr(items, 'train', items) if train_setup else items)

    def _call(self, fn, x, split_idx=None, **kwargs):
        "run fn(x) or return x based on split_idx"
        if split_idx!=self.split_idx and self.split_idx is not None: return x
        return self._do_call(getattr(self, fn), x, **kwargs)

    def _do_call(self, f, x, **kwargs):
        "run f(x) and make result the same type as x"
        if not _is_tuple(x):
            if f is None: return x
            ret = f.returns(x) if hasattr(f,'returns') else None
            return retain_type(f(x, **kwargs), x, ret)
        res = tuple(self._do_call(f, x_, **kwargs) for x_ in x)
        return retain_type(res, x)
# File:           ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py
# Type:           _TfmMeta
# Subclasses:     InplaceTransform, DisplayedTransform, ItemTransform, ToTensor, AddMaskCodes, PointScaler, BBoxLabeler
```


```
Transform.__dict__
```




    mappingproxy({'__module__': '__main__',
                  '__doc__': 'Delegates (`__call__`,`decode`,`setup`) to (<code>encodes</code>,<code>decodes</code>,<code>setups</code>) if `split_idx` matches',
                  'split_idx': None,
                  'init_enc': None,
                  'order': 0,
                  'train_setup': None,
                  '__init__': <function __main__.Transform.__init__(self, enc=None, dec=None, split_idx=None, order=None)>,
                  'name': <property>,
                  '__call__': <function __main__.Transform.__call__(self, x, **kwargs)>,
                  'decode': <function __main__.Transform.decode(self, x, **kwargs)>,
                  '__repr__': <function __main__.Transform.__repr__(self)>,
                  'setup': <function __main__.Transform.setup(self, items=None, train_setup=False)>,
                  '_call': <function __main__.Transform._call(self, fn, x, split_idx=None, **kwargs)>,
                  '_do_call': <function __main__.Transform._do_call(self, f, x, **kwargs)>,
                  '__dict__': <attribute '__dict__' of 'Transform' objects>,
                  '__weakref__': <attribute '__weakref__' of 'Transform' objects>,
                  'encodes': ,
                  'decodes': ,
                  'setups': ,
                  '__signature__': <Signature (self, enc=None, dec=None, split_idx=None, order=None)>})




```
_tfm_methods
```




    ('encodes', 'decodes', 'setups')



### how Transform created by TfmMeta
see how TfmMeta is at work when Transform is created as a class


```
class _TfmMeta(type):
    @snoop
    def __new__(cls, name, bases, dict):
        res = super().__new__(cls, name, bases, dict)
        for nm in _tfm_methods:
            base_td = [getattr(b,nm,None) for b in bases]
            if nm in res.__dict__: getattr(res,nm).bases = base_td
            else: setattr(res, nm, TypeDispatch(bases=base_td))
        # _TfmMeta.__call__ shadows the signature of inheriting classes, set it back
        res.__signature__ = inspect.signature(res.__init__)
        return res


    def __call__(cls, *args, **kwargs):
        "if no tfm_method given, just return type.__call__ as everyone else; otherwise, store the tfm func in Transform class and return it"
        f = first(args)
        n = getattr(f, '__name__', None)
        if _is_tfm_method(n, f):
            getattr(cls,n).add(f)
            return f
        obj = super().__call__(*args, **kwargs)
        # _TfmMeta.__new__ replaces cls.__signature__ which breaks the signature of a callable
        # instances of cls, fix it
        if hasattr(obj, '__call__'): obj.__signature__ = inspect.signature(obj.__call__)
        return obj

    @classmethod
    def __prepare__(cls, name, bases): return _TfmDict()
# File:           ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py
# Type:           type
# Subclasses:     
```


```
class Transform(metaclass=_TfmMeta):pass
```

    20:17:44.73 >>> Call to _TfmMeta.__new__ in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/2173750754.py", line 3
    20:17:44.73 .......... cls = <class '__main__._TfmMeta'>
    20:17:44.73 .......... name = 'Transform'
    20:17:44.73 .......... bases = ()
    20:17:44.73 .......... dict = {'__module__': '__main__', '__qualname__': 'Transform'}
    20:17:44.73 .......... __class__ = <class '__main__._TfmMeta'>
    20:17:44.73    3 |     def __new__(cls, name, bases, dict):
    20:17:44.73    4 |         res = super().__new__(cls, name, bases, dict)
    20:17:44.73 .............. res = <class '__main__.Transform'>
    20:17:44.73    5 |         for nm in _tfm_methods:
    20:17:44.74 .............. nm = 'encodes'
    20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
        20:17:44.74 List comprehension:
        20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
        20:17:44.74 .................. Iterating over <tuple_iterator object>
        20:17:44.74 .................. Values of nm: 'encodes'
        20:17:44.74 Result: []
    20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
    20:17:44.74 .................. base_td = []
    20:17:44.74    7 |             if nm in res.__dict__: getattr(res,nm).bases = base_td
    20:17:44.74    8 |             else: setattr(res, nm, TypeDispatch(bases=base_td))
    20:17:44.74    5 |         for nm in _tfm_methods:
    20:17:44.74 .............. nm = 'decodes'
    20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
        20:17:44.74 List comprehension:
        20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
        20:17:44.74 .................. Iterating over <tuple_iterator object>
        20:17:44.74 .................. Values of nm: 'decodes'
        20:17:44.74 Result: []
    20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
    20:17:44.74    7 |             if nm in res.__dict__: getattr(res,nm).bases = base_td
    20:17:44.74    8 |             else: setattr(res, nm, TypeDispatch(bases=base_td))
    20:17:44.74    5 |         for nm in _tfm_methods:
    20:17:44.74 .............. nm = 'setups'
    20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
        20:17:44.74 List comprehension:
        20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
        20:17:44.74 .................. Iterating over <tuple_iterator object>
        20:17:44.74 .................. Values of nm: 'setups'
        20:17:44.74 Result: []
    20:17:44.74    6 |             base_td = [getattr(b,nm,None) for b in bases]
    20:17:44.74    7 |             if nm in res.__dict__: getattr(res,nm).bases = base_td
    20:17:44.74    8 |             else: setattr(res, nm, TypeDispatch(bases=base_td))
    20:17:44.74    5 |         for nm in _tfm_methods:
    20:17:44.74   10 |         res.__signature__ = inspect.signature(res.__init__)
    20:17:44.74   11 |         return res
    20:17:44.74 <<< Return value from _TfmMeta.__new__: <class '__main__.Transform'>



```
Transform.__dict__
```




    mappingproxy({'__module__': '__main__',
                  '__dict__': <attribute '__dict__' of 'Transform' objects>,
                  '__weakref__': <attribute '__weakref__' of 'Transform' objects>,
                  '__doc__': None,
                  'encodes': ,
                  'decodes': ,
                  'setups': ,
                  '__signature__': <Signature (self, /, *args, **kwargs)>})




```
from fastcore.transform import _TfmMeta
# _TfmMeta??
```


```
class _TfmMeta(type):
    def __new__(cls, name, bases, dict):
        res = super().__new__(cls, name, bases, dict)
        for nm in _tfm_methods:
            base_td = [getattr(b,nm,None) for b in bases]
            if nm in res.__dict__: getattr(res,nm).bases = base_td
            else: setattr(res, nm, TypeDispatch(bases=base_td))
        # _TfmMeta.__call__ shadows the signature of inheriting classes, set it back
        res.__signature__ = inspect.signature(res.__init__)
        return res

    def __call__(cls, *args, **kwargs):
        f = first(args)
        n = getattr(f, '__name__', None)
        if _is_tfm_method(n, f):
            getattr(cls,n).add(f)
            return f
        obj = super().__call__(*args, **kwargs)
        # _TfmMeta.__new__ replaces cls.__signature__ which breaks the signature of a callable
        # instances of cls, fix it
        if hasattr(obj, '__call__'): obj.__signature__ = inspect.signature(obj.__call__)
        return obj

    @classmethod
    def __prepare__(cls, name, bases): return _TfmDict()
# File:           ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py
# Type:           type
```

### ImageBlock(cls:PILBase=PILImage) and TransformBlock()
to create an instance of `TransformBlock`, which is passed to DataBlock and dataloaders for use later

set `PILImage.create` as `type_tfms`, and `IntToFloatTensor` as `batch_tfms` for this `TransformBlock`

A `TransformBlock` is just an object/container which stores a particular set of transform functions such as type_tfms, item_tfms, batch_tfms, 

and also store specific properties or args like dl_type, dls_kwargs


```
def ImageBlock(cls:PILBase=PILImage):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=cls.create, batch_tfms=IntToFloatTensor)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/data.py
# Type:      function
```


```
class TransformBlock():
    "A basic wrapper that links defaults transforms for the data block API"
    def __init__(self, 
        type_tfms:list=None, # One or more `Transform`s
        item_tfms:list=None, # `ItemTransform`s, applied on an item
        batch_tfms:list=None, # `Transform`s or `RandTransform`s, applied by batch
        dl_type:TfmdDL=None, # Task specific `TfmdDL`, defaults to `TfmdDL`
        dls_kwargs:dict=None, # Additional arguments to be passed to `DataLoaders`
    ):
        self.type_tfms  =            L(type_tfms)
        self.item_tfms  = ToTensor + L(item_tfms)
        self.batch_tfms =            L(batch_tfms)
        self.dl_type,self.dls_kwargs = dl_type,({} if dls_kwargs is None else dls_kwargs)
# File:           ~/mambaforge/lib/python3.9/site-packages/fastai/data/block.py
# Type:           type
# Subclasses: 
```

### CategoryBlock(vocab: 'list | pd.Series' = None, sort: 'bool' = True, add_na: 'bool' = False)
to create an instance of `TransformBlock`, which is passed to DataBlock and dataloaders for use later

set an instance of Categorize i.e., `Categorize(vocab=vocab, sort=sort, add_na=add_na)` as `type_tfms` for this `TransformBlock`

A `TransformBlock` is just an object/container which stores a particular set of transform functions such as type_tfms, item_tfms, batch_tfms, 

and also store specific properties or args like dl_type, dls_kwargs


```
def CategoryBlock(
    vocab:list|pd.Series=None, # List of unique class names
    sort:bool=True, # Sort the classes alphabetically
    add_na:bool=False, # Add `#na#` to `vocab`
):
    "`TransformBlock` for single-label categorical targets"
    return TransformBlock(type_tfms=Categorize(vocab=vocab, sort=sort, add_na=add_na))
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/block.py
# Type:      function
```

### DataBlock.__init__(blocks:list=None, dl_type:TfmdDL=None, getters:list=None, n_inp:int=None, item_tfms:list=None, batch_tfms:list=None, get_items=None, splitter=None, get_y=None, get_x=None)
Prepare and organise all the funcs needed to split and transform x and getting y or labels


```
# fastnbs("ImageBlock")
# fastnbs("funcs_kwargs")
```


```
from fastai.data.block import  _merge_tfms, _merge_grouper
```


```
@docs
@funcs_kwargs # it is working with _methods to add more args to __init__ method below
class DataBlock():
    "Generic container to quickly build `Datasets` and `DataLoaders`."
    get_x=get_items=splitter=get_y = None # class properties
    blocks,dl_type = (TransformBlock,TransformBlock),TfmdDL 
    
    _methods = 'get_items splitter get_y get_x'.split() # so that __init__ args will include get_items, splitter, get_y, get_x
    
    _msg = "If you wanted to compose several transforms in your getter don't forget to wrap them in a `Pipeline`."
    
    @snoop
    def __init__(self, 
        blocks:list=None, # One or more `TransformBlock`s, e.g., ImageBlock, CategoryBlock
        dl_type:TfmdDL=None, # Task specific `TfmdDL`, defaults to `block`'s dl_type or`TfmdDL`
        getters:list=None, # Getter functions applied to results of `get_items`
        n_inp:int=None, # Number of inputs
        item_tfms:list=None, # `ItemTransform`s, applied on an item 
        batch_tfms:list=None, # `Transform`s or `RandTransform`s, applied by batch
        **kwargs, 
    ):
        "Prepare and organise all the funcs needed to split and transform x and getting y or labels"
        blocks = L(self.blocks if blocks is None else blocks)
        pp(blocks)
        blocks = L(b() if callable(b) else b for b in blocks)
        pp(blocks)        
        
        pp(inspect.getdoc(blocks.attrgot), inspect.signature(blocks.attrgot))
        pp(blocks.map(lambda x: x.__dict__))
        self.type_tfms = blocks.attrgot('type_tfms', L())
           
        pp(inspect.getdoc(_merge_tfms), inspect.signature(_merge_tfms))
        self.default_item_tfms  = _merge_tfms(*blocks.attrgot('item_tfms',  L()))
        pp(self.default_item_tfms)
        
        self.default_batch_tfms = _merge_tfms(*blocks.attrgot('batch_tfms', L()))
        pp(self.default_batch_tfms)
        
        for b in blocks:
            if getattr(b, 'dl_type', None) is not None: 
                self.dl_type = pp(b.dl_type)
        if dl_type is not None: 
            self.dl_type = pp(dl_type)
        pp(self.dl_type)
            
        self.dataloaders = delegates(self.dl_type.__init__)(self.dataloaders) # get kwargs from dl_type.__init__ to self.dataloaders
        pp(self.dataloaders)
        
        self.dls_kwargs = merge(*blocks.attrgot('dls_kwargs', {}))
        pp(self.dls_kwargs)

        self.n_inp = ifnone(n_inp, max(1, len(blocks)-1)) # n_inp is dependent on the number of blocks
        pp(self.n_inp)
        
        self.getters = ifnone(getters, [noop]*len(self.type_tfms))
        pp(self.getters)
        
        if self.get_x:
            if len(L(self.get_x)) != self.n_inp:
                raise ValueError(f'get_x contains {len(L(self.get_x))} functions, but must contain {self.n_inp} (one for each input)\n{self._msg}')
            self.getters[:self.n_inp] = L(self.get_x)
        pp(self.get_x)
            
        if self.get_y:
            n_targs = len(self.getters) - self.n_inp
            if len(L(self.get_y)) != n_targs:
                raise ValueError(f'get_y contains {len(L(self.get_y))} functions, but must contain {n_targs} (one for each target)\n{self._msg}')
            self.getters[self.n_inp:] = L(self.get_y)
        pp(self.getters)

        if kwargs: 
            raise TypeError(f'invalid keyword arguments: {", ".join(kwargs.keys())}')
        
        pp(item_tfms, batch_tfms)
        pp(inspect.getdoc(self.new), inspect.signature(self.new))
        self.new(item_tfms, batch_tfms)

    def _combine_type_tfms(self): return L([self.getters, self.type_tfms]).map_zip(
        lambda g,tt: (g.fs if isinstance(g, Pipeline) else L(g)) + tt)

    def new(self, 
        item_tfms:list=None, # `ItemTransform`s, applied on an item
        batch_tfms:list=None, # `Transform`s or `RandTransform`s, applied by batch 
    ):
        self.item_tfms  = _merge_tfms(self.default_item_tfms,  item_tfms)
        self.batch_tfms = _merge_tfms(self.default_batch_tfms, batch_tfms)
        return self

    @classmethod
    def from_columns(cls, 
        blocks:list =None, # One or more `TransformBlock`s
        getters:list =None, # Getter functions applied to results of `get_items`
        get_items:callable=None, # A function to get items
        **kwargs,
    ):
        if getters is None: getters = L(ItemGetter(i) for i in range(2 if blocks is None else len(L(blocks))))
        get_items = _zip if get_items is None else compose(get_items, _zip)
        return cls(blocks=blocks, getters=getters, get_items=get_items, **kwargs)

    def datasets(self, 
        source, # The data source
        verbose:bool=False, # Show verbose messages
    ) -> Datasets:
        self.source = source                     ; pv(f"Collecting items from {source}", verbose)
        items = (self.get_items or noop)(source) ; pv(f"Found {len(items)} items", verbose)
        splits = (self.splitter or RandomSplitter())(items)
        pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)

    def dataloaders(self, 
        source, # The data source
        path:str='.', # Data source and default `Learner` path 
        verbose:bool=False, # Show verbose messages
        **kwargs
    ) -> DataLoaders:
        dsets = self.datasets(source, verbose=verbose)
        kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
        return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)

    _docs = dict(new="Create a new `DataBlock` with other `item_tfms` and `batch_tfms`",
                 datasets="Create a `Datasets` object from `source`",
                 dataloaders="Create a `DataLoaders` object from `source`")
# File:           ~/mambaforge/lib/python3.9/site-packages/fastai/data/block.py
# Type:           type
```


```
#|eval: false
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(bird)
```

    20:17:45.00 >>> Call to DataBlock.__init__ in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/1228990494.py", line 13
    20:17:45.00 .......... self = <__main__.DataBlock object>
    20:17:45.00 .......... blocks = (<function ImageBlock>, <function CategoryBlock>)
    20:17:45.00 .......... dl_type = None
    20:17:45.00 .......... getters = None
    20:17:45.00 .......... n_inp = None
    20:17:45.00 .......... item_tfms = [Resize -- {'size': (192, 192), 'method': 'squish...encodes
    20:17:45.00                        (TensorPoint,object) -> encodes
    20:17:45.00                        decodes: ]
    20:17:45.00 .......... batch_tfms = None
    20:17:45.00 .......... kwargs = {}
    20:17:45.00   13 |     def __init__(self, 
    20:17:45.00   23 |         blocks = L(self.blocks if blocks is None else blocks)
    20:17:45.00 .............. blocks = [<function ImageBlock>, <function CategoryBlock>]
    20:17:45.00   24 |         pp(blocks)
    20:17:45.01 LOG:
    20:17:45.06 .... blocks = [<function ImageBlock>, <function CategoryBlock>]
    20:17:45.06   25 |         blocks = L(b() if callable(b) else b for b in blocks)
    20:17:45.06 .............. blocks = [<__main__.TransformBlock object>, <__main__.TransformBlock object>]
    20:17:45.06   26 |         pp(blocks)        
    20:17:45.06 LOG:
    20:17:45.07 .... blocks = [<__main__.TransformBlock object>, <__main__.TransformBlock object>]
    20:17:45.07   28 |         pp(inspect.getdoc(blocks.attrgot), inspect.signature(blocks.attrgot))
    20:17:45.08 LOG:
    20:17:45.09 .... inspect.getdoc(blocks.attrgot) = 'Create new `L` with attr `k` (or value `k` for dicts) of all `items`.'
    20:17:45.09 .... inspect.signature(blocks.attrgot) = <Signature (k, default=None)>
    20:17:45.09   29 |         pp(blocks.map(lambda x: x.__dict__))
    20:17:45.09 LOG:
    20:17:45.11 .... blocks.map(lambda x: x.__dict__) = [{'type_tfms': [<bound method PILBase.create of <class 'fastai.vision.core.PILImage'>>], 'item_tfms': [<class 'fastai.data.transforms.ToTensor'>], 'batch_tfms': [<class 'fastai.data.transforms.IntToFloatTensor'>], 'dl_type': None, 'dls_kwargs': {}}, {'type_tfms': [Categorize -- {'vocab': None, 'sort': True, 'add_na': False}:
    20:17:45.11                                         encodes: (Tabular,object) -> encodes
    20:17:45.11                                         (object,object) -> encodes
    20:17:45.11                                         decodes: (Tabular,object) -> decodes
    20:17:45.11                                         (object,object) -> decodes
    20:17:45.11                                         ], 'item_tfms': [<class 'fastai.data.transforms.ToTensor'>], 'batch_tfms': [], 'dl_type': None, 'dls_kwargs': {}}]
    20:17:45.11   30 |         self.type_tfms = blocks.attrgot('type_tfms', L())
    20:17:45.11   32 |         pp(inspect.getdoc(_merge_tfms), inspect.signature(_merge_tfms))
    20:17:45.11 LOG:
    20:17:45.13 .... inspect.getdoc(_merge_tfms) = ('Group the `tfms` in a single list, removing duplicates (from the same class) '
    20:17:45.13                                     'and instantiating')
    20:17:45.13 .... inspect.signature(_merge_tfms) = <Signature (*tfms)>
    20:17:45.13   33 |         self.default_item_tfms  = _merge_tfms(*blocks.attrgot('item_tfms',  L()))
    20:17:45.13   34 |         pp(self.default_item_tfms)
    20:17:45.13 LOG:
    20:17:45.14 .... self.default_item_tfms = [ToTensor:
    20:17:45.14                               encodes: (PILMask,object) -> encodes
    20:17:45.14                               (PILBase,object) -> encodes
    20:17:45.14                               decodes: ]
    20:17:45.14   36 |         self.default_batch_tfms = _merge_tfms(*blocks.attrgot('batch_tfms', L()))
    20:17:45.14   37 |         pp(self.default_batch_tfms)
    20:17:45.14 LOG:
    20:17:45.15 .... self.default_batch_tfms = [IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}:
    20:17:45.15                                encodes: (TensorImage,object) -> encodes
    20:17:45.15                                (TensorMask,object) -> encodes
    20:17:45.15                                decodes: (TensorImage,object) -> decodes
    20:17:45.15                                ]
    20:17:45.15   39 |         for b in blocks:
    20:17:45.15 .............. b = <__main__.TransformBlock object>
    20:17:45.15   40 |             if getattr(b, 'dl_type', None) is not None: 
    20:17:45.15   39 |         for b in blocks:
    20:17:45.15 .............. b = <__main__.TransformBlock object>
    20:17:45.15   40 |             if getattr(b, 'dl_type', None) is not None: 
    20:17:45.15   39 |         for b in blocks:
    20:17:45.15   42 |         if dl_type is not None: 
    20:17:45.15   44 |         pp(self.dl_type)
    20:17:45.15 LOG:
    20:17:45.16 .... self.dl_type = <class 'fastai.data.core.TfmdDL'>
    20:17:45.16   46 |         self.dataloaders = delegates(self.dl_type.__init__)(self.dataloaders) # get kwargs from dl_type.__init__ to self.dataloaders
    20:17:45.16   47 |         pp(self.dataloaders)
    20:17:45.16 LOG:
    20:17:45.18 .... self.dataloaders = <bound method DataBlock.dataloaders of <__main__.DataBlock object>>
    20:17:45.18   49 |         self.dls_kwargs = merge(*blocks.attrgot('dls_kwargs', {}))
    20:17:45.18   50 |         pp(self.dls_kwargs)
    20:17:45.18 LOG:
    20:17:45.19 .... self.dls_kwargs = {}
    20:17:45.19   52 |         self.n_inp = ifnone(n_inp, max(1, len(blocks)-1)) # n_inp is dependent on the number of blocks
    20:17:45.19   53 |         pp(self.n_inp)
    20:17:45.19 LOG:
    20:17:45.20 .... self.n_inp = 1
    20:17:45.20   55 |         self.getters = ifnone(getters, [noop]*len(self.type_tfms))
    20:17:45.20   56 |         pp(self.getters)
    20:17:45.20 LOG:
    20:17:45.21 .... self.getters = [<function noop>, <function noop>]
    20:17:45.21   58 |         if self.get_x:
    20:17:45.21   62 |         pp(self.get_x)
    20:17:45.21 LOG:
    20:17:45.22 .... self.get_x = None
    20:17:45.22   64 |         if self.get_y:
    20:17:45.22   65 |             n_targs = len(self.getters) - self.n_inp
    20:17:45.22 .................. n_targs = 1
    20:17:45.22   66 |             if len(L(self.get_y)) != n_targs:
    20:17:45.22   68 |             self.getters[self.n_inp:] = L(self.get_y)
    20:17:45.22   69 |         pp(self.getters)
    20:17:45.22 LOG:
    20:17:45.23 .... self.getters = [<function noop>, <function parent_label>]
    20:17:45.23   71 |         if kwargs: 
    20:17:45.23   74 |         pp(item_tfms, batch_tfms)
    20:17:45.23 LOG:
    20:17:45.25 .... item_tfms = [Resize -- {'size': (192, 192), 'method': 'squish', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0}:
    20:17:45.25                  encodes: (Image,object) -> encodes
    20:17:45.25                  (TensorBBox,object) -> encodes
    20:17:45.25                  (TensorPoint,object) -> encodes
    20:17:45.25                  decodes: ]
    20:17:45.25 .... batch_tfms = None
    20:17:45.25   75 |         pp(inspect.getdoc(self.new), inspect.signature(self.new))
    20:17:45.25 LOG:
    20:17:45.26 .... inspect.getdoc(self.new) = 'Create a new `DataBlock` with other `item_tfms` and `batch_tfms`'
    20:17:45.26 .... inspect.signature(self.new) = <Signature (item_tfms: 'list' = None, batch_tfms: 'list' = None)>
    20:17:45.26   76 |         self.new(item_tfms, batch_tfms)
    20:17:45.26 <<< Return value from DataBlock.__init__: None


### doc: DataBlock.datasets(source, verbose)
get data items from the source, and split the items and use `fastai.data.core.Datasets` to create the datasets

### src: DataBlock.datasets((source, verbose)


```
# DataBlock.dataloaders??
# from fastdebug.utils import *
# chk
```


```
from fastai.vision.all import DataBlock # so that @snoop and pp can remain inside the DataBlock.__init__ source code
```


```
@patch
@snoop
def datasets(self:DataBlock, 
        source, # The data source
        verbose:bool=False, # Show verbose messages
    ) -> Datasets:
        self.source = source 
        
        pp(doc_sig(pv))
        pv(f"Collecting items from {source}", verbose)
        pp(pv(f"Collecting items from {source}", verbose))
        
        pp((None or noop))
        pp((self.get_items or noop))
        items = (self.get_items or noop)(source)
        pp(chk(items))  
        
        pv(f"Found {len(items)} items", verbose)
        splits = (self.splitter or RandomSplitter())(items)
        pp(chk(splits))
        pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        
        pp(doc_sig(Datasets))
        pp(doc_sig(Datasets.__init__))
        res = Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)
        return res
```


```
#|eval: false
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).datasets(bird, True)
```

    20:17:45.82 >>> Call to datasets in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/3647961006.py", line 3
    20:17:45.82 ...... self = <fastai.data.block.DataBlock object>
    20:17:45.82 ...... source = Path('forest_or_bird')
    20:17:45.82 ...... verbose = True
    20:17:45.82    3 | def datasets(self:DataBlock, 
    20:17:45.82    7 |         self.source = source 
    20:17:45.82    9 |         pp(doc_sig(pv))
    20:17:45.82 LOG:
    20:17:45.84 .... doc_sig(pv) = ('no mro', 'no doc', <Signature (text, verbose)>)
    20:17:45.84   10 |         pv(f"Collecting items from {source}", verbose)
    20:17:45.84   11 |         pp(pv(f"Collecting items from {source}", verbose))
    20:17:45.84 LOG:
    20:17:45.85 .... pv(f"Collecting items from {source}", verbose) = None
    20:17:45.85   13 |         pp((None or noop))
    20:17:45.85 LOG:
    20:17:45.85 .... None or noop = <function noop>
    20:17:45.85   14 |         pp((self.get_items or noop))
    20:17:45.85 LOG:
    20:17:45.86 .... self.get_items or noop = <function get_image_files>
    20:17:45.86   15 |         items = (self.get_items or noop)(source)
    20:17:45.86 .............. items = [Path('forest_or_bird/forest/82e9179d-2dd6-4144-.../bird/0bc8d1a8-e443-40b7-a148-f043c158400f.jpg')]
    20:17:45.86   16 |         pp(chk(items))  
    20:17:45.86 LOG:
    20:17:45.87 .... chk(items) = (<class 'fastcore.foundation.L'>, 251, 'no shape')
    20:17:45.87   18 |         pv(f"Found {len(items)} items", verbose)
    20:17:45.87   19 |         splits = (self.splitter or RandomSplitter())(items)
    20:17:45.87 .............. splits = ([83, 160, 7, 134, 147, 94, 126, 237, 55, 95, 244...8, 200, 229, 213, 206, 115, 121, 66, 195, 1, 168], [128, 68, 120, 17, 177, 16, 221, 131, 102, 22, 6..., 4, 2, 203, 129, 218, 137, 118, 249, 93, 34, 36])
    20:17:45.87   20 |         pp(chk(splits))
    20:17:45.87 LOG:
    20:17:45.88 .... chk(splits) = (<class 'tuple'>, 2, 'no shape')
    20:17:45.88   21 |         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        20:17:45.88 List comprehension:
        20:17:45.88   21 |         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        20:17:45.88 .............. Iterating over <tuple_iterator object>
        20:17:45.88 .............. Values of s: [83, 160, 7, 134, 147, 94, 126, 237, 55, 95, 244...8, 200, 229, 213, 206, 115, 121, 66, 195, 1, 168], [128, 68, 120, 17, 177, 16, 221, 131, 102, 22, 6..., 4, 2, 203, 129, 218, 137, 118, 249, 93, 34, 36]
        20:17:45.88 Result: ['201', '50']
    20:17:45.88   21 |         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
    20:17:45.88   23 |         pp(doc_sig(Datasets))
    20:17:45.88 LOG:
    20:17:45.89 .... doc_sig(Datasets) = ((<class 'fastai.data.core.Datasets'>,
    20:17:45.89                            <class 'fastai.data.core.FilteredBase'>,
    20:17:45.89                            <class 'object'>),
    20:17:45.89                           'A dataset that creates a tuple from each `tfms`',
    20:17:45.89                           <Signature (items: 'list' = None, tfms: 'list | Pipeline' = None, tls: 'TfmdLists' = None, n_inp: 'int' = None, dl_type=None, *, use_list: 'bool' = None, do_setup: 'bool' = True, split_idx: 'int' = None, train_setup: 'bool' = True, splits: 'list' = None, types=None, verbose: 'bool' = False)>)
    20:17:45.89   24 |         pp(doc_sig(Datasets.__init__))
    20:17:45.89 LOG:
    20:17:45.90 .... doc_sig(Datasets.__init__) = ('no mro',
    20:17:45.90                                    'Initialize self.  See help(type(self)) for accurate signature.',
    20:17:45.90                                    <Signature (self, items: 'list' = None, tfms: 'list | Pipeline' = None, tls: 'TfmdLists' = None, n_inp: 'int' = None, dl_type=None, **kwargs)>)
    20:17:45.90   25 |         res = Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)


    Collecting items from forest_or_bird
    Collecting items from forest_or_bird
    Found 251 items
    2 datasets of sizes 201,50
    Setting up Pipeline: PILBase.create


    20:17:46.16 .............. res = (#251) [(PILImage mode=RGB size=400x250, TensorC...age mode=RGB size=400x400, TensorCategory(1))...]
    20:17:46.16   26 |         return res


    Setting up Pipeline: parent_label -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}


    20:17:46.31 <<< Return value from datasets: (#251) [(PILImage mode=RGB size=400x250, TensorC...age mode=RGB size=400x400, TensorCategory(1))...]



```
@patch
def datasets(self:DataBlock, 
    source, # The data source
    verbose:bool=False, # Show verbose messages
) -> Datasets:
    self.source = source                     ; pv(f"Collecting items from {source}", verbose)
    items = (self.get_items or noop)(source) ; pv(f"Found {len(items)} items", verbose)
    splits = (self.splitter or RandomSplitter())(items)
    pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
    return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)
```

### doc: DataBlock.dataloaders
use `DataBlock.datasets(source, verbose=verbose)` to create a ```fastai.data.core.Datasets``` first and then use ```Datasets.dataloaders``` to create a ```fastai.data.core.DataLoaders```

### src: DataBlock.dataloaders


```
from fastai.vision.all import DataBlock # so that @snoop and pp can remain inside the DataBlock.__init__ source code
```


```
@patch
@snoop
def dataloaders(self:DataBlock, 
    source, # The data source
    path:str='.', # Data source and default `Learner` path 
    verbose:bool=False, # Show verbose messages
    **kwargs
) -> DataLoaders:
    pp(doc_sig(self.datasets))
    
    dsets = self.datasets(source, verbose=verbose)
    pp(doc_sig(type(dsets)))
#     pp(inspect_class(dsets.__class__))
    
    kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    pp(kwargs)
    
    pp(doc_sig(dsets.dataloaders))
    res = dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
    pp(doc_sig(res))
    pp(doc_sig(res.__class__))
    return res
```


```
#|eval: false
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(bird)
```

    20:17:46.40 >>> Call to dataloaders in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/2346295921.py", line 3
    20:17:46.40 ...... self = <fastai.data.block.DataBlock object>
    20:17:46.40 ...... source = Path('forest_or_bird')
    20:17:46.40 ...... path = '.'
    20:17:46.40 ...... verbose = False
    20:17:46.40 ...... kwargs = {}
    20:17:46.40    3 | def dataloaders(self:DataBlock, 
    20:17:46.40    9 |     pp(doc_sig(self.datasets))
    20:17:46.40 LOG:
    20:17:46.41 .... doc_sig(self.datasets) = ('no mro',
    20:17:46.41                                'no doc',
    20:17:46.41                                <Signature (source, verbose: 'bool' = False) -> 'Datasets'>)
    20:17:46.41   11 |     dsets = self.datasets(source, verbose=verbose)
    20:17:46.68 .......... dsets = (#251) [(PILImage mode=RGB size=400x250, TensorC...age mode=RGB size=400x400, TensorCategory(1))...]
    20:17:46.68   12 |     pp(doc_sig(type(dsets)))
    20:17:46.68 LOG:
    20:17:46.68 .... doc_sig(type(dsets)) = ((<class 'fastai.data.core.Datasets'>,
    20:17:46.68                               <class 'fastai.data.core.FilteredBase'>,
    20:17:46.68                               <class 'object'>),
    20:17:46.68                              'A dataset that creates a tuple from each `tfms`',
    20:17:46.68                              <Signature (items: 'list' = None, tfms: 'list | Pipeline' = None, tls: 'TfmdLists' = None, n_inp: 'int' = None, dl_type=None, *, use_list: 'bool' = None, do_setup: 'bool' = True, split_idx: 'int' = None, train_setup: 'bool' = True, splits: 'list' = None, types=None, verbose: 'bool' = False)>)
    20:17:46.76   15 |     kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    20:17:46.83 .......... kwargs = {'verbose': False}
    20:17:46.83   16 |     pp(kwargs)
    20:17:46.83 LOG:
    20:17:46.83 .... kwargs = {'verbose': False}
    20:17:46.91   18 |     pp(doc_sig(dsets.dataloaders))
    20:17:46.91 LOG:
    20:17:46.91 .... doc_sig(dsets.dataloaders) = ('no mro',
    20:17:46.91                                    'Get a `DataLoaders`',
    20:17:46.91                                    <Signature (bs: 'int' = 64, shuffle_train: 'bool' = None, shuffle: 'bool' = True, val_shuffle: 'bool' = False, n: 'int' = None, path: 'str | Path' = '.', dl_type: 'TfmdDL' = None, dl_kwargs: 'list' = None, device: 'torch.device' = None, drop_last: 'bool' = None, val_bs: 'int' = None, *, num_workers: 'int' = None, verbose: 'bool' = False, do_setup: 'bool' = True, pin_memory=False, timeout=0, batch_size=None, indexed=None, persistent_workers=False, pin_memory_device='', wif=None, before_iter=None, after_item=None, before_batch=None, after_batch=None, after_iter=None, create_batches=None, create_item=None, create_batch=None, retain=None, get_idxs=None, sample=None, shuffle_fn=None, do_batch=None) -> 'DataLoaders'>)
    20:17:46.99   19 |     res = dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
    20:17:47.32 .......... res = <fastai.data.core.DataLoaders object>
    20:17:47.32   20 |     pp(doc_sig(res))
    20:17:47.32 LOG:
    20:17:47.32 .... doc_sig(res) = ('no mro', 'Basic wrapper around several `DataLoader`s.', 'no signature')
    20:17:47.40   21 |     pp(doc_sig(res.__class__))
    20:17:47.40 LOG:
    20:17:47.40 .... doc_sig(res.__class__) = ((<class 'fastai.data.core.DataLoaders'>,
    20:17:47.40                                 <class 'fastcore.basics.GetAttr'>,
    20:17:47.40                                 <class 'object'>),
    20:17:47.40                                'Basic wrapper around several `DataLoader`s.',
    20:17:47.40                                <Signature (*loaders, path: 'str | Path' = '.', device=None)>)
    20:17:47.48   22 |     return res
    20:17:47.55 <<< Return value from dataloaders: <fastai.data.core.DataLoaders object>



```
@patch # return dataloaders back to normal
def dataloaders(self:DataBlock, 
    source, # The data source
    path:str='.', # Data source and default `Learner` path 
    verbose:bool=False, # Show verbose messages
    **kwargs
) -> DataLoaders:
    dsets = self.datasets(source, verbose=verbose)
    kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
```


```

```


```

```


```

```

### vision_learner(dls, arch...)
Build a vision learner from `dls` and `arch`

A learner is a container to prepare model and args, and put model, dls, loss func together


```
from fastai.vision.learner import *
from fastai.vision.learner import  _default_meta, _add_norm
```


```
@snoop
@delegates(create_vision_model) 
def vision_learner(dls, arch, normalize=True, n_out=None, pretrained=True, 
        # learner args
        loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
        model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95),
        # model & head args
        cut=None, init=nn.init.kaiming_normal_, custom_head=None, concat_pool=True, pool=True,
        lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None, **kwargs):
    "Build a vision learner from `dls` and `arch`"
    pp(doc_sig(get_c))
    if n_out is None: 
        n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    pp(arch, _default_meta, model_meta, model_meta.get)
    meta = model_meta.get(arch, _default_meta)
    model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
                      first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    pp(model_args)
    if isinstance(arch, str):
        model,cfg = create_timm_model(arch, n_out, default_split, pretrained, **model_args)
        pp(doc_sig(create_timm_model))
        pp(model, cfg)
        if normalize: 
            _timm_norm(dls, cfg, pretrained)
        pp(doc_sig(_timm_norm))
    else:
        pp(dls, meta, pretrained)
        pp(doc_sig(_add_norm))
        if normalize: 
            _add_norm(dls, meta, pretrained)
        pp(arch, n_out, pretrained, model_args)
        pp(doc_sig(create_vision_model))
        model = create_vision_model(arch, n_out, pretrained=pretrained, **model_args)
        pp(model)
    pp.deep(lambda: ifnone(splitter, meta['split']))
    splitter=ifnone(splitter, meta['split'])
    pp(dict(dls=dls, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms))
    pp(doc_sig(Learner))
    learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
    pp(doc_sig(learn.freeze))
    if pretrained: 
        learn.freeze()
    # keep track of args for loggers
    pp(doc_sig(store_attr), kwargs)
    pp(learn.__dict__.keys())
    store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    pp(learn.__dict__.keys())
    return learn
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/learner.py
# Type:      function
```


```
learn = vision_learner(dls, resnet18, metrics=error_rate)
```

    20:17:47.66 >>> Call to vision_learner in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/3098971962.py", line 3
    20:17:47.66 ...... dls = <fastai.data.core.DataLoaders object>
    20:17:47.66 ...... arch = <function resnet18>
    20:17:47.66 ...... normalize = True
    20:17:47.66 ...... n_out = None
    20:17:47.66 ...... pretrained = True
    20:17:47.66 ...... loss_func = None
    20:17:47.66 ...... opt_func = <function Adam>
    20:17:47.66 ...... lr = 0.001
    20:17:47.66 ...... splitter = None
    20:17:47.66 ...... cbs = None
    20:17:47.66 ...... metrics = <function error_rate>
    20:17:47.66 ...... path = None
    20:17:47.66 ...... model_dir = 'models'
    20:17:47.66 ...... wd = None
    20:17:47.66 ...... wd_bn_bias = False
    20:17:47.66 ...... train_bn = True
    20:17:47.66 ...... moms = (0.95, 0.85, 0.95)
    20:17:47.66 ...... cut = None
    20:17:47.66 ...... init = <function kaiming_normal_>
    20:17:47.66 ...... custom_head = None
    20:17:47.66 ...... concat_pool = True
    20:17:47.66 ...... pool = True
    20:17:47.66 ...... lin_ftrs = None
    20:17:47.66 ...... ps = 0.5
    20:17:47.66 ...... first_bn = True
    20:17:47.66 ...... bn_final = False
    20:17:47.66 ...... lin_first = False
    20:17:47.66 ...... y_range = None
    20:17:47.66 ...... kwargs = {}
    20:17:47.66    3 | def vision_learner(dls, arch, normalize=True, n_out=None, pretrained=True, 
    20:17:47.66   11 |     pp(doc_sig(get_c))
    20:17:47.66 LOG:
    20:17:47.69 .... doc_sig(get_c) = ('no mro', 'no doc', <Signature (dls)>)
    20:17:47.69   12 |     if n_out is None: 
    20:17:47.69   13 |         n_out = get_c(dls)
    20:17:47.70 .............. n_out = 2
    20:17:47.70   14 |     assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    20:17:47.70   15 |     pp(arch, _default_meta, model_meta, model_meta.get)
    20:17:47.70 LOG:
    20:17:47.70 .... arch = <function resnet18>
    20:17:47.70 .... _default_meta = {'cut': None, 'split': <function default_split>}
    20:17:47.71 .... model_meta = {<function alexnet>: {'cut': -2,
    20:17:47.71                                                        'split': <function _alexnet_split>,
    20:17:47.71                                                        'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                  [0.229, 0.224, 0.225])},
    20:17:47.71                    <function densenet121>: {'cut': -1,
    20:17:47.71                                                            'split': <function _densenet_split>,
    20:17:47.71                                                            'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                      [0.229, 0.224, 0.225])},
    20:17:47.71                    <function densenet161>: {'cut': -1,
    20:17:47.71                                                            'split': <function _densenet_split>,
    20:17:47.71                                                            'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                      [0.229, 0.224, 0.225])},
    20:17:47.71                    <function densenet169>: {'cut': -1,
    20:17:47.71                                                            'split': <function _densenet_split>,
    20:17:47.71                                                            'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                      [0.229, 0.224, 0.225])},
    20:17:47.71                    <function densenet201>: {'cut': -1,
    20:17:47.71                                                            'split': <function _densenet_split>,
    20:17:47.71                                                            'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                      [0.229, 0.224, 0.225])},
    20:17:47.71                    <function resnet18>: {'cut': -2,
    20:17:47.71                                                         'split': <function _resnet_split>,
    20:17:47.71                                                         'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                   [0.229, 0.224, 0.225])},
    20:17:47.71                    <function squeezenet1_0>: {'cut': -1,
    20:17:47.71                                                              'split': <function _squeezenet_split>,
    20:17:47.71                                                              'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                        [0.229, 0.224, 0.225])},
    20:17:47.71                    <function resnet34>: {'cut': -2,
    20:17:47.71                                                         'split': <function _resnet_split>,
    20:17:47.71                                                         'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                   [0.229, 0.224, 0.225])},
    20:17:47.71                    <function resnet50>: {'cut': -2,
    20:17:47.71                                                         'split': <function _resnet_split>,
    20:17:47.71                                                         'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                   [0.229, 0.224, 0.225])},
    20:17:47.71                    <function resnet101>: {'cut': -2,
    20:17:47.71                                                          'split': <function _resnet_split>,
    20:17:47.71                                                          'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                    [0.229, 0.224, 0.225])},
    20:17:47.71                    <function resnet152>: {'cut': -2,
    20:17:47.71                                                          'split': <function _resnet_split>,
    20:17:47.71                                                          'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                    [0.229, 0.224, 0.225])},
    20:17:47.71                    <function squeezenet1_1>: {'cut': -1,
    20:17:47.71                                                              'split': <function _squeezenet_split>,
    20:17:47.71                                                              'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                        [0.229, 0.224, 0.225])},
    20:17:47.71                    <function vgg11_bn>: {'cut': -2,
    20:17:47.71                                                         'split': <function _vgg_split>,
    20:17:47.71                                                         'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                   [0.229, 0.224, 0.225])},
    20:17:47.71                    <function vgg13_bn>: {'cut': -2,
    20:17:47.71                                                         'split': <function _vgg_split>,
    20:17:47.71                                                         'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                   [0.229, 0.224, 0.225])},
    20:17:47.71                    <function vgg16_bn>: {'cut': -2,
    20:17:47.71                                                         'split': <function _vgg_split>,
    20:17:47.71                                                         'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                   [0.229, 0.224, 0.225])},
    20:17:47.71                    <function vgg19_bn>: {'cut': -2,
    20:17:47.71                                                         'split': <function _vgg_split>,
    20:17:47.71                                                         'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                   [0.229, 0.224, 0.225])},
    20:17:47.71                    <function xresnet101>: {'cut': -4,
    20:17:47.71                                                           'split': <function _xresnet_split>,
    20:17:47.71                                                           'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                     [0.229, 0.224, 0.225])},
    20:17:47.71                    <function xresnet152>: {'cut': -4,
    20:17:47.71                                                           'split': <function _xresnet_split>,
    20:17:47.71                                                           'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                     [0.229, 0.224, 0.225])},
    20:17:47.71                    <function xresnet18>: {'cut': -4,
    20:17:47.71                                                          'split': <function _xresnet_split>,
    20:17:47.71                                                          'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                    [0.229, 0.224, 0.225])},
    20:17:47.71                    <function xresnet34>: {'cut': -4,
    20:17:47.71                                                          'split': <function _xresnet_split>,
    20:17:47.71                                                          'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                    [0.229, 0.224, 0.225])},
    20:17:47.71                    <function xresnet50>: {'cut': -4,
    20:17:47.71                                                          'split': <function _xresnet_split>,
    20:17:47.71                                                          'stats': ([0.485, 0.456, 0.406],
    20:17:47.71                                                                    [0.229, 0.224, 0.225])}}
    20:17:47.71 .... model_meta.get = <built-in method get of dict object>
    20:17:47.71   16 |     meta = model_meta.get(arch, _default_meta)
    20:17:47.71 .......... meta = {'cut': -2, 'split': <function _resnet_split>, 'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
    20:17:47.71   17 |     model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
    20:17:47.71   18 |                       first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    20:17:47.71   17 |     model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
    20:17:47.71   18 |                       first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    20:17:47.71   17 |     model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
    20:17:47.71 .......... model_args = {'init': <function kaiming_normal_>, 'custom_head': None, 'concat_pool': True, 'pool': True, ...}
    20:17:47.71   19 |     pp(model_args)
    20:17:47.71 LOG:
    20:17:47.72 .... model_args = {'bn_final': False,
    20:17:47.72                    'concat_pool': True,
    20:17:47.72                    'custom_head': None,
    20:17:47.72                    'first_bn': True,
    20:17:47.72                    'init': <function kaiming_normal_>,
    20:17:47.72                    'lin_first': False,
    20:17:47.72                    'lin_ftrs': None,
    20:17:47.72                    'pool': True,
    20:17:47.72                    'ps': 0.5,
    20:17:47.72                    'y_range': None}
    20:17:47.72   20 |     if isinstance(arch, str):
    20:17:47.72   28 |         pp(dls, meta, pretrained)
    20:17:47.72 LOG:
    20:17:47.73 .... dls = <fastai.data.core.DataLoaders object>
    20:17:47.73 .... meta = {'cut': -2,
    20:17:47.73              'split': <function _resnet_split>,
    20:17:47.73              'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
    20:17:47.73 .... pretrained = True
    20:17:47.73   29 |         pp(doc_sig(_add_norm))
    20:17:47.73 LOG:
    20:17:47.74 .... doc_sig(_add_norm) = ('no mro', 'no doc', <Signature (dls, meta, pretrained, n_in=3)>)
    20:17:47.74   30 |         if normalize: 
    20:17:47.74   31 |             _add_norm(dls, meta, pretrained)
    20:17:47.74   32 |         pp(arch, n_out, pretrained, model_args)
    20:17:47.74 LOG:
    20:17:47.75 .... arch = <function resnet18>
    20:17:47.75 .... n_out = 2
    20:17:47.75 .... pretrained = True
    20:17:47.75 .... model_args = {'bn_final': False,
    20:17:47.75                    'concat_pool': True,
    20:17:47.75                    'custom_head': None,
    20:17:47.75                    'first_bn': True,
    20:17:47.75                    'init': <function kaiming_normal_>,
    20:17:47.75                    'lin_first': False,
    20:17:47.75                    'lin_ftrs': None,
    20:17:47.75                    'pool': True,
    20:17:47.75                    'ps': 0.5,
    20:17:47.75                    'y_range': None}
    20:17:47.75   33 |         pp(doc_sig(create_vision_model))
    20:17:47.75 LOG:
    20:17:47.76 .... doc_sig(create_vision_model) = ('no mro',
    20:17:47.76                                      'Create custom vision architecture',
    20:17:47.76                                      <Signature (arch, n_out, pretrained=True, cut=None, n_in=3, init=<function kaiming_normal_>, custom_head=None, concat_pool=True, pool=True, lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None)>)
    20:17:47.76   34 |         model = create_vision_model(arch, n_out, pretrained=pretrained, **model_args)
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
      warnings.warn(
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)
    20:17:48.02 .............. model = Sequential(
    20:17:48.02                          (0): Sequential(
    20:17:48.02                            (0): Conv2d(3...n_features=512, out_features=2, bias=False)
    20:17:48.02                          )
    20:17:48.02                        )
    20:17:48.02   35 |         pp(model)
    20:17:48.02 LOG:
    20:17:48.03 .... model = Sequential(
    20:17:48.03                (0): Sequential(
    20:17:48.03                  (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    20:17:48.03                  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                  (2): ReLU(inplace=True)
    20:17:48.03                  (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    20:17:48.03                  (4): Sequential(
    20:17:48.03                    (0): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                    )
    20:17:48.03                    (1): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                    )
    20:17:48.03                  )
    20:17:48.03                  (5): Sequential(
    20:17:48.03                    (0): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (downsample): Sequential(
    20:17:48.03                        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    20:17:48.03                        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      )
    20:17:48.03                    )
    20:17:48.03                    (1): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                    )
    20:17:48.03                  )
    20:17:48.03                  (6): Sequential(
    20:17:48.03                    (0): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (downsample): Sequential(
    20:17:48.03                        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    20:17:48.03                        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      )
    20:17:48.03                    )
    20:17:48.03                    (1): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                    )
    20:17:48.03                  )
    20:17:48.03                  (7): Sequential(
    20:17:48.03                    (0): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (downsample): Sequential(
    20:17:48.03                        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    20:17:48.03                        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      )
    20:17:48.03                    )
    20:17:48.03                    (1): BasicBlock(
    20:17:48.03                      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                      (relu): ReLU(inplace=True)
    20:17:48.03                      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    20:17:48.03                      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                    )
    20:17:48.03                  )
    20:17:48.03                )
    20:17:48.03                (1): Sequential(
    20:17:48.03                  (0): AdaptiveConcatPool2d(
    20:17:48.03                    (ap): AdaptiveAvgPool2d(output_size=1)
    20:17:48.03                    (mp): AdaptiveMaxPool2d(output_size=1)
    20:17:48.03                  )
    20:17:48.03                  (1): fastai.layers.Flatten(full=False)
    20:17:48.03                  (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                  (3): Dropout(p=0.25, inplace=False)
    20:17:48.03                  (4): Linear(in_features=1024, out_features=512, bias=False)
    20:17:48.03                  (5): ReLU(inplace=True)
    20:17:48.03                  (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    20:17:48.03                  (7): Dropout(p=0.5, inplace=False)
    20:17:48.03                  (8): Linear(in_features=512, out_features=2, bias=False)
    20:17:48.03                )
    20:17:48.03              )
    20:17:48.03   36 |     pp.deep(lambda: ifnone(splitter, meta['split']))
    20:17:48.03 LOG:
    20:17:48.07 ........ ifnone = <function ifnone>
    20:17:48.07 ........ splitter = None
    20:17:48.07 ............ meta = {'cut': -2, 'split': <function _resnet_split>, 'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
    20:17:48.07 ........ meta['split'] = <function _resnet_split>
    20:17:48.07 .... ifnone(splitter, meta['split']) = <function _resnet_split>
    20:17:48.07   37 |     splitter=ifnone(splitter, meta['split'])
    20:17:48.07 .......... splitter = <function _resnet_split>
    20:17:48.07   38 |     pp(dict(dls=dls, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    20:17:48.08   39 |                    metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms))
    20:17:48.08   38 |     pp(dict(dls=dls, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    20:17:48.08 LOG:
    20:17:48.09 .... dict(dls=dls, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    20:17:48.09                  metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms) = {'cbs': None,
    20:17:48.09                                                                                                                                  'dls': <fastai.data.core.DataLoaders object>,
    20:17:48.09                                                                                                                                  'loss_func': None,
    20:17:48.09                                                                                                                                  'lr': 0.001,
    20:17:48.09                                                                                                                                  'metrics': <function error_rate>,
    20:17:48.09                                                                                                                                  'model_dir': 'models',
    20:17:48.09                                                                                                                                  'moms': (0.95, 0.85, 0.95),
    20:17:48.09                                                                                                                                  'opt_func': <function Adam>,
    20:17:48.09                                                                                                                                  'path': None,
    20:17:48.09                                                                                                                                  'splitter': <function _resnet_split>,
    20:17:48.09                                                                                                                                  'train_bn': True,
    20:17:48.09                                                                                                                                  'wd': None,
    20:17:48.09                                                                                                                                  'wd_bn_bias': False}
    20:17:48.09   40 |     pp(doc_sig(Learner))
    20:17:48.09 LOG:
    20:17:48.10 .... doc_sig(Learner) = ((<class 'fastai.learner.Learner'>,
    20:17:48.10                           <class 'fastcore.basics.GetAttr'>,
    20:17:48.10                           <class 'object'>),
    20:17:48.10                          'Group together a `model`, some `dls` and a `loss_func` to handle training',
    20:17:48.10                          <Signature (dls, model: 'callable', loss_func: 'callable | None' = None, opt_func=<function Adam>, lr=0.001, splitter: 'callable' = <function trainable_params>, cbs=None, metrics=None, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95, 0.85, 0.95), default_cbs: 'bool' = True)>)
    20:17:48.10   41 |     learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    20:17:48.10   42 |                    metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
    20:17:48.10   41 |     learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    20:17:48.10 .......... learn = <fastai.learner.Learner object>
    20:17:48.10   43 |     pp(doc_sig(learn.freeze))
    20:17:48.10 LOG:
    20:17:48.11 .... doc_sig(learn.freeze) = ('no mro', 'Freeze up to last parameter group', <Signature ()>)
    20:17:48.11   44 |     if pretrained: 
    20:17:48.11   45 |         learn.freeze()
    20:17:48.12   47 |     pp(doc_sig(store_attr), kwargs)
    20:17:48.12 LOG:
    20:17:48.13 .... doc_sig(store_attr) = ('no mro',
    20:17:48.13                             'Store params named in comma-separated `names` from calling context into '
    20:17:48.13                             'attrs in `self`',
    20:17:48.13                             <Signature (names=None, self=None, but='', cast=False, store_args=None, **attrs)>)
    20:17:48.13 .... kwargs = {}
    20:17:48.14   48 |     pp(learn.__dict__.keys())
    20:17:48.14 LOG:
    20:17:48.15 .... learn.__dict__.keys() = dict_keys(['dls', 'model', '__stored_args__', 'loss_func', 'opt_func', 'lr', 'splitter', '_metrics', 'path', 'model_dir', 'wd', 'wd_bn_bias', 'train_bn', 'moms', 'default_cbs', 'training', 'create_mbar', 'logger', 'opt', 'cbs', 'train_eval', 'recorder', 'cast_to_tensor', 'progress', 'lock', 'n_epoch'])
    20:17:48.15   49 |     store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    20:17:48.15   50 |     pp(learn.__dict__.keys())
    20:17:48.15 LOG:
    20:17:48.16 .... learn.__dict__.keys() = dict_keys(['dls', 'model', '__stored_args__', 'loss_func', 'opt_func', 'lr', 'splitter', '_metrics', 'path', 'model_dir', 'wd', 'wd_bn_bias', 'train_bn', 'moms', 'default_cbs', 'training', 'create_mbar', 'logger', 'opt', 'cbs', 'train_eval', 'recorder', 'cast_to_tensor', 'progress', 'lock', 'n_epoch', 'arch', 'normalize', 'n_out', 'pretrained'])
    20:17:48.16   51 |     return learn
    20:17:48.16 <<< Return value from vision_learner: <fastai.learner.Learner object>



```
# the official source
@delegates(create_vision_model) 
def vision_learner(dls, arch, normalize=True, n_out=None, pretrained=True, 
        # learner args
        loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
        model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95),
        # model & head args
        cut=None, init=nn.init.kaiming_normal_, custom_head=None, concat_pool=True, pool=True,
        lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None, **kwargs):
    "Build a vision learner from `dls` and `arch`"
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    meta = model_meta.get(arch, _default_meta)
    model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
                      first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    if isinstance(arch, str):
        model,cfg = create_timm_model(arch, n_out, default_split, pretrained, **model_args)
        if normalize: _timm_norm(dls, cfg, pretrained)
    else:
        if normalize: _add_norm(dls, meta, pretrained)
        model = create_vision_model(arch, n_out, pretrained=pretrained, **model_args)
    
    splitter=ifnone(splitter, meta['split'])
    learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
    if pretrained: learn.freeze()
    # keep track of args for loggers
    store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    return learn
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/learner.py
# Type:      function
```


```
learn.__dict__.keys()
```




    dict_keys(['dls', 'model', '__stored_args__', 'loss_func', 'opt_func', 'lr', 'splitter', '_metrics', 'path', 'model_dir', 'wd', 'wd_bn_bias', 'train_bn', 'moms', 'default_cbs', 'training', 'create_mbar', 'logger', 'opt', 'cbs', 'train_eval', 'recorder', 'cast_to_tensor', 'progress', 'lock', 'n_epoch', 'arch', 'normalize', 'n_out', 'pretrained'])



### fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100...)
Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR.


```
@patch
@snoop
@delegates(Learner.fit_one_cycle)
def fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,
              pct_start=0.3, div=5.0, **kwargs):
    "Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR."
    pp(doc_sig(self.freeze))
    self.freeze()
    pp(doc_sig(self.fit_one_cycle))
    pp(freeze_epochs, slice(base_lr), kwargs)
    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    base_lr /= 2
    pp(doc_sig(self.unfreeze))
    self.unfreeze()
    pp(epochs, slice(base_lr/lr_mult, base_lr), pct_start, div, kwargs)
    self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/callback/schedule.py
# Type:      method
```


```
learn.fine_tune(1)
```

    20:17:48.26 >>> Call to fine_tune in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_13247/3127910192.py", line 4
    20:17:48.26 ...... self = <fastai.learner.Learner object>
    20:17:48.26 ...... epochs = 1
    20:17:48.26 ...... base_lr = 0.002
    20:17:48.26 ...... freeze_epochs = 1
    20:17:48.26 ...... lr_mult = 100
    20:17:48.26 ...... pct_start = 0.3
    20:17:48.26 ...... div = 5.0
    20:17:48.26 ...... kwargs = {}
    20:17:48.26    4 | def fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,
    20:17:48.26    7 |     pp(doc_sig(self.freeze))
    20:17:48.26 LOG:
    20:17:48.27 .... doc_sig(self.freeze) = ('no mro', 'Freeze up to last parameter group', <Signature ()>)
    20:17:48.27    8 |     self.freeze()
    20:17:48.27    9 |     pp(doc_sig(self.fit_one_cycle))
    20:17:48.27 LOG:
    20:17:48.28 .... doc_sig(self.fit_one_cycle) = ('no mro',
    20:17:48.28                                     'Fit `self.model` for `n_epoch` using the 1cycle policy.',
    20:17:48.28                                     <Signature (n_epoch, lr_max=None, div=25.0, div_final=100000.0, pct_start=0.25, wd=None, moms=None, cbs=None, reset_opt=False, start_epoch=0)>)
    20:17:48.28   10 |     pp(freeze_epochs, slice(base_lr), kwargs)
    20:17:48.28 LOG:
    20:17:48.28 .... freeze_epochs = 1
    20:17:48.28 .... slice(base_lr) = slice(None, 0.002, None)
    20:17:48.28 .... kwargs = {}
    20:17:48.28   11 |     self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





    <div>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00&lt;?]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
      <progress value='1' class='' max='3' style='width:300px; height:20px; vertical-align: middle;'></progress>
      33.33% [1/3 00:07&lt;00:15]
    </div>



    20:17:56.71 !!! KeyboardInterrupt
    20:17:56.71 !!! When calling: self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    20:17:56.71 !!! Call ended by exception



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Input In [124], in <cell line: 1>()
    ----> 1 learn.fine_tune(1)


    File ~/mambaforge/lib/python3.9/site-packages/snoop/tracer.py:170, in Tracer.__call__.<locals>.simple_wrapper(*args, **kwargs)
        167 @functools.wraps(function)
        168 def simple_wrapper(*args, **kwargs):
        169     with self:
    --> 170         return function(*args, **kwargs)


    Input In [123], in fine_tune(self, epochs, base_lr, freeze_epochs, lr_mult, pct_start, div, **kwargs)
          9 pp(doc_sig(self.fit_one_cycle))
         10 pp(freeze_epochs, slice(base_lr), kwargs)
    ---> 11 self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
         12 base_lr /= 2
         13 pp(doc_sig(self.unfreeze))


    File ~/mambaforge/lib/python3.9/site-packages/fastai/callback/schedule.py:119, in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
        116 lr_max = np.array([h['lr'] for h in self.opt.hypers])
        117 scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
        118           'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
    --> 119 self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:256, in Learner.fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
        254 self.opt.set_hypers(lr=self.lr if lr is None else lr)
        255 self.n_epoch = n_epoch
    --> 256 self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:245, in Learner._do_fit(self)
        243 for epoch in range(self.n_epoch):
        244     self.epoch=epoch
    --> 245     self._with_events(self._do_epoch, 'epoch', CancelEpochException)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:239, in Learner._do_epoch(self)
        238 def _do_epoch(self):
    --> 239     self._do_epoch_train()
        240     self._do_epoch_validate()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:231, in Learner._do_epoch_train(self)
        229 def _do_epoch_train(self):
        230     self.dl = self.dls.train
    --> 231     self._with_events(self.all_batches, 'train', CancelTrainException)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:199, in Learner.all_batches(self)
        197 def all_batches(self):
        198     self.n_iter = len(self.dl)
    --> 199     for o in enumerate(self.dl): self.one_batch(*o)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/load.py:127, in DataLoader.__iter__(self)
        125 self.before_iter()
        126 self.__idxs=self.get_idxs() # called in context of main process (not workers/subprocesses)
    --> 127 for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
        128     # pin_memory causes tuples to be converted to lists, so convert them back to tuples
        129     if self.pin_memory and type(b) == list: b = tuple(b)
        130     if self.device is not None: b = to_device(b, self.device)


    File ~/mambaforge/lib/python3.9/site-packages/torch/utils/data/dataloader.py:681, in _BaseDataLoaderIter.__next__(self)
        678 if self._sampler_iter is None:
        679     # TODO(https://github.com/pytorch/pytorch/issues/76750)
        680     self._reset()  # type: ignore[call-arg]
    --> 681 data = self._next_data()
        682 self._num_yielded += 1
        683 if self._dataset_kind == _DatasetKind.Iterable and \
        684         self._IterableDataset_len_called is not None and \
        685         self._num_yielded > self._IterableDataset_len_called:


    File ~/mambaforge/lib/python3.9/site-packages/torch/utils/data/dataloader.py:721, in _SingleProcessDataLoaderIter._next_data(self)
        719 def _next_data(self):
        720     index = self._next_index()  # may raise StopIteration
    --> 721     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        722     if self._pin_memory:
        723         data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)


    File ~/mambaforge/lib/python3.9/site-packages/torch/utils/data/_utils/fetch.py:39, in _IterableDatasetFetcher.fetch(self, possibly_batched_index)
         37         raise StopIteration
         38 else:
    ---> 39     data = next(self.dataset_iter)
         40 return self.collate_fn(data)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/load.py:138, in DataLoader.create_batches(self, samps)
        136 if self.dataset is not None: self.it = iter(self.dataset)
        137 res = filter(lambda o:o is not None, map(self.do_item, samps))
    --> 138 yield from map(self.do_batch, self.chunkify(res))


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/basics.py:230, in chunked(it, chunk_sz, drop_last, n_chunks)
        228 if not isinstance(it, Iterator): it = iter(it)
        229 while True:
    --> 230     res = list(itertools.islice(it, chunk_sz))
        231     if res and (len(res)==chunk_sz or not drop_last): yield res
        232     if len(res)<chunk_sz: return


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/load.py:153, in DataLoader.do_item(self, s)
        152 def do_item(self, s):
    --> 153     try: return self.after_item(self.create_item(s))
        154     except SkipItemException: return None


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/load.py:160, in DataLoader.create_item(self, s)
        159 def create_item(self, s):
    --> 160     if self.indexed: return self.dataset[s or 0]
        161     elif s is None:  return next(self.it)
        162     else: raise IndexError("Cannot index an iterable dataset numerically - must use `None`.")


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/core.py:455, in Datasets.__getitem__(self, it)
        454 def __getitem__(self, it):
    --> 455     res = tuple([tl[it] for tl in self.tls])
        456     return res if is_indexer(it) else list(zip(*res))


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/core.py:455, in <listcomp>(.0)
        454 def __getitem__(self, it):
    --> 455     res = tuple([tl[it] for tl in self.tls])
        456     return res if is_indexer(it) else list(zip(*res))


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/core.py:414, in TfmdLists.__getitem__(self, idx)
        412 res = super().__getitem__(idx)
        413 if self._after_item is None: return res
    --> 414 return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/data/core.py:374, in TfmdLists._after_item(self, o)
    --> 374 def _after_item(self, o): return self.tfms(o)


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py:208, in Pipeline.__call__(self, o)
    --> 208 def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py:158, in compose_tfms(x, tfms, is_enc, reverse, **kwargs)
        156 for f in tfms:
        157     if not is_enc: f = f.decode
    --> 158     x = f(x, **kwargs)
        159 return x


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py:81, in Transform.__call__(self, x, **kwargs)
    ---> 81 def __call__(self, x, **kwargs): return self._call('encodes', x, **kwargs)


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py:91, in Transform._call(self, fn, x, split_idx, **kwargs)
         89 def _call(self, fn, x, split_idx=None, **kwargs):
         90     if split_idx!=self.split_idx and self.split_idx is not None: return x
    ---> 91     return self._do_call(getattr(self, fn), x, **kwargs)


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/transform.py:97, in Transform._do_call(self, f, x, **kwargs)
         95     if f is None: return x
         96     ret = f.returns(x) if hasattr(f,'returns') else None
    ---> 97     return retain_type(f(x, **kwargs), x, ret)
         98 res = tuple(self._do_call(f, x_, **kwargs) for x_ in x)
         99 return retain_type(res, x)


    File ~/mambaforge/lib/python3.9/site-packages/fastcore/dispatch.py:120, in TypeDispatch.__call__(self, *args, **kwargs)
        118 elif self.inst is not None: f = MethodType(f, self.inst)
        119 elif self.owner is not None: f = MethodType(f, self.owner)
    --> 120 return f(*args, **kwargs)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/vision/core.py:123, in PILBase.create(cls, fn, **kwargs)
        121 if isinstance(fn,ndarray): return cls(Image.fromarray(fn))
        122 if isinstance(fn,bytes): fn = io.BytesIO(fn)
    --> 123 return cls(load_image(fn, **merge(cls._open_args, kwargs)))


    File ~/mambaforge/lib/python3.9/site-packages/fastai/vision/core.py:99, in load_image(fn, mode)
         97 "Open and load a `PIL.Image` and convert to `mode`"
         98 im = Image.open(fn)
    ---> 99 im.load()
        100 im = im._new(im.im)
        101 return im.convert(mode) if mode else im


    File ~/mambaforge/lib/python3.9/site-packages/PIL/ImageFile.py:257, in ImageFile.load(self)
        251         raise OSError(
        252             "image file is truncated "
        253             f"({len(b)} bytes not processed)"
        254         )
        256 b = b + s
    --> 257 n, err_code = decoder.decode(b)
        258 if n < 0:
        259     break


    KeyboardInterrupt: 



```
@patch
@delegates(Learner.fit_one_cycle)
def fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,
              pct_start=0.3, div=5.0, **kwargs):
    "Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR."
    self.freeze()
    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    base_lr /= 2
    self.unfreeze()
    self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/callback/schedule.py
# Type:      method
```


```

```

### CPU is fine and quick with small dataset like this one

Now we're ready to train our model. The fastest widely used computer vision model is resnet18. You can train this in a few minutes, even on a CPU! (On a GPU, it generally takes under 10 seconds...)

fastai comes with a helpful `fine_tune()` method which automatically uses best practices for fine tuning a pre-trained model, so we'll use that.


```
#|eval: false
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```


```
# Path.unlink(Path('/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird/bird/82a964e3-e3db-4f05-895a-27306fe11a71.jpg'))
```

### Learner.predict(self:Learner, item, rm_type_tfms=None, with_input=False)
use learner to predict on a single item, and return 3 things: target (eg., bird or forest), 0 or 1, prob of bird and prob of forest


```
@patch
@snoop
def predict(self:Learner, item, rm_type_tfms=None, with_input=False):
    pp(doc_sig(self.dls.test_dl))
    dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
    pp(doc_sig(self.get_preds))
    inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
    i = getattr(self.dls, 'n_inp', -1)
    if i==1:
        inp = (inp,) 
    else:
        inp = tuplify(inp)
    pp(doc_sig(self.dls.decode_batch))
    dec = self.dls.decode_batch(inp + tuplify(dec_preds))[0]
    pp(doc_sig(tuplify), doc_sig(detuplify))
    dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
    res = dec_targ,dec_preds[0],preds[0]
    if with_input: 
        res = (dec_inp,) + res
    return res
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py
# Type:      method
```


```
randomdisplay(bird/"bird")
```


```
is_bird,_,probs = learn.predict(PILImage.create(Path('forest_or_bird/bird/fa32d017-01fc-4175-be53-16014ea7f683.jpg')))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
```


```

```


```
@patch
def predict(self:Learner, item, rm_type_tfms=None, with_input=False):
    dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
    inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
    i = getattr(self.dls, 'n_inp', -1)
    inp = (inp,) if i==1 else tuplify(inp)
    dec = self.dls.decode_batch(inp + tuplify(dec_preds))[0]
    dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
    res = dec_targ,dec_preds[0],preds[0]
    if with_input: res = (dec_inp,) + res
    return res
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py
# Type:      method
```


```

```


```
# fastnbs("DataBlock.", strict=True)
```


```

```
