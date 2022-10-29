# 0001_fastai_Is it a bird? Creating a model from your own data
---
skip_exec: true
---
## Useful Course sites

**Official course site**:  for lesson [1](https://course.fast.ai/Lessons/lesson1.html)    

**Official notebooks** [repo](https://github.com/fastai/course22), on [nbviewer](https://nbviewer.org/github/fastai/course22/tree/master/)

Official **Is it a bird** [notebook](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data) on kaggle     



```
from fastdebug.utils import *
from __future__ import annotations
```


<style>.container { width:100% !important; }</style>


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
whichversion("fastai")
```

    fastai: 2.7.9 
    fastai simplifies training fast and accurate neural nets using modern best practices    
    Jeremy Howard, Sylvain Gugger, and contributors 
    https://github.com/fastai/fastai/tree/master/     
    python_version: >=3.7     
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastai



```
whatinside(fastai, lib=True)
```

    The library has 24 modules
    ['_modidx',
     '_nbdev',
     '_pytorch_doc',
     'basics',
     'callback',
     'collab',
     'data',
     'distributed',
     'fp16_utils',
     'imports',
     'interpret',
     'layers',
     'learner',
     'losses',
     'medical',
     'metrics',
     'optimizer',
     'tabular',
     'test_utils',
     'text',
     'torch_basics',
     'torch_core',
     'torch_imports',
     'vision']



```
import duckduckgo_search
```


```
whichversion("duckduckgo_search")
```

    duckduckgo-search: 2.2.2 
    Search for words, documents, images, news, maps and text translation using the DuckDuckGo.com search engine.    
    deedy5 
    https://github.com/deedy5/duckduckgo_search     
    python_version: >=3.7     
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/duckduckgo_search



```
whatinside(duckduckgo_search)
```

    duckduckgo_search has: 
    0 items in its __all__, and 
    6 user defined functions, 
    0 classes or class objects, 
    0 builtin funcs and methods, and
    6 callables.
    
    Duckduckgo_search
    ~~~~~~~~~~~~~~
    Search for words, documents, images, videos, news, maps and text translation
    using the DuckDuckGo.com search engine.



```
whatinside(duckduckgo_search, func=True)
```

    duckduckgo_search has: 
    0 items in its __all__, and 
    6 user defined functions, 
    0 classes or class objects, 
    0 builtin funcs and methods, and
    6 callables.
    
    Duckduckgo_search
    ~~~~~~~~~~~~~~
    Search for words, documents, images, videos, news, maps and text translation
    using the DuckDuckGo.com search engine.
    The user defined functions are:
    ddg:               function    (keywords, region='wt-wt', safesearch='Moderate', time=None, max_results=25, output=None)
    ddg_images:        function    (keywords, region='wt-wt', safesearch='Moderate', time=None, size=None, color=None, type_image=None, layout=None, license_image=None, max_results=100, output=None, download=False)
    ddg_maps:          function    (keywords, place=None, street=None, city=None, county=None, state=None, country=None, postalcode=None, latitude=None, longitude=None, radius=0, max_results=None, output=None)
    ddg_news:          function    (keywords, region='wt-wt', safesearch='Moderate', time=None, max_results=25, output=None)
    ddg_translate:     function    (keywords, from_=None, to='en', output=None)
    ddg_videos:        function    (keywords, region='wt-wt', safesearch='Moderate', time=None, resolution=None, duration=None, license_videos=None, max_results=50, output=None)



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

### ```itemgot(self:L, *idxs)```
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

### ```itemgetter```
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





    (#3) ['https://www.highreshdwallpapers.com/wp-content/uploads/2014/05/Colourful-Flying-Bird.jpg','http://2.bp.blogspot.com/-LZ4VixDdVoE/Tq0ZhPycLsI/AAAAAAAADDM/OKyayfW-z4U/s1600/beautiful_Birds_wallpapers_pictures_Kingfisher_Lilac+Breasted+Roller+Bird.JPG','https://amazinganimalphotos.com/wp-content/uploads/2016/11/beautiful-birds.jpeg']






    'https://www.highreshdwallpapers.com/wp-content/uploads/2014/05/Colourful-Flying-Bird.jpg'




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
  <progress value='311296' class='' max='308309' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.97% [311296/308309 00:01&lt;00:00]
</div>






    Path('bird.jpg')



### ```Image.open(filename)```


```
from fastai.vision.all import *
```


```
im = Image.open(dest)
im.width
im.height
```




    1600






    1200



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

    22:38:39.68 >>> Call to to_thumb in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/1866859134.py", line 3
    22:38:39.68 ...... self = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1600x1200>
    22:38:39.68 ...... h = 256
    22:38:39.68 ...... w = 256
    22:38:39.68    3 | def to_thumb(self:Image.Image, h, w=None):
    22:38:39.68    5 |     if w is None: 
    22:38:39.69    7 |     im = self.copy()
    22:38:39.70 .......... im = <PIL.Image.Image image mode=RGB size=1600x1200>
    22:38:39.70    8 |     im.thumbnail((w,h))
    22:38:39.70 .......... im = <PIL.Image.Image image mode=RGB size=256x192>
    22:38:39.70    9 |     return im
    22:38:39.70 <<< Return value from to_thumb: <PIL.Image.Image image mode=RGB size=256x192>





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_38_1.png)
    




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

    Searching for 'Spinosaurus aegyptiacus photos'





    Path('forest.jpg')






    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_40_2.png)
    




```

```

### PILImage.create or PILBase.create(fn, **kwargs)
Open an `Image` from `fn`, which can be path or str, Tensor, numpy, ndarray, bytes


```
doc(PILImage.create)
```


<hr/>
<h3>PILBase.create</h3>
<blockquote><pre><code>PILBase.create(fn:Union[pathlib.Path,str,torch.Tensor,numpy.ndarray,bytes], **kwargs)</code></pre></blockquote><p>Open an `Image` from path `fn`</p>
<p><a href="https://docs.fast.ai/vision.core.html#pilbase.create" target="_blank" rel="noreferrer noopener">Show in docs</a></p>



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

    22:38:43.91 >>> Call to PILBase.create in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/1026981879.py", line 7
    22:38:43.91 .......... cls = <class '__main__.PILBase'>
    22:38:43.91 .......... fn = 'forest.jpg'
    22:38:43.91 .......... kwargs = {}
    22:38:43.91    7 |     def create(cls, fn:Path|str|Tensor|ndarray|bytes, **kwargs)->None:
    22:38:43.91    9 |         if isinstance(fn,TensorImage): 
    22:38:43.91   11 |         if isinstance(fn, TensorMask): 
    22:38:43.91   13 |         if isinstance(fn,Tensor): 
    22:38:43.91   15 |         if isinstance(fn,ndarray): 
    22:38:43.91   17 |         if isinstance(fn,bytes): 
    22:38:43.91   19 |         pp(cls._open_args, kwargs, merge(cls._open_args, kwargs))
    22:38:43.91 LOG:
    22:38:44.21 .... cls._open_args = {'mode': 'RGB'}
    22:38:44.22 .... kwargs = {}
    22:38:44.22 .... merge(cls._open_args, kwargs) = {'mode': 'RGB'}
    22:38:44.22   20 |         res = cls(load_image(fn, **merge(cls._open_args, kwargs)))
    22:38:44.23 .............. res = PILBase mode=RGB size=1920x1080
    22:38:44.23   21 |         return res
    22:38:44.23 <<< Return value from PILBase.create: PILBase mode=RGB size=1920x1080





    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_45_1.png)
    




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

    22:38:44.81 >>> Call to resize_image in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/4048456158.py", line 2
    22:38:44.81 ...... file = 'bird.jpg'
    22:38:44.81 ...... dest = Path('resized')
    22:38:44.81 ...... src = Path('.')
    22:38:44.81 ...... max_size = 400
    22:38:44.81 ...... n_channels = 3
    22:38:44.81 ...... ext = None
    22:38:44.81 ...... img_format = None
    22:38:44.81 ...... resample = <Resampling.BILINEAR: 2>
    22:38:44.81 ...... resume = False
    22:38:44.81 ...... kwargs = {}
    22:38:44.81    2 | def resize_image(file, # str for image filename
    22:38:44.81    9 |     dest = Path(dest)
    22:38:44.81   11 |     dest_fname = dest/file
    22:38:44.81 .......... dest_fname = Path('resized/bird.jpg')
    22:38:44.81   12 |     dest_fname.parent.mkdir(exist_ok=True, parents=True)
    22:38:44.82   13 |     file = Path(src)/file
    22:38:44.82 .......... file = Path('bird.jpg')
    22:38:44.82   14 |     if resume and dest_fname.exists(): return
    22:38:44.82   15 |     if not verify_image(file): return
    22:38:44.82   17 |     img = Image.open(file)
    22:38:44.82 .......... img = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080>
    22:38:44.82   18 |     imgarr = np.array(img)
    22:38:44.84 .......... imgarr = array([[[37, 39, 28],
    22:38:44.84                             [36, 38, 27],
    22:38:44.84                             ...,
    22:38:44.84                             [33, 35, 24],
    22:38:44.84                             [34, 36, 25]],
    22:38:44.84                     
    22:38:44.84                            [[37, 39, 28],
    22:38:44.84                             [35, 37, 26],
    22:38:44.84                             ...,
    22:38:44.84                             [34, 36, 25],
    22:38:44.84                             [34, 36, 25]],
    22:38:44.84                     
    22:38:44.84                            ...,
    22:38:44.84                     
    22:38:44.84                            [[ 7,  7,  7],
    22:38:44.84                             [ 7,  7,  7],
    22:38:44.84                             ...,
    22:38:44.84                             [ 6,  6,  6],
    22:38:44.84                             [ 6,  6,  6]],
    22:38:44.84                     
    22:38:44.84                            [[ 7,  7,  7],
    22:38:44.84                             [ 7,  7,  7],
    22:38:44.84                             ...,
    22:38:44.84                             [ 6,  6,  6],
    22:38:44.84                             [ 6,  6,  6]]], dtype=uint8)
    22:38:44.84   19 |     img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
    22:38:44.84 .......... img_channels = 3
    22:38:44.84   20 |     if ext is not None: dest_fname=dest_fname.with_suffix(ext) # specify file extensions
    22:38:44.84   21 |     if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
    22:38:44.84   22 |         if max_size is not None:
    22:38:44.84   23 |             pp(doc_sig(resize_to))
    22:38:44.84 LOG:
    22:38:44.86 .... doc_sig(resize_to) = ('no mro',
    22:38:44.86                            'Size to resize to, to hit `targ_sz` at same aspect ratio, in PIL coords (i.e '
    22:38:44.86                            'w*h)',
    22:38:44.86                            <Signature (img, targ_sz, use_min=False)>)
    22:38:44.86   24 |             pp(img.height, img.width)
    22:38:44.87 LOG:
    22:38:44.87 .... img.height = 1080
    22:38:44.87 .... img.width = 1920
    22:38:44.87   25 |             new_sz = resize_to(img, max_size) # keep the ratio
    22:38:44.87 .................. new_sz = (400, 225)
    22:38:44.87   26 |             pp(doc_sig(img.resize))
    22:38:44.87 LOG:
    22:38:44.88 .... doc_sig(img.resize) = ('no mro',
    22:38:44.88                             'Returns a resized copy of this image.\n'
    22:38:44.88                             '\n'
    22:38:44.88                             ':param size: The requested size in pixels, as a 2-tuple:\n'
    22:38:44.88                             '   (width, height).\n'
    22:38:44.88                             ':param resample: An optional resampling filter.  This can be\n'
    22:38:44.88                             '   one of :py:data:`PIL.Image.Resampling.NEAREST`,\n'
    22:38:44.88                             '   :py:data:`PIL.Image.Resampling.BOX`,\n'
    22:38:44.88                             '   :py:data:`PIL.Image.Resampling.BILINEAR`,\n'
    22:38:44.88                             '   :py:data:`PIL.Image.Resampling.HAMMING`,\n'
    22:38:44.88                             '   :py:data:`PIL.Image.Resampling.BICUBIC` or\n'
    22:38:44.88                             '   :py:data:`PIL.Image.Resampling.LANCZOS`.\n'
    22:38:44.88                             '   If the image has mode "1" or "P", it is always set to\n'
    22:38:44.88                             '   :py:data:`PIL.Image.Resampling.NEAREST`.\n'
    22:38:44.88                             '   If the image mode specifies a number of bits, such as "I;16", then the\n'
    22:38:44.88                             '   default filter is :py:data:`PIL.Image.Resampling.NEAREST`.\n'
    22:38:44.88                             '   Otherwise, the default filter is\n'
    22:38:44.88                             '   :py:data:`PIL.Image.Resampling.BICUBIC`. See: :ref:`concept-filters`.\n'
    22:38:44.88                             ':param box: An optional 4-tuple of floats providing\n'
    22:38:44.88                             '   the source image region to be scaled.\n'
    22:38:44.88                             '   The values must be within (0, 0, width, height) rectangle.\n'
    22:38:44.88                             '   If omitted or None, the entire source is used.\n'
    22:38:44.88                             ':param reducing_gap: Apply optimization by resizing the image\n'
    22:38:44.88                             '   in two steps. First, reducing the image by integer times\n'
    22:38:44.88                             '   using :py:meth:`~PIL.Image.Image.reduce`.\n'
    22:38:44.88                             '   Second, resizing using regular resampling. The last step\n'
    22:38:44.88                             '   changes size no less than by ``reducing_gap`` times.\n'
    22:38:44.88                             '   ``reducing_gap`` may be None (no first step is performed)\n'
    22:38:44.88                             '   or should be greater than 1.0. The bigger ``reducing_gap``,\n'
    22:38:44.88                             '   the closer the result to the fair resampling.\n'
    22:38:44.88                             '   The smaller ``reducing_gap``, the faster resizing.\n'
    22:38:44.88                             '   With ``reducing_gap`` greater or equal to 3.0, the result is\n'
    22:38:44.88                             '   indistinguishable from fair resampling in most cases.\n'
    22:38:44.88                             '   The default value is None (no optimization).\n'
    22:38:44.88                             ':returns: An :py:class:`~PIL.Image.Image` object.',
    22:38:44.88                             <Signature (size, resample=None, box=None, reducing_gap=None)>)
    22:38:44.88   27 |             img = img.resize(new_sz, resample=resample)
    22:38:44.89 .................. img = <PIL.Image.Image image mode=RGB size=400x225>
    22:38:44.89   28 |         if n_channels == 3: 
    22:38:44.89   29 |             img = img.convert("RGB")
    22:38:44.89   30 |         pp(doc_sig(img.save))
    22:38:44.89 LOG:
    22:38:44.90 .... doc_sig(img.save) = ('no mro',
    22:38:44.90                           'Saves this image under the given filename.  If no format is\n'
    22:38:44.90                           'specified, the format to use is determined from the filename\n'
    22:38:44.90                           'extension, if possible.\n'
    22:38:44.90                           '\n'
    22:38:44.90                           'Keyword options can be used to provide additional instructions\n'
    22:38:44.90                           "to the writer. If a writer doesn't recognise an option, it is\n"
    22:38:44.90                           'silently ignored. The available options are described in the\n'
    22:38:44.90                           ':doc:`image format documentation\n'
    22:38:44.90                           '<../handbook/image-file-formats>` for each writer.\n'
    22:38:44.90                           '\n'
    22:38:44.90                           'You can use a file object instead of a filename. In this case,\n'
    22:38:44.90                           'you must always specify the format. The file object must\n'
    22:38:44.90                           'implement the ``seek``, ``tell``, and ``write``\n'
    22:38:44.90                           'methods, and be opened in binary mode.\n'
    22:38:44.90                           '\n'
    22:38:44.90                           ':param fp: A filename (string), pathlib.Path object or file object.\n'
    22:38:44.90                           ':param format: Optional format override.  If omitted, the\n'
    22:38:44.90                           '   format to use is determined from the filename extension.\n'
    22:38:44.90                           '   If a file object was used instead of a filename, this\n'
    22:38:44.90                           '   parameter should always be used.\n'
    22:38:44.90                           ':param params: Extra parameters to the image writer.\n'
    22:38:44.90                           ':returns: None\n'
    22:38:44.90                           ':exception ValueError: If the output format could not be determined\n'
    22:38:44.90                           '   from the file name.  Use the format option to solve this.\n'
    22:38:44.90                           ':exception OSError: If the file could not be written.  The file\n'
    22:38:44.90                           '   may have been created, and may contain partial data.',
    22:38:44.90                           <Signature (fp, format=None, **params)>)
    22:38:44.90   31 |         img.save(dest_fname, img_format, **kwargs)
    22:38:44.90 <<< Return value from resize_image: None



```
file = 'bird.jpg'
src = Path('.')
dest = src/"resized"
resize_image(file, dest, src=src, max_size=None) # just copy not size changed
im = Image.open(dest/file)
test_eq(im.shape[1],1920)
```

    22:38:44.92 >>> Call to resize_image in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/4048456158.py", line 2
    22:38:44.92 ...... file = 'bird.jpg'
    22:38:44.92 ...... dest = Path('resized')
    22:38:44.92 ...... src = Path('.')
    22:38:44.92 ...... max_size = None
    22:38:44.92 ...... n_channels = 3
    22:38:44.92 ...... ext = None
    22:38:44.92 ...... img_format = None
    22:38:44.92 ...... resample = <Resampling.BILINEAR: 2>
    22:38:44.92 ...... resume = False
    22:38:44.92 ...... kwargs = {}
    22:38:44.92    2 | def resize_image(file, # str for image filename
    22:38:44.92    9 |     dest = Path(dest)
    22:38:44.92   11 |     dest_fname = dest/file
    22:38:44.92 .......... dest_fname = Path('resized/bird.jpg')
    22:38:44.92   12 |     dest_fname.parent.mkdir(exist_ok=True, parents=True)
    22:38:44.92   13 |     file = Path(src)/file
    22:38:44.93 .......... file = Path('bird.jpg')
    22:38:44.93   14 |     if resume and dest_fname.exists(): return
    22:38:44.93   15 |     if not verify_image(file): return
    22:38:44.93   17 |     img = Image.open(file)
    22:38:44.93 .......... img = <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080>
    22:38:44.93   18 |     imgarr = np.array(img)
    22:38:44.95 .......... imgarr = array([[[37, 39, 28],
    22:38:44.95                             [36, 38, 27],
    22:38:44.95                             ...,
    22:38:44.95                             [33, 35, 24],
    22:38:44.95                             [34, 36, 25]],
    22:38:44.95                     
    22:38:44.95                            [[37, 39, 28],
    22:38:44.95                             [35, 37, 26],
    22:38:44.95                             ...,
    22:38:44.95                             [34, 36, 25],
    22:38:44.95                             [34, 36, 25]],
    22:38:44.95                     
    22:38:44.95                            ...,
    22:38:44.95                     
    22:38:44.95                            [[ 7,  7,  7],
    22:38:44.95                             [ 7,  7,  7],
    22:38:44.95                             ...,
    22:38:44.95                             [ 6,  6,  6],
    22:38:44.95                             [ 6,  6,  6]],
    22:38:44.95                     
    22:38:44.95                            [[ 7,  7,  7],
    22:38:44.95                             [ 7,  7,  7],
    22:38:44.95                             ...,
    22:38:44.95                             [ 6,  6,  6],
    22:38:44.95                             [ 6,  6,  6]]], dtype=uint8)
    22:38:44.95   19 |     img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
    22:38:44.95 .......... img_channels = 3
    22:38:44.95   20 |     if ext is not None: dest_fname=dest_fname.with_suffix(ext) # specify file extensions
    22:38:44.95   21 |     if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
    22:38:44.95   32 |     elif file != dest_fname : 
    22:38:44.95   33 |         shutil.copy2(file, dest_fname)
    22:38:44.96 <<< Return value from resize_image: None



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

    22:38:45.03 >>> Call to resize_images in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/1408674014.py", line 2
    22:38:45.03 ...... path = 'bird_or_not'
    22:38:45.03 ...... max_workers = 0
    22:38:45.03 ...... max_size = 100
    22:38:45.03 ...... recurse = True
    22:38:45.03 ...... dest = Path('try_resize_images')
    22:38:45.03 ...... n_channels = 3
    22:38:45.03 ...... ext = None
    22:38:45.03 ...... img_format = None
    22:38:45.03 ...... resample = <Resampling.BILINEAR: 2>
    22:38:45.03 ...... resume = None
    22:38:45.03 ...... kwargs = {}
    22:38:45.03    2 | def resize_images(path, max_workers=defaults.cpus, max_size=None, recurse=False,
    22:38:45.03    6 |     path = Path(path)
    22:38:45.03 .......... path = Path('bird_or_not')
    22:38:45.03    7 |     if resume is None and dest != Path('.'): 
    22:38:45.03    8 |         resume=False
    22:38:45.03 .............. resume = False
    22:38:45.03    9 |     os.makedirs(dest, exist_ok=True)
    22:38:45.03   10 |     files = get_image_files(path, recurse=recurse)
    22:38:45.03 .......... files = [Path('bird_or_not/0b8fcba5-91a5-4689-999c-008e1.../bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')]
    22:38:45.03   11 |     files = [o.relative_to(path) for o in files]
        22:38:45.03 List comprehension:
        22:38:45.03   11 |     files = [o.relative_to(path) for o in files]
        22:38:45.03 .......... Iterating over <list_iterator object>
        22:38:45.03 .......... Values of path: Path('bird_or_not')
        22:38:45.03 .......... Values of o: Path('bird_or_not/0b8fcba5-91a5-4689-999c-008e108828f1.jpg'), Path('bird_or_not/037e9e61-3731-4876-9745-98758ae21be3.jpg'), Path('bird_or_not/forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg'), Path('bird_or_not/bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')
        22:38:45.03 Result: [Path('0b8fcba5-91a5-4689-999c-008e108828f1.jpg'), Path('037e9e61-3731-4876-9745-98758ae21be3.jpg'), Path('forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg'), Path('bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')]
    22:38:45.03   11 |     files = [o.relative_to(path) for o in files]
    22:38:45.04 .......... files = [Path('0b8fcba5-91a5-4689-999c-008e108828f1.jpg'), Path('037e9e61-3731-4876-9745-98758ae21be3.jpg'), Path('forest/02af1f04-3387-4bc8-a108-e209e2ae69cc.jpg'), Path('bird/003ca626-2352-4ddb-9ead-69041ec99473.jpg')]
    22:38:45.04   12 |     parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
    22:38:45.04   13 |                    img_format=img_format, resample=resample, resume=resume, **kwargs)
    22:38:45.04   12 |     parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
    22:38:45.04   13 |                    img_format=img_format, resample=resample, resume=resume, **kwargs)
    22:38:45.04   12 |     parallel(resize_image, files, src=path, n_workers=max_workers, max_size=max_size, dest=dest, n_channels=n_channels, ext=ext,
    22:38:45.05 <<< Return value from resize_images: None



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
#| export utils
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

    22:38:45.14 LOG:
    22:38:45.14 .... Path(folder_name) = Path('T-rex_or_Brachiosaurus')


    T-rex_or_Brachiosaurus/T-rex
    174
    T-rex_or_Brachiosaurus/Brachiosaurus
    111



```
bird = prepare_images_dataset_binary("forest", "bird")
```

    22:38:45.16 LOG:
    22:38:45.17 .... Path(folder_name) = Path('forest_or_bird')


    forest_or_bird/forest
    175
    forest_or_bird/bird
    82



```
dino = prepare_images_dataset_binary("T-rex", "Spinosaurus aegyptiacus")
```

    22:38:45.19 LOG:
    22:38:45.19 .... Path(folder_name) = Path('T-rex_or_Spinosaurus aegyptiacus')


    T-rex_or_Spinosaurus aegyptiacus/T-rex
    84
    T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus
    82


### ```randomdisplay(path)```
to randomly display images in their path


```
# @snoop
def randomdisplay(path, db=False):
# https://www.geeksforgeeks.org/python-random-module/
    import random
    rand = random.randint(0,len(path.ls())-1) # choose a random int between 0 and len(T-rex)-1
    file = path.ls()[rand]
    im = PILImage.create(file)
    if db: pp(im.width, im.height, file)
    return file, im

```


```
dino.ls()
bird.ls()
```




    (#4) [Path('T-rex_or_Spinosaurus aegyptiacus/crying'),Path('T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex'),Path('T-rex_or_Spinosaurus aegyptiacus/fierce')]






    (#2) [Path('forest_or_bird/forest'),Path('forest_or_bird/bird')]




```
randomdisplay(bird/'bird')
randomdisplay(bird/'forest')
```




    (Path('forest_or_bird/bird/97afca4b-236b-4935-96f1-fc78345b1479.jpg'),
     PILImage mode=RGB size=400x300)






    (Path('forest_or_bird/forest/cfea9288-5ed3-490a-b933-b52a714cd915.jpg'),
     PILImage mode=RGB size=1920x1200)




```
randomdisplay(dino/"Spinosaurus aegyptiacus")
```




    (Path('T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus/9141bdb2-0f06-4b20-88af-3bc317f6eb85.jpg'),
     PILImage mode=RGB size=400x225)




```
randomdisplay(dino/"T-rex")
```




    (Path('T-rex_or_Spinosaurus aegyptiacus/T-rex/db19c338-032f-45be-99f4-3431a407e54e.jpeg'),
     PILImage mode=RGB size=400x225)




```
randomdisplay(cry_dino/"T-rex")
```




    (Path('T-rex_or_Brachiosaurus/T-rex/7a0d5b8f-bd8f-4362-afcf-c1cbcd3fa334.jpg'),
     PILImage mode=RGB size=400x300)



### file_type, file_exts, and n_max in Path.ls
to list all the files inside the path

use file_type = 'binary' to find folders, images, videos; use file_type = 'text' to find `.py` and `.ipynb` files

use n_max=10 if you just want to list out 10 files or items


```
dino.ls(file_type="text")
dino.ls(file_type="binary")
```




    (#0) []






    (#4) [Path('T-rex_or_Spinosaurus aegyptiacus/crying'),Path('T-rex_or_Spinosaurus aegyptiacus/Spinosaurus aegyptiacus'),Path('T-rex_or_Spinosaurus aegyptiacus/T-rex'),Path('T-rex_or_Spinosaurus aegyptiacus/fierce')]




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

### count_images_in_subfolders(path)
check all subfolders for images and print out the number of images they have recursively


```
# @snoop
def count_files_in_subfolders(path, db=False):
    from pathlib import Path
    for entry in path.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and len(entry.ls(file_exts=image_extensions)) > 5:
            print(entry.name, f': {str(entry.parent.absolute()).split("/Users/Natsume/Documents/fastdebug/")[1]}')
            print(entry.name, f': {len(entry.ls(file_exts=image_extensions))}')
#             print(entry.name, f': {len(entry.ls(file_exts=[".jpg", ".png", ".jpeg", ".JPG", ".jpg!d"]))}') # how to include both png and jpg
            if db:
                for e in entry.ls(): # check any image file which has a different suffix from those above
                    if e.is_file() and not e.name.startswith(".") and e.suffix not in image_extensions and e.suffix not in [".ipynb", ".py"]:
    #                 if e.suffix not in [".jpg", ".png", ".jpeg", ".JPG", ".jpg!d"]:
                        pp(e.suffix, e)
                        try:
                            pp(Image.open(e).width)
                        except:
                            print(f"{e} can't be opened")
    #                     pp(Image.open(e).width if e.suffix in image_extensions)
        elif entry.is_dir() and not entry.name.startswith("."): 
#             with snoop:
            count_files_in_subfolders(entry)
                    
```


```
count_files_in_subfolders(dino)
count_files_in_subfolders(bird)
count_files_in_subfolders(cry_dino)
```

    crying : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    crying : 83
    Spinosaurus aegyptiacus : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    Spinosaurus aegyptiacus : 82
    T-rex : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    T-rex : 83
    fierce : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    fierce : 82
    forest : nbs/fastai_notebooks/forest_or_bird
    forest : 170
    bird : nbs/fastai_notebooks/forest_or_bird
    bird : 79
    T-rex : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    T-rex : 174
    Brachiosaurus : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    Brachiosaurus : 111



```
# list(dino.parent.absolute().ls())
dino.parent.absolute().parent
count_files_in_subfolders(dino.parent.absolute().parent)
```




    Path('/Users/Natsume/Documents/fastdebug/nbs')



    images : nbs/lib
    images : 16
    crying : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    crying : 83
    Spinosaurus aegyptiacus : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    Spinosaurus aegyptiacus : 82
    T-rex : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    T-rex : 83
    fierce : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    fierce : 82
    T-rex : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    T-rex : 174
    Brachiosaurus : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    Brachiosaurus : 111
    forest : nbs/fastai_notebooks/forest_or_bird
    forest : 170
    bird : nbs/fastai_notebooks/forest_or_bird
    bird : 79


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


### remove_failed(path)
find all images inside a path which can't be opened and unlink them

Some photos might not download correctly which could cause our model training to fail, so we'll remove them:


```
def remove_failed(path):
    print("before running remove_failed:")
    count_files_in_subfolders(path)
    failed = verify_images(get_image_files(dino))
    failed.map(Path.unlink)
    print()
    print("after running remove_failed:")
    count_files_in_subfolders(path)
```


```
remove_failed(dino)
```

    before running remove_failed:
    crying : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    crying : 83
    Spinosaurus aegyptiacus : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    Spinosaurus aegyptiacus : 82
    T-rex : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    T-rex : 83
    fierce : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    fierce : 82
    
    after running remove_failed:
    crying : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    crying : 83
    Spinosaurus aegyptiacus : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    Spinosaurus aegyptiacus : 82
    T-rex : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    T-rex : 83
    fierce : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    fierce : 82



```
remove_failed(bird)
remove_failed(cry_dino)
```

    before running remove_failed:
    forest : nbs/fastai_notebooks/forest_or_bird
    forest : 170
    bird : nbs/fastai_notebooks/forest_or_bird
    bird : 79
    
    after running remove_failed:
    forest : nbs/fastai_notebooks/forest_or_bird
    forest : 170
    bird : nbs/fastai_notebooks/forest_or_bird
    bird : 79
    before running remove_failed:
    T-rex : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    T-rex : 174
    Brachiosaurus : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    Brachiosaurus : 111
    
    after running remove_failed:
    T-rex : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    T-rex : 174
    Brachiosaurus : nbs/fastai_notebooks/T-rex_or_Brachiosaurus
    Brachiosaurus : 111



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
randomdisplay(cry_dino.ls()[0])
```




    (Path('T-rex_or_Brachiosaurus/T-rex/38fdfbb4-0fc2-4bac-bba9-af838a87e2d5.jpg'),
     PILImage mode=RGB size=400x225)




```
randomdisplay(cry_dino.ls()[1])
```




    (Path('T-rex_or_Brachiosaurus/Brachiosaurus/e2f548b8-c6cb-408d-9e2e-5997bca8170c.png'),
     PILImage mode=RGB size=400x225)




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
count_files_in_subfolders(dino)
get_image_files(dino)
count_files_in_subfolders(bird)
get_image_files(bird)
```

    crying : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    crying : 83
    Spinosaurus aegyptiacus : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    Spinosaurus aegyptiacus : 82
    T-rex : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    T-rex : 83
    fierce : nbs/fastai_notebooks/T-rex_or_Spinosaurus aegyptiacus
    fierce : 82





    (#331) [Path('T-rex_or_Spinosaurus aegyptiacus/crying/c4016dd8-bde7-4dd4-b2b0-a9a4514ac834.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/20591e18-54dd-49e7-b00c-722dbd8872ab.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/553f845b-c5d9-4d10-94ac-9230da42b866.png'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/6c8a2034-8d0c-4935-a5af-7cb73dd75b63.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/481399dc-7dc5-4f20-b651-bf29e1085611.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/d19616c2-bf32-448a-879a-b409732b1cdb.png'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/2a4e59cd-511f-4a70-a5e4-26998160031a.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/9d2c2803-1403-4e0f-ab37-b16015d506db.jpg'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/c176c0f3-c4b5-42e6-ab52-5a4fc8b4d766.png'),Path('T-rex_or_Spinosaurus aegyptiacus/crying/b806226d-287c-46ec-9a0c-882ea3be438c.jpg')...]



    forest : nbs/fastai_notebooks/forest_or_bird
    forest : 170
    bird : nbs/fastai_notebooks/forest_or_bird
    bird : 79





    (#251) [Path('forest_or_bird/forest/82e9179d-2dd6-4144-8f66-891533ec6467.jpg'),Path('forest_or_bird/forest/6e84b01f-c9a3-466b-b0eb-527ea360d517.jpg'),Path('forest_or_bird/forest/c929266f-7495-434f-bd07-0b1a1d4a9051.jpg'),Path('forest_or_bird/forest/3601c5f2-dc4a-4e56-ada4-58943c0190b3.jpg'),Path('forest_or_bird/forest/eec11827-0f51-429d-84f0-84b3c24c2e1b.JPG'),Path('forest_or_bird/forest/f1742fe6-770d-45b7-a14f-d35ec8514704.jpg'),Path('forest_or_bird/forest/4dd1840a-1a62-43b0-acdc-de476c881e4a.jpg'),Path('forest_or_bird/forest/beb09236-e6d9-49eb-a522-f8c2e9d3ffb3.jpeg'),Path('forest_or_bird/forest/ea756b81-0e41-4554-830b-858fef1e2275.jpg'),Path('forest_or_bird/forest/16ffe691-aa2d-493e-8f1c-1dea9db419e4.jpg')...]




```
Image.open(Path('forest_or_bird/bird/3b7a4112-9d77-4d8f-8b4c-a01e2ca1ecea.aspx'))
```




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_119_0.png)
    



### parent_label(o)
extract the label from the filename's parent folder name


```
def parent_label(o):
    "Label `item` with the parent folder name."
    return Path(o).parent.name
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```


```
dino.ls()[0].ls()[0]
```




    Path('T-rex_or_Spinosaurus aegyptiacus/crying/c4016dd8-bde7-4dd4-b2b0-a9a4514ac834.jpg')




```
parent_label(dino.ls()[0].ls()[0])
parent_label(dino.ls()[1].ls()[0])
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




    ((#80) [60,80,75,33,1,91,35,15,93,58...],
     (#20) [51,57,3,55,65,20,94,46,76,70...])




```
rs = RandomSplitter(seed=42)
rs(x)
```




    ((#80) [89,86,18,40,5,38,9,82,83,43...],
     (#20) [42,96,62,98,46,95,60,24,78,16...])



### Resize(RandTransform)
Resize image to `size` using `method` such as 'crop', 'squish' and 'pad'

What does `crop`, `squish` and `pad` resize effects look like


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




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_134_1.png)
    



```
rsz = Resize(256, method='crop')
rsz(img, split_idx=0)
```




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_135_0.png)
    




```
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):
    rsz = Resize(256, method=method)
    show_image(rsz(img, split_idx=0), ctx=ax, title=method);
```




    <AxesSubplot:title={'center':'squish'}>






    <AxesSubplot:title={'center':'pad'}>






    <AxesSubplot:title={'center':'crop'}>




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_136_3.png)
    



```
_,axs = plt.subplots(1,3,figsize=(12,4))
for ax,method in zip(axs.flatten(), [ResizeMethod.Squish, ResizeMethod.Pad, ResizeMethod.Crop]):
    rsz = Resize(256, method=method)
    show_image(rsz(img, split_idx=1), ctx=ax, title=method);
```




    <AxesSubplot:title={'center':'squish'}>






    <AxesSubplot:title={'center':'pad'}>






    <AxesSubplot:title={'center':'crop'}>




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_137_3.png)
    


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

    22:38:47.92 >>> Call to _TfmMeta.__new__ in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/2173750754.py", line 3
    22:38:47.92 .......... cls = <class '__main__._TfmMeta'>
    22:38:47.92 .......... name = 'Transform'
    22:38:47.92 .......... bases = ()
    22:38:47.92 .......... dict = {'__module__': '__main__', '__qualname__': 'Transform'}
    22:38:47.92 .......... __class__ = <class '__main__._TfmMeta'>
    22:38:47.92    3 |     def __new__(cls, name, bases, dict):
    22:38:47.92    4 |         res = super().__new__(cls, name, bases, dict)
    22:38:47.92 .............. res = <class '__main__.Transform'>
    22:38:47.92    5 |         for nm in _tfm_methods:
    22:38:47.92 .............. nm = 'encodes'
    22:38:47.92    6 |             base_td = [getattr(b,nm,None) for b in bases]
        22:38:47.92 List comprehension:
        22:38:47.92    6 |             base_td = [getattr(b,nm,None) for b in bases]
        22:38:47.92 .................. Iterating over <tuple_iterator object>
        22:38:47.92 .................. Values of nm: 'encodes'
        22:38:47.92 Result: []
    22:38:47.92    6 |             base_td = [getattr(b,nm,None) for b in bases]
    22:38:47.92 .................. base_td = []
    22:38:47.92    7 |             if nm in res.__dict__: getattr(res,nm).bases = base_td
    22:38:47.92    8 |             else: setattr(res, nm, TypeDispatch(bases=base_td))
    22:38:47.92    5 |         for nm in _tfm_methods:
    22:38:47.92 .............. nm = 'decodes'
    22:38:47.92    6 |             base_td = [getattr(b,nm,None) for b in bases]
        22:38:47.92 List comprehension:
        22:38:47.92    6 |             base_td = [getattr(b,nm,None) for b in bases]
        22:38:47.92 .................. Iterating over <tuple_iterator object>
        22:38:47.92 .................. Values of nm: 'decodes'
        22:38:47.92 Result: []
    22:38:47.92    6 |             base_td = [getattr(b,nm,None) for b in bases]
    22:38:47.92    7 |             if nm in res.__dict__: getattr(res,nm).bases = base_td
    22:38:47.92    8 |             else: setattr(res, nm, TypeDispatch(bases=base_td))
    22:38:47.93    5 |         for nm in _tfm_methods:
    22:38:47.93 .............. nm = 'setups'
    22:38:47.93    6 |             base_td = [getattr(b,nm,None) for b in bases]
        22:38:47.93 List comprehension:
        22:38:47.93    6 |             base_td = [getattr(b,nm,None) for b in bases]
        22:38:47.93 .................. Iterating over <tuple_iterator object>
        22:38:47.93 .................. Values of nm: 'setups'
        22:38:47.93 Result: []
    22:38:47.93    6 |             base_td = [getattr(b,nm,None) for b in bases]
    22:38:47.93    7 |             if nm in res.__dict__: getattr(res,nm).bases = base_td
    22:38:47.93    8 |             else: setattr(res, nm, TypeDispatch(bases=base_td))
    22:38:47.93    5 |         for nm in _tfm_methods:
    22:38:47.93   10 |         res.__signature__ = inspect.signature(res.__init__)
    22:38:47.93   11 |         return res
    22:38:47.93 <<< Return value from _TfmMeta.__new__: <class '__main__.Transform'>



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

### DataBlock.```__init__```(blocks:list=None, dl_type:TfmdDL=None, getters:list=None, n_inp:int=None, item_tfms:list=None, batch_tfms:list=None, get_items=None, splitter=None, get_y=None, get_x=None)
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

    22:38:48.17 >>> Call to DataBlock.__init__ in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/1228990494.py", line 13
    22:38:48.17 .......... self = <__main__.DataBlock object>
    22:38:48.17 .......... blocks = (<function ImageBlock>, <function CategoryBlock>)
    22:38:48.17 .......... dl_type = None
    22:38:48.17 .......... getters = None
    22:38:48.17 .......... n_inp = None
    22:38:48.17 .......... item_tfms = [Resize -- {'size': (192, 192), 'method': 'squish...encodes
    22:38:48.17                        (TensorPoint,object) -> encodes
    22:38:48.17                        decodes: ]
    22:38:48.17 .......... batch_tfms = None
    22:38:48.17 .......... kwargs = {}
    22:38:48.17   13 |     def __init__(self, 
    22:38:48.17   23 |         blocks = L(self.blocks if blocks is None else blocks)
    22:38:48.17 .............. blocks = [<function ImageBlock>, <function CategoryBlock>]
    22:38:48.17   24 |         pp(blocks)
    22:38:48.17 LOG:
    22:38:48.22 .... blocks = [<function ImageBlock>, <function CategoryBlock>]
    22:38:48.22   25 |         blocks = L(b() if callable(b) else b for b in blocks)
    22:38:48.22 .............. blocks = [<__main__.TransformBlock object>, <__main__.TransformBlock object>]
    22:38:48.22   26 |         pp(blocks)        
    22:38:48.22 LOG:
    22:38:48.23 .... blocks = [<__main__.TransformBlock object>, <__main__.TransformBlock object>]
    22:38:48.24   28 |         pp(inspect.getdoc(blocks.attrgot), inspect.signature(blocks.attrgot))
    22:38:48.24 LOG:
    22:38:48.25 .... inspect.getdoc(blocks.attrgot) = 'Create new `L` with attr `k` (or value `k` for dicts) of all `items`.'
    22:38:48.25 .... inspect.signature(blocks.attrgot) = <Signature (k, default=None)>
    22:38:48.25   29 |         pp(blocks.map(lambda x: x.__dict__))
    22:38:48.25 LOG:
    22:38:48.27 .... blocks.map(lambda x: x.__dict__) = [{'type_tfms': [<bound method PILBase.create of <class 'fastai.vision.core.PILImage'>>], 'item_tfms': [<class 'fastai.data.transforms.ToTensor'>], 'batch_tfms': [<class 'fastai.data.transforms.IntToFloatTensor'>], 'dl_type': None, 'dls_kwargs': {}}, {'type_tfms': [Categorize -- {'vocab': None, 'sort': True, 'add_na': False}:
    22:38:48.27                                         encodes: (Tabular,object) -> encodes
    22:38:48.27                                         (object,object) -> encodes
    22:38:48.27                                         decodes: (Tabular,object) -> decodes
    22:38:48.27                                         (object,object) -> decodes
    22:38:48.27                                         ], 'item_tfms': [<class 'fastai.data.transforms.ToTensor'>], 'batch_tfms': [], 'dl_type': None, 'dls_kwargs': {}}]
    22:38:48.27   30 |         self.type_tfms = blocks.attrgot('type_tfms', L())
    22:38:48.27   32 |         pp(inspect.getdoc(_merge_tfms), inspect.signature(_merge_tfms))
    22:38:48.27 LOG:
    22:38:48.28 .... inspect.getdoc(_merge_tfms) = ('Group the `tfms` in a single list, removing duplicates (from the same class) '
    22:38:48.28                                     'and instantiating')
    22:38:48.28 .... inspect.signature(_merge_tfms) = <Signature (*tfms)>
    22:38:48.28   33 |         self.default_item_tfms  = _merge_tfms(*blocks.attrgot('item_tfms',  L()))
    22:38:48.28   34 |         pp(self.default_item_tfms)
    22:38:48.28 LOG:
    22:38:48.30 .... self.default_item_tfms = [ToTensor:
    22:38:48.30                               encodes: (PILMask,object) -> encodes
    22:38:48.30                               (PILBase,object) -> encodes
    22:38:48.30                               decodes: ]
    22:38:48.30   36 |         self.default_batch_tfms = _merge_tfms(*blocks.attrgot('batch_tfms', L()))
    22:38:48.30   37 |         pp(self.default_batch_tfms)
    22:38:48.30 LOG:
    22:38:48.31 .... self.default_batch_tfms = [IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}:
    22:38:48.31                                encodes: (TensorImage,object) -> encodes
    22:38:48.31                                (TensorMask,object) -> encodes
    22:38:48.31                                decodes: (TensorImage,object) -> decodes
    22:38:48.31                                ]
    22:38:48.31   39 |         for b in blocks:
    22:38:48.31 .............. b = <__main__.TransformBlock object>
    22:38:48.31   40 |             if getattr(b, 'dl_type', None) is not None: 
    22:38:48.31   39 |         for b in blocks:
    22:38:48.31 .............. b = <__main__.TransformBlock object>
    22:38:48.31   40 |             if getattr(b, 'dl_type', None) is not None: 
    22:38:48.31   39 |         for b in blocks:
    22:38:48.31   42 |         if dl_type is not None: 
    22:38:48.31   44 |         pp(self.dl_type)
    22:38:48.31 LOG:
    22:38:48.32 .... self.dl_type = <class 'fastai.data.core.TfmdDL'>
    22:38:48.32   46 |         self.dataloaders = delegates(self.dl_type.__init__)(self.dataloaders) # get kwargs from dl_type.__init__ to self.dataloaders
    22:38:48.32   47 |         pp(self.dataloaders)
    22:38:48.32 LOG:
    22:38:48.33 .... self.dataloaders = <bound method DataBlock.dataloaders of <__main__.DataBlock object>>
    22:38:48.33   49 |         self.dls_kwargs = merge(*blocks.attrgot('dls_kwargs', {}))
    22:38:48.33   50 |         pp(self.dls_kwargs)
    22:38:48.33 LOG:
    22:38:48.34 .... self.dls_kwargs = {}
    22:38:48.34   52 |         self.n_inp = ifnone(n_inp, max(1, len(blocks)-1)) # n_inp is dependent on the number of blocks
    22:38:48.34   53 |         pp(self.n_inp)
    22:38:48.34 LOG:
    22:38:48.35 .... self.n_inp = 1
    22:38:48.35   55 |         self.getters = ifnone(getters, [noop]*len(self.type_tfms))
    22:38:48.35   56 |         pp(self.getters)
    22:38:48.36 LOG:
    22:38:48.37 .... self.getters = [<function noop>, <function noop>]
    22:38:48.37   58 |         if self.get_x:
    22:38:48.37   62 |         pp(self.get_x)
    22:38:48.37 LOG:
    22:38:48.38 .... self.get_x = None
    22:38:48.38   64 |         if self.get_y:
    22:38:48.38   65 |             n_targs = len(self.getters) - self.n_inp
    22:38:48.38 .................. n_targs = 1
    22:38:48.38   66 |             if len(L(self.get_y)) != n_targs:
    22:38:48.38   68 |             self.getters[self.n_inp:] = L(self.get_y)
    22:38:48.38   69 |         pp(self.getters)
    22:38:48.38 LOG:
    22:38:48.39 .... self.getters = [<function noop>, <function parent_label>]
    22:38:48.39   71 |         if kwargs: 
    22:38:48.39   74 |         pp(item_tfms, batch_tfms)
    22:38:48.39 LOG:
    22:38:48.40 .... item_tfms = [Resize -- {'size': (192, 192), 'method': 'squish', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0}:
    22:38:48.40                  encodes: (Image,object) -> encodes
    22:38:48.40                  (TensorBBox,object) -> encodes
    22:38:48.40                  (TensorPoint,object) -> encodes
    22:38:48.40                  decodes: ]
    22:38:48.40 .... batch_tfms = None
    22:38:48.40   75 |         pp(inspect.getdoc(self.new), inspect.signature(self.new))
    22:38:48.40 LOG:
    22:38:48.42 .... inspect.getdoc(self.new) = 'Create a new `DataBlock` with other `item_tfms` and `batch_tfms`'
    22:38:48.42 .... inspect.signature(self.new) = <Signature (item_tfms: 'list' = None, batch_tfms: 'list' = None)>
    22:38:48.42   76 |         self.new(item_tfms, batch_tfms)
    22:38:48.42 <<< Return value from DataBlock.__init__: None


### DataBlock.datasets(source, verbose)
get data items from the source, and split the items and use `fastai.data.core.Datasets` to create the datasets


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

    22:38:48.97 >>> Call to datasets in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/3647961006.py", line 3
    22:38:48.97 ...... self = <fastai.data.block.DataBlock object>
    22:38:48.97 ...... source = Path('forest_or_bird')
    22:38:48.97 ...... verbose = True
    22:38:48.97    3 | def datasets(self:DataBlock, 
    22:38:48.97    7 |         self.source = source 
    22:38:48.97    9 |         pp(doc_sig(pv))
    22:38:48.97 LOG:
    22:38:48.98 .... doc_sig(pv) = ('no mro', 'no doc', <Signature (text, verbose)>)
    22:38:48.98   10 |         pv(f"Collecting items from {source}", verbose)
    22:38:48.98   11 |         pp(pv(f"Collecting items from {source}", verbose))
    22:38:48.98 LOG:
    22:38:48.99 .... pv(f"Collecting items from {source}", verbose) = None
    22:38:48.99   13 |         pp((None or noop))
    22:38:48.99 LOG:
    22:38:48.99 .... None or noop = <function noop>
    22:38:48.99   14 |         pp((self.get_items or noop))
    22:38:48.99 LOG:
    22:38:49.00 .... self.get_items or noop = <function get_image_files>
    22:38:49.00   15 |         items = (self.get_items or noop)(source)
    22:38:49.00 .............. items = [Path('forest_or_bird/forest/82e9179d-2dd6-4144-.../bird/0bc8d1a8-e443-40b7-a148-f043c158400f.jpg')]
    22:38:49.00   16 |         pp(chk(items))  
    22:38:49.00 LOG:
    22:38:49.01 .... chk(items) = (<class 'fastcore.foundation.L'>, 251, 'no shape')
    22:38:49.01   18 |         pv(f"Found {len(items)} items", verbose)
    22:38:49.01   19 |         splits = (self.splitter or RandomSplitter())(items)
    22:38:49.01 .............. splits = ([83, 160, 7, 134, 147, 94, 126, 237, 55, 95, 244...8, 200, 229, 213, 206, 115, 121, 66, 195, 1, 168], [128, 68, 120, 17, 177, 16, 221, 131, 102, 22, 6..., 4, 2, 203, 129, 218, 137, 118, 249, 93, 34, 36])
    22:38:49.01   20 |         pp(chk(splits))
    22:38:49.01 LOG:
    22:38:49.02 .... chk(splits) = (<class 'tuple'>, 2, 'no shape')
    22:38:49.02   21 |         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        22:38:49.02 List comprehension:
        22:38:49.02   21 |         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        22:38:49.02 .............. Iterating over <tuple_iterator object>
        22:38:49.02 .............. Values of s: [83, 160, 7, 134, 147, 94, 126, 237, 55, 95, 244...8, 200, 229, 213, 206, 115, 121, 66, 195, 1, 168], [128, 68, 120, 17, 177, 16, 221, 131, 102, 22, 6..., 4, 2, 203, 129, 218, 137, 118, 249, 93, 34, 36]
        22:38:49.02 Result: ['201', '50']
    22:38:49.02   21 |         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
    22:38:49.02   23 |         pp(doc_sig(Datasets))
    22:38:49.02 LOG:
    22:38:49.03 .... doc_sig(Datasets) = ((<class 'fastai.data.core.Datasets'>,
    22:38:49.03                            <class 'fastai.data.core.FilteredBase'>,
    22:38:49.03                            <class 'object'>),
    22:38:49.03                           'A dataset that creates a tuple from each `tfms`',
    22:38:49.03                           <Signature (items: 'list' = None, tfms: 'list | Pipeline' = None, tls: 'TfmdLists' = None, n_inp: 'int' = None, dl_type=None, *, use_list: 'bool' = None, do_setup: 'bool' = True, split_idx: 'int' = None, train_setup: 'bool' = True, splits: 'list' = None, types=None, verbose: 'bool' = False)>)
    22:38:49.03   24 |         pp(doc_sig(Datasets.__init__))
    22:38:49.03 LOG:
    22:38:49.03 .... doc_sig(Datasets.__init__) = ('no mro',
    22:38:49.03                                    'Initialize self.  See help(type(self)) for accurate signature.',
    22:38:49.03                                    <Signature (self, items: 'list' = None, tfms: 'list | Pipeline' = None, tls: 'TfmdLists' = None, n_inp: 'int' = None, dl_type=None, **kwargs)>)
    22:38:49.04   25 |         res = Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)


    Collecting items from forest_or_bird
    Collecting items from forest_or_bird
    Found 251 items
    2 datasets of sizes 201,50
    Setting up Pipeline: PILBase.create


    22:38:49.30 .............. res = (#251) [(PILImage mode=RGB size=400x250, TensorC...age mode=RGB size=400x400, TensorCategory(1))...]
    22:38:49.30   26 |         return res


    Setting up Pipeline: parent_label -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}


    22:38:49.45 <<< Return value from datasets: (#251) [(PILImage mode=RGB size=400x250, TensorC...age mode=RGB size=400x400, TensorCategory(1))...]



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

### DataBlock.dataloaders
use `DataBlock.datasets(source, verbose=verbose)` to create a ```fastai.data.core.Datasets``` first and then use ```Datasets.dataloaders``` to create a ```fastai.data.core.DataLoaders```


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

    22:38:49.54 >>> Call to dataloaders in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/2346295921.py", line 3
    22:38:49.54 ...... self = <fastai.data.block.DataBlock object>
    22:38:49.54 ...... source = Path('forest_or_bird')
    22:38:49.54 ...... path = '.'
    22:38:49.54 ...... verbose = False
    22:38:49.54 ...... kwargs = {}
    22:38:49.54    3 | def dataloaders(self:DataBlock, 
    22:38:49.54    9 |     pp(doc_sig(self.datasets))
    22:38:49.54 LOG:
    22:38:49.55 .... doc_sig(self.datasets) = ('no mro',
    22:38:49.55                                'no doc',
    22:38:49.55                                <Signature (source, verbose: 'bool' = False) -> 'Datasets'>)
    22:38:49.55   11 |     dsets = self.datasets(source, verbose=verbose)
    22:38:49.81 .......... dsets = (#251) [(PILImage mode=RGB size=400x250, TensorC...age mode=RGB size=400x400, TensorCategory(1))...]
    22:38:49.81   12 |     pp(doc_sig(type(dsets)))
    22:38:49.81 LOG:
    22:38:49.81 .... doc_sig(type(dsets)) = ((<class 'fastai.data.core.Datasets'>,
    22:38:49.81                               <class 'fastai.data.core.FilteredBase'>,
    22:38:49.81                               <class 'object'>),
    22:38:49.81                              'A dataset that creates a tuple from each `tfms`',
    22:38:49.81                              <Signature (items: 'list' = None, tfms: 'list | Pipeline' = None, tls: 'TfmdLists' = None, n_inp: 'int' = None, dl_type=None, *, use_list: 'bool' = None, do_setup: 'bool' = True, split_idx: 'int' = None, train_setup: 'bool' = True, splits: 'list' = None, types=None, verbose: 'bool' = False)>)
    22:38:49.89   15 |     kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    22:38:49.96 .......... kwargs = {'verbose': False}
    22:38:49.96   16 |     pp(kwargs)
    22:38:49.96 LOG:
    22:38:49.96 .... kwargs = {'verbose': False}
    22:38:50.03   18 |     pp(doc_sig(dsets.dataloaders))
    22:38:50.04 LOG:
    22:38:50.04 .... doc_sig(dsets.dataloaders) = ('no mro',
    22:38:50.04                                    'Get a `DataLoaders`',
    22:38:50.04                                    <Signature (bs: 'int' = 64, shuffle_train: 'bool' = None, shuffle: 'bool' = True, val_shuffle: 'bool' = False, n: 'int' = None, path: 'str | Path' = '.', dl_type: 'TfmdDL' = None, dl_kwargs: 'list' = None, device: 'torch.device' = None, drop_last: 'bool' = None, val_bs: 'int' = None, *, num_workers: 'int' = None, verbose: 'bool' = False, do_setup: 'bool' = True, pin_memory=False, timeout=0, batch_size=None, indexed=None, persistent_workers=False, pin_memory_device='', wif=None, before_iter=None, after_item=None, before_batch=None, after_batch=None, after_iter=None, create_batches=None, create_item=None, create_batch=None, retain=None, get_idxs=None, sample=None, shuffle_fn=None, do_batch=None) -> 'DataLoaders'>)
    22:38:50.11   19 |     res = dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
    22:38:50.44 .......... res = <fastai.data.core.DataLoaders object>
    22:38:50.44   20 |     pp(doc_sig(res))
    22:38:50.45 LOG:
    22:38:50.45 .... doc_sig(res) = ('no mro', 'Basic wrapper around several `DataLoader`s.', 'no signature')
    22:38:50.53   21 |     pp(doc_sig(res.__class__))
    22:38:50.53 LOG:
    22:38:50.53 .... doc_sig(res.__class__) = ((<class 'fastai.data.core.DataLoaders'>,
    22:38:50.53                                 <class 'fastcore.basics.GetAttr'>,
    22:38:50.53                                 <class 'object'>),
    22:38:50.53                                'Basic wrapper around several `DataLoader`s.',
    22:38:50.53                                <Signature (*loaders, path: 'str | Path' = '.', device=None)>)
    22:38:50.61   22 |     return res
    22:38:50.68 <<< Return value from dataloaders: <fastai.data.core.DataLoaders object>



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

    22:38:50.77 >>> Call to vision_learner in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/3098971962.py", line 3
    22:38:50.77 ...... dls = <fastai.data.core.DataLoaders object>
    22:38:50.77 ...... arch = <function resnet18>
    22:38:50.77 ...... normalize = True
    22:38:50.77 ...... n_out = None
    22:38:50.77 ...... pretrained = True
    22:38:50.77 ...... loss_func = None
    22:38:50.77 ...... opt_func = <function Adam>
    22:38:50.77 ...... lr = 0.001
    22:38:50.77 ...... splitter = None
    22:38:50.77 ...... cbs = None
    22:38:50.77 ...... metrics = <function error_rate>
    22:38:50.77 ...... path = None
    22:38:50.77 ...... model_dir = 'models'
    22:38:50.77 ...... wd = None
    22:38:50.77 ...... wd_bn_bias = False
    22:38:50.77 ...... train_bn = True
    22:38:50.77 ...... moms = (0.95, 0.85, 0.95)
    22:38:50.77 ...... cut = None
    22:38:50.77 ...... init = <function kaiming_normal_>
    22:38:50.77 ...... custom_head = None
    22:38:50.77 ...... concat_pool = True
    22:38:50.77 ...... pool = True
    22:38:50.77 ...... lin_ftrs = None
    22:38:50.77 ...... ps = 0.5
    22:38:50.77 ...... first_bn = True
    22:38:50.77 ...... bn_final = False
    22:38:50.77 ...... lin_first = False
    22:38:50.77 ...... y_range = None
    22:38:50.77 ...... kwargs = {}
    22:38:50.77    3 | def vision_learner(dls, arch, normalize=True, n_out=None, pretrained=True, 
    22:38:50.77   11 |     pp(doc_sig(get_c))
    22:38:50.77 LOG:
    22:38:50.80 .... doc_sig(get_c) = ('no mro', 'no doc', <Signature (dls)>)
    22:38:50.80   12 |     if n_out is None: 
    22:38:50.80   13 |         n_out = get_c(dls)
    22:38:50.80 .............. n_out = 2
    22:38:50.80   14 |     assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    22:38:50.80   15 |     pp(arch, _default_meta, model_meta, model_meta.get)
    22:38:50.80 LOG:
    22:38:50.81 .... arch = <function resnet18>
    22:38:50.81 .... _default_meta = {'cut': None, 'split': <function default_split>}
    22:38:50.81 .... model_meta = {<function alexnet>: {'cut': -2,
    22:38:50.81                                                        'split': <function _alexnet_split>,
    22:38:50.81                                                        'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                  [0.229, 0.224, 0.225])},
    22:38:50.81                    <function densenet121>: {'cut': -1,
    22:38:50.81                                                            'split': <function _densenet_split>,
    22:38:50.81                                                            'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                      [0.229, 0.224, 0.225])},
    22:38:50.81                    <function densenet161>: {'cut': -1,
    22:38:50.81                                                            'split': <function _densenet_split>,
    22:38:50.81                                                            'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                      [0.229, 0.224, 0.225])},
    22:38:50.81                    <function densenet169>: {'cut': -1,
    22:38:50.81                                                            'split': <function _densenet_split>,
    22:38:50.81                                                            'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                      [0.229, 0.224, 0.225])},
    22:38:50.81                    <function densenet201>: {'cut': -1,
    22:38:50.81                                                            'split': <function _densenet_split>,
    22:38:50.81                                                            'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                      [0.229, 0.224, 0.225])},
    22:38:50.81                    <function resnet18>: {'cut': -2,
    22:38:50.81                                                         'split': <function _resnet_split>,
    22:38:50.81                                                         'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                   [0.229, 0.224, 0.225])},
    22:38:50.81                    <function resnet34>: {'cut': -2,
    22:38:50.81                                                         'split': <function _resnet_split>,
    22:38:50.81                                                         'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                   [0.229, 0.224, 0.225])},
    22:38:50.81                    <function resnet50>: {'cut': -2,
    22:38:50.81                                                         'split': <function _resnet_split>,
    22:38:50.81                                                         'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                   [0.229, 0.224, 0.225])},
    22:38:50.81                    <function resnet101>: {'cut': -2,
    22:38:50.81                                                          'split': <function _resnet_split>,
    22:38:50.81                                                          'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                    [0.229, 0.224, 0.225])},
    22:38:50.81                    <function resnet152>: {'cut': -2,
    22:38:50.81                                                          'split': <function _resnet_split>,
    22:38:50.81                                                          'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                    [0.229, 0.224, 0.225])},
    22:38:50.81                    <function squeezenet1_0>: {'cut': -1,
    22:38:50.81                                                              'split': <function _squeezenet_split>,
    22:38:50.81                                                              'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                        [0.229, 0.224, 0.225])},
    22:38:50.81                    <function squeezenet1_1>: {'cut': -1,
    22:38:50.81                                                              'split': <function _squeezenet_split>,
    22:38:50.81                                                              'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                        [0.229, 0.224, 0.225])},
    22:38:50.81                    <function vgg11_bn>: {'cut': -2,
    22:38:50.81                                                         'split': <function _vgg_split>,
    22:38:50.81                                                         'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                   [0.229, 0.224, 0.225])},
    22:38:50.81                    <function vgg13_bn>: {'cut': -2,
    22:38:50.81                                                         'split': <function _vgg_split>,
    22:38:50.81                                                         'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                   [0.229, 0.224, 0.225])},
    22:38:50.81                    <function vgg16_bn>: {'cut': -2,
    22:38:50.81                                                         'split': <function _vgg_split>,
    22:38:50.81                                                         'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                   [0.229, 0.224, 0.225])},
    22:38:50.81                    <function vgg19_bn>: {'cut': -2,
    22:38:50.81                                                         'split': <function _vgg_split>,
    22:38:50.81                                                         'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                   [0.229, 0.224, 0.225])},
    22:38:50.81                    <function xresnet18>: {'cut': -4,
    22:38:50.81                                                          'split': <function _xresnet_split>,
    22:38:50.81                                                          'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                    [0.229, 0.224, 0.225])},
    22:38:50.81                    <function xresnet34>: {'cut': -4,
    22:38:50.81                                                          'split': <function _xresnet_split>,
    22:38:50.81                                                          'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                    [0.229, 0.224, 0.225])},
    22:38:50.81                    <function xresnet50>: {'cut': -4,
    22:38:50.81                                                          'split': <function _xresnet_split>,
    22:38:50.81                                                          'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                    [0.229, 0.224, 0.225])},
    22:38:50.81                    <function xresnet101>: {'cut': -4,
    22:38:50.81                                                           'split': <function _xresnet_split>,
    22:38:50.81                                                           'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                     [0.229, 0.224, 0.225])},
    22:38:50.81                    <function xresnet152>: {'cut': -4,
    22:38:50.81                                                           'split': <function _xresnet_split>,
    22:38:50.81                                                           'stats': ([0.485, 0.456, 0.406],
    22:38:50.81                                                                     [0.229, 0.224, 0.225])}}
    22:38:50.81 .... model_meta.get = <built-in method get of dict object>
    22:38:50.81   16 |     meta = model_meta.get(arch, _default_meta)
    22:38:50.81 .......... meta = {'cut': -2, 'split': <function _resnet_split>, 'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
    22:38:50.81   17 |     model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
    22:38:50.82   18 |                       first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    22:38:50.82   17 |     model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
    22:38:50.82   18 |                       first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    22:38:50.82   17 |     model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
    22:38:50.82 .......... model_args = {'init': <function kaiming_normal_>, 'custom_head': None, 'concat_pool': True, 'pool': True, ...}
    22:38:50.82   19 |     pp(model_args)
    22:38:50.82 LOG:
    22:38:50.82 .... model_args = {'bn_final': False,
    22:38:50.82                    'concat_pool': True,
    22:38:50.82                    'custom_head': None,
    22:38:50.82                    'first_bn': True,
    22:38:50.82                    'init': <function kaiming_normal_>,
    22:38:50.82                    'lin_first': False,
    22:38:50.82                    'lin_ftrs': None,
    22:38:50.82                    'pool': True,
    22:38:50.82                    'ps': 0.5,
    22:38:50.82                    'y_range': None}
    22:38:50.82   20 |     if isinstance(arch, str):
    22:38:50.83   28 |         pp(dls, meta, pretrained)
    22:38:50.83 LOG:
    22:38:50.83 .... dls = <fastai.data.core.DataLoaders object>
    22:38:50.83 .... meta = {'cut': -2,
    22:38:50.83              'split': <function _resnet_split>,
    22:38:50.83              'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
    22:38:50.83 .... pretrained = True
    22:38:50.83   29 |         pp(doc_sig(_add_norm))
    22:38:50.84 LOG:
    22:38:50.84 .... doc_sig(_add_norm) = ('no mro', 'no doc', <Signature (dls, meta, pretrained)>)
    22:38:50.85   30 |         if normalize: 
    22:38:50.85   31 |             _add_norm(dls, meta, pretrained)
    22:38:50.85   32 |         pp(arch, n_out, pretrained, model_args)
    22:38:50.85 LOG:
    22:38:50.86 .... arch = <function resnet18>
    22:38:50.86 .... n_out = 2
    22:38:50.86 .... pretrained = True
    22:38:50.86 .... model_args = {'bn_final': False,
    22:38:50.86                    'concat_pool': True,
    22:38:50.86                    'custom_head': None,
    22:38:50.86                    'first_bn': True,
    22:38:50.86                    'init': <function kaiming_normal_>,
    22:38:50.86                    'lin_first': False,
    22:38:50.86                    'lin_ftrs': None,
    22:38:50.86                    'pool': True,
    22:38:50.86                    'ps': 0.5,
    22:38:50.86                    'y_range': None}
    22:38:50.86   33 |         pp(doc_sig(create_vision_model))
    22:38:50.86 LOG:
    22:38:50.87 .... doc_sig(create_vision_model) = ('no mro',
    22:38:50.87                                      'Create custom vision architecture',
    22:38:50.87                                      <Signature (arch, n_out, pretrained=True, cut=None, n_in=3, init=<function kaiming_normal_>, custom_head=None, concat_pool=True, pool=True, lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None)>)
    22:38:50.87   34 |         model = create_vision_model(arch, n_out, pretrained=pretrained, **model_args)
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
      warnings.warn(
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)
    22:38:51.11 .............. model = Sequential(
    22:38:51.11                          (0): Sequential(
    22:38:51.11                            (0): Conv2d(3...n_features=512, out_features=2, bias=False)
    22:38:51.11                          )
    22:38:51.11                        )
    22:38:51.11   35 |         pp(model)
    22:38:51.11 LOG:
    22:38:51.12 .... model = Sequential(
    22:38:51.12                (0): Sequential(
    22:38:51.12                  (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    22:38:51.12                  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                  (2): ReLU(inplace=True)
    22:38:51.12                  (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    22:38:51.12                  (4): Sequential(
    22:38:51.12                    (0): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                    )
    22:38:51.12                    (1): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                    )
    22:38:51.12                  )
    22:38:51.12                  (5): Sequential(
    22:38:51.12                    (0): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (downsample): Sequential(
    22:38:51.12                        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
    22:38:51.12                        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      )
    22:38:51.12                    )
    22:38:51.12                    (1): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                    )
    22:38:51.12                  )
    22:38:51.12                  (6): Sequential(
    22:38:51.12                    (0): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (downsample): Sequential(
    22:38:51.12                        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
    22:38:51.12                        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      )
    22:38:51.12                    )
    22:38:51.12                    (1): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                    )
    22:38:51.12                  )
    22:38:51.12                  (7): Sequential(
    22:38:51.12                    (0): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (downsample): Sequential(
    22:38:51.12                        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
    22:38:51.12                        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      )
    22:38:51.12                    )
    22:38:51.12                    (1): BasicBlock(
    22:38:51.12                      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                      (relu): ReLU(inplace=True)
    22:38:51.12                      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    22:38:51.12                      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                    )
    22:38:51.12                  )
    22:38:51.12                )
    22:38:51.12                (1): Sequential(
    22:38:51.12                  (0): AdaptiveConcatPool2d(
    22:38:51.12                    (ap): AdaptiveAvgPool2d(output_size=1)
    22:38:51.12                    (mp): AdaptiveMaxPool2d(output_size=1)
    22:38:51.12                  )
    22:38:51.12                  (1): fastai.layers.Flatten(full=False)
    22:38:51.12                  (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                  (3): Dropout(p=0.25, inplace=False)
    22:38:51.12                  (4): Linear(in_features=1024, out_features=512, bias=False)
    22:38:51.12                  (5): ReLU(inplace=True)
    22:38:51.12                  (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    22:38:51.12                  (7): Dropout(p=0.5, inplace=False)
    22:38:51.12                  (8): Linear(in_features=512, out_features=2, bias=False)
    22:38:51.12                )
    22:38:51.12              )
    22:38:51.12   36 |     pp.deep(lambda: ifnone(splitter, meta['split']))
    22:38:51.12 LOG:
    22:38:51.16 ........ ifnone = <function ifnone>
    22:38:51.16 ........ splitter = None
    22:38:51.16 ............ meta = {'cut': -2, 'split': <function _resnet_split>, 'stats': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}
    22:38:51.16 ........ meta['split'] = <function _resnet_split>
    22:38:51.16 .... ifnone(splitter, meta['split']) = <function _resnet_split>
    22:38:51.16   37 |     splitter=ifnone(splitter, meta['split'])
    22:38:51.16 .......... splitter = <function _resnet_split>
    22:38:51.16   38 |     pp(dict(dls=dls, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    22:38:51.16   39 |                    metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms))
    22:38:51.16   38 |     pp(dict(dls=dls, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    22:38:51.16 LOG:
    22:38:51.18 .... dict(dls=dls, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    22:38:51.18                  metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms) = {'cbs': None,
    22:38:51.18                                                                                                                                  'dls': <fastai.data.core.DataLoaders object>,
    22:38:51.18                                                                                                                                  'loss_func': None,
    22:38:51.18                                                                                                                                  'lr': 0.001,
    22:38:51.18                                                                                                                                  'metrics': <function error_rate>,
    22:38:51.18                                                                                                                                  'model_dir': 'models',
    22:38:51.18                                                                                                                                  'moms': (0.95, 0.85, 0.95),
    22:38:51.18                                                                                                                                  'opt_func': <function Adam>,
    22:38:51.18                                                                                                                                  'path': None,
    22:38:51.18                                                                                                                                  'splitter': <function _resnet_split>,
    22:38:51.18                                                                                                                                  'train_bn': True,
    22:38:51.18                                                                                                                                  'wd': None,
    22:38:51.18                                                                                                                                  'wd_bn_bias': False}
    22:38:51.18   40 |     pp(doc_sig(Learner))
    22:38:51.18 LOG:
    22:38:51.19 .... doc_sig(Learner) = ((<class 'fastai.learner.Learner'>,
    22:38:51.19                           <class 'fastcore.basics.GetAttr'>,
    22:38:51.19                           <class 'object'>),
    22:38:51.19                          'Group together a `model`, some `dls` and a `loss_func` to handle training',
    22:38:51.19                          <Signature (dls, model: 'callable', loss_func: 'callable | None' = None, opt_func=<function Adam>, lr=0.001, splitter: 'callable' = <function trainable_params>, cbs=None, metrics=None, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95, 0.85, 0.95), default_cbs: 'bool' = True)>)
    22:38:51.19   41 |     learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    22:38:51.19   42 |                    metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
    22:38:51.19   41 |     learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
    22:38:51.19 .......... learn = <fastai.learner.Learner object>
    22:38:51.19   43 |     pp(doc_sig(learn.freeze))
    22:38:51.19 LOG:
    22:38:51.20 .... doc_sig(learn.freeze) = ('no mro', 'Freeze up to last parameter group', <Signature ()>)
    22:38:51.20   44 |     if pretrained: 
    22:38:51.20   45 |         learn.freeze()
    22:38:51.21   47 |     pp(doc_sig(store_attr), kwargs)
    22:38:51.21 LOG:
    22:38:51.22 .... doc_sig(store_attr) = ('no mro',
    22:38:51.22                             'Store params named in comma-separated `names` from calling context into '
    22:38:51.22                             'attrs in `self`',
    22:38:51.22                             <Signature (names=None, self=None, but='', cast=False, store_args=None, **attrs)>)
    22:38:51.22 .... kwargs = {}
    22:38:51.22   48 |     pp(learn.__dict__.keys())
    22:38:51.22 LOG:
    22:38:51.23 .... learn.__dict__.keys() = dict_keys(['dls', 'model', '__stored_args__', 'loss_func', 'opt_func', 'lr', 'splitter', '_metrics', 'path', 'model_dir', 'wd', 'wd_bn_bias', 'train_bn', 'moms', 'default_cbs', 'training', 'create_mbar', 'logger', 'opt', 'cbs', 'train_eval', 'recorder', 'cast_to_tensor', 'progress', 'lock', 'n_epoch'])
    22:38:51.23   49 |     store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    22:38:51.23   50 |     pp(learn.__dict__.keys())
    22:38:51.23 LOG:
    22:38:51.24 .... learn.__dict__.keys() = dict_keys(['dls', 'model', '__stored_args__', 'loss_func', 'opt_func', 'lr', 'splitter', '_metrics', 'path', 'model_dir', 'wd', 'wd_bn_bias', 'train_bn', 'moms', 'default_cbs', 'training', 'create_mbar', 'logger', 'opt', 'cbs', 'train_eval', 'recorder', 'cast_to_tensor', 'progress', 'lock', 'n_epoch', 'arch', 'normalize', 'n_out', 'pretrained'])
    22:38:51.25   51 |     return learn
    22:38:51.25 <<< Return value from vision_learner: <fastai.learner.Learner object>



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

    22:38:51.34 >>> Call to fine_tune in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/3127910192.py", line 4
    22:38:51.34 ...... self = <fastai.learner.Learner object>
    22:38:51.34 ...... epochs = 1
    22:38:51.34 ...... base_lr = 0.002
    22:38:51.34 ...... freeze_epochs = 1
    22:38:51.34 ...... lr_mult = 100
    22:38:51.34 ...... pct_start = 0.3
    22:38:51.34 ...... div = 5.0
    22:38:51.34 ...... kwargs = {}
    22:38:51.34    4 | def fine_tune(self:Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,
    22:38:51.34    7 |     pp(doc_sig(self.freeze))
    22:38:51.34 LOG:
    22:38:51.35 .... doc_sig(self.freeze) = ('no mro', 'Freeze up to last parameter group', <Signature ()>)
    22:38:51.35    8 |     self.freeze()
    22:38:51.35    9 |     pp(doc_sig(self.fit_one_cycle))
    22:38:51.35 LOG:
    22:38:51.35 .... doc_sig(self.fit_one_cycle) = ('no mro',
    22:38:51.35                                     'Fit `self.model` for `n_epoch` using the 1cycle policy.',
    22:38:51.35                                     <Signature (n_epoch, lr_max=None, div=25.0, div_final=100000.0, pct_start=0.25, wd=None, moms=None, cbs=None, reset_opt=False, start_epoch=0)>)
    22:38:51.35   10 |     pp(freeze_epochs, slice(base_lr), kwargs)
    22:38:51.35 LOG:
    22:38:51.36 .... freeze_epochs = 1
    22:38:51.36 .... slice(base_lr) = slice(None, 0.002, None)
    22:38:51.36 .... kwargs = {}
    22:38:51.36   11 |     self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)




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
    <tr>
      <td>0</td>
      <td>0.691961</td>
      <td>1.092685</td>
      <td>0.420000</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>


    22:38:57.85   12 |     base_lr /= 2
    22:38:57.85 .......... base_lr = 0.001
    22:38:57.85   13 |     pp(doc_sig(self.unfreeze))
    22:38:57.85 LOG:
    22:38:57.85 .... doc_sig(self.unfreeze) = ('no mro', 'Unfreeze the entire model', <Signature ()>)
    22:38:57.85   14 |     self.unfreeze()
    22:38:57.86   15 |     pp(epochs, slice(base_lr/lr_mult, base_lr), pct_start, div, kwargs)
    22:38:57.86 LOG:
    22:38:57.86 .... epochs = 1
    22:38:57.86 .... slice(base_lr/lr_mult, base_lr) = slice(1e-05, 0.001, None)
    22:38:57.86 .... pct_start = 0.3
    22:38:57.86 .... div = 5.0
    22:38:57.86 .... kwargs = {}
    22:38:57.86   16 |     self.fit_one_cycle(epochs, slice(base_lr/lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)




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
    <tr>
      <td>0</td>
      <td>0.247853</td>
      <td>0.153514</td>
      <td>0.060000</td>
      <td>00:07</td>
    </tr>
  </tbody>
</table>


    22:39:05.01 <<< Return value from fine_tune: None



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
    <tr>
      <td>0</td>
      <td>1.177711</td>
      <td>0.171992</td>
      <td>0.060000</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>




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
    <tr>
      <td>0</td>
      <td>0.403473</td>
      <td>0.052708</td>
      <td>0.020000</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.276206</td>
      <td>0.019374</td>
      <td>0.000000</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.187863</td>
      <td>0.011193</td>
      <td>0.000000</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>



```
# Path.unlink(Path('/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/forest_or_bird/bird/82a964e3-e3db-4f05-895a-27306fe11a71.jpg'))
```

### Learner.predict(self:Learner, item, rm_type_tfms=None, with_input=False)
use learner to predict on a single item, and return 3 things: target (eg., bird or forest), 0 or 1, prob of bird and prob of forest


```
#| export
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
file,im = randomdisplay(bird/"bird")
im
is_bird,_,probs = learn.predict(PILImage.create(file))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")

```




    
![png](0001_fastai_is_it_a_bird_files/0001_fastai_is_it_a_bird_191_0.png)
    



    22:39:33.92 >>> Call to predict in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_93170/1656489701.py", line 4
    22:39:33.92 ...... self = <fastai.learner.Learner object>
    22:39:33.92 ...... item = PILImage mode=RGB size=400x300
    22:39:33.92 ...... rm_type_tfms = None
    22:39:33.92 ...... with_input = False
    22:39:33.92    4 | def predict(self:Learner, item, rm_type_tfms=None, with_input=False):
    22:39:33.92    5 |     pp(doc_sig(self.dls.test_dl))
    22:39:33.92 LOG:
    22:39:33.93 .... doc_sig(self.dls.test_dl) = ('no mro',
    22:39:33.93                                   'Create a test dataloader from `test_items` using validation transforms of '
    22:39:33.93                                   '`dls`',
    22:39:33.93                                   <Signature (test_items, rm_type_tfms=None, with_labels: 'bool' = False, *, bs: 'int' = 64, shuffle: 'bool' = False, num_workers: 'int' = None, verbose: 'bool' = False, do_setup: 'bool' = True, pin_memory=False, timeout=0, batch_size=None, drop_last=False, indexed=None, n=None, device=None, persistent_workers=False, pin_memory_device='', wif=None, before_iter=None, after_item=None, before_batch=None, after_batch=None, after_iter=None, create_batches=None, create_item=None, create_batch=None, retain=None, get_idxs=None, sample=None, shuffle_fn=None, do_batch=None)>)
    22:39:33.93    6 |     dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
    22:39:33.94 .......... dl = <fastai.data.core.TfmdDL object>
    22:39:33.94    7 |     pp(doc_sig(self.get_preds))
    22:39:33.94 LOG:
    22:39:33.94 .... doc_sig(self.get_preds) = ('no mro',
    22:39:33.94                                 'Get the predictions and targets on the `ds_idx`-th dbunchset or `dl`, '
    22:39:33.94                                 'optionally `with_input` and `with_loss`',
    22:39:33.94                                 <Signature (ds_idx=1, dl=None, with_input=False, with_decoded=False, with_loss=False, act=None, inner=False, reorder=True, cbs=None, *, save_preds: 'Path' = None, save_targs: 'Path' = None, with_preds: 'bool' = True, with_targs: 'bool' = True, concat_dim: 'int' = 0, pickle_protocol: 'int' = 2)>)
    22:39:33.94    8 |     inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)




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







    22:39:34.08 .......... inp = TensorImage([[[[ 1.0673,  0.7591,  0.2796,  ...,...15, -0.8284,  ..., -1.4559, -1.3861, -1.3861]]]])
    22:39:34.08 .......... preds = TensorBase([[1.0000e+00, 3.9171e-08]])
    22:39:34.08 .......... _ = None
    22:39:34.08 .......... dec_preds = TensorBase([0])
    22:39:34.08    9 |     i = getattr(self.dls, 'n_inp', -1)
    22:39:34.08 .......... i = 1
    22:39:34.08   10 |     if i==1:
    22:39:34.08   11 |         inp = (inp,) 
    22:39:34.08 .............. inp = (TensorImage([[[[ 1.0673,  0.7591,  0.2796,  ...,...15, -0.8284,  ..., -1.4559, -1.3861, -1.3861]]]]),)
    22:39:34.08   14 |     pp(doc_sig(self.dls.decode_batch))
    22:39:34.08 LOG:
    22:39:34.09 .... doc_sig(self.dls.decode_batch) = ('no mro',
    22:39:34.09                                        'Decode `b` entirely',
    22:39:34.09                                        <Signature (b, max_n: 'int' = 9, full: 'bool' = True)>)
    22:39:34.09   15 |     dec = self.dls.decode_batch(inp + tuplify(dec_preds))[0]
    22:39:34.10 .......... dec = (TensorImage([[[186, 168, 140,  ...,  37,  67,  9...          [ 74,  65,  56,  ...,  19,  24,  24]]]), 'bird')
    22:39:34.10   16 |     pp(doc_sig(tuplify), doc_sig(detuplify))
    22:39:34.10 LOG:
    22:39:34.10 .... doc_sig(tuplify) = ('no mro', 'Make `o` a tuple', <Signature (o, use_list=False, match=None)>)
    22:39:34.11 .... doc_sig(detuplify) = ('no mro', 'If `x` is a tuple with one thing, extract it', <Signature (x)>)
    22:39:34.11   17 |     dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
    22:39:34.11 .......... dec_inp = TensorImage([[[186, 168, 140,  ...,  37,  67,  9...          [ 74,  65,  56,  ...,  19,  24,  24]]])
    22:39:34.11 .......... dec_targ = 'bird'
    22:39:34.11   18 |     res = dec_targ,dec_preds[0],preds[0]
    22:39:34.11 .......... res = ('bird', TensorBase(0), TensorBase([1.0000e+00, 3.9171e-08]))
    22:39:34.11   19 |     if with_input: 
    22:39:34.12   21 |     return res
    22:39:34.12 <<< Return value from predict: ('bird', TensorBase(0), TensorBase([1.0000e+00, 3.9171e-08]))


    This is a: bird.
    Probability it's a bird: 1.0000



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

```
