# 0012_fastcore_foundation_L


```

```

## Document `L` with fastdebug


```
fastnbs("how to download images")
```


keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a section: in  <mark style="background-color: #FFFF00">0001_is_it_a_bird.md</mark> 



##  <mark style="background-color: #ffff00">how</mark>   <mark style="background-color: #ffff00">to</mark>   <mark style="background-color: #ffff00">download</mark>   <mark style="background-color: #FFFF00">images</mark> 



    



## how to download images


```python
from duckduckgo_search import ddg_images
from fastcore.all import *
```

```python
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
```

```python
#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('bird photos', max_images=1)
urls[0]
```

```python
from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```

```python
download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```

```python

```





[Open `0001_is_it_a_bird` in Jupyter Notebook](http://localhost:8888/tree/nbs/2022part1/0001_is_it_a_bird.ipynb)


I want to know what is `L` and how does `L.itemgot` do?


```
from nbdev.showdoc import * # how to get doc(func) ready
```


```
doc(L)
```


<hr/>
<h3>L</h3>
<blockquote><pre><code>L(items=None, *rest, use_list=False, match=None)</code></pre></blockquote><p>Behaves like a list of `items` but can also index with list of indices or masks</p>
<p><a href="https://fastcore.fast.ai/foundation.html#l" target="_blank" rel="noreferrer noopener">Show in docs</a></p>



```
doc(L.itemgot)
```


<hr/>
<h3>L.itemgot</h3>
<blockquote><pre><code>L.itemgot(*idxs)</code></pre></blockquote><p>Create new `L` with item `idx` of all `items`</p>
<p><a href="https://fastcore.fast.ai/foundation.html#itemgot" target="_blank" rel="noreferrer noopener">Show in docs</a></p>



```
inspect_class(L)
```

    
    is L a metaclass: False
    is L created by a metaclass: True
    L is created by metaclass <class 'fastcore.foundation._L_Meta'>
    L.__new__ is object.__new__: True
    L.__new__ is type.__new__: False
    L.__new__: <built-in method __new__ of type object>
    L.__init__ is object.__init__: False
    L.__init__ is type.__init__: False
    L.__init__: <function L.__init__>
    L.__call__ is object.__call__: False
    L.__call__ is type.__call__: False
    L.__call__: <bound method _L_Meta.__call__ of <class 'fastcore.foundation.L'>>
    L.__class__: <class 'fastcore.foundation._L_Meta'>
    L.__bases__: (<class 'fastcore.basics.GetAttr'>, <class 'fastcore.foundation.CollBase'>)
    L.__mro__: (<class 'fastcore.foundation.L'>, <class 'fastcore.basics.GetAttr'>, <class 'fastcore.foundation.CollBase'>, <class 'object'>)
    
    L's metaclass <class 'fastcore.foundation._L_Meta'>'s function members are:
    {'__call__': <function _L_Meta.__call__>}
    
    L's function members are:
    __add__: None
    __addi__: None
    __contains__: None
    __delitem__: None
    __dir__: Default dir() implementation.
    __eq__: Return self==value.
    __getattr__: None
    __getitem__: Retrieve `idx` (can be list of indices, or mask, or int) items
    __init__: Initialize self.  See help(type(self)) for accurate signature.
    __invert__: None
    __iter__: None
    __len__: None
    __mul__: None
    __radd__: None
    __repr__: Return repr(self).
    __reversed__: None
    __setitem__: Set `idx` (can be list of indices, or mask, or int) items to `o` (which is broadcast if not iterable)
    __setstate__: None
    _component_attr_filter: None
    _dir: None
    _get: None
    _new: None
    _repr_pretty_: None
    argfirst: Return index of first matching item
    argwhere: Like `filter`, but return indices for matching items
    attrgot: Create new `L` with attr `k` (or value `k` for dicts) of all `items`.
    concat: Concatenate all elements of list
    copy: Same as `list.copy`, but returns an `L`
    cycle: Same as `itertools.cycle`
    enumerate: Same as `enumerate`
    filter: Create new `L` filtered by predicate `f`, passing `args` and `kwargs` to `f`
    itemgot: Create new `L` with item `idx` of all `items`
    map: Create new `L` with `f` applied to all `items`, passing `args` and `kwargs` to `f`
    map_dict: Like `map`, but creates a dict from `items` to function results
    map_first: First element of `map_filter`
    map_zip: Combine `zip` and `starmap`
    map_zipwith: Combine `zipwith` and `starmap`
    product: Product of the items
    reduce: Wrapper for `functools.reduce`
    renumerate: Same as `renumerate`
    setattrs: Call `setattr` on all items
    shuffle: Same as `random.shuffle`, but not inplace
    sorted: New `L` sorted by `key`. If key is str use `attrgetter`; if int use `itemgetter`
    starmap: Like `map`, but use `itertools.starmap`
    sum: Sum of the items
    unique: Unique items, in stable order
    val2idx: Dict from value to index
    zip: Create new `L` with `zip(*items)`
    zipwith: Create new `L` with `self` zip with each of `*rest`
    
    L's method members are:
    {'range': <bound method L.range of <class 'fastcore.foundation.L'>>,
     'split': <bound method L.split of <class 'fastcore.foundation.L'>>}
    
    L's class members are:
    {'__class__': <class 'fastcore.foundation._L_Meta'>}
    
    L's namespace are:
    mappingproxy({'__add__': <function L.__add__>,
                  '__addi__': <function L.__addi__>,
                  '__contains__': <function L.__contains__>,
                  '__doc__': 'Behaves like a list of `items` but can also index '
                             'with list of indices or masks',
                  '__eq__': <function L.__eq__>,
                  '__getitem__': <function L.__getitem__>,
                  '__hash__': None,
                  '__init__': <function L.__init__>,
                  '__invert__': <function L.__invert__>,
                  '__iter__': <function L.__iter__>,
                  '__module__': 'fastcore.foundation',
                  '__mul__': <function L.__mul__>,
                  '__radd__': <function L.__radd__>,
                  '__repr__': <function L.__repr__>,
                  '__reversed__': <function L.__reversed__>,
                  '__setitem__': <function L.__setitem__>,
                  '__signature__': <Signature (items=None, *rest, use_list=False, match=None)>,
                  '_default': 'items',
                  '_get': <function L._get>,
                  '_new': <function L._new>,
                  '_repr_pretty_': <function L._repr_pretty_>,
                  '_xtra': <property object>,
                  'argfirst': <function L.argfirst>,
                  'argwhere': <function L.argwhere>,
                  'attrgot': <function L.attrgot>,
                  'concat': <function L.concat>,
                  'copy': <function L.copy>,
                  'cycle': <function L.cycle>,
                  'enumerate': <function L.enumerate>,
                  'filter': <function L.filter>,
                  'itemgot': <function L.itemgot>,
                  'map': <function L.map>,
                  'map_dict': <function L.map_dict>,
                  'map_first': <function L.map_first>,
                  'map_zip': <function L.map_zip>,
                  'map_zipwith': <function L.map_zipwith>,
                  'product': <function L.product>,
                  'range': <classmethod object>,
                  'reduce': <function L.reduce>,
                  'renumerate': <function L.renumerate>,
                  'setattrs': <function L.setattrs>,
                  'shuffle': <function L.shuffle>,
                  'sorted': <function L.sorted>,
                  'split': <classmethod object>,
                  'starmap': <function L.starmap>,
                  'sum': <function L.sum>,
                  'unique': <function L.unique>,
                  'val2idx': <function L.val2idx>,
                  'zip': <function L.zip>,
                  'zipwith': <function L.zipwith>})



```

```


```

```
