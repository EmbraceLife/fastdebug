```
#|default_exp delete_fastcore_foundation
```


```
#|export
from fastcore.imports import *
from fastcore.basics import *
from functools import lru_cache
from contextlib import contextmanager
from copy import copy
from configparser import ConfigParser
import random,pickle,inspect
```


```
#|hide
from fastcore.test import *
from nbdev.showdoc import *
from fastcore.nb_imports import *
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>


# Foundation

> The `L` class and helpers for it

## Foundational Functions


```
#|export
@contextmanager
def working_directory(path):
    "Change working directory to `path` and return to previous on exit."
    prev_cwd = Path.cwd()
    os.chdir(path)
    try: yield
    finally: os.chdir(prev_cwd)
```


```
#|export
def add_docs(cls, cls_doc=None, **docs):
    "Copy values from `docs` to `cls` docstrings, and confirm all public methods are documented"
    if cls_doc is not None: cls.__doc__ = cls_doc
    for k,v in docs.items():
        f = getattr(cls,k)
        if hasattr(f,'__func__'): f = f.__func__ # required for class methods
        f.__doc__ = v
    # List of public callables without docstring
    nodoc = [c for n,c in vars(cls).items() if callable(c)
             and not n.startswith('_') and c.__doc__ is None]
    assert not nodoc, f"Missing docs: {nodoc}"
    assert cls.__doc__ is not None, f"Missing class docs: {cls}"
```

`add_docs` allows you to add docstrings to a class and its associated methods.  This function allows you to group docstrings together seperate from your code, which enables you to define one-line functions as well as organize your code more succintly. We believe this confers a number of benefits which we discuss in [our style guide](https://docs.fast.ai/dev/style.html).

Suppose you have the following undocumented class:


```
class T:
    def foo(self): pass
    def bar(self): pass
```

You can add documentation to this class like so:


```
add_docs(T, cls_doc="A docstring for the class.",
            foo="The foo method.",
            bar="The bar method.")
```

Now, docstrings will appear as expected:


```
test_eq(T.__doc__, "A docstring for the class.")
test_eq(T.foo.__doc__, "The foo method.")
test_eq(T.bar.__doc__, "The bar method.")
```

`add_docs` also validates that all of  your public methods contain a docstring.  If one of your methods is not documented, it will raise an error:


```
class T:
    def foo(self): pass
    def bar(self): pass

f=lambda: add_docs(T, "A docstring for the class.", foo="The foo method.")
test_fail(f, contains="Missing docs")
```


```
#|hide
class _T:
    def f(self): pass
    @classmethod
    def g(cls): pass
add_docs(_T, "a", f="f", g="g")

test_eq(_T.__doc__, "a")
test_eq(_T.f.__doc__, "f")
test_eq(_T.g.__doc__, "g")
```


```
#|export
def docs(cls):
    "Decorator version of `add_docs`, using `_docs` dict"
    add_docs(cls, **cls._docs)
    return cls
```

Instead of using `add_docs`, you can use the decorator `docs` as shown below.  Note that the docstring for the class can be set with the argument `cls_doc`:


```
@docs
class _T:
    def f(self): pass
    def g(cls): pass
    
    _docs = dict(cls_doc="The class docstring", 
                 f="The docstring for method f.",
                 g="A different docstring for method g.")

    
test_eq(_T.__doc__, "The class docstring")
test_eq(_T.f.__doc__, "The docstring for method f.")
test_eq(_T.g.__doc__, "A different docstring for method g.")
```

For either the `docs` decorator or the `add_docs` function, you can still define your docstrings in the normal way.  Below we set the docstring for the class as usual, but define the method docstrings through the `_docs` attribute:


```
@docs
class _T:
    "The class docstring"
    def f(self): pass
    _docs = dict(f="The docstring for method f.")

    
test_eq(_T.__doc__, "The class docstring")
test_eq(_T.f.__doc__, "The docstring for method f.")
```


```
show_doc(is_iter)
```




---

### is_iter

>      is_iter (o)

Test whether `o` can be used in a `for` loop




```
assert is_iter([1])
assert not is_iter(array(1))
assert is_iter(array([1,2]))
assert (o for o in range(3))
```


```
#|export
def coll_repr(c, max_n=10):
    "String repr of up to `max_n` items of (possibly lazy) collection `c`"
    return f'(#{len(c)}) [' + ','.join(itertools.islice(map(repr,c), max_n)) + (
        '...' if len(c)>max_n else '') + ']'
```

`coll_repr` is used to provide a more informative [`__repr__`](https://stackoverflow.com/questions/1984162/purpose-of-pythons-repr) about list-like objects.  `coll_repr` and is used by `L` to build a `__repr__` that displays the length of a list in addition to a preview of a list.

Below is an example of the `__repr__` string created for a list of 1000 elements:


```
test_eq(coll_repr(range(1000)),    '(#1000) [0,1,2,3,4,5,6,7,8,9...]')
test_eq(coll_repr(range(1000), 5), '(#1000) [0,1,2,3,4...]')
test_eq(coll_repr(range(10),   5),   '(#10) [0,1,2,3,4...]')
test_eq(coll_repr(range(5),    5),    '(#5) [0,1,2,3,4]')
```

We can set the option `max_n` to optionally preview a specified number of items instead of the default:


```
test_eq(coll_repr(range(1000), max_n=5), '(#1000) [0,1,2,3,4...]')
```


```
#|export
def is_bool(x):
    "Check whether `x` is a bool or None"
    return isinstance(x,(bool,NoneType)) or risinstance('bool_', x)
```


```
#|export
def mask2idxs(mask):
    "Convert bool mask or index list to index `L`"
    if isinstance(mask,slice): return mask
    mask = list(mask)
    if len(mask)==0: return []
    it = mask[0]
    if hasattr(it,'item'): it = it.item()
    if is_bool(it): return [i for i,m in enumerate(mask) if m]
    return [int(i) for i in mask]
```


```
test_eq(mask2idxs([False,True,False,True]), [1,3])
test_eq(mask2idxs(array([False,True,False,True])), [1,3])
test_eq(mask2idxs(array([1,2,3])), [1,2,3])
```


```
#|export
def cycle(o):
    "Like `itertools.cycle` except creates list of `None`s if `o` is empty"
    o = listify(o)
    return itertools.cycle(o) if o is not None and len(o) > 0 else itertools.cycle([None])
```


```
test_eq(itertools.islice(cycle([1,2,3]),5), [1,2,3,1,2])
test_eq(itertools.islice(cycle([]),3), [None]*3)
test_eq(itertools.islice(cycle(None),3), [None]*3)
test_eq(itertools.islice(cycle(1),3), [1,1,1])
```


```
#|export
def zip_cycle(x, *args):
    "Like `itertools.zip_longest` but `cycle`s through elements of all but first argument"
    return zip(x, *map(cycle,args))
```


```
test_eq(zip_cycle([1,2,3,4],list('abc')), [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'a')])
```


```
#|export
def is_indexer(idx):
    "Test whether `idx` will index a single item in a list"
    return isinstance(idx,int) or not getattr(idx,'ndim',1)
```

You can, for example index a single item in a list with an integer or a 0-dimensional numpy array:


```
assert is_indexer(1)
assert is_indexer(np.array(1))
```

However, you cannot index into single item in a list with another list or a numpy array with ndim > 0. 


```
assert not is_indexer([1, 2])
assert not is_indexer(np.array([[1, 2], [3, 4]]))
```

## `L` helpers


```
#|export
class CollBase:
    "Base class for composing a list of `items`"
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, k): return self.items[list(k) if isinstance(k,CollBase) else k]
    def __setitem__(self, k, v): self.items[list(k) if isinstance(k,CollBase) else k] = v
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self): return self.items.__repr__()
    def __iter__(self): return self.items.__iter__()
```

`ColBase` is a base class that emulates the functionality of a python `list`:


```
class _T(CollBase): pass
l = _T([1,2,3,4,5])

test_eq(len(l), 5) # __len__
test_eq(l[-1], 5); test_eq(l[0], 1) #__getitem__
l[2] = 100; test_eq(l[2], 100)      # __set_item__
del l[0]; test_eq(len(l), 4)        # __delitem__
test_eq(str(l), '[2, 100, 4, 5]')   # __repr__
```

## L -


```
#|export
class _L_Meta(type):
    def __call__(cls, x=None, *args, **kwargs):
        if not args and not kwargs and x is not None and isinstance(x,cls): return x
        return super().__call__(x, *args, **kwargs)
```


```
#|export
class L(GetAttr, CollBase, metaclass=_L_Meta):
    "Behaves like a list of `items` but can also index with list of indices or masks"
    _default='items'
    def __init__(self, items=None, *rest, use_list=False, match=None):
        if (use_list is not None) or not is_array(items):
            items = listify(items, *rest, use_list=use_list, match=match)
        super().__init__(items)

    @property
    def _xtra(self): return None
    def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    def __getitem__(self, idx): return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    def copy(self): return self._new(self.items.copy())

    def _get(self, i):
        if is_indexer(i) or isinstance(i,slice): return getattr(self.items,'iloc',self.items)[i]
        i = mask2idxs(i)
        return (self.items.iloc[list(i)] if hasattr(self.items,'iloc')
                else self.items.__array__()[(i,)] if hasattr(self.items,'__array__')
                else [self.items[i_] for i_ in i])

    def __setitem__(self, idx, o):
        "Set `idx` (can be list of indices, or mask, or int) items to `o` (which is broadcast if not iterable)"
        if isinstance(idx, int): self.items[idx] = o
        else:
            idx = idx if isinstance(idx,L) else listify(idx)
            if not is_iter(o): o = [o]*len(idx)
            for i,o_ in zip(idx,o): self.items[i] = o_

    def __eq__(self,b):
        if b is None: return False
        if risinstance('ndarray', b): return array_equal(b, self)
        if isinstance(b, (str,dict)): return False
        return all_equal(b,self)

    def sorted(self, key=None, reverse=False): return self._new(sorted_ex(self, key=key, reverse=reverse))
    def __iter__(self): return iter(self.items.itertuples() if hasattr(self.items,'iloc') else self.items)
    def __contains__(self,b): return b in self.items
    def __reversed__(self): return self._new(reversed(self.items))
    def __invert__(self): return self._new(not i for i in self)
    def __repr__(self): return repr(self.items)
    def _repr_pretty_(self, p, cycle):
        p.text('...' if cycle else repr(self.items) if is_array(self.items) else coll_repr(self))
    def __mul__ (a,b): return a._new(a.items*b)
    def __add__ (a,b): return a._new(a.items+listify(b))
    def __radd__(a,b): return a._new(b)+a
    def __addi__(a,b):
        a.items += list(b)
        return a

    @classmethod
    def split(cls, s, sep=None, maxsplit=-1): return cls(s.split(sep,maxsplit))
    @classmethod
    def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))

    def map(self, f, *args, gen=False, **kwargs): return self._new(map_ex(self, f, *args, gen=gen, **kwargs))
    def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    def argfirst(self, f, negate=False): return first(i for i,o in self.enumerate() if f(o))
    def filter(self, f=noop, negate=False, gen=False, **kwargs):
        return self._new(filter_ex(self, f=f, negate=negate, gen=gen, **kwargs))

    def enumerate(self): return L(enumerate(self))
    def renumerate(self): return L(renumerate(self))
    def unique(self, sort=False, bidir=False, start=None): return L(uniqueify(self, sort=sort, bidir=bidir, start=start))
    def val2idx(self): return val2idx(self)
    def cycle(self): return cycle(self)
    def map_dict(self, f=noop, *args, gen=False, **kwargs): return {k:f(k, *args,**kwargs) for k in self}
    def map_first(self, f=noop, g=noop, *args, **kwargs):
        return first(self.map(f, *args, gen=True, **kwargs), g)

    def itemgot(self, *idxs):
        x = self
        for idx in idxs: x = x.map(itemgetter(idx))
        return x
    def attrgot(self, k, default=None):
        return self.map(lambda o: o.get(k,default) if isinstance(o, dict) else nested_attr(o,k,default))

    def starmap(self, f, *args, **kwargs): return self._new(itertools.starmap(partial(f,*args,**kwargs), self))
    def zip(self, cycled=False): return self._new((zip_cycle if cycled else zip)(*self))
    def zipwith(self, *rest, cycled=False): return self._new([self, *rest]).zip(cycled=cycled)
    def map_zip(self, f, *args, cycled=False, **kwargs): return self.zip(cycled=cycled).starmap(f, *args, **kwargs)
    def map_zipwith(self, f, *rest, cycled=False, **kwargs): return self.zipwith(*rest, cycled=cycled).starmap(f, **kwargs)
    def shuffle(self):
        it = copy(self.items)
        random.shuffle(it)
        return self._new(it)

    def concat(self): return self._new(itertools.chain.from_iterable(self.map(L)))
    def reduce(self, f, initial=None): return reduce(f, self) if initial is None else reduce(f, self, initial)
    def sum(self): return self.reduce(operator.add, 0)
    def product(self): return self.reduce(operator.mul, 1)
    def setattrs(self, attr, val): [setattr(o,attr,val) for o in self]
```


```
#|export
add_docs(L,
         __getitem__="Retrieve `idx` (can be list of indices, or mask, or int) items",
         range="Class Method: Same as `range`, but returns `L`. Can pass collection for `a`, to use `len(a)`",
         split="Class Method: Same as `str.split`, but returns an `L`",
         copy="Same as `list.copy`, but returns an `L`",
         sorted="New `L` sorted by `key`. If key is str use `attrgetter`; if int use `itemgetter`",
         unique="Unique items, in stable order",
         val2idx="Dict from value to index",
         filter="Create new `L` filtered by predicate `f`, passing `args` and `kwargs` to `f`",
         argwhere="Like `filter`, but return indices for matching items",
         argfirst="Return index of first matching item",
         map="Create new `L` with `f` applied to all `items`, passing `args` and `kwargs` to `f`",
         map_first="First element of `map_filter`",
         map_dict="Like `map`, but creates a dict from `items` to function results",
         starmap="Like `map`, but use `itertools.starmap`",
         itemgot="Create new `L` with item `idx` of all `items`",
         attrgot="Create new `L` with attr `k` (or value `k` for dicts) of all `items`.",
         cycle="Same as `itertools.cycle`",
         enumerate="Same as `enumerate`",
         renumerate="Same as `renumerate`",
         zip="Create new `L` with `zip(*items)`",
         zipwith="Create new `L` with `self` zip with each of `*rest`",
         map_zip="Combine `zip` and `starmap`",
         map_zipwith="Combine `zipwith` and `starmap`",
         concat="Concatenate all elements of list",
         shuffle="Same as `random.shuffle`, but not inplace",
         reduce="Wrapper for `functools.reduce`",
         sum="Sum of the items",
         product="Product of the items",
         setattrs="Call `setattr` on all items"
        )
```


```
#|export
#|hide
# Here we are fixing the signature of L. What happens is that the __call__ method on the MetaClass of L shadows the __init__
# giving the wrong signature (https://stackoverflow.com/questions/49740290/call-from-metaclass-shadows-signature-of-init).
def _f(items=None, *rest, use_list=False, match=None): ...
L.__signature__ = inspect.signature(_f)
```


```
#|export
Sequence.register(L);
```

`L` is a drop in replacement for a python `list`.  Inspired by [NumPy](http://www.numpy.org/), `L`,  supports advanced indexing and has additional methods (outlined below) that provide additional functionality and encourage simple expressive code. For example, the code below takes a list of pairs, selects the second item of each pair, takes its absolute value, filters items greater than 4, and adds them up:


```
from fastcore.utils import gt
```


```
d = dict(a=1,b=-5,d=6,e=9).items()
test_eq(L(d).itemgot(1).map(abs).filter(gt(4)).sum(), 20) # abs(-5) + abs(6) + abs(9) = 20; 1 was filtered out.
```

Read [this overview section](https://fastcore.fast.ai/#L) for a quick tutorial of `L`, as well as background on the name.  

You can create an `L` from an existing iterable (e.g. a list, range, etc) and access or modify it with an int list/tuple index, mask, int, or slice. All `list` methods can also be used with `L`.


```
t = L(range(12))
test_eq(t, list(range(12)))
test_ne(t, list(range(11)))
t.reverse()
test_eq(t[0], 11)
t[3] = "h"
test_eq(t[3], "h")
t[3,5] = ("j","k")
test_eq(t[3,5], ["j","k"])
test_eq(t, L(t))
test_eq(L(L(1,2),[3,4]), ([1,2],[3,4]))
t
```




    (#12) [11,10,9,'j',7,'k',5,4,3,2...]



Any `L` is a `Sequence` so you can use it with methods like `random.sample`:


```
assert isinstance(t, Sequence)
```


```
import random
```


```
random.seed(0)
random.sample(t, 3)
```




    [5, 0, 11]




```
#|hide
# test set items with L of collections
x = L([[1,2,3], [4,5], [6,7]])
x[0] = [1,2]
test_eq(x, L([[1,2], [4,5], [6,7]]))

# non-idiomatic None-ness check - avoid infinite recursion
some_var = L(['a', 'b'])
assert some_var != None, "L != None"
```

There are optimized indexers for arrays, tensors, and DataFrames.


```
import pandas as pd
```


```
arr = np.arange(9).reshape(3,3)
t = L(arr, use_list=None)
test_eq(t[1,2], arr[[1,2]])

df = pd.DataFrame({'a':[1,2,3]})
t = L(df, use_list=None)
test_eq(t[1,2], L(pd.DataFrame({'a':[2,3]}, index=[1,2]), use_list=None))
```

You can also modify an `L` with `append`, `+`, and `*`.


```
t = L()
test_eq(t, [])
t.append(1)
test_eq(t, [1])
t += [3,2]
test_eq(t, [1,3,2])
t = t + [4]
test_eq(t, [1,3,2,4])
t = 5 + t
test_eq(t, [5,1,3,2,4])
test_eq(L(1,2,3), [1,2,3])
test_eq(L(1,2,3), L(1,2,3))
t = L(1)*5
t = t.map(operator.neg)
test_eq(t,[-1]*5)
test_eq(~L([True,False,False]), L([False,True,True]))
t = L(range(4))
test_eq(zip(t, L(1).cycle()), zip(range(4),(1,1,1,1)))
t = L.range(100)
test_shuffled(t,t.shuffle())
```


```
test_eq(L([]).sum(), 0)
test_eq(L([]).product(), 1)
```


```
def _f(x,a=0): return x+a
t = L(1)*5
test_eq(t.map(_f), t)
test_eq(t.map(_f,1), [2]*5)
test_eq(t.map(_f,a=2), [3]*5)
```

An `L` can be constructed from anything iterable, although tensors and arrays will not be iterated over on construction, unless you pass `use_list` to the constructor.


```
test_eq(L([1,2,3]),[1,2,3])
test_eq(L(L([1,2,3])),[1,2,3])
test_ne(L([1,2,3]),[1,2,])
test_eq(L('abc'),['abc'])
test_eq(L(range(0,3)),[0,1,2])
test_eq(L(o for o in range(0,3)),[0,1,2])
test_eq(L(array(0)),[array(0)])
test_eq(L([array(0),array(1)]),[array(0),array(1)])
test_eq(L(array([0.,1.1]))[0],array([0.,1.1]))
test_eq(L(array([0.,1.1]), use_list=True), [array(0.),array(1.1)])  # `use_list=True` to unwrap arrays/arrays
```

If `match` is not `None` then the created list is same len as `match`, either by:

- If `len(items)==1` then `items` is replicated,
- Otherwise an error is raised if `match` and `items` are not already the same size.


```
test_eq(L(1,match=[1,2,3]),[1,1,1])
test_eq(L([1,2],match=[2,3]),[1,2])
test_fail(lambda: L([1,2],match=[1,2,3]))
```

If you create an `L` from an existing `L` then you'll get back the original object (since `L` uses the `NewChkMeta` metaclass).


```
test_is(L(t), t)
```

An `L` is considred equal to a list if they have the same elements. It's never considered equal to a `str` a `set` or a `dict` even if they have the same elements/keys.


```
test_eq(L(['a', 'b']), ['a', 'b'])
test_ne(L(['a', 'b']), 'ab')
test_ne(L(['a', 'b']), {'a':1, 'b':2})
```

### `L` Methods


```
show_doc(L.__getitem__)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L112){target="_blank" style="float:right; font-size:smaller"}

### L.__getitem__

>      L.__getitem__ (idx)

Retrieve `idx` (can be list of indices, or mask, or int) items




```
t = L(range(12))
test_eq(t[1,2], [1,2])                # implicit tuple
test_eq(t[[1,2]], [1,2])              # list
test_eq(t[:3], [0,1,2])               # slice
test_eq(t[[False]*11 + [True]], [11]) # mask
test_eq(t[array(3)], 3)
```


```
show_doc(L.__setitem__)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L122){target="_blank" style="float:right; font-size:smaller"}

### L.__setitem__

>      L.__setitem__ (idx, o)

Set `idx` (can be list of indices, or mask, or int) items to `o` (which is broadcast if not iterable)




```
t[4,6] = 0
test_eq(t[4,6], [0,0])
t[4,6] = [1,2]
test_eq(t[4,6], [1,2])
```


```
show_doc(L.unique)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L164){target="_blank" style="float:right; font-size:smaller"}

### L.unique

>      L.unique (sort=False, bidir=False, start=None)

Unique items, in stable order




```
test_eq(L(4,1,2,3,4,4).unique(), [4,1,2,3])
```


```
show_doc(L.val2idx)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L165){target="_blank" style="float:right; font-size:smaller"}

### L.val2idx

>      L.val2idx ()

Dict from value to index




```
test_eq(L(1,2,3).val2idx(), {3:2,1:0,2:1})
```


```
show_doc(L.filter)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L159){target="_blank" style="float:right; font-size:smaller"}

### L.filter

>      L.filter (f=<function noop>, negate=False, gen=False, **kwargs)

Create new `L` filtered by predicate `f`, passing `args` and `kwargs` to `f`




```
list(t)
```




    [0, 1, 2, 3, 1, 5, 2, 7, 8, 9, 10, 11]




```
test_eq(t.filter(lambda o:o<5), [0,1,2,3,1,2])
test_eq(t.filter(lambda o:o<5, negate=True), [5,7,8,9,10,11])
```


```
show_doc(L.argwhere)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L157){target="_blank" style="float:right; font-size:smaller"}

### L.argwhere

>      L.argwhere (f, negate=False, **kwargs)

Like `filter`, but return indices for matching items




```
test_eq(t.argwhere(lambda o:o<5), [0,1,2,3,4,6])
```


```
show_doc(L.argfirst)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L158){target="_blank" style="float:right; font-size:smaller"}

### L.argfirst

>      L.argfirst (f, negate=False)

Return index of first matching item




```
test_eq(t.argfirst(lambda o:o>4), 5)
```


```
show_doc(L.map)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L156){target="_blank" style="float:right; font-size:smaller"}

### L.map

>      L.map (f, *args, gen=False, **kwargs)

Create new `L` with `f` applied to all `items`, passing `args` and `kwargs` to `f`




```
test_eq(L.range(4).map(operator.neg), [0,-1,-2,-3])
```

If `f` is a string then it is treated as a format string to create the mapping:


```
test_eq(L.range(4).map('#{}#'), ['#0#','#1#','#2#','#3#'])
```

If `f` is a dictionary (or anything supporting `__getitem__`) then it is indexed to create the mapping:


```
test_eq(L.range(4).map(list('abcd')), list('abcd'))
```

You can also pass the same `arg` params that `bind` accepts:


```
def f(a=None,b=None): return b
test_eq(L.range(4).map(f, b=arg0), range(4))
```


```
show_doc(L.map_dict)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L167){target="_blank" style="float:right; font-size:smaller"}

### L.map_dict

>      L.map_dict (f=<function noop>, *args, gen=False, **kwargs)

Like `map`, but creates a dict from `items` to function results




```
test_eq(L(range(1,5)).map_dict(), {1:1, 2:2, 3:3, 4:4})
test_eq(L(range(1,5)).map_dict(operator.neg), {1:-1, 2:-2, 3:-3, 4:-4})
```


```
show_doc(L.zip)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L179){target="_blank" style="float:right; font-size:smaller"}

### L.zip

>      L.zip (cycled=False)

Create new `L` with `zip(*items)`




```
t = L([[1,2,3],'abc'])
test_eq(t.zip(), [(1, 'a'),(2, 'b'),(3, 'c')])
```


```
t = L([[1,2,3,4],['a','b','c']])
test_eq(t.zip(cycled=True ), [(1, 'a'),(2, 'b'),(3, 'c'),(4, 'a')])
test_eq(t.zip(cycled=False), [(1, 'a'),(2, 'b'),(3, 'c')])
```


```
show_doc(L.map_zip)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L181){target="_blank" style="float:right; font-size:smaller"}

### L.map_zip

>      L.map_zip (f, *args, cycled=False, **kwargs)

Combine `zip` and `starmap`




```
t = L([1,2,3],[2,3,4])
test_eq(t.map_zip(operator.mul), [2,6,12])
```


```
show_doc(L.zipwith)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L180){target="_blank" style="float:right; font-size:smaller"}

### L.zipwith

>      L.zipwith (*rest, cycled=False)

Create new `L` with `self` zip with each of `*rest`




```
b = [[0],[1],[2,2]]
t = L([1,2,3]).zipwith(b)
test_eq(t, [(1,[0]), (2,[1]), (3,[2,2])])
```


```
show_doc(L.map_zipwith)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L182){target="_blank" style="float:right; font-size:smaller"}

### L.map_zipwith

>      L.map_zipwith (f, *rest, cycled=False, **kwargs)

Combine `zipwith` and `starmap`




```
test_eq(L(1,2,3).map_zipwith(operator.mul, [2,3,4]), [2,6,12])
```


```
show_doc(L.itemgot)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L171){target="_blank" style="float:right; font-size:smaller"}

### L.itemgot

>      L.itemgot (*idxs)

Create new `L` with item `idx` of all `items`




```
test_eq(t.itemgot(1), b)
```


```
show_doc(L.attrgot)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L175){target="_blank" style="float:right; font-size:smaller"}

### L.attrgot

>      L.attrgot (k, default=None)

Create new `L` with attr `k` (or value `k` for dicts) of all `items`.




```
# Example when items are not a dict
a = [SimpleNamespace(a=3,b=4),SimpleNamespace(a=1,b=2)]
test_eq(L(a).attrgot('b'), [4,2])

#Example of when items are a dict
b =[{'id': 15, 'name': 'nbdev'}, {'id': 17, 'name': 'fastcore'}]
test_eq(L(b).attrgot('id'), [15, 17])
```


```
show_doc(L.sorted)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L136){target="_blank" style="float:right; font-size:smaller"}

### L.sorted

>      L.sorted (key=None, reverse=False)

New `L` sorted by `key`. If key is str use `attrgetter`; if int use `itemgetter`




```
test_eq(L(a).sorted('a').attrgot('b'), [2,4])
```


```
show_doc(L.split)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L152){target="_blank" style="float:right; font-size:smaller"}

### L.split

>      L.split (s, sep=None, maxsplit=-1)

Class Method: Same as `str.split`, but returns an `L`




```
test_eq(L.split('a b c'), list('abc'))
```


```
show_doc(L.range)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L154){target="_blank" style="float:right; font-size:smaller"}

### L.range

>      L.range (a, b=None, step=None)

Class Method: Same as `range`, but returns `L`. Can pass collection for `a`, to use `len(a)`




```
test_eq_type(L.range([1,1,1]), L(range(3)))
test_eq_type(L.range(5,2,2), L(range(5,2,2)))
```


```
show_doc(L.concat)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L188){target="_blank" style="float:right; font-size:smaller"}

### L.concat

>      L.concat ()

Concatenate all elements of list




```
test_eq(L([0,1,2,3],4,L(5,6)).concat(), range(7))
```


```
show_doc(L.copy)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L113){target="_blank" style="float:right; font-size:smaller"}

### L.copy

>      L.copy ()

Same as `list.copy`, but returns an `L`




```
t = L([0,1,2,3],4,L(5,6)).copy()
test_eq(t.concat(), range(7))
```


```
show_doc(L.map_first)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L168){target="_blank" style="float:right; font-size:smaller"}

### L.map_first

>      L.map_first (f=<function noop>, g=<function noop>, *args, **kwargs)

First element of `map_filter`




```
t = L(0,1,2,3)
test_eq(t.map_first(lambda o:o*2 if o>2 else None), 6)
```


```
show_doc(L.setattrs)
```




---

[source](https://github.com/fastai/fastcore/blob/master/fastcore/foundation.py#L192){target="_blank" style="float:right; font-size:smaller"}

### L.setattrs

>      L.setattrs (attr, val)

Call `setattr` on all items




```
t = L(SimpleNamespace(),SimpleNamespace())
t.setattrs('foo', 'bar')
test_eq(t.attrgot('foo'), ['bar','bar'])
```

## Config -


```
#|export
def save_config_file(file, d, **kwargs):
    "Write settings dict to a new config file, or overwrite the existing one."
    config = ConfigParser(**kwargs)
    config['DEFAULT'] = d
    config.write(open(file, 'w'))
```


```
#|export
def read_config_file(file, **kwargs):
    config = ConfigParser(**kwargs)
    config.read(file, encoding='utf8')
    return config['DEFAULT']
```

Config files are saved and read using Python's `configparser.ConfigParser`, inside the `DEFAULT` section.


```
_d = dict(user='fastai', lib_name='fastcore', some_path='test', some_bool=True, some_num=3)
try:
    save_config_file('tmp.ini', _d)
    res = read_config_file('tmp.ini')
finally: os.unlink('tmp.ini')
dict(res)
```




    {'user': 'fastai',
     'lib_name': 'fastcore',
     'some_path': 'test',
     'some_bool': 'True',
     'some_num': '3'}




```
#|export
class Config:
    "Reading and writing `ConfigParser` ini files"
    def __init__(self, cfg_path, cfg_name, create=None, save=True, extra_files=None, types=None):
        self.types = types or {}
        cfg_path = Path(cfg_path).expanduser().absolute()
        self.config_path,self.config_file = cfg_path,cfg_path/cfg_name
        self._cfg = ConfigParser()
        self.d = self._cfg['DEFAULT']
        found = [Path(o) for o in self._cfg.read(L(extra_files)+[self.config_file], encoding='utf8')]
        if self.config_file not in found and create is not None:
            self._cfg.read_dict({'DEFAULT':create})
            if save:
                cfg_path.mkdir(exist_ok=True, parents=True)
                save_config_file(self.config_file, create)

    def __repr__(self): return repr(dict(self._cfg.items('DEFAULT', raw=True)))
    def __setitem__(self,k,v): self.d[k] = str(v)
    def __contains__(self,k):  return k in self.d
    def save(self):            save_config_file(self.config_file,self.d)
    def __getattr__(self,k):   return stop(AttributeError(k)) if k=='d' or k not in self.d else self.get(k)
    def __getitem__(self,k):   return stop(IndexError(k)) if k not in self.d else self.get(k)

    def get(self,k,default=None):
        v = self.d.get(k, default)
        if v is None: return None
        typ = self.types.get(k, None)
        if typ==bool: return str2bool(v)
        if not typ: return str(v)
        if typ==Path: return self.config_path/v
        return typ(v)

    def path(self,k,default=None):
        v = self.get(k, default)
        return v if v is None else self.config_path/v
```

`Config` is a convenient wrapper around `ConfigParser` ini files with a single section (`DEFAULT`).

Instantiate a `Config` from an ini file at `cfg_path/cfg_name`:


```
save_config_file('../tmp.ini', _d)
try: cfg = Config('..', 'tmp.ini')
finally: os.unlink('../tmp.ini')
cfg
```




    {'user': 'fastai', 'lib_name': 'fastcore', 'some_path': 'test', 'some_bool': 'True', 'some_num': '3'}



You can create a new file if one doesn't exist by providing a `create` dict:


```
try: cfg = Config('..', 'tmp.ini', create=_d)
finally: os.unlink('../tmp.ini')
cfg
```




    {'user': 'fastai', 'lib_name': 'fastcore', 'some_path': 'test', 'some_bool': 'True', 'some_num': '3'}



If you additionally pass `save=False`, the `Config` will contain the items from `create` without writing a new file:


```
cfg = Config('..', 'tmp.ini', create=_d, save=False)
test_eq(cfg.user,'fastai')
assert not Path('../tmp.ini').exists()
```

Keys can be accessed as attributes, items, or with `get` and an optional default:


```
test_eq(cfg.user,'fastai')
test_eq(cfg['some_path'], 'test')
test_eq(cfg.get('foo','bar'),'bar')
```

Extra files can be read _before_ `cfg_path/cfg_name` using `extra_files`, in the order they appear:


```
with tempfile.TemporaryDirectory() as d:
    a = Config(d, 'a.ini', {'a':0,'b':0})
    b = Config(d, 'b.ini', {'a':1,'c':0})
    c = Config(d, 'c.ini', {'a':2,'d':0}, extra_files=[a.config_file,b.config_file])
    test_eq(c.d, {'a':'2','b':'0','c':'0','d':'0'})
```

If you pass a dict `types`, then the values of that dict will be used as types to instantiate all values returned. `Path` is a special case -- in that case, the path returned will be relative to the path containing the config file (assuming the value is relative). `bool` types use `str2bool` to convert to boolean.


```
_types = dict(some_path=Path, some_bool=bool, some_num=int)
cfg = Config('..', 'tmp.ini', create=_d, save=False, types=_types)

test_eq(cfg.user,'fastai')
test_eq(cfg['some_path'].resolve(), (Path('..')/'test').resolve())
test_eq(cfg.get('some_num'), 3)
```

# Export -


```
#|hide
import nbdev; nbdev.nbdev_export()
```


```

```
