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

# 00_quesolved_anno_dict

```python
from fastcore.meta import *
from fastcore.test import *
import inspect
```

## `anno_dict` docs

```python
inspect.getdoc(anno_dict)
```

I have to confess I don't undersatnd the docs statement very well. So, I look into the source code of `anno_dict` and `empty2none`.

```python
print(inspect.getsource(anno_dict))
```

```python
print(inspect.getsource(empty2none))
```

## Dive in


If a parameter's default value is `Parameter.empty`, then `empty2none` is to replace `Parameter.empty` with `None` . So, I think it is reasonable to assume `p` is primarily used as a parameter's default value. The cell below supports this assumption.

```python
def foo(a, b:int=1): pass
sig = inspect.signature(foo)
for k,v in sig.parameters.items():
    print(f'{k} is a parameter {v}, whose default value is {v.default}, \
if apply empty2none to default value, then the default value is {empty2none(v.default)}')
    print(f'{k} is a parameter {v}, whose default value is {v.default}, \
if apply empty2none to parameter, then we get: {empty2none(v)}')
```

So, **what is odd** is that in `anno_dict`, `empty2none` is applied to `v` which is not parameter's default value, but mostly classes like `int`, `list` ect, as in `__annotations__`.

Then I experimented the section below and didn't find `anno_dict` doing anything new than `__annotations__`. 




## `anno_dict` seems not add anything new to `__annotations__`

```python
def foo(a, b:int=1): pass
test_eq(foo.__annotations__, {'b': int})
test_eq(anno_dict(foo), {'b': int})
def foo(a:bool, b:int=1): pass
test_eq(foo.__annotations__, {'a': bool, 'b': int})
test_eq(anno_dict(foo), {'a': bool, 'b': int})
def foo(a, d:list, b:int=1, c:bool=True): pass
test_eq(foo.__annotations__, {'d': list, 'b': int, 'c': bool})
test_eq(anno_dict(foo), {'d': list, 'b': int, 'c': bool})
```

```python
from fastcore.foundation import L
```

```python
def foo(a, b): pass
test_eq(foo.__annotations__, {})
test_eq(anno_dict(foo), {})

def _f(a:int, b:L)->str: ...
test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})
```

**Question!** so far above anno_dict has done nothing new or more, so what am I missing here?


## use fastdebug to double check

```python
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```

```python
fdb = Fastdb(anno_dict)
fdb.eg = """
def foo(a, b): pass
test_eq(foo.__annotations__, {})
test_eq(anno_dict(foo), {})

from fastcore.foundation import L
def _f(a:int, b:L)->str: ...
test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})
"""
```

```python
fdb.snoop(['empty2none(v)'])
```

```python
fdb.docsrc(1, "empty2none works on paramter.default especially when the default is Parameter.empty; anno_dict works on the types \
of params, not the value of params; so it is odd to use empty2none in anno_dict;")
```

```python
fdb.print()
```

## Does fastcore want anno_dict to include params with no annos?


If so, I have written a lengthy `anno_dict_maybe` to do it. (can be shorter if needed)

```python
def anno_dict_maybe(f):
    "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist"
    new_anno = {}
    for k, v in inspect.signature(f).parameters.items():
        if k not in f.__annotations__:
            new_anno[k] = None
        else: 
            new_anno[k] = f.__annotations__[k]
    if 'return' in f.__annotations__:
        new_anno['return'] = f.__annotations__['return']
#     if hasattr(f, '__annotations__'):
    if True in [bool(v) for k,v in new_anno.items()]:
        return new_anno
    else:
        return {}
```

```python
def foo(a:int, b, c:bool=True)->str: pass
```

```python
test_eq(foo.__annotations__, {'a': int, 'c': bool, 'return': str})
```

```python
test_eq(anno_dict(foo), {'a': int, 'c': bool, 'return': str})
```

```python
test_eq(anno_dict_maybe(foo), {'a': int, 'b': None, 'c': bool, 'return': str})
```

```python
def foo(a, b, c): pass
```

```python
test_eq(foo.__annotations__, {})
```

```python
test_eq(anno_dict(foo), {})
```

```python
test_eq(anno_dict_maybe(foo), {})
```

## Jeremy's response


A supportive and confirmative [response](https://forums.fast.ai/t/help-reading-fastcore-docs/100168/3?u=daniel) from Jeremy on this issue

```python

```
