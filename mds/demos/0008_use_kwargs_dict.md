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

# 08_use_kwargs_dict


## Imports

```python
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```

## Reading official docs

```python
from fastcore.meta import _mk_param # not included in __all__
```

## empty2none

```python
fdbe = Fastdb(empty2none)
fdbe.docsrc(0, "p is the Parameter.default value")
fdbe.docsrc(1, "to use empty2none, I need to make sure p is not a parameter, but parameter.default")
fdbe.docsrc(2, "how to check whether a parameter default value is empty")
```

```python
# def foo(a, b=1): pass
# sig = inspect.signature(foo)
# print(sig.parameters.items())
# for k,v in sig.parameters.items():
#     print(f'{k} : {v.default} => empty2none => {empty2none(v.default)}')
```

```python
fdbe.eg = """
def foo(a, b=1): pass
sig = inspect.signature(foo)
print(sig.parameters.items())
for k,v in sig.parameters.items():
    print(f'{k} : {v.default} => empty2none => {empty2none(v.default)}')
"""
```

```python
fdbe.snoop()
```

## `_mk_param`

```python
fdb = Fastdb(_mk_param)
fdb.print()
```

```python
fdb.eg = """
print(_mk_param("a", 1))
"""
```

```python
fdb.snoop()
```

```python
fdb.docsrc(0, "_mk_param is to create a new parameter as KEYWORD_ONLY kind; n is its name in string; d is its default value")
```

## use_kwargs_dict


### Reading docs

<!-- #region -->
Replace all **kwargs with named arguments like so:
```python
@use_kwargs_dict(y=1,z=None)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None)')
```
Add named arguments, but optionally keep **kwargs by setting keep=True:
```python
@use_kwargs_dict(y=1,z=None, keep=True)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
```
<!-- #endregion -->

```python
print(inspect.getsource(use_kwargs_dict))
```

```python
fdb = Fastdb(use_kwargs_dict)
fdb.eg = """
@use_kwargs_dict(y=1,z=None)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None)')
"""

fdb.eg = """
@use_kwargs_dict(y=1,z=None, keep=True)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
"""
```

```python
fdb.print()
```

```python
fdb.docsrc(1, "how to use use_kwargs_dict; use_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict; \
f's signature is saved inside f.__signature__")
fdb.docsrc(3, "how to get the signature from an object")
fdb.docsrc(4, "how to get all parameters of a signature; how to make it into a dict; ")
fdb.docsrc(5, "how to pop out an item from a dict")
fdb.docsrc(6, "how to create a dict of params based on a dict")
fdb.docsrc(7, "how to udpate one dict into another dict")
fdb.docsrc(8, "how to create a new item in a dict")
fdb.docsrc(9, "how to update a signature with a new set of parameters in the form of a dict values")
```

```python
fdb.snoop(deco=True) # how to use snoop on decorator
```

## use_kwargs


### Reading docs

<!-- #region -->
use_kwargs is different than use_kwargs_dict as it only replaces **kwargs with named parameters without any default values:
```python
@use_kwargs(['y', 'z'])
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=None, z=None)')
```
You may optionally keep the **kwargs argument in your signature by setting keep=True:
```python
@use_kwargs(['y', 'z'], keep=True)
def foo(a, *args, b=1, **kwargs): pass
test_sig(foo, '(a, *args, b=1, y=None, z=None, **kwargs)')
```
<!-- #endregion -->

```python
print(inspect.getsource(use_kwargs))
```

```python
fdb = Fastdb(use_kwargs)
fdb.eg = """
@use_kwargs(['y', 'z'])
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=None, z=None)')
"""

fdb.eg = """
@use_kwargs(['y', 'z'], keep=True)
def foo(a, *args, b=1, **kwargs): pass
test_sig(foo, '(a, *args, b=1, y=None, z=None, **kwargs)')
"""
```

```python
fdb.print()
```

```python
fdb.docsrc(0, "How to use use_kwargs; use_kwargs has names as a list of strings; all the newly created params have None as default value; f's signature \
is saved inside f.__signature__")
```

```python
fdb.snoop(deco=True)
```

```python

```
