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

# Explore and Document BypassNewMeta

```python
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```

## Reading official docs


BypassNewMeta
> BypassNewMeta (name, bases, dict)     

Metaclass: casts x to this class if it's of type cls._bypass_type

BypassNewMeta is identical to NewChkMeta, except for checking for a class as the same type, we instead check for a class of type specified in attribute _bypass_type.

In NewChkMeta, objects of the same type passed to the constructor (without arguments) would result into a new variable referencing the same object. 

However, with BypassNewMeta this only occurs if the type matches the `_bypass_type` of the class you are defining:

```python
# class _TestA: pass
# class _TestB: pass

# class _T(_TestA, metaclass=BypassNewMeta):
#     _bypass_type=_TestB
#     def __init__(self,x): self.x=x
```

In the below example, t does not refer to t2 because t is of type _TestA while _T._bypass_type is of type TestB:

```python
# t = _TestA()
# t2 = _T(t)
# assert t is not t2
```

However, if t is set to _TestB to match _T._bypass_type, then both t and t2 will refer to the same object.

```python
# t = _TestB()
# t2 = _T(t)
# t2.new_attr = 15

# test_is(t, t2)
# # since t2 just references t these will be the same
# test_eq(t.new_attr, t2.new_attr)

# # likewise, chaning an attribute on t will also affect t2 because they both point to the same object.
# t.new_attr = 9
# test_eq(t2.new_attr, 9)
```

```python
# t = _TestB(); t
# isinstance(t, _TestB)
# id(_TestB)
# # t2 = _T(t)
# # t, t2
```

## Inspecting class

```python
inspect_class(BypassNewMeta)
```

## Initiating with examples

```python
g = locals()
fdb = Fastdb(BypassNewMeta, outloc=g)
fdb.eg = """
class _TestA: pass
class _TestB: pass

class _T(_TestA, metaclass=BypassNewMeta):
    _bypass_type=_TestB
    def __init__(self,x): self.x=x

t = _TestA()
print(t)
t2 = _T(t)
print(t2)
assert t is not t2
"""

fdb.eg = """
class _TestA: pass
class _TestB: pass

class _T(_TestA, metaclass=BypassNewMeta):
    _bypass_type=_TestB
    def __init__(self,x): self.x=x

t = _TestB()
t2 = _T(t)
t2.new_attr = 15

test_is(t, t2)
# since t2 just references t these will be the same
test_eq(t.new_attr, t2.new_attr)

# likewise, chaning an attribute on t will also affect t2 because they both point to the same object.
t.new_attr = 9
test_eq(t2.new_attr, 9)

# both t and t2's __class__ is _T
test_eq(t.__class__, t2.__class__)
test_eq(t.__class__, _T)
"""
```

## Snoop

```python
fdb.snoop()
```

```python
fdb.debug()
```

## Document

```python
fdb.docsrc(3, "If the instance class like _T has attr '_new_meta', then run it with param x;", "x", \
           "cls", "getattr(cls,'_bypass_type',object)", "isinstance(x, _TestB)", "isinstance(x,getattr(cls,'_bypass_type',object))")
fdb.docsrc(4, "when x is not an instance of _T's _bypass_type; or when a positional param is given; or when a keyword arg is given; \
let's run _T's super's __call__ function with x as param; and assign the result to x")
fdb.docsrc(6, "If x.__class__ is not cls or _T, then make it so")
fdb.docsrc(1, "BypassNewMeta allows its instance class e.g., _T to choose a specific class e.g., _TestB and \
change `__class__` of an object e.g., t of _TestB to _T without creating a new object")
```

```python
fdb.snoop(['cls._bypass_type', "isinstance(x,getattr(cls,'_bypass_type',object))"])
```

```python
fdb.debug()
```

```python
fdb.print()
```

```python

```
