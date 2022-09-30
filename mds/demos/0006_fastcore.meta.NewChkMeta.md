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

# 06_NewChkMeta


## Import and Initalization

```python
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```

```python
g = locals()
fdb = Fastdb(NewChkMeta, outloc = g)
```

```python
fdb.print()
```

## Official docs
The official docs (at first, it does not make sense to me)


`NewChkMeta` is used when an object of the same type is the first argument to your class's constructor (i.e. the __init__ function), and you would rather it not create a new object but point to the same exact object.

This is used in L, for example, to avoid creating a new object when the object is already of type L. This allows the users to defenisvely instantiate an L object and just return a reference to the same object if it already happens to be of type L.

For example, the below class _T optionally accepts an object o as its first argument. A new object is returned upon instantiation per usual:

```python
class _T():
    "Testing"
    def __init__(self, o): 
        # if `o` is not an object without an attribute `foo`, set foo = 1
        self.foo = getattr(o,'foo',1)
        
t = _T(3)
test_eq(t.foo,1) # 1 was not of type _T, so foo = 1

t2 = _T(t) #t1 is of type _T
assert t is not t2 # t1 and t2 are different objects
```

However, if we want _T to return a reference to the same object when passed an an object of type _T we can inherit from the NewChkMeta class as illustrated below:

```python
class _T(metaclass=NewChkMeta):
    "Testing with metaclass NewChkMeta"
    def __init__(self, o=None, b=1):
        # if `o` is not an object without an attribute `foo`, set foo = 1
        self.foo = getattr(o,'foo',1)
        self.b = b

t = _T(3)
test_eq(t.foo,1) # 1 was not of type _T, so foo = 1

t2 = _T(t) # t2 will now reference t

test_is(t, t2) # t and t2 are the same object
t2.foo = 5 # this will also change t.foo to 5 because it is the same object
test_eq(t.foo, 5)
test_eq(t2.foo, 5)

t3 = _T(t, b=1)
assert t3 is not t

t4 = _T(t) # without any arguments the constructor will return a reference to the same object
assert t4 is t
```

```python

```

## Prepare Example

```python
fdb.eg = """
class _T(metaclass=NewChkMeta):
    "Testing with metaclass NewChkMeta"
    def __init__(self, o=None, b=1):
        # if `o` is not an object without an attribute `foo`, set foo = 1
        self.foo = getattr(o,'foo',1)
        self.b = b

t = _T(3)
test_eq(t.foo,1) # 1 was not of type _T, so foo = 1

t2 = _T(t) # t2 will now reference t

test_is(t, t2) # t and t2 are the same object
t2.foo = 5 # this will also change t.foo to 5 because it is the same object
test_eq(t.foo, 5)
test_eq(t2.foo, 5)

t3 = _T(t, b=1)
assert t3 is not t

t4 = _T(t) # without any arguments the constructor will return a reference to the same object
assert t4 is t
"""
```

## Inspect classes

```python
inspect_class(NewChkMeta)
```

```python
inspect_class(_T)
```

## Snoop

```python
fdb.snoop()
```

## Document

```python
fdb.docsrc(1, "NewChkMeta is a metaclass inherited from FixSigMea; it makes its own __call__; \
when its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, \
and x is an object of the instance class, then return x; otherwise, create and return a new object created by \
the instance class's super class' __call__ method with x as param; In other words, t = _T(3) will create a new obj; \
_T(t) will return t; _T(t, 1) or _T(t, b=1) will also return a new obj")
fdb.docsrc(2, "how to create a __call__ method with param cls, x, *args, **kwargs;")
fdb.docsrc(3, "how to express no args and no kwargs and x is an instance of cls?")
fdb.docsrc(4, "how to call __call__ of super class with x and consider all possible situations of args and kwargs")
```

```python
fdb.print()
```

```python

```
