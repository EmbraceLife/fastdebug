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

## Explore and document `inspect._signature_from_callable` (a long src)


## Expand cell

```python
# from IPython.core.display import display, HTML # a depreciated import
from IPython.display import display, HTML 
```

```python
display(HTML("<style>.container { width:100% !important; }</style>"))
```

## Imports and initiate

```python
from fastdebug.core import *
from fastcore.meta import *
```

```python
g = locals()
fdb = Fastdb(inspect._signature_from_callable, outloc=g)
fdbF = Fastdb(FixSigMeta, outloc=g)
```

## Examples

```python
from fastdebug.utils import whatinside
```

```python
inspect._signature_from_callable(whatinside, sigcls=inspect.Signature)
```

```python
fdb.eg = "inspect._signature_from_callable(whatinside, sigcls=inspect.Signature)"

fdb.eg = """
class Base: # pass
    def __new__(self, **args): pass  # defines a __new__ 

class Foo_new(Base):
    def __init__(self, d, e, f): pass
    
pprint(inspect._signature_from_callable(Foo_new, sigcls=inspect.Signature))
"""
fdb.eg = """
class Base: # pass
    def __new__(self, **args): pass  # defines a __new__ 

class Foo_new_fix(Base, metaclass=FixSigMeta):
    def __init__(self, d, e, f): pass
    
pprint(inspect._signature_from_callable(Foo_new_fix, sigcls=inspect.Signature))
"""

fdb.eg = """
class BaseMeta(type): 
    # using __new__ from type
    def __call__(cls, *args, **kwargs): pass
class Foo_call(metaclass=BaseMeta): 
    def __init__(self, d, e, f): pass

pprint(inspect._signature_from_callable(Foo_call, sigcls=inspect.Signature))
"""

fdbF.eg = """
class BaseMeta(FixSigMeta): 
    # using __new__ of  FixSigMeta instead of type
    def __call__(cls, *args, **kwargs): pass

class Foo_call_fix(metaclass=BaseMeta): # Base
    def __init__(self, d, e, f): pass

pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
"""

fdb.eg = """
class Foo_init:
    def __init__(self, a, b, c): pass

pprint(inspect._signature_from_callable(Foo_init, sigcls=inspect.Signature))
"""
```

```python
fdbF.docsrc(2, "how does a metaclass create a class instance; what does super().__new__() do here;", "inspect.getdoc(super)")
fdbF.docsrc(4, "how to remove self from a signature; how to check whether a class' __init__ is inherited from object or not;",\
            "res", "res.__init__ is not object.__init__")
fdbF.docsrc(1, "Any class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;\
FixSigMeta uses its __new__ to create a class instance; then check whether its class instance has its own __init__;\
if so, remove self from the sig of __init__; then assign this new sig to __signature__ for the class instance;")
```

```python
fdbF.snoop()
```

```python
fdb.docsrc(29, "How to check whether a class has __signature__?", "hasattr(obj, '__signature__')")
fdb.docsrc(82, "how to check whether obj whose signature is builtins;", "inspect.getdoc(_signature_is_builtin)")
fdb.docsrc(7, "inspect.signature is calling inspect._signature_from_callable; \
create _get_signature_of using functools.partial to call on _signature_from_callable itself;\
obj is first tested for callable; then test obj for classmethod; then unwrap to the end unless obj has __signature__;\
if obj has __signature__, assign __signature__ to sig; then test obj for function, is true calling _signature_from_function; \
then test obj whose signature is builtins or not; test whether obj created by functools.partial; test obj is a class or not; \
if obj is a class, then check obj has its own __call__ first; then its own __new__; then its own __init__; then inherited __new__; \
finally inherited __init__; and then get sig from either of them by calling _get_signature_of on them; \
FixSigMeta assigns __init__ function to __signature__ attr for the instance class it creates; \
so that class with FixSigMeta as metaclass can have sig from __init__ through __signature__; \
no more worry about interference of sig from __call__ or __new__.")
```

```python
fdb.snoop()
```

```python
fdbF.print()
```

```python
fdb.print(30, 1)
```

```python
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/Demos/_signature_from_callable_with_FixSigMeta.ipynb
!mv /Users/Natsume/Documents/fastdebug/Demos/_signature_from_callable_with_FixSigMeta.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/

!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```
