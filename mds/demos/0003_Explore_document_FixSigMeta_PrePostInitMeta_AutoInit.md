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

# 03_FixSigMeta_PrePostInitMeta_AutoInit

```python
from fastdebug.utils import *
from fastdebug.core import *
import inspect
```

```python
from fastcore.meta import *
from fastcore.test import *
```

## Initialize fastdebug objects

```python
g = locals() # g can update itself as more cells get run, like globals() in the __main__
fdbF = Fastdb(FixSigMeta, outloc=g)
fdbP = Fastdb(PrePostInitMeta, outloc=g)
fdbA = Fastdb(AutoInit, outloc=g)
```

## class FixSigMeta(type) vs class Foo(type)


FixSigMeta inherits `__init__`, and `__call__` from `type`, but writes its own `__new__`    
Foo inherits all three from `type`    
FixSigMeta is used to create class instance not object instance

```python
fdbF.docsrc(1, "FixSigMeta inherits __init__, and __call__ from type; but writes its own __new__; Foo inherits all three from type; \
FixSigMeta is used to create class instance not object instance.")
# fdbF.print()
```

```python
print(inspect.getsource(FixSigMeta))
```

```python
inspect_class(FixSigMeta)
```

```python
class Foo(type): pass
inspect_class(Foo)
```

## class Foo()


When Foo inherit `__new__` and `__new__` from `object`    
but `__call__` of Foo and `__call__` of `object` maybe the same but different objects    
Foo is to create object instance not class instance

```python
class Foo(): pass
inspect_class(Foo)
```

## class PrePostInitMeta(FixSigMeta)


`PrePostInitMeta` inherit `__new__` and `__init__` from `FixSigMeta` as a metaclass (a different type), not from `type`, nor from `object`    
`PrePostInitMeta` is itself a metaclass, which is used to create class instance not object instance     
`PrePostInitMeta` writes its own `__call__` which regulates how its class instance create and initialize object instance 

```python
fdbP.docsrc(1, "PrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type); \
not from type, nor from object; PrePostInitMeta is itself a metaclass, which is used to create class instance not object instance; \
PrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance")
```

```python
inspect_class(PrePostInitMeta)
```

```python
class Foo(FixSigMeta): pass
inspect_class(Foo)
```

## class Foo(metaclass=FixSigMeta)


Foo inherit `__new__`, `__init__` from object to create object instance     
Foo uses FixSigMeta not type to create class instance
FixSigMeta.`__new__` determine what kind of a class is Foo    
In this case, FixSigMeta.`__new__` create Foo class and an attr `__signature__` if Foo has its own `__init__`    
FixSigMeta.`__new__` create Foo the class, has nothing to do with the instance method Foo.`__init__`

```python
class Foo(metaclass=FixSigMeta): pass
inspect_class(Foo)
```

```python
class Foo(metaclass=FixSigMeta): 
    def __init__(self, a, b): pass
inspect_class(Foo)
```

## class AutoInit(metaclass=PrePostInitMeta)


AutoInit inherit `__new__` and `__init__` from `object` to create and initialize object instances     
AutoInit uses PrePostInitMeta.`__new__` or in fact FixSigMeta.`__new__` to create its own class instance, which can have `__signature__`    
AutoInit uses PrePostInitMeta.`__call__` to specify how its object instance to be created and initialized (with pre_init, init, post_init)     
AutoInit as a normal or non-metaclass, it writes its own `__pre_init__` instance method

```python
fdbA.docsrc(1, "AutoInit inherit __new__ and __init__ from object to create and initialize object instances; \
AutoInit uses PrePostInitMeta.__new__ or in fact FixSigMeta.__new__ to create its own class instance, which can have __signature__; \
AutoInit uses PrePostInitMeta.__call__ to specify how its object instance to be created and initialized (with pre_init, init, post_init)); \
AutoInit as a normal or non-metaclass, it writes its own __pre_init__ method")
```

```python

```

```python
inspect_class(AutoInit)
```

```python
class Foo(AutoInit): pass
inspect_class(Foo)
```

```python
class Foo(AutoInit): 
    def __init__(self): pass # to enable __signature__ by FixSigMeta.__new__
inspect_class(Foo)
```

```python
class TestParent():
    def __init__(self): self.h = 10
        
class TestChild(AutoInit, TestParent):
    def __init__(self): self.k = self.h + 2
inspect_class(TestChild)
```

```python
class _T(metaclass=PrePostInitMeta):
    def __pre_init__(self):  self.a  = 0; 
    def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
    def __post_init__(self): self.c = self.b + 2; assert self.c==3
inspect_class(_T)
```

```python

```

## Prepare examples for FixSigMeta, PrePostInitMeta, AutoInit 

```python
# g = locals() 
# fdbF = Fastdb(FixSigMeta, outloc=g)
fdbF.eg = """
class Foo(metaclass=FixSigMeta):
    def __init__(self): pass
"""

# fdbP = Fastdb(PrePostInitMeta, outloc=g)
fdbP.eg = """
class _T(metaclass=PrePostInitMeta):
    def __pre_init__(self):  self.a  = 0; 
    def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
    def __post_init__(self): self.c = self.b + 2; assert self.c==3

t = _T()
test_eq(t.a, 0) # set with __pre_init__
test_eq(t.b, 1) # set with __init__
test_eq(t.c, 3) # set with __post_init__
inspect.signature(_T)
"""

# fdbA = Fastdb(AutoInit, outloc=g)
fdbA.eg = """
class TestParent():
    def __init__(self): self.h = 10
        
class TestChild(AutoInit, TestParent):
    def __init__(self): self.k = self.h + 2
    
t = TestChild()
test_eq(t.h, 10) # h=10 is initialized in the parent class
test_eq(t.k, 12)
"""
```

## Snoop them together in one go

```python

fdbF.snoop(watch=['res', 'type(res)', 'res.__class__', 'res.__dict__'])
```

### embed the dbsrc of FixSigMeta into PrePostInitMeta


**Important!**    
FixSigMeta is untouched, fdbF.dbsrc.`__new__` is the actual dbsrc     
To use fdbF.dbsrc in other functions or classes which uses FixSigMeta, we need to assign `fdbF.dbsrc` to `fm.FixSigMeta`

```python
import fastcore.meta as fm
```

```python
fm.FixSigMeta = fdbF.dbsrc
```

```python
fdbP.snoop(['res.__dict__'])
```

### embed dbsrc of PrePostInitMeta into AutoInit

```python
fm.PrePostInitMeta = fdbP.dbsrc
```

```python
fdbA.snoop()
```

## Explore and Document on them together 

```python
fdbF.docsrc(4, "FixSigMeta: what is res", "'inside FixSigMeta, line 4'", "res.__name__")
fm.FixSigMeta = fdbF.dbsrc
```

```python
fdbP.docsrc(6, "what inside res.__dict__", "'inside PrePostInitMeta: '", "res.__dict__")
fm.PrePostInitMeta = fdbP.dbsrc
```

```python
fdbA.docsrc(2, "what is cls", "'Inside AutoInit'", "cls") # need to run it twice (a little bug here)
```

```python
fdbF.docsrc(3, "how to create a new class instance with type dynamically; \
the rest below is how FixSigMeta as a metaclass create its own instance classes")
fdbF.docsrc(4, "how to check whether a class has its own __init__ function; how to remove self param from a signature")
```

```python
fdbF.print()
```

```python
fdbP.docsrc(3, "how to create an object instance with a cls; how to check the type of an object is cls; \
how to run a function without knowing its params;")
fdbP.print()
```

```python
fdbA.docsrc(2, "how to run superclass' __init__ function")
```

```python
fdbA.print()
```

```python
fdbP.docsrc(6, "how to run __init__ without knowing its params")
```

```python
fdbP.print()
```

```python
fdbF.print()
```

```python

```
