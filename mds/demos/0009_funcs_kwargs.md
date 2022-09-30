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

# 09_method_funcs_kwargs


## fastcore.meta.method


### Reading Docs

<!-- #region -->
```python
#|export
def method(f):
    "Mark `f` as a method"
    # `1` is a dummy instance since Py3 doesn't allow `None` any more
    return MethodType(f, 1)
```

The method function is used to change a function's type to a method. In the below example we change the type of a from a function to a method:

```python
def a(x=2): return x + 1
assert type(a).__name__ == 'function'

a = method(a)
assert type(a).__name__ == 'method'
```
<!-- #endregion -->

### Running codes

```python
from fastcore.meta import method
```

```python
def a(x=2): return x + 1
assert type(a).__name__ == 'function' # how to test on the type of function or method

a = method(a)
assert type(a).__name__ == 'method'
```

### Document

```python
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```

```python
#| column: screen
fdb = Fastdb(method)
fdb.eg = """
def a(x=2): return x + 1
assert type(a).__name__ == 'function' # how to test on the type of function or method

a = method(a)
assert type(a).__name__ == 'method'
"""
fdb.print()
```

```python
#| column: screen
fdb.docsrc(2, "How to use fastcore.meta.method; method(function, instance); f needs to be a function; \
1 is a dummy instance to which the newly created method belongs; no need to worry about instance here")
```

```python
fdb.debug()
```

### snoop

```python
#| column: screen
fdb.snoop()
```

```python
fdb.debug()
```

## funcs_kwargs


### Official docs


The func_kwargs decorator allows you to add a list of functions or methods to an existing class. You must set this list as a class attribute named _methods when defining your class. Additionally, you must incldue the `**kwargs` argument in the ___init__ method of your class.

After defining your class this way, you can add functions to your class upon instantation as illusrated below.

For example, we define class T to allow adding the function b to class T as follows (note that this function is stored as an attribute of T and doesn't have access to cls or self):

```python
@funcs_kwargs
class T:
    _methods=['b'] # allows you to add method b upon instantiation
    def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
    def a(self): return 1
    def b(self): return 2
    
t = T()
test_eq(t.a(), 1)
test_eq(t.b(), 2)

test_sig(T, '(f=1, *, b=None)')
inspect.signature(T)

def _new_func(): return 5

t = T(b = _new_func)
test_eq(t.b(), 5)

t = T(a = lambda:3)
test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
```

### snoop: from _funcs_kwargs to funcs_kwargs


how to snoop on two functions one wrap around another: `funcs_kwargs` is a wrapper around `_funcs_kwargs`, so I can first snoop on `_funcs_kwargs` and assign its snoop dbsrc to \
`fm._funcs_kwargs` so that when I snoop on `funcs_kwargs`, it can use the snoop dbsrc of `_funcs_kwargs` and no example codes need to change.

```python
from fastcore.meta import _funcs_kwargs
```

```python
fdb_ = Fastdb(_funcs_kwargs)
fdb_.eg = """
@funcs_kwargs
class T:
    _methods=['b'] # allows you to add method b upon instantiation
    def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
    def a(self): return 1
    def b(self): return 2
    
t = T()
test_eq(t.a(), 1)
test_eq(t.b(), 2)

test_sig(T, '(f=1, *, b=None)')
inspect.signature(T)

def _new_func(): return 5

t = T(b = _new_func)
test_eq(t.b(), 5)

t = T(a = lambda:3)
test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
"""
```

```python
#| column: screen
# no snoop result, it is expected, because the example is not calling _funcs_kwargs, but funcs_kwargs
fdb_.snoop(deco=True) # how to snoop decorator: _funcs_kwargs is a decorator, so set deco=True to see running codes in inner f
```

```python
import fastcore.meta as fm
```

```python
fm._funcs_kwargs = fdb_.dbsrc # how to snoop on two functions one wrap around another
```

```python
fdb = Fastdb(funcs_kwargs)
fdb.eg = """
@funcs_kwargs
class T:
    _methods=['b'] # allows you to add method b upon instantiation
    def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
    def a(self): return 1
    def b(self): return 2
    
t = T()
test_eq(t.a(), 1)
test_eq(t.b(), 2)

test_sig(T, '(f=1, *, b=None)')
inspect.signature(T)

def _new_func(): return 5

t = T(b = _new_func)
test_eq(t.b(), 5)

t = T(a = lambda:3)
test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.

def _f(self,a=1): return self.num + a # access the num attribute from the instance

@funcs_kwargs(as_method=True)
class T: 
    _methods=['b']
    num = 5
    
t = T(b = _f) # adds method b
test_eq(t.b(5), 10) # self.num + 5 = 10

def _f(self,a=1): return self.num * a #multiply instead of add 

class T2(T):
    def __init__(self,num):
        super().__init__(b = _f) # add method b from the super class
        self.num=num
        
t = T2(num=3)
test_eq(t.b(a=5), 15) # 3 * 5 = 15
test_sig(T2, '(num)')
"""
```

```python
#| column: screen
fdb.print()
```

```python
#| column: screen
fdb_.print()
```

```python
#| column: screen
fdb.docsrc(1, "how funcs_kwargs works; it is a wrapper around _funcs_kwargs; it offers two ways of running _funcs_kwargs; \
the first, default way, is to add a func to a class without using self; second way is to add func to class enabling self use;")
fdb.docsrc(2, "how to check whether an object is callable; how to return a result of running a func; ")
fdb.docsrc(3, "how to custom the params of `_funcs_kwargs` for a particular use with partial")
```

```python
#| column: screen
fdb_.print()
```

```python
#| column: screen
fdb_.docsrc(0, "how does _funcs_kwargs work: _funcs_kwargs is a decorator; it helps class e.g., T to add more methods; \
I need to give the method a name, \
and put the name e.g., 'b' inside a list called _methods=['b'] inside class T; \
then after writing a func e.g., _new_func, I can add it by T(b = _new_func); if I want the func added to class to use self, \
I shall write @funcs_kwargs(as_method=True)")
fdb_.docsrc(2, "how to define a method which can use self and accept any parameters")
fdb_.docsrc(3, "how to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead; ")
fdb_.docsrc(6, "how to turn a func into a method")
fdb_.docsrc(7, "how to give a method a different instance, like self")
fdb_.docsrc(8, "how to add a method to a class as an attribute")
fdb_.docsrc(10, "how to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself")
fdb_.docsrc(11, "how to add a list of names with None as default value to function `_init` to repalce its kwargs param")
fdb_.docsrc(12, "how to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`")

```

```python
#| column: screen
fdb.snoop() # how to snoop together with docsrc: snoop first and docsrc above it
```

```python
fdb_.debug()
```

```python
#| column: screen
fdb.print()
```

```python
#| column: screen
fdb_.print()
```

```python

```

### snoop only '_funcs_kwargs' by breaking up 'funcs_kwargs'


I could do it this way, but I have to change more codes which leads to more efforts and potential errors. So not recommended.

```python
fdb = Fastdb(funcs_kwargs)
fdb_ = Fastdb(_funcs_kwargs)
```

```python
fdb_.eg = """

class T:
    _methods=['b'] # allows you to add method b upon instantiation
    def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
    def a(self): return 1
    def b(self): return 2
    
t = _funcs_kwargs(T, False)()
test_eq(t.a(), 1)
test_eq(t.b(), 2)

test_sig(T, '(f=1, *, b=None)')
inspect.signature(T)
"""
fdb_.eg = """
class T:
    _methods=['b'] # allows you to add method b upon instantiation
    def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
    def a(self): return 1
    def b(self): return 2
    
def _new_func(): return 5

t = _funcs_kwargs(T, False)(b = _new_func)
test_eq(t.b(), 5)
"""
# fdb_.eg = """
# class T:
#     _methods=['b'] # allows you to add method b upon instantiation
#     def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
#     def a(self): return 1
#     def b(self): return 2
    
# t = _funcs_kwargs(T, False)(a = lambda:3)
# test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
# """
```

```python
fdb_.snoop(watch=["cls.__init__"], deco=True) # can't do this expression
```

```python

```
