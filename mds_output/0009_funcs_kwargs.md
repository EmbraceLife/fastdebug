# 09_method_funcs_kwargs

## fastcore.meta.method

### Reading Docs

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

### Running codes


```
from fastcore.meta import method
```


```
def a(x=2): return x + 1
assert type(a).__name__ == 'function' # how to test on the type of function or method

a = method(a)
assert type(a).__name__ == 'method'
```

### Document


```
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```


<style>.container { width:100% !important; }</style>



```
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

    ==========================================================     Investigating [91;1mmethod[0m     ==========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    def a(x=2): return x + 1
    assert type(a).__name__ == 'function' # how to test on the type of function or method
    
    a = method(a)
    assert type(a).__name__ == 'method'
    [0m     
    
    def method(f):============================================================================(0)       
        "Mark `f` as a method"================================================================(1)       
        # `1` is a dummy instance since Py3 doesn't allow `None` any more=====================(2)       
        return MethodType(f, 1)===============================================================(3)       
                                                                                                                                                            (4)



```
#| column: screen
fdb.docsrc(2, "How to use fastcore.meta.method; method(function, instance); f needs to be a function; \
1 is a dummy instance to which the newly created method belongs; no need to worry about instance here")
```

    ==========================================================     Investigating [91;1mmethod[0m     ==========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
    def a(x=2): return x + 1
    assert type(a).__name__ == 'function' # how to test on the type of function or method
    
    a = method(a)
    assert type(a).__name__ == 'method'
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def method(f):                                                                                                                                          (0)
        "Mark `f` as a method"                                                                                                                              (1)
        # `1` is a dummy instance since Py3 doesn't allow `None` any more===================================================================================(2)
    [91;1mHow to use fastcore.meta.method; method(function, instance); f needs to be a function; 1 is a dummy instance to which the newly created method belongs; no need to worry about instance here[0m
        return MethodType(f, 1)                                                                                                                             (3)
                                                                                                                                                            (4)



```
fdb.debug()
```

    method's dbsrc code: ==============
    import snoop
    @snoop
    def method(f):
        "Mark `f` as a method"
        # `1` is a dummy instance since Py3 doesn't allow `None` any more
        return MethodType(f, 1)
    
    
    
    method's example processed with dbsrc: ===============
    
    def a(x=2): return x + 1
    assert type(a).__name__ == 'function' # how to test on the type of function or method
    
    a = method(a)
    assert type(a).__name__ == 'method'
    


### snoop


```
#| column: screen
fdb.snoop()
```

    22:28:59.92 >>> Call to method in File "/tmp/method.py", line 3
    22:28:59.92 ...... f = <function a>
    22:28:59.92    3 | def method(f):
    22:28:59.92    6 |     return MethodType(f, 1)
    22:28:59.92 <<< Return value from method: <bound method int.a of 1>


    ==========================================================     Investigating [91;1mmethod[0m     ==========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    def a(x=2): return x + 1
    assert type(a).__name__ == 'function' # how to test on the type of function or method
    
    a = method(a)
    assert type(a).__name__ == 'method'
    [0m     
    



```
fdb.debug()
```

    method's dbsrc code: ==============
    import snoop
    @snoop
    def method(f):
        "Mark `f` as a method"
        # `1` is a dummy instance since Py3 doesn't allow `None` any more
        return MethodType(f, 1)
    
    
    
    method's example processed with dbsrc: ===============
    
    def a(x=2): return x + 1
    assert type(a).__name__ == 'function' # how to test on the type of function or method
    
    a = self.dbsrc(a)
    assert type(a).__name__ == 'method'
    
    


## funcs_kwargs

### Official docs

The func_kwargs decorator allows you to add a list of functions or methods to an existing class. You must set this list as a class attribute named _methods when defining your class. Additionally, you must incldue the `**kwargs` argument in the ___init__ method of your class.

After defining your class this way, you can add functions to your class upon instantation as illusrated below.

For example, we define class T to allow adding the function b to class T as follows (note that this function is stored as an attribute of T and doesn't have access to cls or self):


```
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


```
from fastcore.meta import _funcs_kwargs
```


```
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


```
#| column: screen
# no snoop result, it is expected, because the example is not calling _funcs_kwargs, but funcs_kwargs
fdb_.snoop(deco=True) # how to snoop decorator: _funcs_kwargs is a decorator, so set deco=True to see running codes in inner f
```

    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    



```
import fastcore.meta as fm
```


```
fm._funcs_kwargs = fdb_.dbsrc # how to snoop on two functions one wrap around another
```


```
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


```
#| column: screen
fdb.print()
```

    =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    
    def funcs_kwargs(as_method=False):========================================================(0)       
        "Replace methods in `cls._methods` with those from `kwargs`"==========================(1)       
        if callable(as_method): return _funcs_kwargs(as_method, False)========================(2)       
        return partial(_funcs_kwargs, as_method=as_method)====================================(3)       
                                                                                                                                                            (4)



```
#| column: screen
fdb_.print()
```

    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    
    [0m     
    
    def _funcs_kwargs(cls, as_method):========================================================(0)       
        old_init = cls.__init__===============================================================(1)       
        def _init(self, *args, **kwargs):=====================================================(2)       
            for k in cls._methods:============================================================(3)       
                arg = kwargs.pop(k,None)======================================================(4)       
                if arg is not None:===========================================================(5)       
                    if as_method: arg = method(arg)===========================================(6)       
                    if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=======(7)       
                    setattr(self, k, arg)=====================================================(8)       
            old_init(self, *args, **kwargs)===================================================(9)       
        functools.update_wrapper(_init, old_init)=============================================(10)      
        cls.__init__ = use_kwargs(cls._methods)(_init)========================================(11)      
        if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
        return cls============================================================================(13)      
                                                                                                                                                            (14)



```
#| column: screen
fdb.docsrc(1, "how funcs_kwargs works; it is a wrapper around _funcs_kwargs; it offers two ways of running _funcs_kwargs; \
the first, default way, is to add a func to a class without using self; second way is to add func to class enabling self use;")
fdb.docsrc(2, "how to check whether an object is callable; how to return a result of running a func; ")
fdb.docsrc(3, "how to custom the params of `_funcs_kwargs` for a particular use with partial")
```

    =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def funcs_kwargs(as_method=False):                                                                                                                      (0)
        "Replace methods in `cls._methods` with those from `kwargs`"========================================================================================(1)
    [91;1mhow funcs_kwargs works; it is a wrapper around _funcs_kwargs; it offers two ways of running _funcs_kwargs; the first, default way, is to add a func to a class without using self; second way is to add func to class enabling self use;[0m
        if callable(as_method): return _funcs_kwargs(as_method, False)                                                                                      (2)
        return partial(_funcs_kwargs, as_method=as_method)                                                                                                  (3)
    =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def funcs_kwargs(as_method=False):                                                                                                                      (0)
        "Replace methods in `cls._methods` with those from `kwargs`"                                                                                        (1)
        if callable(as_method): return _funcs_kwargs(as_method, False)======================================================================================(2)
                                                                           [91;1mhow to check whether an object is callable; how to return a result of running a func; [0m
        return partial(_funcs_kwargs, as_method=as_method)                                                                                                  (3)
                                                                                                                                                            (4)
    =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        "Replace methods in `cls._methods` with those from `kwargs`"                                                                                        (1)
        if callable(as_method): return _funcs_kwargs(as_method, False)                                                                                      (2)
        return partial(_funcs_kwargs, as_method=as_method)==================================================================================================(3)
                                                                                    [91;1mhow to custom the params of `_funcs_kwargs` for a particular use with partial[0m
                                                                                                                                                            (4)



```
#| column: screen
fdb_.print()
```

    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    
    [0m     
    
    def _funcs_kwargs(cls, as_method):========================================================(0)       
        old_init = cls.__init__===============================================================(1)       
        def _init(self, *args, **kwargs):=====================================================(2)       
            for k in cls._methods:============================================================(3)       
                arg = kwargs.pop(k,None)======================================================(4)       
                if arg is not None:===========================================================(5)       
                    if as_method: arg = method(arg)===========================================(6)       
                    if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=======(7)       
                    setattr(self, k, arg)=====================================================(8)       
            old_init(self, *args, **kwargs)===================================================(9)       
        functools.update_wrapper(_init, old_init)=============================================(10)      
        cls.__init__ = use_kwargs(cls._methods)(_init)========================================(11)      
        if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
        return cls============================================================================(13)      
                                                                                                                                                            (14)



```
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

    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m0[0m     ================================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def _funcs_kwargs(cls, as_method):======================================================================================================================(0)
    [91;1mhow does _funcs_kwargs work: _funcs_kwargs is a decorator; it helps class e.g., T to add more methods; I need to give the method a name, and put the name e.g., 'b' inside a list called _methods=['b'] inside class T; then after writing a func e.g., _new_func, I can add it by T(b = _new_func); if I want the func added to class to use self, I shall write @funcs_kwargs(as_method=True)[0m
        old_init = cls.__init__                                                                                                                             (1)
        def _init(self, *args, **kwargs):                                                                                                                   (2)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def _funcs_kwargs(cls, as_method):                                                                                                                      (0)
        old_init = cls.__init__                                                                                                                             (1)
        def _init(self, *args, **kwargs):===================================================================================================================(2)
                                                                                              [91;1mhow to define a method which can use self and accept any parameters[0m
            for k in cls._methods:                                                                                                                          (3)
                arg = kwargs.pop(k,None)                                                                                                                    (4)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        old_init = cls.__init__                                                                                                                             (1)
        def _init(self, *args, **kwargs):                                                                                                                   (2)
            for k in cls._methods:==========================================================================================================================(3)
                                  [91;1mhow to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead; [0m
                arg = kwargs.pop(k,None)                                                                                                                    (4)
                if arg is not None:                                                                                                                         (5)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m6[0m     ================================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
                arg = kwargs.pop(k,None)                                                                                                                    (4)
                if arg is not None:                                                                                                                         (5)
                    if as_method: arg = method(arg)=========================================================================================================(6)
                                                                                                                                 [91;1mhow to turn a func into a method[0m
                    if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)                                                                     (7)
                    setattr(self, k, arg)                                                                                                                   (8)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m7[0m     ================================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
                if arg is not None:                                                                                                                         (5)
                    if as_method: arg = method(arg)                                                                                                         (6)
                    if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=====================================================================(7)
                                                                                                             [91;1mhow to give a method a different instance, like self[0m
                    setattr(self, k, arg)                                                                                                                   (8)
            old_init(self, *args, **kwargs)                                                                                                                 (9)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m8[0m     ================================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
                    if as_method: arg = method(arg)                                                                                                         (6)
                    if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)                                                                     (7)
                    setattr(self, k, arg)===================================================================================================================(8)
                                                                                                                   [91;1mhow to add a method to a class as an attribute[0m
            old_init(self, *args, **kwargs)                                                                                                                 (9)
        functools.update_wrapper(_init, old_init)                                                                                                           (10)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m10[0m     ===============================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
                    setattr(self, k, arg)                                                                                                                   (8)
            old_init(self, *args, **kwargs)                                                                                                                 (9)
        functools.update_wrapper(_init, old_init)===========================================================================================================(10)
                                                                          [91;1mhow to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself[0m
        cls.__init__ = use_kwargs(cls._methods)(_init)                                                                                                      (11)
        if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m11[0m     ===============================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            old_init(self, *args, **kwargs)                                                                                                                 (9)
        functools.update_wrapper(_init, old_init)                                                                                                           (10)
        cls.__init__ = use_kwargs(cls._methods)(_init)======================================================================================================(11)
                                                            [91;1mhow to add a list of names with None as default value to function `_init` to repalce its kwargs param[0m
        if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
        return cls                                                                                                                                          (13)
    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m12[0m     ===============================================================
         with example [91;1m
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
    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        functools.update_wrapper(_init, old_init)                                                                                                           (10)
        cls.__init__ = use_kwargs(cls._methods)(_init)                                                                                                      (11)
        if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))=====================================================(12)
                                                   [91;1mhow to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`[0m
        return cls                                                                                                                                          (13)
                                                                                                                                                            (14)



```
#| column: screen
fdb.snoop() # how to snoop together with docsrc: snoop first and docsrc above it
```

    22:28:59.99 >>> Call to funcs_kwargs in File "/tmp/funcs_kwargs.py", line 3
    22:28:59.99 ...... as_method = <class 'fastcore.meta.T'>
    22:28:59.99    3 | def funcs_kwargs(as_method=False):
    22:28:59.99    5 |     if callable(as_method): return _funcs_kwargs(as_method, False)
        22:28:59.99 >>> Call to _funcs_kwargs in File "/tmp/_funcs_kwargs.py", line 3
        22:28:59.99 ...... cls = <class 'fastcore.meta.T'>
        22:28:59.99 ...... as_method = False
        22:28:59.99    3 | def _funcs_kwargs(cls, as_method):
        22:28:59.99    4 |     old_init = cls.__init__
        22:28:59.99 .......... old_init = <function T.__init__>
        22:28:59.99    5 |     import snoop
        22:28:59.99 .......... snoop = <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>
        22:28:59.99    6 |     @snoop
        22:28:59.99    7 |     def _init(self, *args, **kwargs):
        22:28:59.99 .......... _init = <function _funcs_kwargs.<locals>._init>
        22:28:59.99   15 |     functools.update_wrapper(_init, old_init)
        22:28:59.99 .......... _init = <function T.__init__>
        22:28:59.99   16 |     cls.__init__ = use_kwargs(cls._methods)(_init)
        22:28:59.99   17 |     if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))
        22:28:59.99   18 |     return cls
        22:28:59.99 <<< Return value from _funcs_kwargs: <class 'fastcore.meta.T'>
    22:28:59.99    5 |     if callable(as_method): return _funcs_kwargs(as_method, False)
    22:28:59.99 <<< Return value from funcs_kwargs: <class 'fastcore.meta.T'>
    22:28:59.99 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
    22:28:59.99 .......... self = <fastcore.meta.T object>
    22:28:59.99 .......... args = ()
    22:28:59.99 .......... kwargs = {}
    22:28:59.99 .......... as_method = False
    22:28:59.99 .......... cls = <class 'fastcore.meta.T'>
    22:28:59.99 .......... old_init = <function T.__init__>
    22:28:59.99    7 |     def _init(self, *args, **kwargs):
    22:29:00.00    8 |         for k in cls._methods:
    22:29:00.00 .............. k = 'b'
    22:29:00.00    9 |             arg = kwargs.pop(k,None)
    22:29:00.00 .................. arg = None
    22:29:00.00   10 |             if arg is not None:
    22:29:00.00    8 |         for k in cls._methods:
    22:29:00.00   14 |         old_init(self, *args, **kwargs)
    22:29:00.00 <<< Return value from _funcs_kwargs.<locals>._init: None
    22:29:00.00 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
    22:29:00.00 .......... self = <fastcore.meta.T object>
    22:29:00.00 .......... args = ()
    22:29:00.00 .......... kwargs = {'b': <function _new_func>}
    22:29:00.00 .......... len(kwargs) = 1
    22:29:00.00 .......... as_method = False
    22:29:00.00 .......... cls = <class 'fastcore.meta.T'>
    22:29:00.00 .......... old_init = <function T.__init__>
    22:29:00.00    7 |     def _init(self, *args, **kwargs):
    22:29:00.00    8 |         for k in cls._methods:
    22:29:00.00 .............. k = 'b'
    22:29:00.00    9 |             arg = kwargs.pop(k,None)
    22:29:00.00 .................. kwargs = {}
    22:29:00.00 .................. arg = <function _new_func>
    22:29:00.00   10 |             if arg is not None:
    22:29:00.00   11 |                 if as_method: arg = method(arg)
    22:29:00.00   12 |                 if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
    22:29:00.00   13 |                 setattr(self, k, arg)
    22:29:00.00    8 |         for k in cls._methods:
    22:29:00.00   14 |         old_init(self, *args, **kwargs)
    22:29:00.00 <<< Return value from _funcs_kwargs.<locals>._init: None
    22:29:00.00 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
    22:29:00.00 .......... self = <fastcore.meta.T object>
    22:29:00.00 .......... args = ()
    22:29:00.00 .......... kwargs = {'a': <function <lambda>>}
    22:29:00.00 .......... len(kwargs) = 1
    22:29:00.00 .......... as_method = False
    22:29:00.00 .......... cls = <class 'fastcore.meta.T'>
    22:29:00.00 .......... old_init = <function T.__init__>
    22:29:00.00    7 |     def _init(self, *args, **kwargs):
    22:29:00.00    8 |         for k in cls._methods:
    22:29:00.00 .............. k = 'b'
    22:29:00.00    9 |             arg = kwargs.pop(k,None)
    22:29:00.00 .................. arg = None
    22:29:00.00   10 |             if arg is not None:
    22:29:00.00    8 |         for k in cls._methods:
    22:29:00.00   14 |         old_init(self, *args, **kwargs)
    22:29:00.00 <<< Return value from _funcs_kwargs.<locals>._init: None
    22:29:00.00 >>> Call to funcs_kwargs in File "/tmp/funcs_kwargs.py", line 3
    22:29:00.00 ...... as_method = True
    22:29:00.00    3 | def funcs_kwargs(as_method=False):
    22:29:00.00    5 |     if callable(as_method): return _funcs_kwargs(as_method, False)
    22:29:00.00    6 |     return partial(_funcs_kwargs, as_method=as_method)
    22:29:00.00 <<< Return value from funcs_kwargs: functools.partial(<function _funcs_kwargs>, as_method=True)
    22:29:00.00 >>> Call to _funcs_kwargs in File "/tmp/_funcs_kwargs.py", line 3
    22:29:00.00 ...... cls = <class 'fastcore.meta.T'>
    22:29:00.00 ...... as_method = True
    22:29:00.00    3 | def _funcs_kwargs(cls, as_method):
    22:29:00.00    4 |     old_init = cls.__init__
    22:29:00.00 .......... old_init = <slot wrapper '__init__' of 'object' objects>
    22:29:00.00    5 |     import snoop
    22:29:00.00 .......... snoop = <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>
    22:29:00.00    6 |     @snoop
    22:29:00.00    7 |     def _init(self, *args, **kwargs):
    22:29:00.00 .......... _init = <function _funcs_kwargs.<locals>._init>
    22:29:00.00   15 |     functools.update_wrapper(_init, old_init)
    22:29:00.00 .......... _init = <function object.__init__>
    22:29:00.00   16 |     cls.__init__ = use_kwargs(cls._methods)(_init)
    22:29:00.01   17 |     if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))
    22:29:00.01   18 |     return cls
    22:29:00.01 <<< Return value from _funcs_kwargs: <class 'fastcore.meta.T'>
    22:29:00.01 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
    22:29:00.01 .......... self = <fastcore.meta.T object>
    22:29:00.01 .......... args = ()
    22:29:00.01 .......... kwargs = {'b': <function _f>}
    22:29:00.01 .......... len(kwargs) = 1
    22:29:00.01 .......... as_method = True
    22:29:00.01 .......... cls = <class 'fastcore.meta.T'>
    22:29:00.01 .......... old_init = <slot wrapper '__init__' of 'object' objects>
    22:29:00.01    7 |     def _init(self, *args, **kwargs):
    22:29:00.01    8 |         for k in cls._methods:
    22:29:00.01 .............. k = 'b'
    22:29:00.01    9 |             arg = kwargs.pop(k,None)
    22:29:00.01 .................. kwargs = {}
    22:29:00.01 .................. arg = <function _f>
    22:29:00.01   10 |             if arg is not None:
    22:29:00.01   11 |                 if as_method: arg = method(arg)
    22:29:00.01 ...... arg = <bound method int._f of 1>
    22:29:00.01   12 |                 if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
    22:29:00.01 ...... arg = <bound method T._f of <fastcore.meta.T object>>
    22:29:00.01   13 |                 setattr(self, k, arg)
    22:29:00.01    8 |         for k in cls._methods:
    22:29:00.01   14 |         old_init(self, *args, **kwargs)
    22:29:00.01 <<< Return value from _funcs_kwargs.<locals>._init: None
    22:29:00.01 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
    22:29:00.01 .......... self = <fastcore.meta.T2 object>
    22:29:00.01 .......... args = ()
    22:29:00.01 .......... kwargs = {'b': <function _f>}
    22:29:00.01 .......... len(kwargs) = 1
    22:29:00.01 .......... as_method = True
    22:29:00.01 .......... cls = <class 'fastcore.meta.T'>
    22:29:00.01 .......... old_init = <slot wrapper '__init__' of 'object' objects>
    22:29:00.01    7 |     def _init(self, *args, **kwargs):
    22:29:00.01    8 |         for k in cls._methods:
    22:29:00.01 .............. k = 'b'
    22:29:00.01    9 |             arg = kwargs.pop(k,None)
    22:29:00.01 .................. kwargs = {}
    22:29:00.01 .................. arg = <function _f>
    22:29:00.01   10 |             if arg is not None:
    22:29:00.01   11 |                 if as_method: arg = method(arg)
    22:29:00.01 ...... arg = <bound method int._f of 1>
    22:29:00.01   12 |                 if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
    22:29:00.01 ...... arg = <bound method T2._f of <fastcore.meta.T2 object>>
    22:29:00.01   13 |                 setattr(self, k, arg)
    22:29:00.01    8 |         for k in cls._methods:
    22:29:00.01   14 |         old_init(self, *args, **kwargs)
    22:29:00.01 <<< Return value from _funcs_kwargs.<locals>._init: None


    =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    



```
fdb_.debug()
```

    _funcs_kwargs's dbsrc code: ==============
    import snoop
    @snoop
    def _funcs_kwargs(cls, as_method):
        old_init = cls.__init__
        import snoop
        @snoop
        def _init(self, *args, **kwargs):
            for k in cls._methods:
                arg = kwargs.pop(k,None)
                if arg is not None:
                    if as_method: arg = method(arg)
                    if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
                    setattr(self, k, arg)
            old_init(self, *args, **kwargs)
        functools.update_wrapper(_init, old_init)
        cls.__init__ = use_kwargs(cls._methods)(_init)
        if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))
        return cls
    
    
    
    _funcs_kwargs's example processed with dbsrc: ===============
    
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
#| column: screen
fdb.print()
```

    =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    
    def funcs_kwargs(as_method=False):========================================================(0)       
        "Replace methods in `cls._methods` with those from `kwargs`"==========================(1) # [93;1mhow funcs_kwargs works[0m; [93;1mit is a wrapper around _funcs_kwargs[0m; [91;1mit offers two ways of running _funcs_kwargs[0m; [37;1mthe first, default way, is to add a func to a class without using self[0m; [91;1msecond way is to add func to class enabling self use;[0m; 
        if callable(as_method): return _funcs_kwargs(as_method, False)========================(2) # [37;1mhow to check whether an object is callable[0m; [93;1mhow to return a result of running a func[0m; [37;1m[0m; 
        return partial(_funcs_kwargs, as_method=as_method)====================================(3) # [92;1mhow to custom the params of `_funcs_kwargs` for a particular use with partial[0m; 
                                                                                                                                                            (4)



```
#| column: screen
fdb_.print()
```

    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ===============================================================     on line [91;1m12[0m     ===============================================================
         with example [91;1m
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
    
    [0m     
    
    def _funcs_kwargs(cls, as_method):========================================================(0) # [93;1mhow does _funcs_kwargs work: _funcs_kwargs is a decorator[0m; [35;1mit helps class e.g., T to add more methods[0m; [93;1mI need to give the method a name, and put the name e.g., 'b' inside a list called _methods=['b'] inside class T[0m; [36;1mthen after writing a func e.g., _new_func, I can add it by T(b = _new_func)[0m; [36;1mif I want the func added to class to use self, I shall write @funcs_kwargs(as_method=True)[0m; 
        old_init = cls.__init__===============================================================(1)       
        def _init(self, *args, **kwargs):=====================================================(2) # [37;1mhow to define a method which can use self and accept any parameters[0m; 
            for k in cls._methods:============================================================(3) # [35;1mhow to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead[0m; [91;1m[0m; 
                arg = kwargs.pop(k,None)======================================================(4)       
                if arg is not None:===========================================================(5)       
                    if as_method: arg = method(arg)===========================================(6) # [36;1mhow to turn a func into a method[0m; 
                    if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=======(7) # [93;1mhow to give a method a different instance, like self[0m; 
                    setattr(self, k, arg)=====================================================(8) # [36;1mhow to add a method to a class as an attribute[0m; 
            old_init(self, *args, **kwargs)===================================================(9)       
        functools.update_wrapper(_init, old_init)=============================================(10) # [34;1mhow to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself[0m; 
        cls.__init__ = use_kwargs(cls._methods)(_init)========================================(11) # [34;1mhow to add a list of names with None as default value to function `_init` to repalce its kwargs param[0m; 
        if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__)) # [35;1mhow to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`[0m;  (12)
        return cls============================================================================(13)      
                                                                                                                                                            (14)



```

```

### snoop only '_funcs_kwargs' by breaking up 'funcs_kwargs'

I could do it this way, but I have to change more codes which leads to more efforts and potential errors. So not recommended.


```
fdb = Fastdb(funcs_kwargs)
fdb_ = Fastdb(_funcs_kwargs)
```


```
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


```
fdb_.snoop(watch=["cls.__init__"], deco=True) # can't do this expression
```

    22:29:00.04 >>> Call to _funcs_kwargs in File "/tmp/_funcs_kwargs.py", line 3
    22:29:00.04 ...... cls = <class 'fastcore.meta.T'>
    22:29:00.04 ...... as_method = False
    22:29:00.04    3 | def _funcs_kwargs(cls, as_method):
    22:29:00.04    4 |     old_init = cls.__init__
    22:29:00.05 .......... old_init = <function T.__init__>
    22:29:00.05    5 |     import snoop
    22:29:00.05 .......... snoop = <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>
    22:29:00.05    6 |     @snoop
    22:29:00.05    7 |     def _init(self, *args, **kwargs):
    22:29:00.05 .......... _init = <function _funcs_kwargs.<locals>._init>
    22:29:00.05   15 |     functools.update_wrapper(_init, old_init)
    22:29:00.05 .......... _init = <function T.__init__>
    22:29:00.05   16 |     cls.__init__ = use_kwargs(cls._methods)(_init)
    22:29:00.05   17 |     if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))
    22:29:00.05   18 |     return cls
    22:29:00.05 <<< Return value from _funcs_kwargs: <class 'fastcore.meta.T'>
    22:29:00.05 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
    22:29:00.05 .......... self = <fastcore.meta.T object>
    22:29:00.05 .......... args = ()
    22:29:00.05 .......... kwargs = {'b': <function _new_func>}
    22:29:00.05 .......... len(kwargs) = 1
    22:29:00.05 .......... as_method = False
    22:29:00.05 .......... cls = <class 'fastcore.meta.T'>
    22:29:00.05 .......... old_init = <function T.__init__>
    22:29:00.05    7 |     def _init(self, *args, **kwargs):
    22:29:00.05    8 |         for k in cls._methods:
    22:29:00.05 .............. k = 'b'
    22:29:00.05    9 |             arg = kwargs.pop(k,None)
    22:29:00.05 .................. kwargs = {}
    22:29:00.05 .................. arg = <function _new_func>
    22:29:00.05   10 |             if arg is not None:
    22:29:00.05   11 |                 if as_method: arg = method(arg)
    22:29:00.05   12 |                 if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
    22:29:00.05   13 |                 setattr(self, k, arg)
    22:29:00.05    8 |         for k in cls._methods:
    22:29:00.05   14 |         old_init(self, *args, **kwargs)
    22:29:00.05 <<< Return value from _funcs_kwargs.<locals>._init: None


    ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class T:
        _methods=['b'] # allows you to add method b upon instantiation
        def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
        def a(self): return 1
        def b(self): return 2
        
    def _new_func(): return 5
    
    t = _funcs_kwargs(T, False)(b = _new_func)
    test_eq(t.b(), 5)
    [0m     
    



```

```
