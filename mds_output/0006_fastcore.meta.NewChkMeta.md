# 06_NewChkMeta

## Import and Initalization


```
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```


<style>.container { width:100% !important; }</style>



```
g = locals()
fdb = Fastdb(NewChkMeta, outloc = g)
```


```
fdb.print()
```

    ========================================================     Investigating [91;1mNewChkMeta[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    class NewChkMeta(FixSigMeta):=============================================================(0)       
        "Metaclass to avoid recreating object passed to constructor"==========================(1)       
        def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
            if not args and not kwargs and x is not None and isinstance(x,cls): return x======(3)       
            res = super().__call__(*((x,) + args), **kwargs)==================================(4)       
            return res========================================================================(5)       
                                                                                                                                                            (6)


## Official docs
The official docs (at first, it does not make sense to me)

`NewChkMeta` is used when an object of the same type is the first argument to your class's constructor (i.e. the __init__ function), and you would rather it not create a new object but point to the same exact object.

This is used in L, for example, to avoid creating a new object when the object is already of type L. This allows the users to defenisvely instantiate an L object and just return a reference to the same object if it already happens to be of type L.

For example, the below class _T optionally accepts an object o as its first argument. A new object is returned upon instantiation per usual:


```
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


```
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


```

```

## Prepare Example


```
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


```
inspect_class(NewChkMeta)
```

    class NewChkMeta(FixSigMeta):
        "Metaclass to avoid recreating object passed to constructor"
        def __call__(cls, x=None, *args, **kwargs):
            if not args and not kwargs and x is not None and isinstance(x,cls): return x
            res = super().__call__(*((x,) + args), **kwargs)
            return res
    
    
    is NewChkMeta a metaclass: True
    is NewChkMeta created by a metaclass: False
    NewChkMeta is created by <class 'type'>
    NewChkMeta.__new__ is object.__new__: False
    NewChkMeta.__new__ is type.__new__: False
    NewChkMeta.__new__: <function FixSigMeta.__new__>
    NewChkMeta.__init__ is object.__init__: False
    NewChkMeta.__init__ is type.__init__: True
    NewChkMeta.__init__: <slot wrapper '__init__' of 'type' objects>
    NewChkMeta.__call__ is object.__call__: False
    NewChkMeta.__call__ is type.__call__: False
    NewChkMeta.__call__: <function NewChkMeta.__call__>
    NewChkMeta.__class__: <class 'type'>
    NewChkMeta.__bases__: (<class 'fastcore.meta.FixSigMeta'>,)
    NewChkMeta.__mro__: (<class 'fastcore.meta.NewChkMeta'>, <class 'fastcore.meta.FixSigMeta'>, <class 'type'>, <class 'object'>)
    
    NewChkMeta's function members are:
    {'__call__': <function NewChkMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    NewChkMeta's method members are:
    {}
    
    NewChkMeta's class members are:
    {'__base__': <class 'fastcore.meta.FixSigMeta'>, '__class__': <class 'type'>}
    
    NewChkMeta's namespace are:
    mappingproxy({'__call__': <function NewChkMeta.__call__>,
                  '__doc__': 'Metaclass to avoid recreating object passed to '
                             'constructor',
                  '__module__': 'fastcore.meta'})



```
inspect_class(_T)
```

    
    is _T a metaclass: False
    is _T created by a metaclass: True
    _T is created by metaclass <class 'fastcore.meta.NewChkMeta'>
    _T.__new__ is object.__new__: True
    _T.__new__ is type.__new__: False
    _T.__new__: <built-in method __new__ of type object>
    _T.__init__ is object.__init__: False
    _T.__init__ is type.__init__: False
    _T.__init__: <function _T.__init__>
    _T.__call__ is object.__call__: False
    _T.__call__ is type.__call__: False
    _T.__call__: <bound method NewChkMeta.__call__ of <class '__main__._T'>>
    _T.__class__: <class 'fastcore.meta.NewChkMeta'>
    _T.__bases__: (<class 'object'>,)
    _T.__mro__: (<class '__main__._T'>, <class 'object'>)
    
    _T's metaclass <class 'fastcore.meta.NewChkMeta'>'s function members are:
    {'__call__': <function NewChkMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    _T's function members are:
    {'__init__': <function _T.__init__>}
    
    _T's method members are:
    {}
    
    _T's class members are:
    {'__class__': <class 'fastcore.meta.NewChkMeta'>}
    
    _T's namespace are:
    mappingproxy({'__dict__': <attribute '__dict__' of '_T' objects>,
                  '__doc__': 'Testing with metaclass NewChkMeta',
                  '__init__': <function _T.__init__>,
                  '__module__': '__main__',
                  '__signature__': <Signature (o=None, b=1)>,
                  '__weakref__': <attribute '__weakref__' of '_T' objects>})


## Snoop


```
fdb.snoop()
```

    21:38:05.32 >>> Call to NewChkMeta.__call__ in File "/tmp/NewChkMeta.py", line 5
    21:38:05.32 .......... cls = <class 'fastcore.meta._T'>
    21:38:05.32 .......... x = 3
    21:38:05.32 .......... args = ()
    21:38:05.32 .......... kwargs = {}
    21:38:05.32 .......... __class__ = <class 'fastcore.meta.NewChkMeta'>
    21:38:05.32    5 |     def __call__(cls, x=None, *args, **kwargs):
    21:38:05.32    6 |         if not args and not kwargs and x is not None and isinstance(x,cls): return x
    21:38:05.32    7 |         res = super().__call__(*((x,) + args), **kwargs)
    21:38:05.32 .............. res = <fastcore.meta._T object>
    21:38:05.32    8 |         return res
    21:38:05.32 <<< Return value from NewChkMeta.__call__: <fastcore.meta._T object>
    21:38:05.32 >>> Call to NewChkMeta.__call__ in File "/tmp/NewChkMeta.py", line 5
    21:38:05.32 .......... cls = <class 'fastcore.meta._T'>
    21:38:05.32 .......... x = <fastcore.meta._T object>
    21:38:05.32 .......... args = ()
    21:38:05.32 .......... kwargs = {}
    21:38:05.32 .......... __class__ = <class 'fastcore.meta.NewChkMeta'>
    21:38:05.32    5 |     def __call__(cls, x=None, *args, **kwargs):
    21:38:05.32    6 |         if not args and not kwargs and x is not None and isinstance(x,cls): return x
    21:38:05.32 <<< Return value from NewChkMeta.__call__: <fastcore.meta._T object>
    21:38:05.32 >>> Call to NewChkMeta.__call__ in File "/tmp/NewChkMeta.py", line 5
    21:38:05.32 .......... cls = <class 'fastcore.meta._T'>
    21:38:05.32 .......... x = <fastcore.meta._T object>
    21:38:05.32 .......... args = ()
    21:38:05.32 .......... kwargs = {'b': 1}
    21:38:05.32 .......... len(kwargs) = 1
    21:38:05.32 .......... __class__ = <class 'fastcore.meta.NewChkMeta'>
    21:38:05.32    5 |     def __call__(cls, x=None, *args, **kwargs):
    21:38:05.32    6 |         if not args and not kwargs and x is not None and isinstance(x,cls): return x
    21:38:05.32    7 |         res = super().__call__(*((x,) + args), **kwargs)
    21:38:05.32 .............. res = <fastcore.meta._T object>
    21:38:05.32    8 |         return res
    21:38:05.32 <<< Return value from NewChkMeta.__call__: <fastcore.meta._T object>
    21:38:05.32 >>> Call to NewChkMeta.__call__ in File "/tmp/NewChkMeta.py", line 5
    21:38:05.32 .......... cls = <class 'fastcore.meta._T'>
    21:38:05.32 .......... x = <fastcore.meta._T object>
    21:38:05.32 .......... args = ()
    21:38:05.32 .......... kwargs = {}
    21:38:05.32 .......... __class__ = <class 'fastcore.meta.NewChkMeta'>
    21:38:05.32    5 |     def __call__(cls, x=None, *args, **kwargs):
    21:38:05.32    6 |         if not args and not kwargs and x is not None and isinstance(x,cls): return x
    21:38:05.32 <<< Return value from NewChkMeta.__call__: <fastcore.meta._T object>


    ========================================================     Investigating [91;1mNewChkMeta[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    


## Document


```
fdb.docsrc(1, "NewChkMeta is a metaclass inherited from FixSigMea; it makes its own __call__; \
when its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, \
and x is an object of the instance class, then return x; otherwise, create and return a new object created by \
the instance class's super class' __call__ method with x as param; In other words, t = _T(3) will create a new obj; \
_T(t) will return t; _T(t, 1) or _T(t, b=1) will also return a new obj")
fdb.docsrc(2, "how to create a __call__ method with param cls, x, *args, **kwargs;")
fdb.docsrc(3, "how to express no args and no kwargs and x is an instance of cls?")
fdb.docsrc(4, "how to call __call__ of super class with x and consider all possible situations of args and kwargs")
```

    ========================================================     Investigating [91;1mNewChkMeta[0m     ========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    class NewChkMeta(FixSigMeta):                                                                                                                           (0)
        "Metaclass to avoid recreating object passed to constructor"========================================================================================(1)
    [91;1mNewChkMeta is a metaclass inherited from FixSigMea; it makes its own __call__; when its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x; otherwise, create and return a new object created by the instance class's super class' __call__ method with x as param; In other words, t = _T(3) will create a new obj; _T(t) will return t; _T(t, 1) or _T(t, b=1) will also return a new obj[0m
        def __call__(cls, x=None, *args, **kwargs):                                                                                                         (2)
            if not args and not kwargs and x is not None and isinstance(x,cls): return x                                                                    (3)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class NewChkMeta(FixSigMeta):=============================================================(0)       
        "Metaclass to avoid recreating object passed to constructor"==========================(1) # [34;1mNewChkMeta is a metaclass inherited from FixSigMea[0m; [91;1mit makes its own __call__[0m; [34;1mwhen its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x[0m; [93;1motherwise, create and return a new object created by the instance class's super class' __call__ method with x as param[0m; [91;1mIn other words, t = _T(3) will create a new obj[0m; [93;1m_T(t) will return t[0m; [35;1m_T(t, 1) or _T(t, b=1) will also return a new obj[0m; 
        def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
            if not args and not kwargs and x is not None and isinstance(x,cls): return x======(3)       
            res = super().__call__(*((x,) + args), **kwargs)==================================(4)       
            return res========================================================================(5)       
                                                                                                                                                            (6)
                                                                                                                                         part No.1 out of 1 parts
    
    ========================================================     Investigating [91;1mNewChkMeta[0m     ========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    class NewChkMeta(FixSigMeta):                                                                                                                           (0)
        "Metaclass to avoid recreating object passed to constructor"                                                                                        (1)
        def __call__(cls, x=None, *args, **kwargs):=========================================================================================================(2)
                                                                                              [91;1mhow to create a __call__ method with param cls, x, *args, **kwargs;[0m
            if not args and not kwargs and x is not None and isinstance(x,cls): return x                                                                    (3)
            res = super().__call__(*((x,) + args), **kwargs)                                                                                                (4)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class NewChkMeta(FixSigMeta):=============================================================(0)       
        "Metaclass to avoid recreating object passed to constructor"==========================(1) # [36;1mNewChkMeta is a metaclass inherited from FixSigMea[0m; [92;1mit makes its own __call__[0m; [36;1mwhen its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x[0m; [91;1motherwise, create and return a new object created by the instance class's super class' __call__ method with x as param[0m; [36;1mIn other words, t = _T(3) will create a new obj[0m; [37;1m_T(t) will return t[0m; [37;1m_T(t, 1) or _T(t, b=1) will also return a new obj[0m; 
        def __call__(cls, x=None, *args, **kwargs):===========================================(2) # [34;1mhow to create a __call__ method with param cls, x, *args, **kwargs;[0m; 
            if not args and not kwargs and x is not None and isinstance(x,cls): return x======(3)       
            res = super().__call__(*((x,) + args), **kwargs)==================================(4)       
            return res========================================================================(5)       
                                                                                                                                                            (6)
                                                                                                                                         part No.1 out of 1 parts
    
    ========================================================     Investigating [91;1mNewChkMeta[0m     ========================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        "Metaclass to avoid recreating object passed to constructor"                                                                                        (1)
        def __call__(cls, x=None, *args, **kwargs):                                                                                                         (2)
            if not args and not kwargs and x is not None and isinstance(x,cls): return x====================================================================(3)
                                                                                                [91;1mhow to express no args and no kwargs and x is an instance of cls?[0m
            res = super().__call__(*((x,) + args), **kwargs)                                                                                                (4)
            return res                                                                                                                                      (5)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class NewChkMeta(FixSigMeta):=============================================================(0)       
        "Metaclass to avoid recreating object passed to constructor"==========================(1) # [36;1mNewChkMeta is a metaclass inherited from FixSigMea[0m; [37;1mit makes its own __call__[0m; [37;1mwhen its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x[0m; [35;1motherwise, create and return a new object created by the instance class's super class' __call__ method with x as param[0m; [34;1mIn other words, t = _T(3) will create a new obj[0m; [91;1m_T(t) will return t[0m; [92;1m_T(t, 1) or _T(t, b=1) will also return a new obj[0m; 
        def __call__(cls, x=None, *args, **kwargs):===========================================(2) # [34;1mhow to create a __call__ method with param cls, x, *args, **kwargs;[0m; 
            if not args and not kwargs and x is not None and isinstance(x,cls): return x======(3) # [93;1mhow to express no args and no kwargs and x is an instance of cls?[0m; 
            res = super().__call__(*((x,) + args), **kwargs)==================================(4)       
            return res========================================================================(5)       
                                                                                                                                                            (6)
                                                                                                                                         part No.1 out of 1 parts
    
    ========================================================     Investigating [91;1mNewChkMeta[0m     ========================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        def __call__(cls, x=None, *args, **kwargs):                                                                                                         (2)
            if not args and not kwargs and x is not None and isinstance(x,cls): return x                                                                    (3)
            res = super().__call__(*((x,) + args), **kwargs)================================================================================================(4)
                                                               [91;1mhow to call __call__ of super class with x and consider all possible situations of args and kwargs[0m
            return res                                                                                                                                      (5)
                                                                                                                                                            (6)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class NewChkMeta(FixSigMeta):=============================================================(0)       
        "Metaclass to avoid recreating object passed to constructor"==========================(1) # [36;1mNewChkMeta is a metaclass inherited from FixSigMea[0m; [93;1mit makes its own __call__[0m; [34;1mwhen its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x[0m; [93;1motherwise, create and return a new object created by the instance class's super class' __call__ method with x as param[0m; [93;1mIn other words, t = _T(3) will create a new obj[0m; [91;1m_T(t) will return t[0m; [92;1m_T(t, 1) or _T(t, b=1) will also return a new obj[0m; 
        def __call__(cls, x=None, *args, **kwargs):===========================================(2) # [91;1mhow to create a __call__ method with param cls, x, *args, **kwargs;[0m; 
            if not args and not kwargs and x is not None and isinstance(x,cls): return x======(3) # [92;1mhow to express no args and no kwargs and x is an instance of cls?[0m; 
            res = super().__call__(*((x,) + args), **kwargs)==================================(4) # [91;1mhow to call __call__ of super class with x and consider all possible situations of args and kwargs[0m; 
            return res========================================================================(5)       
                                                                                                                                                            (6)
                                                                                                                                         part No.1 out of 1 parts
    



```
fdb.print()
```

    ========================================================     Investigating [91;1mNewChkMeta[0m     ========================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    class NewChkMeta(FixSigMeta):=============================================================(0)       
        "Metaclass to avoid recreating object passed to constructor"==========================(1) # [36;1mNewChkMeta is a metaclass inherited from FixSigMea[0m; [93;1mit makes its own __call__[0m; [93;1mwhen its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x[0m; [35;1motherwise, create and return a new object created by the instance class's super class' __call__ method with x as param[0m; [93;1mIn other words, t = _T(3) will create a new obj[0m; [34;1m_T(t) will return t[0m; [34;1m_T(t, 1) or _T(t, b=1) will also return a new obj[0m; 
        def __call__(cls, x=None, *args, **kwargs):===========================================(2) # [37;1mhow to create a __call__ method with param cls, x, *args, **kwargs;[0m; 
            if not args and not kwargs and x is not None and isinstance(x,cls): return x======(3) # [37;1mhow to express no args and no kwargs and x is an instance of cls?[0m; 
            res = super().__call__(*((x,) + args), **kwargs)==================================(4) # [34;1mhow to call __call__ of super class with x and consider all possible situations of args and kwargs[0m; 
            return res========================================================================(5)       
                                                                                                                                                            (6)



```

```
