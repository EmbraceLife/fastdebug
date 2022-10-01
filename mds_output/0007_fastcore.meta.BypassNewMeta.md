# 07_BypassNewMeta


```
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```


<style>.container { width:100% !important; }</style>


## Reading official docs

BypassNewMeta
> BypassNewMeta (name, bases, dict)     

Metaclass: casts x to this class if it's of type cls._bypass_type

BypassNewMeta is identical to NewChkMeta, except for checking for a class as the same type, we instead check for a class of type specified in attribute _bypass_type.

In NewChkMeta, objects of the same type passed to the constructor (without arguments) would result into a new variable referencing the same object. 

However, with BypassNewMeta this only occurs if the type matches the `_bypass_type` of the class you are defining:


```
# class _TestA: pass
# class _TestB: pass

# class _T(_TestA, metaclass=BypassNewMeta):
#     _bypass_type=_TestB
#     def __init__(self,x): self.x=x
```

In the below example, t does not refer to t2 because t is of type _TestA while _T._bypass_type is of type TestB:


```
# t = _TestA()
# t2 = _T(t)
# assert t is not t2
```

However, if t is set to _TestB to match _T._bypass_type, then both t and t2 will refer to the same object.


```
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


```
# t = _TestB(); t
# isinstance(t, _TestB)
# id(_TestB)
# # t2 = _T(t)
# # t, t2
```

## Inspecting class


```
inspect_class(BypassNewMeta)
```

    class BypassNewMeta(FixSigMeta):
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"
        def __call__(cls, x=None, *args, **kwargs):
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):
                x = super().__call__(*((x,)+args), **kwargs)
            if cls!=x.__class__: x.__class__ = cls
            return x
    
    
    is BypassNewMeta a metaclass: True
    is BypassNewMeta created by a metaclass: False
    BypassNewMeta is created by <class 'type'>
    BypassNewMeta.__new__ is object.__new__: False
    BypassNewMeta.__new__ is type.__new__: False
    BypassNewMeta.__new__: <function FixSigMeta.__new__>
    BypassNewMeta.__init__ is object.__init__: False
    BypassNewMeta.__init__ is type.__init__: True
    BypassNewMeta.__init__: <slot wrapper '__init__' of 'type' objects>
    BypassNewMeta.__call__ is object.__call__: False
    BypassNewMeta.__call__ is type.__call__: False
    BypassNewMeta.__call__: <function BypassNewMeta.__call__>
    BypassNewMeta.__class__: <class 'type'>
    BypassNewMeta.__bases__: (<class 'fastcore.meta.FixSigMeta'>,)
    BypassNewMeta.__mro__: (<class 'fastcore.meta.BypassNewMeta'>, <class 'fastcore.meta.FixSigMeta'>, <class 'type'>, <class 'object'>)
    
    BypassNewMeta's function members are:
    {'__call__': <function BypassNewMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    BypassNewMeta's method members are:
    {}
    
    BypassNewMeta's class members are:
    {'__base__': <class 'fastcore.meta.FixSigMeta'>, '__class__': <class 'type'>}
    
    BypassNewMeta's namespace are:
    mappingproxy({'__call__': <function BypassNewMeta.__call__>,
                  '__doc__': "Metaclass: casts `x` to this class if it's of type "
                             '`cls._bypass_type`',
                  '__module__': 'fastcore.meta'})


## Initiating with examples


```
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


```
fdb.snoop()
```

    23:28:43.69 >>> Call to BypassNewMeta.__call__ in File "/tmp/BypassNewMeta.py", line 5
    23:28:43.69 .......... cls = <class '__main__._T'>
    23:28:43.69 .......... x = <__main__._TestB object>
    23:28:43.69 .......... args = ()
    23:28:43.69 .......... kwargs = {}
    23:28:43.69 .......... __class__ = <class 'fastcore.meta.BypassNewMeta'>
    23:28:43.69    5 |     def __call__(cls, x=None, *args, **kwargs):
    23:28:43.69    6 |         if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)
    23:28:43.69    7 |         elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):
    23:28:43.69    9 |         if cls!=x.__class__: x.__class__ = cls
    23:28:43.69 ...... x = <__main__._T object>
    23:28:43.69   10 |         return x
    23:28:43.69 <<< Return value from BypassNewMeta.__call__: <__main__._T object>


    ======================================================     Investigating [91;1mBypassNewMeta[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    



```
fdb.debug()
```

    BypassNewMeta's dbsrc code: ==============
    class BypassNewMeta(FixSigMeta):
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"
        import snoop
        @snoop
        def __call__(cls, x=None, *args, **kwargs):
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):
                x = super().__call__(*((x,)+args), **kwargs)
            if cls!=x.__class__: x.__class__ = cls
            return x
    
    
    
    BypassNewMeta's example processed with dbsrc: ===============
    
    class _TestA: pass
    class _TestB: pass
    
    class _T(_TestA, metaclass=self.dbsrc):
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
    
    


## Document


```
fdb.docsrc(3, "If the instance class like _T has attr '_new_meta', then run it with param x;", "x", \
           "cls", "getattr(cls,'_bypass_type',object)", "isinstance(x, _TestB)", "isinstance(x,getattr(cls,'_bypass_type',object))")
fdb.docsrc(4, "when x is not an instance of _T's _bypass_type; or when a positional param is given; or when a keyword arg is given; \
let's run _T's super's __call__ function with x as param; and assign the result to x")
fdb.docsrc(6, "If x.__class__ is not cls or _T, then make it so")
fdb.docsrc(1, "BypassNewMeta allows its instance class e.g., _T to choose a specific class e.g., _TestB and \
change `__class__` of an object e.g., t of _TestB to _T without creating a new object")
```

    ======================================================     Investigating [91;1mBypassNewMeta[0m     =======================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"                                                                             (1)
        def __call__(cls, x=None, *args, **kwargs):                                                                                                         (2)
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)=============================================================================(3)
                                                                                    [91;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):                                                          (4)
                x = super().__call__(*((x,)+args), **kwargs)                                                                                                (5)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                                                                 x => x : <__main__._TestB object>
    
    
                                                                                                                               cls => cls : <class '__main__._T'>
    
    
                                                             getattr(cls,'_bypass_type',object) => getattr(cls,'_bypass_type',object) : <class '__main__._TestB'>
    
    
                                                                                                            isinstance(x, _TestB) => isinstance(x, _TestB) : True
    
    
                                                      isinstance(x,getattr(cls,'_bypass_type',object)) => isinstance(x,getattr(cls,'_bypass_type',object)) : True
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class BypassNewMeta(FixSigMeta):==========================================================(0)       
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"===============(1)       
        def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)===============(3) # [91;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m; 
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):                                                          (4)
                x = super().__call__(*((x,)+args), **kwargs)==================================(5)       
            if cls!=x.__class__: x.__class__ = cls============================================(6)       
            return x==========================================================================(7)       
                                                                                                                                                            (8)
                                                                                                                                         part No.1 out of 1 parts
    
    ======================================================     Investigating [91;1mBypassNewMeta[0m     =======================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        def __call__(cls, x=None, *args, **kwargs):                                                                                                         (2)
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)                                                                             (3)
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):==========================================================(4)
    [91;1mwhen x is not an instance of _T's _bypass_type; or when a positional param is given; or when a keyword arg is given; let's run _T's super's __call__ function with x as param; and assign the result to x[0m
                x = super().__call__(*((x,)+args), **kwargs)                                                                                                (5)
            if cls!=x.__class__: x.__class__ = cls                                                                                                          (6)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class BypassNewMeta(FixSigMeta):==========================================================(0)       
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"===============(1)       
        def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)===============(3) # [36;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m; 
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs): # [91;1mwhen x is not an instance of _T's _bypass_type[0m; [93;1mor when a positional param is given[0m; [91;1mor when a keyword arg is given[0m; [92;1mlet's run _T's super's __call__ function with x as param[0m; [92;1mand assign the result to x[0m;  (4)
                x = super().__call__(*((x,)+args), **kwargs)==================================(5)       
            if cls!=x.__class__: x.__class__ = cls============================================(6)       
            return x==========================================================================(7)       
                                                                                                                                                            (8)
                                                                                                                                         part No.1 out of 1 parts
    
    ======================================================     Investigating [91;1mBypassNewMeta[0m     =======================================================
    ===============================================================     on line [91;1m6[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):                                                          (4)
                x = super().__call__(*((x,)+args), **kwargs)                                                                                                (5)
            if cls!=x.__class__: x.__class__ = cls==========================================================================================================(6)
                                                                                                                 [91;1mIf x.__class__ is not cls or _T, then make it so[0m
            return x                                                                                                                                        (7)
                                                                                                                                                            (8)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class BypassNewMeta(FixSigMeta):==========================================================(0)       
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"===============(1)       
        def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)===============(3) # [36;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m; 
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs): # [93;1mwhen x is not an instance of _T's _bypass_type[0m; [34;1mor when a positional param is given[0m; [91;1mor when a keyword arg is given[0m; [35;1mlet's run _T's super's __call__ function with x as param[0m; [34;1mand assign the result to x[0m;  (4)
                x = super().__call__(*((x,)+args), **kwargs)==================================(5)       
            if cls!=x.__class__: x.__class__ = cls============================================(6) # [93;1mIf x.__class__ is not cls or _T, then make it so[0m; 
            return x==========================================================================(7)       
                                                                                                                                                            (8)
                                                                                                                                         part No.1 out of 1 parts
    
    ======================================================     Investigating [91;1mBypassNewMeta[0m     =======================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
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
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    class BypassNewMeta(FixSigMeta):                                                                                                                        (0)
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"=============================================================================(1)
    [91;1mBypassNewMeta allows its instance class e.g., _T to choose a specific class e.g., _TestB and change `__class__` of an object e.g., t of _TestB to _T without creating a new object[0m
        def __call__(cls, x=None, *args, **kwargs):                                                                                                         (2)
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)                                                                             (3)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class BypassNewMeta(FixSigMeta):==========================================================(0)       
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"===============(1) # [91;1mBypassNewMeta allows its instance class e.g., _T to choose a specific class e.g., _TestB and change `__class__` of an object e.g., t of _TestB to _T without creating a new object[0m; 
        def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)===============(3) # [34;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m; 
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs): # [35;1mwhen x is not an instance of _T's _bypass_type[0m; [92;1mor when a positional param is given[0m; [93;1mor when a keyword arg is given[0m; [36;1mlet's run _T's super's __call__ function with x as param[0m; [92;1mand assign the result to x[0m;  (4)
                x = super().__call__(*((x,)+args), **kwargs)==================================(5)       
            if cls!=x.__class__: x.__class__ = cls============================================(6) # [36;1mIf x.__class__ is not cls or _T, then make it so[0m; 
            return x==========================================================================(7)       
                                                                                                                                                            (8)
                                                                                                                                         part No.1 out of 1 parts
    



```
fdb.snoop(['cls._bypass_type', "isinstance(x,getattr(cls,'_bypass_type',object))"])
```

    23:28:43.75 >>> Call to BypassNewMeta.__call__ in File "/tmp/BypassNewMeta.py", line 5
    23:28:43.75 .......... cls = <class '__main__._T'>
    23:28:43.75 .......... x = <__main__._TestB object>
    23:28:43.75 .......... args = ()
    23:28:43.75 .......... kwargs = {}
    23:28:43.75 .......... __class__ = <class 'fastcore.meta.BypassNewMeta'>
    23:28:43.75 .......... cls._bypass_type = <class '__main__._TestB'>
    23:28:43.75 .......... isinstance(x,getattr(cls,'_bypass_type',object)) = True
    23:28:43.75    5 |     def __call__(cls, x=None, *args, **kwargs):
    23:28:43.76    6 |         if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)
    23:28:43.76    7 |         elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):
    23:28:43.76    9 |         if cls!=x.__class__: x.__class__ = cls
    23:28:43.76 ...... x = <__main__._T object>
    23:28:43.76 ...... isinstance(x,getattr(cls,'_bypass_type',object)) = False
    23:28:43.76   10 |         return x
    23:28:43.76 <<< Return value from BypassNewMeta.__call__: <__main__._T object>


    ======================================================     Investigating [91;1mBypassNewMeta[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    



```
fdb.debug()
```

    BypassNewMeta's dbsrc code: ==============
    class BypassNewMeta(FixSigMeta):
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"
        import snoop
        @snoop(watch=("cls._bypass_type","isinstance(x,getattr(cls,'_bypass_type',object))"))
        def __call__(cls, x=None, *args, **kwargs):
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs):
                x = super().__call__(*((x,)+args), **kwargs)
            if cls!=x.__class__: x.__class__ = cls
            return x
    
    
    
    BypassNewMeta's example processed with dbsrc: ===============
    
    class _TestA: pass
    class _TestB: pass
    
    class _T(_TestA, metaclass=self.dbsrc):
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
    
    
    
    
    
    
    



```
fdb.print()
```

    ======================================================     Investigating [91;1mBypassNewMeta[0m     =======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
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
    [0m     
    
    class BypassNewMeta(FixSigMeta):==========================================================(0)       
        "Metaclass: casts `x` to this class if it's of type `cls._bypass_type`"===============(1) # [93;1mBypassNewMeta allows its instance class e.g., _T to choose a specific class e.g., _TestB and change `__class__` of an object e.g., t of _TestB to _T without creating a new object[0m; 
        def __call__(cls, x=None, *args, **kwargs):===========================================(2)       
            if hasattr(cls, '_new_meta'): x = cls._new_meta(x, *args, **kwargs)===============(3) # [91;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m; 
            elif not isinstance(x,getattr(cls,'_bypass_type',object)) or len(args) or len(kwargs): # [37;1mwhen x is not an instance of _T's _bypass_type[0m; [93;1mor when a positional param is given[0m; [36;1mor when a keyword arg is given[0m; [93;1mlet's run _T's super's __call__ function with x as param[0m; [35;1mand assign the result to x[0m;  (4)
                x = super().__call__(*((x,)+args), **kwargs)==================================(5)       
            if cls!=x.__class__: x.__class__ = cls============================================(6) # [36;1mIf x.__class__ is not cls or _T, then make it so[0m; 
            return x==========================================================================(7)       
                                                                                                                                                            (8)



```

```
