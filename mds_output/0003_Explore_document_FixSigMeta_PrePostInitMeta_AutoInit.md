# 03_FixSigMeta_PrePostInitMeta_AutoInit


```
from fastdebug.utils import *
from fastdebug.core import *
import inspect
```


<style>.container { width:100% !important; }</style>



```
from fastcore.meta import *
from fastcore.test import *
```

## Initialize fastdebug objects


```
g = locals() # g can update itself as more cells get run, like globals() in the __main__
fdbF = Fastdb(FixSigMeta, outloc=g)
fdbP = Fastdb(PrePostInitMeta, outloc=g)
fdbA = Fastdb(AutoInit, outloc=g)
```

## class FixSigMeta(type) vs class Foo(type)

FixSigMeta inherits `__init__`, and `__call__` from `type`, but writes its own `__new__`    
Foo inherits all three from `type`    
FixSigMeta is used to create class instance not object instance


```
fdbF.docsrc(1, "FixSigMeta inherits __init__, and __call__ from type; but writes its own __new__; Foo inherits all three from type; \
FixSigMeta is used to create class instance not object instance.")
# fdbF.print()
```

    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    [93;1mprint selected srcline with expands below[0m--------
    class FixSigMeta(type):                                                                                                                                 (0)
        "A metaclass that fixes the signature on classes that override `__new__`"===========================================================================(1)
    [91;1mFixSigMeta inherits __init__, and __call__ from type; but writes its own __new__; Foo inherits all three from type; FixSigMeta is used to create class instance not object instance.[0m
        def __new__(cls, name, bases, dict):                                                                                                                (2)
            res = super().__new__(cls, name, bases, dict)                                                                                                   (3)



```
print(inspect.getsource(FixSigMeta))
```

    class FixSigMeta(type):
        "A metaclass that fixes the signature on classes that override `__new__`"
        def __new__(cls, name, bases, dict):
            res = super().__new__(cls, name, bases, dict)
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))
            return res
    



```
inspect_class(FixSigMeta)
```

    class FixSigMeta(type):
        "A metaclass that fixes the signature on classes that override `__new__`"
        def __new__(cls, name, bases, dict):
            res = super().__new__(cls, name, bases, dict)
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))
            return res
    
    
    is FixSigMeta a metaclass: True
    is FixSigMeta created by a metaclass: False
    FixSigMeta is created by <class 'type'>
    FixSigMeta.__new__ is object.__new__: False
    FixSigMeta.__new__ is type.__new__: False
    FixSigMeta.__new__: <function FixSigMeta.__new__>
    FixSigMeta.__init__ is object.__init__: False
    FixSigMeta.__init__ is type.__init__: True
    FixSigMeta.__init__: <slot wrapper '__init__' of 'type' objects>
    FixSigMeta.__call__ is object.__call__: False
    FixSigMeta.__call__ is type.__call__: True
    FixSigMeta.__call__: <slot wrapper '__call__' of 'type' objects>
    FixSigMeta.__class__: <class 'type'>
    FixSigMeta.__bases__: (<class 'type'>,)
    FixSigMeta.__mro__: (<class 'fastcore.meta.FixSigMeta'>, <class 'type'>, <class 'object'>)
    
    FixSigMeta's function members are:
    {'__new__': <function FixSigMeta.__new__>}
    
    FixSigMeta's method members are:
    {}
    
    FixSigMeta's class members are:
    {'__base__': <class 'type'>, '__class__': <class 'type'>}
    
    FixSigMeta's namespace are:
    mappingproxy({'__doc__': 'A metaclass that fixes the signature on classes that '
                             'override `__new__`',
                  '__module__': 'fastcore.meta',
                  '__new__': <staticmethod object>})



```
class Foo(type): pass
inspect_class(Foo)
```

    
    is Foo a metaclass: True
    is Foo created by a metaclass: False
    Foo is created by <class 'type'>
    Foo.__new__ is object.__new__: False
    Foo.__new__ is type.__new__: True
    Foo.__new__: <built-in method __new__ of type object>
    Foo.__init__ is object.__init__: False
    Foo.__init__ is type.__init__: True
    Foo.__init__: <slot wrapper '__init__' of 'type' objects>
    Foo.__call__ is object.__call__: False
    Foo.__call__ is type.__call__: True
    Foo.__call__: <slot wrapper '__call__' of 'type' objects>
    Foo.__class__: <class 'type'>
    Foo.__bases__: (<class 'type'>,)
    Foo.__mro__: (<class '__main__.Foo'>, <class 'type'>, <class 'object'>)
    
    Foo's function members are:
    {}
    
    Foo's method members are:
    {}
    
    Foo's class members are:
    {'__base__': <class 'type'>, '__class__': <class 'type'>}
    
    Foo's namespace are:
    mappingproxy({'__module__': '__main__', '__doc__': None})


## class Foo()

When Foo inherit `__new__` and `__new__` from `object`    
but `__call__` of Foo and `__call__` of `object` maybe the same but different objects    
Foo is to create object instance not class instance


```
class Foo(): pass
inspect_class(Foo)
```

    
    is Foo a metaclass: False
    is Foo created by a metaclass: False
    Foo is created by <class 'type'>
    Foo.__new__ is object.__new__: True
    Foo.__new__ is type.__new__: False
    Foo.__new__: <built-in method __new__ of type object>
    Foo.__init__ is object.__init__: True
    Foo.__init__ is type.__init__: False
    Foo.__init__: <slot wrapper '__init__' of 'object' objects>
    Foo.__call__ is object.__call__: False
    Foo.__call__ is type.__call__: False
    Foo.__call__: <method-wrapper '__call__' of type object>
    Foo.__class__: <class 'type'>
    Foo.__bases__: (<class 'object'>,)
    Foo.__mro__: (<class '__main__.Foo'>, <class 'object'>)
    
    Foo's function members are:
    {}
    
    Foo's method members are:
    {}
    
    Foo's class members are:
    {'__class__': <class 'type'>}
    
    Foo's namespace are:
    mappingproxy({'__dict__': <attribute '__dict__' of 'Foo' objects>,
                  '__doc__': None,
                  '__module__': '__main__',
                  '__weakref__': <attribute '__weakref__' of 'Foo' objects>})


## class PrePostInitMeta(FixSigMeta)

`PrePostInitMeta` inherit `__new__` and `__init__` from `FixSigMeta` as a metaclass (a different type), not from `type`, nor from `object`    
`PrePostInitMeta` is itself a metaclass, which is used to create class instance not object instance     
`PrePostInitMeta` writes its own `__call__` which regulates how its class instance create and initialize object instance 


```
fdbP.docsrc(1, "PrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type); \
not from type, nor from object; PrePostInitMeta is itself a metaclass, which is used to create class instance not object instance; \
PrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance")
```

    =====================================================     Investigating [91;1mPrePostInitMeta[0m     ======================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    [93;1mprint selected srcline with expands below[0m--------
    class PrePostInitMeta(FixSigMeta):                                                                                                                      (0)
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"========================================================================(1)
    [91;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type); not from type, nor from object; PrePostInitMeta is itself a metaclass, which is used to create class instance not object instance; PrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m
        def __call__(cls, *args, **kwargs):                                                                                                                 (2)
            res = cls.__new__(cls)                                                                                                                          (3)



```
inspect_class(PrePostInitMeta)
```

    class PrePostInitMeta(FixSigMeta):
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"
        def __call__(cls, *args, **kwargs):
            res = cls.__new__(cls)
            if type(res)==cls:
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)
                res.__init__(*args,**kwargs)
                if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)
            return res
    
    
    is PrePostInitMeta a metaclass: True
    is PrePostInitMeta created by a metaclass: False
    PrePostInitMeta is created by <class 'type'>
    PrePostInitMeta.__new__ is object.__new__: False
    PrePostInitMeta.__new__ is type.__new__: False
    PrePostInitMeta.__new__: <function FixSigMeta.__new__>
    PrePostInitMeta.__init__ is object.__init__: False
    PrePostInitMeta.__init__ is type.__init__: True
    PrePostInitMeta.__init__: <slot wrapper '__init__' of 'type' objects>
    PrePostInitMeta.__call__ is object.__call__: False
    PrePostInitMeta.__call__ is type.__call__: False
    PrePostInitMeta.__call__: <function PrePostInitMeta.__call__>
    PrePostInitMeta.__class__: <class 'type'>
    PrePostInitMeta.__bases__: (<class 'fastcore.meta.FixSigMeta'>,)
    PrePostInitMeta.__mro__: (<class 'fastcore.meta.PrePostInitMeta'>, <class 'fastcore.meta.FixSigMeta'>, <class 'type'>, <class 'object'>)
    
    PrePostInitMeta's function members are:
    {'__call__': <function PrePostInitMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    PrePostInitMeta's method members are:
    {}
    
    PrePostInitMeta's class members are:
    {'__base__': <class 'fastcore.meta.FixSigMeta'>, '__class__': <class 'type'>}
    
    PrePostInitMeta's namespace are:
    mappingproxy({'__call__': <function PrePostInitMeta.__call__>,
                  '__doc__': 'A metaclass that calls optional `__pre_init__` and '
                             '`__post_init__` methods',
                  '__module__': 'fastcore.meta'})



```
class Foo(FixSigMeta): pass
inspect_class(Foo)
```

    
    is Foo a metaclass: True
    is Foo created by a metaclass: False
    Foo is created by <class 'type'>
    Foo.__new__ is object.__new__: False
    Foo.__new__ is type.__new__: False
    Foo.__new__: <function FixSigMeta.__new__>
    Foo.__init__ is object.__init__: False
    Foo.__init__ is type.__init__: True
    Foo.__init__: <slot wrapper '__init__' of 'type' objects>
    Foo.__call__ is object.__call__: False
    Foo.__call__ is type.__call__: True
    Foo.__call__: <slot wrapper '__call__' of 'type' objects>
    Foo.__class__: <class 'type'>
    Foo.__bases__: (<class 'fastcore.meta.FixSigMeta'>,)
    Foo.__mro__: (<class '__main__.Foo'>, <class 'fastcore.meta.FixSigMeta'>, <class 'type'>, <class 'object'>)
    
    Foo's function members are:
    {'__new__': <function FixSigMeta.__new__>}
    
    Foo's method members are:
    {}
    
    Foo's class members are:
    {'__base__': <class 'fastcore.meta.FixSigMeta'>, '__class__': <class 'type'>}
    
    Foo's namespace are:
    mappingproxy({'__module__': '__main__', '__doc__': None})


## class Foo(metaclass=FixSigMeta)

Foo inherit `__new__`, `__init__` from object to create object instance     
Foo uses FixSigMeta not type to create class instance
FixSigMeta.`__new__` determine what kind of a class is Foo    
In this case, FixSigMeta.`__new__` create Foo class and an attr `__signature__` if Foo has its own `__init__`    
FixSigMeta.`__new__` create Foo the class, has nothing to do with the instance method Foo.`__init__`


```
class Foo(metaclass=FixSigMeta): pass
inspect_class(Foo)
```

    
    is Foo a metaclass: False
    is Foo created by a metaclass: True
    Foo is created by metaclass <class 'fastcore.meta.FixSigMeta'>
    Foo.__new__ is object.__new__: True
    Foo.__new__ is type.__new__: False
    Foo.__new__: <built-in method __new__ of type object>
    Foo.__init__ is object.__init__: True
    Foo.__init__ is type.__init__: False
    Foo.__init__: <slot wrapper '__init__' of 'object' objects>
    Foo.__call__ is object.__call__: False
    Foo.__call__ is type.__call__: False
    Foo.__call__: <method-wrapper '__call__' of FixSigMeta object>
    Foo.__class__: <class 'fastcore.meta.FixSigMeta'>
    Foo.__bases__: (<class 'object'>,)
    Foo.__mro__: (<class '__main__.Foo'>, <class 'object'>)
    
    Foo's metaclass <class 'fastcore.meta.FixSigMeta'>'s function members are:
    {'__new__': <function FixSigMeta.__new__>}
    
    Foo's function members are:
    {}
    
    Foo's method members are:
    {}
    
    Foo's class members are:
    {'__class__': <class 'fastcore.meta.FixSigMeta'>}
    
    Foo's namespace are:
    mappingproxy({'__dict__': <attribute '__dict__' of 'Foo' objects>,
                  '__doc__': None,
                  '__module__': '__main__',
                  '__weakref__': <attribute '__weakref__' of 'Foo' objects>})



```
class Foo(metaclass=FixSigMeta): 
    def __init__(self, a, b): pass
inspect_class(Foo)
```

    
    is Foo a metaclass: False
    is Foo created by a metaclass: True
    Foo is created by metaclass <class 'fastcore.meta.FixSigMeta'>
    Foo.__new__ is object.__new__: True
    Foo.__new__ is type.__new__: False
    Foo.__new__: <built-in method __new__ of type object>
    Foo.__init__ is object.__init__: False
    Foo.__init__ is type.__init__: False
    Foo.__init__: <function Foo.__init__>
    Foo.__call__ is object.__call__: False
    Foo.__call__ is type.__call__: False
    Foo.__call__: <method-wrapper '__call__' of FixSigMeta object>
    Foo.__class__: <class 'fastcore.meta.FixSigMeta'>
    Foo.__bases__: (<class 'object'>,)
    Foo.__mro__: (<class '__main__.Foo'>, <class 'object'>)
    
    Foo's metaclass <class 'fastcore.meta.FixSigMeta'>'s function members are:
    {'__new__': <function FixSigMeta.__new__>}
    
    Foo's function members are:
    {'__init__': <function Foo.__init__>}
    
    Foo's method members are:
    {}
    
    Foo's class members are:
    {'__class__': <class 'fastcore.meta.FixSigMeta'>}
    
    Foo's namespace are:
    mappingproxy({'__dict__': <attribute '__dict__' of 'Foo' objects>,
                  '__doc__': None,
                  '__init__': <function Foo.__init__>,
                  '__module__': '__main__',
                  '__signature__': <Signature (a, b)>,
                  '__weakref__': <attribute '__weakref__' of 'Foo' objects>})


## class AutoInit(metaclass=PrePostInitMeta)

AutoInit inherit `__new__` and `__init__` from `object` to create and initialize object instances     
AutoInit uses PrePostInitMeta.`__new__` or in fact FixSigMeta.`__new__` to create its own class instance, which can have `__signature__`    
AutoInit uses PrePostInitMeta.`__call__` to specify how its object instance to be created and initialized (with pre_init, init, post_init)     
AutoInit as a normal or non-metaclass, it writes its own `__pre_init__` instance method


```
fdbA.docsrc(1, "AutoInit inherit __new__ and __init__ from object to create and initialize object instances; \
AutoInit uses PrePostInitMeta.__new__ or in fact FixSigMeta.__new__ to create its own class instance, which can have __signature__; \
AutoInit uses PrePostInitMeta.__call__ to specify how its object instance to be created and initialized (with pre_init, init, post_init)); \
AutoInit as a normal or non-metaclass, it writes its own __pre_init__ method")
```

    =========================================================     Investigating [91;1mAutoInit[0m     =========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    [93;1mprint selected srcline with expands below[0m--------
    class AutoInit(metaclass=PrePostInitMeta):                                                                                                              (0)
        "Same as `object`, but no need for subclasses to call `super().__init__`"===========================================================================(1)
    [91;1mAutoInit inherit __new__ and __init__ from object to create and initialize object instances; AutoInit uses PrePostInitMeta.__new__ or in fact FixSigMeta.__new__ to create its own class instance, which can have __signature__; AutoInit uses PrePostInitMeta.__call__ to specify how its object instance to be created and initialized (with pre_init, init, post_init)); AutoInit as a normal or non-metaclass, it writes its own __pre_init__ method[0m
        def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)                                                                          (2)
                                                                                                                                                            (3)



```

```


```
inspect_class(AutoInit)
```

    class AutoInit(metaclass=PrePostInitMeta):
        "Same as `object`, but no need for subclasses to call `super().__init__`"
        def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    
    
    is AutoInit a metaclass: False
    is AutoInit created by a metaclass: True
    AutoInit is created by metaclass <class 'fastcore.meta.PrePostInitMeta'>
    AutoInit.__new__ is object.__new__: True
    AutoInit.__new__ is type.__new__: False
    AutoInit.__new__: <built-in method __new__ of type object>
    AutoInit.__init__ is object.__init__: True
    AutoInit.__init__ is type.__init__: False
    AutoInit.__init__: <slot wrapper '__init__' of 'object' objects>
    AutoInit.__call__ is object.__call__: False
    AutoInit.__call__ is type.__call__: False
    AutoInit.__call__: <bound method PrePostInitMeta.__call__ of <class 'fastcore.meta.AutoInit'>>
    AutoInit.__class__: <class 'fastcore.meta.PrePostInitMeta'>
    AutoInit.__bases__: (<class 'object'>,)
    AutoInit.__mro__: (<class 'fastcore.meta.AutoInit'>, <class 'object'>)
    
    AutoInit's metaclass <class 'fastcore.meta.PrePostInitMeta'>'s function members are:
    {'__call__': <function PrePostInitMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    AutoInit's function members are:
    {'__pre_init__': <function AutoInit.__pre_init__>}
    
    AutoInit's method members are:
    {}
    
    AutoInit's class members are:
    {'__class__': <class 'fastcore.meta.PrePostInitMeta'>}
    
    AutoInit's namespace are:
    mappingproxy({'__dict__': <attribute '__dict__' of 'AutoInit' objects>,
                  '__doc__': 'Same as `object`, but no need for subclasses to call '
                             '`super().__init__`',
                  '__module__': 'fastcore.meta',
                  '__pre_init__': <function AutoInit.__pre_init__>,
                  '__weakref__': <attribute '__weakref__' of 'AutoInit' objects>})



```
class Foo(AutoInit): pass
inspect_class(Foo)
```

    
    is Foo a metaclass: False
    is Foo created by a metaclass: True
    Foo is created by metaclass <class 'fastcore.meta.PrePostInitMeta'>
    Foo.__new__ is object.__new__: True
    Foo.__new__ is type.__new__: False
    Foo.__new__: <built-in method __new__ of type object>
    Foo.__init__ is object.__init__: True
    Foo.__init__ is type.__init__: False
    Foo.__init__: <slot wrapper '__init__' of 'object' objects>
    Foo.__call__ is object.__call__: False
    Foo.__call__ is type.__call__: False
    Foo.__call__: <bound method PrePostInitMeta.__call__ of <class '__main__.Foo'>>
    Foo.__class__: <class 'fastcore.meta.PrePostInitMeta'>
    Foo.__bases__: (<class 'fastcore.meta.AutoInit'>,)
    Foo.__mro__: (<class '__main__.Foo'>, <class 'fastcore.meta.AutoInit'>, <class 'object'>)
    
    Foo's metaclass <class 'fastcore.meta.PrePostInitMeta'>'s function members are:
    {'__call__': <function PrePostInitMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    Foo's function members are:
    {'__pre_init__': <function AutoInit.__pre_init__>}
    
    Foo's method members are:
    {}
    
    Foo's class members are:
    {'__class__': <class 'fastcore.meta.PrePostInitMeta'>}
    
    Foo's namespace are:
    mappingproxy({'__module__': '__main__', '__doc__': None})



```
class Foo(AutoInit): 
    def __init__(self): pass # to enable __signature__ by FixSigMeta.__new__
inspect_class(Foo)
```

    
    is Foo a metaclass: False
    is Foo created by a metaclass: True
    Foo is created by metaclass <class 'fastcore.meta.PrePostInitMeta'>
    Foo.__new__ is object.__new__: True
    Foo.__new__ is type.__new__: False
    Foo.__new__: <built-in method __new__ of type object>
    Foo.__init__ is object.__init__: False
    Foo.__init__ is type.__init__: False
    Foo.__init__: <function Foo.__init__>
    Foo.__call__ is object.__call__: False
    Foo.__call__ is type.__call__: False
    Foo.__call__: <bound method PrePostInitMeta.__call__ of <class '__main__.Foo'>>
    Foo.__class__: <class 'fastcore.meta.PrePostInitMeta'>
    Foo.__bases__: (<class 'fastcore.meta.AutoInit'>,)
    Foo.__mro__: (<class '__main__.Foo'>, <class 'fastcore.meta.AutoInit'>, <class 'object'>)
    
    Foo's metaclass <class 'fastcore.meta.PrePostInitMeta'>'s function members are:
    {'__call__': <function PrePostInitMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    Foo's function members are:
    {'__init__': <function Foo.__init__>,
     '__pre_init__': <function AutoInit.__pre_init__>}
    
    Foo's method members are:
    {}
    
    Foo's class members are:
    {'__class__': <class 'fastcore.meta.PrePostInitMeta'>}
    
    Foo's namespace are:
    mappingproxy({'__doc__': None,
                  '__init__': <function Foo.__init__>,
                  '__module__': '__main__',
                  '__signature__': <Signature ()>})



```
class TestParent():
    def __init__(self): self.h = 10
        
class TestChild(AutoInit, TestParent):
    def __init__(self): self.k = self.h + 2
inspect_class(TestChild)
```

    
    is TestChild a metaclass: False
    is TestChild created by a metaclass: True
    TestChild is created by metaclass <class 'fastcore.meta.PrePostInitMeta'>
    TestChild.__new__ is object.__new__: True
    TestChild.__new__ is type.__new__: False
    TestChild.__new__: <built-in method __new__ of type object>
    TestChild.__init__ is object.__init__: False
    TestChild.__init__ is type.__init__: False
    TestChild.__init__: <function TestChild.__init__>
    TestChild.__call__ is object.__call__: False
    TestChild.__call__ is type.__call__: False
    TestChild.__call__: <bound method PrePostInitMeta.__call__ of <class '__main__.TestChild'>>
    TestChild.__class__: <class 'fastcore.meta.PrePostInitMeta'>
    TestChild.__bases__: (<class 'fastcore.meta.AutoInit'>, <class '__main__.TestParent'>)
    TestChild.__mro__: (<class '__main__.TestChild'>, <class 'fastcore.meta.AutoInit'>, <class '__main__.TestParent'>, <class 'object'>)
    
    TestChild's metaclass <class 'fastcore.meta.PrePostInitMeta'>'s function members are:
    {'__call__': <function PrePostInitMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    TestChild's function members are:
    {'__init__': <function TestChild.__init__>,
     '__pre_init__': <function AutoInit.__pre_init__>}
    
    TestChild's method members are:
    {}
    
    TestChild's class members are:
    {'__class__': <class 'fastcore.meta.PrePostInitMeta'>}
    
    TestChild's namespace are:
    mappingproxy({'__doc__': None,
                  '__init__': <function TestChild.__init__>,
                  '__module__': '__main__',
                  '__signature__': <Signature ()>})



```
class _T(metaclass=PrePostInitMeta):
    def __pre_init__(self):  self.a  = 0; 
    def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
    def __post_init__(self): self.c = self.b + 2; assert self.c==3
inspect_class(_T)
```

    
    is _T a metaclass: False
    is _T created by a metaclass: True
    _T is created by metaclass <class 'fastcore.meta.PrePostInitMeta'>
    _T.__new__ is object.__new__: True
    _T.__new__ is type.__new__: False
    _T.__new__: <built-in method __new__ of type object>
    _T.__init__ is object.__init__: False
    _T.__init__ is type.__init__: False
    _T.__init__: <function _T.__init__>
    _T.__call__ is object.__call__: False
    _T.__call__ is type.__call__: False
    _T.__call__: <bound method PrePostInitMeta.__call__ of <class '__main__._T'>>
    _T.__class__: <class 'fastcore.meta.PrePostInitMeta'>
    _T.__bases__: (<class 'object'>,)
    _T.__mro__: (<class '__main__._T'>, <class 'object'>)
    
    _T's metaclass <class 'fastcore.meta.PrePostInitMeta'>'s function members are:
    {'__call__': <function PrePostInitMeta.__call__>,
     '__new__': <function FixSigMeta.__new__>}
    
    _T's function members are:
    {'__init__': <function _T.__init__>,
     '__post_init__': <function _T.__post_init__>,
     '__pre_init__': <function _T.__pre_init__>}
    
    _T's method members are:
    {}
    
    _T's class members are:
    {'__class__': <class 'fastcore.meta.PrePostInitMeta'>}
    
    _T's namespace are:
    mappingproxy({'__dict__': <attribute '__dict__' of '_T' objects>,
                  '__doc__': None,
                  '__init__': <function _T.__init__>,
                  '__module__': '__main__',
                  '__post_init__': <function _T.__post_init__>,
                  '__pre_init__': <function _T.__pre_init__>,
                  '__signature__': <Signature (b=0)>,
                  '__weakref__': <attribute '__weakref__' of '_T' objects>})



```

```

## Prepare examples for FixSigMeta, PrePostInitMeta, AutoInit 


```
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


```

fdbF.snoop(watch=['res', 'type(res)', 'res.__class__', 'res.__dict__'])
```

    23:04:33.14 >>> Call to FixSigMeta.__new__ in File "/tmp/FixSigMeta.py", line 5
    23:04:33.14 .......... cls = <class 'fastcore.meta.FixSigMeta'>
    23:04:33.14 .......... name = 'Foo'
    23:04:33.14 .......... bases = ()
    23:04:33.14 .......... dict = {'__module__': '__main__', '__qualname__': 'Foo', '__init__': <function Foo.__init__>}
    23:04:33.14 .......... len(dict) = 3
    23:04:33.14 .......... __class__ = <class 'fastcore.meta.FixSigMeta'>
    23:04:33.14    5 |     def __new__(cls, name, bases, dict):
    23:04:33.14    6 |         res = super().__new__(cls, name, bases, dict)
    23:04:33.14 .............. res = <class '__main__.Foo'>
    23:04:33.14 .............. type(res) = <class 'fastcore.meta.FixSigMeta'>
    23:04:33.14 .............. res.__class__ = <class 'fastcore.meta.FixSigMeta'>
    23:04:33.14 .............. res.__dict__ = mappingproxy({'__module__': '__main__', '__init_...__weakref__' of 'Foo' objects>, '__doc__': None})
    23:04:33.14 .............. len(res.__dict__) = 5
    23:04:33.14    7 |         if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))
    23:04:33.14 ...... res.__dict__ = mappingproxy({'__module__': '__main__', '__init_...__doc__': None, '__signature__': <Signature ()>})
    23:04:33.14 ...... len(res.__dict__) = 6
    23:04:33.14    8 |         return res
    23:04:33.14 <<< Return value from FixSigMeta.__new__: <class '__main__.Foo'>


    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    ==============================     with example [91;1m
    class Foo(metaclass=FixSigMeta):
        def __init__(self): pass
    [0m     ==============================
    


### embed the dbsrc of FixSigMeta into PrePostInitMeta

**Important!**    
FixSigMeta is untouched, fdbF.dbsrc.`__new__` is the actual dbsrc     
To use fdbF.dbsrc in other functions or classes which uses FixSigMeta, we need to assign `fdbF.dbsrc` to `fm.FixSigMeta`


```
import fastcore.meta as fm
```


```
fm.FixSigMeta = fdbF.dbsrc
```


```
fdbP.snoop(['res.__dict__'])
```

    23:04:33.18 >>> Call to FixSigMeta.__new__ in File "/tmp/FixSigMeta.py", line 5
    23:04:33.18 .......... cls = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.18 .......... name = '_T'
    23:04:33.18 .......... bases = ()
    23:04:33.18 .......... dict = {'__module__': '__main__', '__qualname__': '_T', '__pre_init__': <function _T.__pre_init__>, '__init__': <function _T.__init__>, ...}
    23:04:33.18 .......... len(dict) = 5
    23:04:33.18 .......... __class__ = <class 'fastcore.meta.FixSigMeta'>
    23:04:33.18    5 |     def __new__(cls, name, bases, dict):
    23:04:33.18    6 |         res = super().__new__(cls, name, bases, dict)
    23:04:33.18 .............. res = <class '__main__._T'>
    23:04:33.18 .............. type(res) = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.18 .............. res.__class__ = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.18 .............. res.__dict__ = mappingproxy({'__module__': '__main__', '__pre_i...'__weakref__' of '_T' objects>, '__doc__': None})
    23:04:33.18 .............. len(res.__dict__) = 7
    23:04:33.18    7 |         if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))
    23:04:33.18 ...... res.__dict__ = mappingproxy({'__module__': '__main__', '__pre_i...oc__': None, '__signature__': <Signature (b=0)>})
    23:04:33.18 ...... len(res.__dict__) = 8
    23:04:33.18    8 |         return res
    23:04:33.18 <<< Return value from FixSigMeta.__new__: <class '__main__._T'>
    23:04:33.19 >>> Call to PrePostInitMeta.__call__ in File "/tmp/PrePostInitMeta.py", line 5
    23:04:33.19 .......... cls = <class '__main__._T'>
    23:04:33.19 .......... args = ()
    23:04:33.19 .......... kwargs = {}
    23:04:33.19    5 |     def __call__(cls, *args, **kwargs):
    23:04:33.19    6 |         res = cls.__new__(cls)
    23:04:33.19 .............. res = <__main__._T object>
    23:04:33.19 .............. res.__dict__ = {}
    23:04:33.19    7 |         if type(res)==cls:
    23:04:33.19    8 |             if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)
    23:04:33.19 ...... res.__dict__ = {'a': 0}
    23:04:33.19 ...... len(res.__dict__) = 1
    23:04:33.19    9 |             res.__init__(*args,**kwargs)
    23:04:33.19 .................. res.__dict__ = {'a': 0, 'b': 1}
    23:04:33.19 .................. len(res.__dict__) = 2
    23:04:33.19   10 |             if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)
    23:04:33.19 ...... res.__dict__ = {'a': 0, 'b': 1, 'c': 3}
    23:04:33.19 ...... len(res.__dict__) = 3
    23:04:33.19   11 |         return res
    23:04:33.19 <<< Return value from PrePostInitMeta.__call__: <__main__._T object>


    =====================================================     Investigating [91;1mPrePostInitMeta[0m     ======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class _T(metaclass=PrePostInitMeta):
        def __pre_init__(self):  self.a  = 0; 
        def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
        def __post_init__(self): self.c = self.b + 2; assert self.c==3
    
    t = _T()
    test_eq(t.a, 0) # set with __pre_init__
    test_eq(t.b, 1) # set with __init__
    test_eq(t.c, 3) # set with __post_init__
    inspect.signature(_T)
    [0m     
    


### embed dbsrc of PrePostInitMeta into AutoInit


```
fm.PrePostInitMeta = fdbP.dbsrc
```


```
fdbA.snoop()
```

    23:04:33.20 >>> Call to FixSigMeta.__new__ in File "/tmp/FixSigMeta.py", line 5
    23:04:33.20 .......... cls = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.20 .......... name = 'AutoInit'
    23:04:33.20 .......... bases = ()
    23:04:33.20 .......... dict = {'__module__': 'fastcore.meta', '__qualname__': 'AutoInit', '__doc__': 'Same as `object`, but no need for subclasses to call `super().__init__`', 'snoop': <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>, ...}
    23:04:33.20 .......... len(dict) = 6
    23:04:33.20 .......... __class__ = <class 'fastcore.meta.FixSigMeta'>
    23:04:33.20    5 |     def __new__(cls, name, bases, dict):
    23:04:33.20    6 |         res = super().__new__(cls, name, bases, dict)
    23:04:33.20 .............. res = <class 'fastcore.meta.AutoInit'>
    23:04:33.20 .............. type(res) = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.20 .............. res.__class__ = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.20 .............. res.__dict__ = mappingproxy({'__module__': 'fastcore.meta', '__...<attribute '__weakref__' of 'AutoInit' objects>})
    23:04:33.20 .............. len(res.__dict__) = 6
    23:04:33.20    7 |         if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))
    23:04:33.20    8 |         return res
    23:04:33.20 <<< Return value from FixSigMeta.__new__: <class 'fastcore.meta.AutoInit'>
    23:04:33.20 >>> Call to FixSigMeta.__new__ in File "/tmp/FixSigMeta.py", line 5
    23:04:33.20 .......... cls = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.20 .......... name = 'TestChild'
    23:04:33.20 .......... bases = (<class 'fastcore.meta.AutoInit'>, <class '__main__.TestParent'>)
    23:04:33.20 .......... len(bases) = 2
    23:04:33.20 .......... dict = {'__module__': '__main__', '__qualname__': 'TestChild', '__init__': <function TestChild.__init__>}
    23:04:33.20 .......... len(dict) = 3
    23:04:33.20 .......... __class__ = <class 'fastcore.meta.FixSigMeta'>
    23:04:33.20    5 |     def __new__(cls, name, bases, dict):
    23:04:33.20    6 |         res = super().__new__(cls, name, bases, dict)
    23:04:33.20 .............. res = <class '__main__.TestChild'>
    23:04:33.20 .............. type(res) = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.20 .............. res.__class__ = <class 'fastcore.meta.PrePostInitMeta'>
    23:04:33.20 .............. res.__dict__ = mappingproxy({'__module__': '__main__', '__init_...Child.__init__ at 0x11fbf51f0>, '__doc__': None})
    23:04:33.20 .............. len(res.__dict__) = 3
    23:04:33.20    7 |         if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))
    23:04:33.20 ...... res.__dict__ = mappingproxy({'__module__': '__main__', '__init_...__doc__': None, '__signature__': <Signature ()>})
    23:04:33.20 ...... len(res.__dict__) = 4
    23:04:33.20    8 |         return res
    23:04:33.20 <<< Return value from FixSigMeta.__new__: <class '__main__.TestChild'>
    23:04:33.20 >>> Call to PrePostInitMeta.__call__ in File "/tmp/PrePostInitMeta.py", line 5
    23:04:33.20 .......... cls = <class '__main__.TestChild'>
    23:04:33.20 .......... args = ()
    23:04:33.20 .......... kwargs = {}
    23:04:33.20    5 |     def __call__(cls, *args, **kwargs):
    23:04:33.20    6 |         res = cls.__new__(cls)
    23:04:33.20 .............. res = <__main__.TestChild object>
    23:04:33.20 .............. res.__dict__ = {}
    23:04:33.20    7 |         if type(res)==cls:
    23:04:33.20    8 |             if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)
        23:04:33.20 >>> Call to AutoInit.__pre_init__ in File "/tmp/AutoInit.py", line 5
        23:04:33.20 .......... self = <__main__.TestChild object>
        23:04:33.20 .......... args = ()
        23:04:33.20 .......... kwargs = {}
        23:04:33.20 .......... __class__ = <class 'fastcore.meta.AutoInit'>
        23:04:33.20    5 |     def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
        23:04:33.20    5 |     def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
        23:04:33.20 <<< Return value from AutoInit.__pre_init__: None
    23:04:33.20    8 |             if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)
    23:04:33.20 ...... res.__dict__ = {'h': 10}
    23:04:33.20 ...... len(res.__dict__) = 1
    23:04:33.20    9 |             res.__init__(*args,**kwargs)
    23:04:33.20 .................. res.__dict__ = {'h': 10, 'k': 12}
    23:04:33.20 .................. len(res.__dict__) = 2
    23:04:33.20   10 |             if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)
    23:04:33.20   11 |         return res
    23:04:33.20 <<< Return value from PrePostInitMeta.__call__: <__main__.TestChild object>


    =========================================================     Investigating [91;1mAutoInit[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class TestParent():
        def __init__(self): self.h = 10
            
    class TestChild(AutoInit, TestParent):
        def __init__(self): self.k = self.h + 2
        
    t = TestChild()
    test_eq(t.h, 10) # h=10 is initialized in the parent class
    test_eq(t.k, 12)
    [0m     
    


## Explore and Document on them together 


```
fdbF.docsrc(4, "FixSigMeta: what is res", "'inside FixSigMeta, line 4'", "res.__name__")
fm.FixSigMeta = fdbF.dbsrc
```

    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
    ==============================     with example [91;1m
    class Foo(metaclass=FixSigMeta):
        def __init__(self): pass
    [0m     ==============================
    
    [93;1mprint selected srcline with expands below[0m--------
        def __new__(cls, name, bases, dict):                                                                                                                (2)
            res = super().__new__(cls, name, bases, dict)                                                                                                   (3)
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))===========================================(4)
                                                                                                                                          [91;1mFixSigMeta: what is res[0m
            return res                                                                                                                                      (5)
                                                                                                                                                            (6)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                           'inside FixSigMeta, line 4' => 'inside FixSigMeta, line 4' : inside FixSigMeta, line 4
    
    
                                                                                                                               res.__name__ => res.__name__ : Foo
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [35;1mFixSigMeta inherits __init__, and __call__ from type[0m; [35;1mbut writes its own __new__[0m; [92;1mFoo inherits all three from type[0m; [34;1mFixSigMeta is used to create class instance not object instance.[0m; 
        def __new__(cls, name, bases, dict):==================================================(2)       
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [35;1mFixSigMeta: what is res[0m;                   (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)
                                                                                                                                         part No.1 out of 1 parts
    



```
fdbP.docsrc(6, "what inside res.__dict__", "'inside PrePostInitMeta: '", "res.__dict__")
fm.PrePostInitMeta = fdbP.dbsrc
```

    =====================================================     Investigating [91;1mPrePostInitMeta[0m     ======================================================
    ===============================================================     on line [91;1m6[0m     ================================================================
         with example [91;1m
    class _T(metaclass=PrePostInitMeta):
        def __pre_init__(self):  self.a  = 0; 
        def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
        def __post_init__(self): self.c = self.b + 2; assert self.c==3
    
    t = _T()
    test_eq(t.a, 0) # set with __pre_init__
    test_eq(t.b, 1) # set with __init__
    test_eq(t.c, 3) # set with __post_init__
    inspect.signature(_T)
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            if type(res)==cls:                                                                                                                              (4)
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)                                                                            (5)
                res.__init__(*args,**kwargs)================================================================================================================(6)
                                                                                                                                         [91;1mwhat inside res.__dict__[0m
                if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)                                                                          (7)
            return res                                                                                                                                      (8)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                           'inside FixSigMeta, line 4' => 'inside FixSigMeta, line 4' : inside FixSigMeta, line 4
    
    
                                                                                                                                res.__name__ => res.__name__ : _T
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                              'inside PrePostInitMeta: ' => 'inside PrePostInitMeta: ' : inside PrePostInitMeta: 
    
    
                                                                                                                          res.__dict__ => res.__dict__ : {'a': 0}
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class PrePostInitMeta(FixSigMeta):========================================================(0)       
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"==========(1) # [92;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type)[0m; [91;1mnot from type, nor from object[0m; [36;1mPrePostInitMeta is itself a metaclass, which is used to create class instance not object instance[0m; [92;1mPrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m; 
        def __call__(cls, *args, **kwargs):===================================================(2)       
            res = cls.__new__(cls)============================================================(3)       
            if type(res)==cls:================================================================(4)       
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)==============(5)       
                res.__init__(*args,**kwargs)==================================================(6) # [91;1mwhat inside res.__dict__[0m; 
                if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)============(7)       
            return res========================================================================(8)       
                                                                                                                                                            (9)
                                                                                                                                         part No.1 out of 1 parts
    



```
fdbA.docsrc(2, "what is cls", "'Inside AutoInit'", "cls") # need to run it twice (a little bug here)
```

    =========================================================     Investigating [91;1mAutoInit[0m     =========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
    class TestParent():
        def __init__(self): self.h = 10
            
    class TestChild(AutoInit, TestParent):
        def __init__(self): self.k = self.h + 2
        
    t = TestChild()
    test_eq(t.h, 10) # h=10 is initialized in the parent class
    test_eq(t.k, 12)
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    class AutoInit(metaclass=PrePostInitMeta):                                                                                                              (0)
        "Same as `object`, but no need for subclasses to call `super().__init__`"                                                                           (1)
        def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)==========================================================================(2)
                                                                                                                                                      [91;1mwhat is cls[0m
                                                                                                                                                            (3)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                                                         'Inside AutoInit' => 'Inside AutoInit' : Inside AutoInit
    
    
                                                                                                                               cls => cls : <class '__main__._T'>
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                           'inside FixSigMeta, line 4' => 'inside FixSigMeta, line 4' : inside FixSigMeta, line 4
    
    
                                                                                                                          res.__name__ => res.__name__ : AutoInit
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                           'inside FixSigMeta, line 4' => 'inside FixSigMeta, line 4' : inside FixSigMeta, line 4
    
    
                                                                                                                         res.__name__ => res.__name__ : TestChild
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                              'inside PrePostInitMeta: ' => 'inside PrePostInitMeta: ' : inside PrePostInitMeta: 
    
    
                                                                                                                         res.__dict__ => res.__dict__ : {'h': 10}
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class AutoInit(metaclass=PrePostInitMeta):================================================(0)       
        "Same as `object`, but no need for subclasses to call `super().__init__`"=============(1) # [92;1mAutoInit inherit __new__ and __init__ from object to create and initialize object instances[0m; [93;1mAutoInit uses PrePostInitMeta.__new__ or in fact FixSigMeta.__new__ to create its own class instance, which can have __signature__[0m; [37;1mAutoInit uses PrePostInitMeta.__call__ to specify how its object instance to be created and initialized (with pre_init, init, post_init))[0m; [36;1mAutoInit as a normal or non-metaclass, it writes its own __pre_init__ method[0m; 
        def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)============(2) # [93;1mwhat is cls[0m; 
                                                                                                                                                            (3)
                                                                                                                                         part No.1 out of 1 parts
    



```
fdbF.docsrc(3, "how to create a new class instance with type dynamically; \
the rest below is how FixSigMeta as a metaclass create its own instance classes")
fdbF.docsrc(4, "how to check whether a class has its own __init__ function; how to remove self param from a signature")
```

    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
    ==============================     with example [91;1m
    class Foo(metaclass=FixSigMeta):
        def __init__(self): pass
    [0m     ==============================
    
    [93;1mprint selected srcline with expands below[0m--------
        "A metaclass that fixes the signature on classes that override `__new__`"                                                                           (1)
        def __new__(cls, name, bases, dict):                                                                                                                (2)
            res = super().__new__(cls, name, bases, dict)===================================================================================================(3)
                        [91;1mhow to create a new class instance with type dynamically; the rest below is how FixSigMeta as a metaclass create its own instance classes[0m
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))                                           (4)
            return res                                                                                                                                      (5)
    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
    ==============================     with example [91;1m
    class Foo(metaclass=FixSigMeta):
        def __init__(self): pass
    [0m     ==============================
    
    [93;1mprint selected srcline with expands below[0m--------
        def __new__(cls, name, bases, dict):                                                                                                                (2)
            res = super().__new__(cls, name, bases, dict)                                                                                                   (3)
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))===========================================(4)
                                                            [91;1mhow to check whether a class has its own __init__ function; how to remove self param from a signature[0m
            return res                                                                                                                                      (5)
                                                                                                                                                            (6)



```
fdbF.print()
```

    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
    ==============================     with example [91;1m
    class Foo(metaclass=FixSigMeta):
        def __init__(self): pass
    [0m     ==============================
    
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [34;1mFixSigMeta inherits __init__, and __call__ from type[0m; [36;1mbut writes its own __new__[0m; [36;1mFoo inherits all three from type[0m; [93;1mFixSigMeta is used to create class instance not object instance.[0m; 
        def __new__(cls, name, bases, dict):==================================================(2)       
            res = super().__new__(cls, name, bases, dict)=====================================(3) # [92;1mhow to create a new class instance with type dynamically[0m; [36;1mthe rest below is how FixSigMeta as a metaclass create its own instance classes[0m; 
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [34;1mhow to check whether a class has its own __init__ function[0m; [92;1mhow to remove self param from a signature[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)



```
fdbP.docsrc(3, "how to create an object instance with a cls; how to check the type of an object is cls; \
how to run a function without knowing its params;")
fdbP.print()
```

    =====================================================     Investigating [91;1mPrePostInitMeta[0m     ======================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
    class _T(metaclass=PrePostInitMeta):
        def __pre_init__(self):  self.a  = 0; 
        def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
        def __post_init__(self): self.c = self.b + 2; assert self.c==3
    
    t = _T()
    test_eq(t.a, 0) # set with __pre_init__
    test_eq(t.b, 1) # set with __init__
    test_eq(t.c, 3) # set with __post_init__
    inspect.signature(_T)
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"                                                                        (1)
        def __call__(cls, *args, **kwargs):                                                                                                                 (2)
            res = cls.__new__(cls)==========================================================================================================================(3)
                        [91;1mhow to create an object instance with a cls; how to check the type of an object is cls; how to run a function without knowing its params;[0m
            if type(res)==cls:                                                                                                                              (4)
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)                                                                            (5)
    =====================================================     Investigating [91;1mPrePostInitMeta[0m     ======================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
    class _T(metaclass=PrePostInitMeta):
        def __pre_init__(self):  self.a  = 0; 
        def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
        def __post_init__(self): self.c = self.b + 2; assert self.c==3
    
    t = _T()
    test_eq(t.a, 0) # set with __pre_init__
    test_eq(t.b, 1) # set with __init__
    test_eq(t.c, 3) # set with __post_init__
    inspect.signature(_T)
    [0m     
    
    class PrePostInitMeta(FixSigMeta):========================================================(0)       
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"==========(1) # [93;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type)[0m; [92;1mnot from type, nor from object[0m; [35;1mPrePostInitMeta is itself a metaclass, which is used to create class instance not object instance[0m; [91;1mPrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m; 
        def __call__(cls, *args, **kwargs):===================================================(2)       
            res = cls.__new__(cls)============================================================(3) # [37;1mhow to create an object instance with a cls[0m; [37;1mhow to check the type of an object is cls[0m; [34;1mhow to run a function without knowing its params;[0m; 
            if type(res)==cls:================================================================(4)       
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)==============(5)       
                res.__init__(*args,**kwargs)==================================================(6) # [35;1mwhat inside res.__dict__[0m; 
                if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)============(7)       
            return res========================================================================(8)       
                                                                                                                                                            (9)



```
fdbA.docsrc(2, "how to run superclass' __init__ function")
```

    =========================================================     Investigating [91;1mAutoInit[0m     =========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
    class TestParent():
        def __init__(self): self.h = 10
            
    class TestChild(AutoInit, TestParent):
        def __init__(self): self.k = self.h + 2
        
    t = TestChild()
    test_eq(t.h, 10) # h=10 is initialized in the parent class
    test_eq(t.k, 12)
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    class AutoInit(metaclass=PrePostInitMeta):                                                                                                              (0)
        "Same as `object`, but no need for subclasses to call `super().__init__`"                                                                           (1)
        def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)==========================================================================(2)
                                                                                                                         [91;1mhow to run superclass' __init__ function[0m
                                                                                                                                                            (3)



```
fdbA.print()
```

    =========================================================     Investigating [91;1mAutoInit[0m     =========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
    class TestParent():
        def __init__(self): self.h = 10
            
    class TestChild(AutoInit, TestParent):
        def __init__(self): self.k = self.h + 2
        
    t = TestChild()
    test_eq(t.h, 10) # h=10 is initialized in the parent class
    test_eq(t.k, 12)
    [0m     
    
    class AutoInit(metaclass=PrePostInitMeta):================================================(0)       
        "Same as `object`, but no need for subclasses to call `super().__init__`"=============(1) # [92;1mAutoInit inherit __new__ and __init__ from object to create and initialize object instances[0m; [91;1mAutoInit uses PrePostInitMeta.__new__ or in fact FixSigMeta.__new__ to create its own class instance, which can have __signature__[0m; [92;1mAutoInit uses PrePostInitMeta.__call__ to specify how its object instance to be created and initialized (with pre_init, init, post_init))[0m; [34;1mAutoInit as a normal or non-metaclass, it writes its own __pre_init__ method[0m; 
        def __pre_init__(self, *args, **kwargs): super().__init__(*args, **kwargs)============(2) # [91;1mhow to run superclass' __init__ function[0m; 
                                                                                                                                                            (3)



```
fdbP.docsrc(6, "how to run __init__ without knowing its params")
```

    =====================================================     Investigating [91;1mPrePostInitMeta[0m     ======================================================
    ===============================================================     on line [91;1m6[0m     ================================================================
         with example [91;1m
    class _T(metaclass=PrePostInitMeta):
        def __pre_init__(self):  self.a  = 0; 
        def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
        def __post_init__(self): self.c = self.b + 2; assert self.c==3
    
    t = _T()
    test_eq(t.a, 0) # set with __pre_init__
    test_eq(t.b, 1) # set with __init__
    test_eq(t.c, 3) # set with __post_init__
    inspect.signature(_T)
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            if type(res)==cls:                                                                                                                              (4)
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)                                                                            (5)
                res.__init__(*args,**kwargs)================================================================================================================(6)
                                                                                                                   [91;1mhow to run __init__ without knowing its params[0m
                if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)                                                                          (7)
            return res                                                                                                                                      (8)



```
fdbP.print()
```

    =====================================================     Investigating [91;1mPrePostInitMeta[0m     ======================================================
    ===============================================================     on line [91;1m6[0m     ================================================================
         with example [91;1m
    class _T(metaclass=PrePostInitMeta):
        def __pre_init__(self):  self.a  = 0; 
        def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
        def __post_init__(self): self.c = self.b + 2; assert self.c==3
    
    t = _T()
    test_eq(t.a, 0) # set with __pre_init__
    test_eq(t.b, 1) # set with __init__
    test_eq(t.c, 3) # set with __post_init__
    inspect.signature(_T)
    [0m     
    
    class PrePostInitMeta(FixSigMeta):========================================================(0)       
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"==========(1) # [35;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type)[0m; [37;1mnot from type, nor from object[0m; [93;1mPrePostInitMeta is itself a metaclass, which is used to create class instance not object instance[0m; [93;1mPrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m; 
        def __call__(cls, *args, **kwargs):===================================================(2)       
            res = cls.__new__(cls)============================================================(3) # [37;1mhow to create an object instance with a cls[0m; [34;1mhow to check the type of an object is cls[0m; [36;1mhow to run a function without knowing its params;[0m; 
            if type(res)==cls:================================================================(4)       
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)==============(5)       
                res.__init__(*args,**kwargs)==================================================(6) # [35;1mhow to run __init__ without knowing its params[0m; 
                if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)============(7)       
            return res========================================================================(8)       
                                                                                                                                                            (9)



```
fdbF.print()
```

    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
    ==============================     with example [91;1m
    class Foo(metaclass=FixSigMeta):
        def __init__(self): pass
    [0m     ==============================
    
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [35;1mFixSigMeta inherits __init__, and __call__ from type[0m; [35;1mbut writes its own __new__[0m; [91;1mFoo inherits all three from type[0m; [35;1mFixSigMeta is used to create class instance not object instance.[0m; 
        def __new__(cls, name, bases, dict):==================================================(2)       
            res = super().__new__(cls, name, bases, dict)=====================================(3) # [34;1mhow to create a new class instance with type dynamically[0m; [35;1mthe rest below is how FixSigMeta as a metaclass create its own instance classes[0m; 
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [36;1mhow to check whether a class has its own __init__ function[0m; [36;1mhow to remove self param from a signature[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)



```

```
