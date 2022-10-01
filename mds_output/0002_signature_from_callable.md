# 0002_signature_from_callable

## Expand cell


```
# from IPython.core.display import display, HTML # a depreciated import
from IPython.display import display, HTML 
```


```
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>


## Imports and initiate


```
from fastdebug.core import *
from fastcore.meta import *
```


```
g = locals()
fdb = Fastdb(inspect._signature_from_callable, outloc=g)
fdbF = Fastdb(FixSigMeta, outloc=g)
```

## Examples


```
from fastdebug.utils import whatinside
```


<style>.container { width:100% !important; }</style>



```
inspect._signature_from_callable(whatinside, sigcls=inspect.Signature)
```




    <Signature (mo, dun: bool = False, func: bool = False, clas: bool = False, bltin: bool = False, lib: bool = False, cal: bool = False)>




```
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


```
fdbF.docsrc(2, "how does a metaclass create a class instance; what does super().__new__() do here;", "inspect.getdoc(super)")
fdbF.docsrc(4, "how to remove self from a signature; how to check whether a class' __init__ is inherited from object or not;",\
            "res", "res.__init__ is not object.__init__")
fdbF.docsrc(1, "Any class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;\
FixSigMeta uses its __new__ to create a class instance; then check whether its class instance has its own __init__;\
if so, remove self from the sig of __init__; then assign this new sig to __signature__ for the class instance;")
```

    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    class FixSigMeta(type):                                                                                                                                 (0)
        "A metaclass that fixes the signature on classes that override `__new__`"                                                                           (1)
        def __new__(cls, name, bases, dict):================================================================================================================(2)
                                                                               [91;1mhow does a metaclass create a class instance; what does super().__new__() do here;[0m
            res = super().__new__(cls, name, bases, dict)                                                                                                   (3)
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))                                           (4)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
    inspect.getdoc(super) => inspect.getdoc(super) : super() -> same as super(__class__, <first argument>)
    super(type) -> unbound super object
    super(type, obj) -> bound super object; requires isinstance(obj, type)
    super(type, type2) -> bound super object; requires issubclass(type2, type)
    Typical use to call a cooperative superclass method:
    class C(B):
        def meth(self, arg):
            super().meth(arg)
    This works for class methods too:
    class C(B):
        @classmethod
        def cmeth(cls, arg):
            super().cmeth(arg)
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (d, e, f)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1)       
        def __new__(cls, name, bases, dict):==================================================(2) # [34;1mhow does a metaclass create a class instance[0m; [37;1mwhat does super().__new__() do here;[0m; 
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))                                           (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)
                                                                                                                                         part No.1 out of 1 parts
    
    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
         with example [91;1m
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        def __new__(cls, name, bases, dict):                                                                                                                (2)
            res = super().__new__(cls, name, bases, dict)                                                                                                   (3)
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))===========================================(4)
                                                     [91;1mhow to remove self from a signature; how to check whether a class' __init__ is inherited from object or not;[0m
            return res                                                                                                                                      (5)
                                                                                                                                                            (6)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                                                                     res => res : <class '__main__.Foo_call_fix'>
    
    
                                                                                res.__init__ is not object.__init__ => res.__init__ is not object.__init__ : True
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (d, e, f)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1)       
        def __new__(cls, name, bases, dict):==================================================(2) # [92;1mhow does a metaclass create a class instance[0m; [34;1mwhat does super().__new__() do here;[0m; 
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [34;1mhow to remove self from a signature[0m; [93;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)
                                                                                                                                         part No.1 out of 1 parts
    
    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    class FixSigMeta(type):                                                                                                                                 (0)
        "A metaclass that fixes the signature on classes that override `__new__`"===========================================================================(1)
    [91;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance; then check whether its class instance has its own __init__;if so, remove self from the sig of __init__; then assign this new sig to __signature__ for the class instance;[0m
        def __new__(cls, name, bases, dict):                                                                                                                (2)
            res = super().__new__(cls, name, bases, dict)                                                                                                   (3)



```
fdbF.snoop()
```

    23:01:29.15 >>> Call to FixSigMeta.__new__ in File "/tmp/FixSigMeta.py", line 5
    23:01:29.15 .......... cls = <class '__main__.BaseMeta'>
    23:01:29.15 .......... name = 'Foo_call_fix'
    23:01:29.15 .......... bases = ()
    23:01:29.15 .......... dict = {'__module__': '__main__', '__qualname__': 'Foo_call_fix', '__init__': <function Foo_call_fix.__init__>}
    23:01:29.15 .......... len(dict) = 3
    23:01:29.15 .......... __class__ = <class 'fastcore.meta.FixSigMeta'>
    23:01:29.15    5 |     def __new__(cls, name, bases, dict):
    23:01:29.15    6 |         res = super().__new__(cls, name, bases, dict)
    23:01:29.15 .............. res = <class '__main__.Foo_call_fix'>
    23:01:29.15    7 |         if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__))
    23:01:29.15    8 |         return res
    23:01:29.15 <<< Return value from FixSigMeta.__new__: <class '__main__.Foo_call_fix'>


    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    [0m     
    
    <Signature (d, e, f)>



```
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

    =================================================     Investigating [91;1m_signature_from_callable[0m     =================================================
    ===============================================================     on line [91;1m29[0m     ===============================================================
         with example [91;1m
    class Foo_init:
        def __init__(self, a, b, c): pass
    
    pprint(inspect._signature_from_callable(Foo_init, sigcls=inspect.Signature))
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        # Was this function wrapped by a decorator?                                                                                                         (27)
        if follow_wrapper_chains:                                                                                                                           (28)
            obj = unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))=================================================================================(29)
                                                                                                                  [91;1mHow to check whether a class has __signature__?[0m
            if isinstance(obj, types.MethodType):                                                                                                           (30)
                # If the unwrapped object is a *method*, we might want to                                                                                   (31)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                                           hasattr(obj, '__signature__') => hasattr(obj, '__signature__') : False
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (a, b, c)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
            sig = _get_signature_of(obj.__func__)=============================================(20)      
                                                                                                                                                            (21)
            if skip_bound_arg:================================================================(22)      
                return _signature_bound_method(sig)===========================================(23)      
            else:=============================================================================(24)      
                return sig====================================================================(25)      
                                                                                                                                                            (26)
        # Was this function wrapped by a decorator?===========================================(27)      
        if follow_wrapper_chains:=============================================================(28)      
            obj = unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))===================(29) # [93;1mHow to check whether a class has __signature__?[0m; 
            if isinstance(obj, types.MethodType):=============================================(30)      
                # If the unwrapped object is a *method*, we might want to=====================(31)      
                # skip its first parameter (self).============================================(32)      
                # See test_signature_wrapped_bound_method for details.========================(33)      
                return _get_signature_of(obj)=================================================(34)      
                                                                                                                                                            (35)
        try:==================================================================================(36)      
            sig = obj.__signature__===========================================================(37)      
        except AttributeError:================================================================(38)      
            pass==============================================================================(39)      
                                                                                                                                        part No.2 out of 10 parts
    
    =================================================     Investigating [91;1m_signature_from_callable[0m     =================================================
    ===============================================================     on line [91;1m82[0m     ===============================================================
         with example [91;1m
    class Foo_init:
        def __init__(self, a, b, c): pass
    
    pprint(inspect._signature_from_callable(Foo_init, sigcls=inspect.Signature))
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
                                            skip_bound_arg=skip_bound_arg)                                                                                  (80)
                                                                                                                                                            (81)
        if _signature_is_builtin(obj):======================================================================================================================(82)
                                                                                                            [91;1mhow to check whether obj whose signature is builtins;[0m
            return _signature_from_builtin(sigcls, obj,                                                                                                     (83)
                                           skip_bound_arg=skip_bound_arg)                                                                                   (84)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
    inspect.getdoc(_signature_is_builtin) => inspect.getdoc(_signature_is_builtin) : Private helper to test if `obj` is a callable that might
    support Argument Clinic's __text_signature__ protocol.
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (a, b, c)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
                                            skip_bound_arg=skip_bound_arg)====================(80)      
                                                                                                                                                            (81)
        if _signature_is_builtin(obj):========================================================(82) # [91;1mhow to check whether obj whose signature is builtins;[0m; 
            return _signature_from_builtin(sigcls, obj,=======================================(83)      
                                           skip_bound_arg=skip_bound_arg)=====================(84)      
                                                                                                                                                            (85)
        if isinstance(obj, functools.partial):================================================(86)      
            wrapped_sig = _get_signature_of(obj.func)=========================================(87)      
            return _signature_get_partial(wrapped_sig, obj)===================================(88)      
                                                                                                                                                            (89)
        sig = None============================================================================(90)      
        if isinstance(obj, type):=============================================================(91)      
            # obj is a class or a metaclass===================================================(92)      
                                                                                                                                                            (93)
            # First, let's see if it has an overloaded __call__ defined=======================(94)      
            # in its metaclass================================================================(95)      
            call = _signature_get_user_defined_method(type(obj), '__call__')==================(96)      
            if call is not None:==============================================================(97)      
                sig = _get_signature_of(call)=================================================(98)      
            else:=============================================================================(99)      
                                                                                                                                        part No.5 out of 10 parts
    
    =================================================     Investigating [91;1m_signature_from_callable[0m     =================================================
    ===============================================================     on line [91;1m7[0m     ================================================================
         with example [91;1m
    class Foo_init:
        def __init__(self, a, b, c): pass
    
    pprint(inspect._signature_from_callable(Foo_init, sigcls=inspect.Signature))
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        """Private helper function to get signature for arbitrary                                                                                           (5)
        callable objects.                                                                                                                                   (6)
        """=================================================================================================================================================(7)
    [91;1minspect.signature is calling inspect._signature_from_callable; create _get_signature_of using functools.partial to call on _signature_from_callable itself;obj is first tested for callable; then test obj for classmethod; then unwrap to the end unless obj has __signature__;if obj has __signature__, assign __signature__ to sig; then test obj for function, is true calling _signature_from_function; then test obj whose signature is builtins or not; test whether obj created by functools.partial; test obj is a class or not; if obj is a class, then check obj has its own __call__ first; then its own __new__; then its own __init__; then inherited __new__; finally inherited __init__; and then get sig from either of them by calling _get_signature_of on them; FixSigMeta assigns __init__ function to __signature__ attr for the instance class it creates; so that class with FixSigMeta as metaclass can have sig from __init__ through __signature__; no more worry about interference of sig from __call__ or __new__.[0m
                                                                                                                                                            (8)
        _get_signature_of = functools.partial(_signature_from_callable,                                                                                     (9)



```
fdb.snoop()
```

    23:01:29.21 >>> Call to _signature_from_callable in File "/tmp/_signature_from_callable.py", line 3
    23:01:29.21 ...... obj = <class '__main__.Foo_init'>
    23:01:29.21 ...... follow_wrapper_chains = True
    23:01:29.21 ...... skip_bound_arg = True
    23:01:29.21 ...... sigcls = <class 'inspect.Signature'>
    23:01:29.21    3 | def _signature_from_callable(obj, *,
    23:01:29.21   12 |     _get_signature_of = functools.partial(_signature_from_callable,
    23:01:29.21   13 |                                 follow_wrapper_chains=follow_wrapper_chains,
    23:01:29.21   14 |                                 skip_bound_arg=skip_bound_arg,
    23:01:29.21   15 |                                 sigcls=sigcls)
    23:01:29.21   12 |     _get_signature_of = functools.partial(_signature_from_callable,
    23:01:29.21 .......... _get_signature_of = functools.partial(<function _signature_from_call...und_arg=True, sigcls=<class 'inspect.Signature'>)
    23:01:29.21   17 |     if not callable(obj):
    23:01:29.22   20 |     if isinstance(obj, types.MethodType):
    23:01:29.22   31 |     if follow_wrapper_chains:
    23:01:29.22   32 |         obj = unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))
    23:01:29.22   33 |         if isinstance(obj, types.MethodType):
    23:01:29.22   39 |     try:
    23:01:29.22   40 |         sig = obj.__signature__


    =================================================     Investigating [91;1m_signature_from_callable[0m     =================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class Foo_init:
        def __init__(self, a, b, c): pass
    
    pprint(inspect._signature_from_callable(Foo_init, sigcls=inspect.Signature))
    [0m     
    


    23:01:29.35 !!! AttributeError: type object 'Foo_init' has no attribute '__signature__'
    23:01:29.35 !!! When getting attribute: obj.__signature__
    23:01:29.35   41 |     except AttributeError:
    23:01:29.35   42 |         pass
    23:01:29.35   51 |     try:
    23:01:29.35   52 |         partialmethod = obj._partialmethod
    23:01:29.36 !!! AttributeError: type object 'Foo_init' has no attribute '_partialmethod'
    23:01:29.36 !!! When getting attribute: obj._partialmethod
    23:01:29.36   53 |     except AttributeError:
    23:01:29.36   54 |         pass
    23:01:29.36   79 |     if isfunction(obj) or _signature_is_functionlike(obj):
    23:01:29.36   85 |     if _signature_is_builtin(obj):
    23:01:29.36   89 |     if isinstance(obj, functools.partial):
    23:01:29.36   93 |     sig = None
    23:01:29.36   94 |     if isinstance(obj, type):
    23:01:29.36   99 |         call = _signature_get_user_defined_method(type(obj), '__call__')
    23:01:29.36 .............. call = None
    23:01:29.36  100 |         if call is not None:
    23:01:29.36  103 |             factory_method = None
    23:01:29.36  104 |             new = _signature_get_user_defined_method(obj, '__new__')
    23:01:29.36 .................. new = None
    23:01:29.36  105 |             init = _signature_get_user_defined_method(obj, '__init__')
    23:01:29.36 .................. init = <function Foo_init.__init__>
    23:01:29.36  107 |             if '__new__' in obj.__dict__:
    23:01:29.36  110 |             elif '__init__' in obj.__dict__:
    23:01:29.36  111 |                 factory_method = init
    23:01:29.36 ...................... factory_method = <function Foo_init.__init__>
    23:01:29.36  118 |             if factory_method is not None:
    23:01:29.36  119 |                 sig = _get_signature_of(factory_method)
    23:01:29.36 ...................... sig = <Signature (self, a, b, c)>
    23:01:29.36  121 |         if sig is None:
    23:01:29.36  170 |     if sig is not None:
    23:01:29.36  173 |         if skip_bound_arg:
    23:01:29.36  174 |             return _signature_bound_method(sig)
    23:01:29.36 <<< Return value from _signature_from_callable: <Signature (a, b, c)>


    <Signature (a, b, c)>



```
fdbF.print()
```

    ========================================================     Investigating [91;1mFixSigMeta[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    [0m     
    
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [92;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance[0m; [37;1mthen check whether its class instance has its own __init__;if so, remove self from the sig of __init__[0m; [35;1mthen assign this new sig to __signature__ for the class instance;[0m; 
        def __new__(cls, name, bases, dict):==================================================(2) # [91;1mhow does a metaclass create a class instance[0m; [92;1mwhat does super().__new__() do here;[0m; 
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [37;1mhow to remove self from a signature[0m; [36;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)



```
fdb.print(30, 1)
```

    def _signature_from_callable(obj, *,======================================================(0)       
                                 follow_wrapper_chains=True,==================================(1)       
                                 skip_bound_arg=True,=========================================(2)       
                                 sigcls):=====================================================(3)       
                                                                                                                                                            (4)
        """Private helper function to get signature for arbitrary=============================(5)       
        callable objects.=====================================================================(6)       
        """===================================================================================(7) # [35;1minspect.signature is calling inspect._signature_from_callable[0m; [34;1mcreate _get_signature_of using functools.partial to call on _signature_from_callable itself;obj is first tested for callable[0m; [93;1mthen test obj for classmethod[0m; [37;1mthen unwrap to the end unless obj has __signature__;if obj has __signature__, assign __signature__ to sig[0m; [37;1mthen test obj for function, is true calling _signature_from_function[0m; [36;1mthen test obj whose signature is builtins or not[0m; [37;1mtest whether obj created by functools.partial[0m; [37;1mtest obj is a class or not[0m; [35;1mif obj is a class, then check obj has its own __call__ first[0m; [91;1mthen its own __new__[0m; [37;1mthen its own __init__[0m; [37;1mthen inherited __new__[0m; [36;1mfinally inherited __init__[0m; [35;1mand then get sig from either of them by calling _get_signature_of on them[0m; [92;1mFixSigMeta assigns __init__ function to __signature__ attr for the instance class it creates[0m; [91;1mso that class with FixSigMeta as metaclass can have sig from __init__ through __signature__[0m; [92;1mno more worry about interference of sig from __call__ or __new__.[0m; 
                                                                                                                                                            (8)
        _get_signature_of = functools.partial(_signature_from_callable,=======================(9)       
                                    follow_wrapper_chains=follow_wrapper_chains,==============(10)      
                                    skip_bound_arg=skip_bound_arg,============================(11)      
                                    sigcls=sigcls)============================================(12)      
                                                                                                                                                            (13)
        if not callable(obj):=================================================================(14)      
            raise TypeError('{!r} is not a callable object'.format(obj))======================(15)      
                                                                                                                                                            (16)
        if isinstance(obj, types.MethodType):=================================================(17)      
            # In this case we skip the first parameter of the underlying======================(18)      
            # function (usually `self` or `cls`).=============================================(19)      
            sig = _get_signature_of(obj.__func__)=============================================(20)      
                                                                                                                                                            (21)
            if skip_bound_arg:================================================================(22)      
                return _signature_bound_method(sig)===========================================(23)      
            else:=============================================================================(24)      
                return sig====================================================================(25)      
                                                                                                                                                            (26)
        # Was this function wrapped by a decorator?===========================================(27)      
        if follow_wrapper_chains:=============================================================(28)      
            obj = unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))===================(29) # [93;1mHow to check whether a class has __signature__?[0m; 
                                                                                                                                         part No.1 out of 7 parts



```
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/Demos/_signature_from_callable_with_FixSigMeta.ipynb
!mv /Users/Natsume/Documents/fastdebug/Demos/_signature_from_callable_with_FixSigMeta.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/

!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```

    [jupytext] Reading /Users/Natsume/Documents/fastdebug/Demos/_signature_from_callable_with_FixSigMeta.ipynb in format ipynb
    Traceback (most recent call last):
      File "/Users/Natsume/mambaforge/bin/jupytext", line 10, in <module>
        sys.exit(jupytext())
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/cli.py", line 488, in jupytext
        exit_code += jupytext_single_file(nb_file, args, log)
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/cli.py", line 552, in jupytext_single_file
        notebook = read(nb_file, fmt=fmt, config=config)
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/jupytext.py", line 411, in read
        with open(fp, encoding="utf-8") as stream:
    FileNotFoundError: [Errno 2] No such file or directory: '/Users/Natsume/Documents/fastdebug/Demos/_signature_from_callable_with_FixSigMeta.ipynb'
    mv: rename /Users/Natsume/Documents/fastdebug/Demos/_signature_from_callable_with_FixSigMeta.md to /Users/Natsume/Documents/divefastai/Debuggable/jupytext/_signature_from_callable_with_FixSigMeta.md: No such file or directory
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/index.ipynb to markdown
    [NbConvertApp] Writing 58088 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/index.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0001_fastcore_meta_delegates.ipynb to markdown
    [NbConvertApp] Writing 75603 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0001_fastcore_meta_delegates.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0005_fastcore.meta.test_sig.ipynb to markdown
    [NbConvertApp] Writing 10361 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0005_fastcore.meta.test_sig.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0002_signature_from_callable_with_FixSigMeta.ipynb to markdown
    [NbConvertApp] Writing 47125 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0002_signature_from_callable_with_FixSigMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0000_tour.ipynb to markdown
    [NbConvertApp] Writing 12617 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0000_tour.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0007_fastcore.meta.BypassNewMeta.ipynb to markdown
    [NbConvertApp] Writing 30610 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0007_fastcore.meta.BypassNewMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0008_use_kwargs_dict.ipynb to markdown
    [NbConvertApp] Writing 56914 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0008_use_kwargs_dict.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0004_fastcore.meta._rm_self.ipynb to markdown
    [NbConvertApp] Writing 16058 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0004_fastcore.meta._rm_self.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0009_funcs_kwargs.ipynb to markdown
    [NbConvertApp] Writing 68929 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0009_funcs_kwargs.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0010_fastcore_meta_summary.ipynb to markdown
    [NbConvertApp] Writing 17132 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0010_fastcore_meta_summary.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0006_fastcore.meta.NewChkMeta.ipynb to markdown
    [NbConvertApp] Writing 30938 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0006_fastcore.meta.NewChkMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb to markdown
    [NbConvertApp] Writing 73586 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0011_Fastdb.ipynb to markdown
    [NbConvertApp] Writing 11924 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0011_Fastdb.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/questions/00_question_anno_dict.ipynb to markdown
    [NbConvertApp] Writing 11779 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_question_anno_dict.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/lib/utils.ipynb to markdown
    [NbConvertApp] Writing 24973 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/utils.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/lib/00_core.ipynb to markdown
    [NbConvertApp] Writing 411001 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_core.md

