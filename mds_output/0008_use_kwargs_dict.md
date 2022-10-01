# 08_use_kwargs_dict

## Imports


```
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```


<style>.container { width:100% !important; }</style>


## Reading official docs


```
from fastcore.meta import _mk_param # not included in __all__
```

## empty2none


```
fdbe = Fastdb(empty2none)
fdbe.docsrc(0, "p is the Parameter.default value")
fdbe.docsrc(1, "to use empty2none, I need to make sure p is not a parameter, but parameter.default")
fdbe.docsrc(2, "how to check whether a parameter default value is empty")
```

    ========================================================     Investigating [91;1mempty2none[0m     ========================================================
    ===============================================================     on line [91;1m0[0m     ================================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    [93;1mprint selected srcline with expands below[0m--------
    def empty2none(p):======================================================================================================================================(0)
                                                                                                                                 [91;1mp is the Parameter.default value[0m
        "Replace `Parameter.empty` with `None`"                                                                                                             (1)
        return None if p==inspect.Parameter.empty else p                                                                                                    (2)
    ========================================================     Investigating [91;1mempty2none[0m     ========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
    =============================================================     with example [91;1m
    [0m     =============================================================
    
    [93;1mprint selected srcline with expands below[0m--------
    def empty2none(p):                                                                                                                                      (0)
        "Replace `Parameter.empty` with `None`"=============================================================================================================(1)
                                                                               [91;1mto use empty2none, I need to make sure p is not a parameter, but parameter.default[0m
        return None if p==inspect.Parameter.empty else p                                                                                                    (2)
                                                                                                                                                            (3)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def empty2none(p):========================================================================(0) # [36;1mp is the Parameter.default value[0m; 
        "Replace `Parameter.empty` with `None`"===============================================(1) # [36;1mto use empty2none, I need to make sure p is not a parameter, but parameter.default[0m; 
        return None if p==inspect.Parameter.empty else p======================================(2)       
                                                                                                                                                            (3)
                                                                                                                                         part No.1 out of 1 parts
    
    ========================================================     Investigating [91;1mempty2none[0m     ========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
    ============================================================     with example [91;1m
    
    [0m     =============================================================
    
    [93;1mprint selected srcline with expands below[0m--------
    def empty2none(p):                                                                                                                                      (0)
        "Replace `Parameter.empty` with `None`"                                                                                                             (1)
        return None if p==inspect.Parameter.empty else p====================================================================================================(2)
                                                                                                          [91;1mhow to check whether a parameter default value is empty[0m
                                                                                                                                                            (3)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def empty2none(p):========================================================================(0) # [93;1mp is the Parameter.default value[0m; 
        "Replace `Parameter.empty` with `None`"===============================================(1) # [37;1mto use empty2none, I need to make sure p is not a parameter, but parameter.default[0m; 
        return None if p==inspect.Parameter.empty else p======================================(2) # [36;1mhow to check whether a parameter default value is empty[0m; 
                                                                                                                                                            (3)
                                                                                                                                         part No.1 out of 1 parts
    



```
# def foo(a, b=1): pass
# sig = inspect.signature(foo)
# print(sig.parameters.items())
# for k,v in sig.parameters.items():
#     print(f'{k} : {v.default} => empty2none => {empty2none(v.default)}')
```


```
fdbe.eg = """
def foo(a, b=1): pass
sig = inspect.signature(foo)
print(sig.parameters.items())
for k,v in sig.parameters.items():
    print(f'{k} : {v.default} => empty2none => {empty2none(v.default)}')
"""
```


```
fdbe.snoop()
```

    22:14:40.57 >>> Call to empty2none in File "/tmp/empty2none.py", line 3
    22:14:40.57 ...... p = <class 'inspect._empty'>
    22:14:40.57    3 | def empty2none(p):
    22:14:40.57    5 |     return None if p==inspect.Parameter.empty else p
    22:14:40.57 <<< Return value from empty2none: None
    22:14:40.57 >>> Call to empty2none in File "/tmp/empty2none.py", line 3
    22:14:40.57 ...... p = 1
    22:14:40.57    3 | def empty2none(p):
    22:14:40.57    5 |     return None if p==inspect.Parameter.empty else p
    22:14:40.57 <<< Return value from empty2none: 1


    ========================================================     Investigating [91;1mempty2none[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    def foo(a, b=1): pass
    sig = inspect.signature(foo)
    print(sig.parameters.items())
    for k,v in sig.parameters.items():
        print(f'{k} : {v.default} => empty2none => {empty2none(v.default)}')
    [0m     
    
    odict_items([('a', <Parameter "a">), ('b', <Parameter "b=1">)])
    a : <class 'inspect._empty'> => empty2none => None
    b : 1 => empty2none => 1


## `_mk_param`


```
fdb = Fastdb(_mk_param)
fdb.print()
```

    ========================================================     Investigating [91;1m_mk_param[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    def _mk_param(n,d=None): return inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=d)                                                         (0)
                                                                                                                                                            (1)



```
fdb.eg = """
print(_mk_param("a", 1))
"""
```


```
fdb.snoop()
```

    22:14:40.58 >>> Call to _mk_param in File "/tmp/_mk_param.py", line 3
    22:14:40.58 ...... n = 'a'
    22:14:40.58 ...... d = 1
    22:14:40.58    3 | def _mk_param(n,d=None): return inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=d)
    22:14:40.58    3 | def _mk_param(n,d=None): return inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=d)
    22:14:40.58 <<< Return value from _mk_param: <Parameter "a=1">


    ========================================================     Investigating [91;1m_mk_param[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    ================================================     with example [91;1m
    print(_mk_param("a", 1))
    [0m     =================================================
    
    a=1



```
fdb.docsrc(0, "_mk_param is to create a new parameter as KEYWORD_ONLY kind; n is its name in string; d is its default value")
```

    ========================================================     Investigating [91;1m_mk_param[0m     =========================================================
    ===============================================================     on line [91;1m0[0m     ================================================================
    ================================================     with example [91;1m
    print(_mk_param("a", 1))
    [0m     =================================================
    
    [93;1mprint selected srcline with expands below[0m--------
    def _mk_param(n,d=None): return inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=d)=========================================================(0)
                                                     [91;1m_mk_param is to create a new parameter as KEYWORD_ONLY kind; n is its name in string; d is its default value[0m
                                                                                                                                                            (1)
    a=1


## use_kwargs_dict

### Reading docs

Replace all **kwargs with named arguments like so:
```python
@use_kwargs_dict(y=1,z=None)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None)')
```
Add named arguments, but optionally keep **kwargs by setting keep=True:
```python
@use_kwargs_dict(y=1,z=None, keep=True)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
```


```
print(inspect.getsource(use_kwargs_dict))
```

    def use_kwargs_dict(keep=False, **kwargs):
        "Decorator: replace `**kwargs` in signature with `names` params"
        def _f(f):
            sig = inspect.signature(f)
            sigd = dict(sig.parameters)
            k = sigd.pop('kwargs')
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}
            sigd.update(s2)
            if keep: sigd['kwargs'] = k
            f.__signature__ = sig.replace(parameters=sigd.values())
            return f
        return _f
    



```
fdb = Fastdb(use_kwargs_dict)
fdb.eg = """
@use_kwargs_dict(y=1,z=None)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None)')
"""

fdb.eg = """
@use_kwargs_dict(y=1,z=None, keep=True)
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
"""
```


```
fdb.print()
```

    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1)       
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3)       
            sigd = dict(sig.parameters)=======================================================(4)       
            k = sigd.pop('kwargs')============================================================(5)       
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6)       
            sigd.update(s2)===================================================================(7)       
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)



```
fdb.docsrc(1, "how to use use_kwargs_dict; use_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict; \
f's signature is saved inside f.__signature__")
fdb.docsrc(3, "how to get the signature from an object")
fdb.docsrc(4, "how to get all parameters of a signature; how to make it into a dict; ")
fdb.docsrc(5, "how to pop out an item from a dict")
fdb.docsrc(6, "how to create a dict of params based on a dict")
fdb.docsrc(7, "how to udpate one dict into another dict")
fdb.docsrc(8, "how to create a new item in a dict")
fdb.docsrc(9, "how to update a signature with a new set of parameters in the form of a dict values")
```

    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def use_kwargs_dict(keep=False, **kwargs):                                                                                                              (0)
        "Decorator: replace `**kwargs` in signature with `names` params"====================================================================================(1)
    [91;1mhow to use use_kwargs_dict; use_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict; f's signature is saved inside f.__signature__[0m
        def _f(f):                                                                                                                                          (2)
            sig = inspect.signature(f)                                                                                                                      (3)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [34;1mhow to use use_kwargs_dict[0m; [92;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [92;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3)       
            sigd = dict(sig.parameters)=======================================================(4)       
            k = sigd.pop('kwargs')============================================================(5)       
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6)       
            sigd.update(s2)===================================================================(7)       
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    
    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        "Decorator: replace `**kwargs` in signature with `names` params"                                                                                    (1)
        def _f(f):                                                                                                                                          (2)
            sig = inspect.signature(f)======================================================================================================================(3)
                                                                                                                          [91;1mhow to get the signature from an object[0m
            sigd = dict(sig.parameters)                                                                                                                     (4)
            k = sigd.pop('kwargs')                                                                                                                          (5)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [34;1mhow to use use_kwargs_dict[0m; [35;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [34;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3) # [93;1mhow to get the signature from an object[0m; 
            sigd = dict(sig.parameters)=======================================================(4)       
            k = sigd.pop('kwargs')============================================================(5)       
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6)       
            sigd.update(s2)===================================================================(7)       
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    
    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m4[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        def _f(f):                                                                                                                                          (2)
            sig = inspect.signature(f)                                                                                                                      (3)
            sigd = dict(sig.parameters)=====================================================================================================================(4)
                                                                                           [91;1mhow to get all parameters of a signature; how to make it into a dict; [0m
            k = sigd.pop('kwargs')                                                                                                                          (5)
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}                                                                              (6)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [92;1mhow to use use_kwargs_dict[0m; [36;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [37;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3) # [35;1mhow to get the signature from an object[0m; 
            sigd = dict(sig.parameters)=======================================================(4) # [37;1mhow to get all parameters of a signature[0m; [36;1mhow to make it into a dict[0m; [34;1m[0m; 
            k = sigd.pop('kwargs')============================================================(5)       
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6)       
            sigd.update(s2)===================================================================(7)       
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    
    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m5[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            sig = inspect.signature(f)                                                                                                                      (3)
            sigd = dict(sig.parameters)                                                                                                                     (4)
            k = sigd.pop('kwargs')==========================================================================================================================(5)
                                                                                                                               [91;1mhow to pop out an item from a dict[0m
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}                                                                              (6)
            sigd.update(s2)                                                                                                                                 (7)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [93;1mhow to use use_kwargs_dict[0m; [36;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [92;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3) # [35;1mhow to get the signature from an object[0m; 
            sigd = dict(sig.parameters)=======================================================(4) # [35;1mhow to get all parameters of a signature[0m; [93;1mhow to make it into a dict[0m; [37;1m[0m; 
            k = sigd.pop('kwargs')============================================================(5) # [36;1mhow to pop out an item from a dict[0m; 
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6)       
            sigd.update(s2)===================================================================(7)       
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    
    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m6[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            sigd = dict(sig.parameters)                                                                                                                     (4)
            k = sigd.pop('kwargs')                                                                                                                          (5)
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}==============================================================================(6)
                                                                                                                   [91;1mhow to create a dict of params based on a dict[0m
            sigd.update(s2)                                                                                                                                 (7)
            if keep: sigd['kwargs'] = k                                                                                                                     (8)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [36;1mhow to use use_kwargs_dict[0m; [35;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [93;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3) # [35;1mhow to get the signature from an object[0m; 
            sigd = dict(sig.parameters)=======================================================(4) # [92;1mhow to get all parameters of a signature[0m; [35;1mhow to make it into a dict[0m; [35;1m[0m; 
            k = sigd.pop('kwargs')============================================================(5) # [37;1mhow to pop out an item from a dict[0m; 
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6) # [36;1mhow to create a dict of params based on a dict[0m; 
            sigd.update(s2)===================================================================(7)       
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    
    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m7[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            k = sigd.pop('kwargs')                                                                                                                          (5)
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}                                                                              (6)
            sigd.update(s2)=================================================================================================================================(7)
                                                                                                                         [91;1mhow to udpate one dict into another dict[0m
            if keep: sigd['kwargs'] = k                                                                                                                     (8)
            f.__signature__ = sig.replace(parameters=sigd.values())                                                                                         (9)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [34;1mhow to use use_kwargs_dict[0m; [92;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [34;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3) # [91;1mhow to get the signature from an object[0m; 
            sigd = dict(sig.parameters)=======================================================(4) # [37;1mhow to get all parameters of a signature[0m; [35;1mhow to make it into a dict[0m; [92;1m[0m; 
            k = sigd.pop('kwargs')============================================================(5) # [36;1mhow to pop out an item from a dict[0m; 
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6) # [36;1mhow to create a dict of params based on a dict[0m; 
            sigd.update(s2)===================================================================(7) # [91;1mhow to udpate one dict into another dict[0m; 
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    
    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m8[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}                                                                              (6)
            sigd.update(s2)                                                                                                                                 (7)
            if keep: sigd['kwargs'] = k=====================================================================================================================(8)
                                                                                                                               [91;1mhow to create a new item in a dict[0m
            f.__signature__ = sig.replace(parameters=sigd.values())                                                                                         (9)
            return f                                                                                                                                        (10)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [37;1mhow to use use_kwargs_dict[0m; [92;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [36;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3) # [93;1mhow to get the signature from an object[0m; 
            sigd = dict(sig.parameters)=======================================================(4) # [91;1mhow to get all parameters of a signature[0m; [34;1mhow to make it into a dict[0m; [34;1m[0m; 
            k = sigd.pop('kwargs')============================================================(5) # [91;1mhow to pop out an item from a dict[0m; 
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6) # [36;1mhow to create a dict of params based on a dict[0m; 
            sigd.update(s2)===================================================================(7) # [35;1mhow to udpate one dict into another dict[0m; 
            if keep: sigd['kwargs'] = k=======================================================(8) # [34;1mhow to create a new item in a dict[0m; 
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    
    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ===============================================================     on line [91;1m9[0m     ================================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            sigd.update(s2)                                                                                                                                 (7)
            if keep: sigd['kwargs'] = k                                                                                                                     (8)
            f.__signature__ = sig.replace(parameters=sigd.values())=========================================================================================(9)
                                                                              [91;1mhow to update a signature with a new set of parameters in the form of a dict values[0m
            return f                                                                                                                                        (10)
        return _f                                                                                                                                           (11)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def use_kwargs_dict(keep=False, **kwargs):================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1) # [91;1mhow to use use_kwargs_dict[0m; [92;1muse_kwargs_dict is to replace **kwargs with newly created KEYWORD_ONLY params based on a dict[0m; [34;1mf's signature is saved inside f.__signature__[0m; 
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3) # [93;1mhow to get the signature from an object[0m; 
            sigd = dict(sig.parameters)=======================================================(4) # [93;1mhow to get all parameters of a signature[0m; [35;1mhow to make it into a dict[0m; [93;1m[0m; 
            k = sigd.pop('kwargs')============================================================(5) # [35;1mhow to pop out an item from a dict[0m; 
            s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}================(6) # [36;1mhow to create a dict of params based on a dict[0m; 
            sigd.update(s2)===================================================================(7) # [37;1mhow to udpate one dict into another dict[0m; 
            if keep: sigd['kwargs'] = k=======================================================(8) # [35;1mhow to create a new item in a dict[0m; 
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9) # [91;1mhow to update a signature with a new set of parameters in the form of a dict values[0m; 
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)
                                                                                                                                         part No.1 out of 1 parts
    



```
fdb.snoop(deco=True) # how to use snoop on decorator
```

    22:14:40.62 >>> Call to use_kwargs_dict in File "/tmp/use_kwargs_dict.py", line 3
    22:14:40.62 ...... keep = True
    22:14:40.62 ...... kwargs = {'y': 1, 'z': None}
    22:14:40.62 ...... len(kwargs) = 2
    22:14:40.62    3 | def use_kwargs_dict(keep=False, **kwargs):
    22:14:40.62    5 |     import snoop
    22:14:40.62 .......... snoop = <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>
    22:14:40.62    6 |     @snoop
    22:14:40.62    7 |     def _f(f):
    22:14:40.62 .......... _f = <function use_kwargs_dict.<locals>._f>
    22:14:40.62   16 |     return _f
    22:14:40.62 <<< Return value from use_kwargs_dict: <function use_kwargs_dict.<locals>._f>
    22:14:40.62 >>> Call to use_kwargs_dict.<locals>._f in File "/tmp/use_kwargs_dict.py", line 7
    22:14:40.62 .......... f = <function foo>
    22:14:40.62 .......... keep = True
    22:14:40.62 .......... kwargs = {'y': 1, 'z': None}
    22:14:40.62 .......... len(kwargs) = 2
    22:14:40.62    7 |     def _f(f):
    22:14:40.62    8 |         sig = inspect.signature(f)
    22:14:40.62 .............. sig = <Signature (a, b=1, **kwargs)>
    22:14:40.62    9 |         sigd = dict(sig.parameters)
    22:14:40.62 .............. sigd = {'a': <Parameter "a">, 'b': <Parameter "b=1">, 'kwargs': <Parameter "**kwargs">}
    22:14:40.62 .............. len(sigd) = 3
    22:14:40.62   10 |         k = sigd.pop('kwargs')
    22:14:40.62 .............. k = <Parameter "**kwargs">
    22:14:40.62 .............. sigd = {'a': <Parameter "a">, 'b': <Parameter "b=1">}
    22:14:40.62 .............. len(sigd) = 2
    22:14:40.62   11 |         s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}
        22:14:40.62 Dict comprehension:
        22:14:40.62   11 |         s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}
        22:14:40.62 .............. Iterating over <dict_itemiterator object>
        22:14:40.62 .............. Values of sigd: {'a': <Parameter "a">, 'b': <Parameter "b=1">}
        22:14:40.62 .............. Values of len(sigd): 2
        22:14:40.62 .............. Values of n: 'y', 'z'
        22:14:40.62 .............. Values of d: 1, None
        22:14:40.62 Result: {'y': <Parameter "y=1">, 'z': <Parameter "z=None">}
    22:14:40.62   11 |         s2 = {n:_mk_param(n,d) for n,d in kwargs.items() if n not in sigd}
    22:14:40.62 .............. s2 = {'y': <Parameter "y=1">, 'z': <Parameter "z=None">}
    22:14:40.62 .............. len(s2) = 2
    22:14:40.62   12 |         sigd.update(s2)
    22:14:40.62 .............. sigd = {'a': <Parameter "a">, 'b': <Parameter "b=1">, 'y': <Parameter "y=1">, 'z': <Parameter "z=None">}
    22:14:40.62 .............. len(sigd) = 4
    22:14:40.62   13 |         if keep: sigd['kwargs'] = k
    22:14:40.62 ...... sigd = {'a': <Parameter "a">, 'b': <Parameter "b=1">, 'y': <Parameter "y=1">, 'z': <Parameter "z=None">, ...}
    22:14:40.62 ...... len(sigd) = 5
    22:14:40.62   14 |         f.__signature__ = sig.replace(parameters=sigd.values())
    22:14:40.63   15 |         return f
    22:14:40.63 <<< Return value from use_kwargs_dict.<locals>._f: <function foo>


    =====================================================     Investigating [91;1muse_kwargs_dict[0m     ======================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    @use_kwargs_dict(y=1,z=None, keep=True)
    def foo(a, b=1, **kwargs): pass
    
    test_sig(foo, '(a, b=1, *, y=1, z=None, **kwargs)')
    [0m     
    


## use_kwargs

### Reading docs

use_kwargs is different than use_kwargs_dict as it only replaces **kwargs with named parameters without any default values:
```python
@use_kwargs(['y', 'z'])
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=None, z=None)')
```
You may optionally keep the **kwargs argument in your signature by setting keep=True:
```python
@use_kwargs(['y', 'z'], keep=True)
def foo(a, *args, b=1, **kwargs): pass
test_sig(foo, '(a, *args, b=1, y=None, z=None, **kwargs)')
```


```
print(inspect.getsource(use_kwargs))
```

    def use_kwargs(names, keep=False):
        "Decorator: replace `**kwargs` in signature with `names` params"
        def _f(f):
            sig = inspect.signature(f)
            sigd = dict(sig.parameters)
            k = sigd.pop('kwargs')
            s2 = {n:_mk_param(n) for n in names if n not in sigd}
            sigd.update(s2)
            if keep: sigd['kwargs'] = k
            f.__signature__ = sig.replace(parameters=sigd.values())
            return f
        return _f
    



```
fdb = Fastdb(use_kwargs)
fdb.eg = """
@use_kwargs(['y', 'z'])
def foo(a, b=1, **kwargs): pass

test_sig(foo, '(a, b=1, *, y=None, z=None)')
"""

fdb.eg = """
@use_kwargs(['y', 'z'], keep=True)
def foo(a, *args, b=1, **kwargs): pass
test_sig(foo, '(a, *args, b=1, y=None, z=None, **kwargs)')
"""
```


```
fdb.print()
```

    ========================================================     Investigating [91;1muse_kwargs[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    @use_kwargs(['y', 'z'], keep=True)
    def foo(a, *args, b=1, **kwargs): pass
    test_sig(foo, '(a, *args, b=1, y=None, z=None, **kwargs)')
    [0m     
    
    def use_kwargs(names, keep=False):========================================================(0)       
        "Decorator: replace `**kwargs` in signature with `names` params"======================(1)       
        def _f(f):============================================================================(2)       
            sig = inspect.signature(f)========================================================(3)       
            sigd = dict(sig.parameters)=======================================================(4)       
            k = sigd.pop('kwargs')============================================================(5)       
            s2 = {n:_mk_param(n) for n in names if n not in sigd}=============================(6)       
            sigd.update(s2)===================================================================(7)       
            if keep: sigd['kwargs'] = k=======================================================(8)       
            f.__signature__ = sig.replace(parameters=sigd.values())===========================(9)       
            return f==========================================================================(10)      
        return _f=============================================================================(11)      
                                                                                                                                                            (12)



```
fdb.docsrc(0, "How to use use_kwargs; use_kwargs has names as a list of strings; all the newly created params have None as default value; f's signature \
is saved inside f.__signature__")
```

    ========================================================     Investigating [91;1muse_kwargs[0m     ========================================================
    ===============================================================     on line [91;1m0[0m     ================================================================
         with example [91;1m
    @use_kwargs(['y', 'z'], keep=True)
    def foo(a, *args, b=1, **kwargs): pass
    test_sig(foo, '(a, *args, b=1, y=None, z=None, **kwargs)')
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def use_kwargs(names, keep=False):======================================================================================================================(0)
    [91;1mHow to use use_kwargs; use_kwargs has names as a list of strings; all the newly created params have None as default value; f's signature is saved inside f.__signature__[0m
        "Decorator: replace `**kwargs` in signature with `names` params"                                                                                    (1)
        def _f(f):                                                                                                                                          (2)



```
fdb.snoop(deco=True)
```

    22:14:40.65 >>> Call to use_kwargs in File "/tmp/use_kwargs.py", line 3
    22:14:40.65 ...... names = ['y', 'z']
    22:14:40.65 ...... len(names) = 2
    22:14:40.65 ...... keep = True
    22:14:40.65    3 | def use_kwargs(names, keep=False):
    22:14:40.65    5 |     import snoop
    22:14:40.65 .......... snoop = <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>
    22:14:40.65    6 |     @snoop
    22:14:40.65    7 |     def _f(f):
    22:14:40.65 .......... _f = <function use_kwargs.<locals>._f>
    22:14:40.65   16 |     return _f
    22:14:40.65 <<< Return value from use_kwargs: <function use_kwargs.<locals>._f>
    22:14:40.65 >>> Call to use_kwargs.<locals>._f in File "/tmp/use_kwargs.py", line 7
    22:14:40.65 .......... f = <function foo>
    22:14:40.65 .......... keep = True
    22:14:40.65 .......... names = ['y', 'z']
    22:14:40.65 .......... len(names) = 2
    22:14:40.65    7 |     def _f(f):
    22:14:40.65    8 |         sig = inspect.signature(f)
    22:14:40.65 .............. sig = <Signature (a, *args, b=1, **kwargs)>
    22:14:40.65    9 |         sigd = dict(sig.parameters)
    22:14:40.65 .............. sigd = {'a': <Parameter "a">, 'args': <Parameter "*args">, 'b': <Parameter "b=1">, 'kwargs': <Parameter "**kwargs">}
    22:14:40.65 .............. len(sigd) = 4
    22:14:40.65   10 |         k = sigd.pop('kwargs')
    22:14:40.65 .............. k = <Parameter "**kwargs">
    22:14:40.65 .............. sigd = {'a': <Parameter "a">, 'args': <Parameter "*args">, 'b': <Parameter "b=1">}
    22:14:40.65 .............. len(sigd) = 3
    22:14:40.65   11 |         s2 = {n:_mk_param(n) for n in names if n not in sigd}
        22:14:40.65 Dict comprehension:
        22:14:40.65   11 |         s2 = {n:_mk_param(n) for n in names if n not in sigd}
        22:14:40.65 .............. Iterating over <list_iterator object>
        22:14:40.65 .............. Values of sigd: {'a': <Parameter "a">, 'args': <Parameter "*args">, 'b': <Parameter "b=1">}
        22:14:40.65 .............. Values of len(sigd): 3
        22:14:40.65 .............. Values of n: 'y', 'z'
        22:14:40.65 Result: {'y': <Parameter "y=None">, 'z': <Parameter "z=None">}
    22:14:40.65   11 |         s2 = {n:_mk_param(n) for n in names if n not in sigd}
    22:14:40.65 .............. s2 = {'y': <Parameter "y=None">, 'z': <Parameter "z=None">}
    22:14:40.65 .............. len(s2) = 2
    22:14:40.65   12 |         sigd.update(s2)
    22:14:40.65 .............. sigd = {'a': <Parameter "a">, 'args': <Parameter "*args">, 'b': <Parameter "b=1">, 'y': <Parameter "y=None">, ...}
    22:14:40.65 .............. len(sigd) = 5
    22:14:40.65   13 |         if keep: sigd['kwargs'] = k
    22:14:40.65 ...... len(sigd) = 6
    22:14:40.65   14 |         f.__signature__ = sig.replace(parameters=sigd.values())
    22:14:40.65   15 |         return f
    22:14:40.65 <<< Return value from use_kwargs.<locals>._f: <function foo>


    ========================================================     Investigating [91;1muse_kwargs[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    @use_kwargs(['y', 'z'], keep=True)
    def foo(a, *args, b=1, **kwargs): pass
    test_sig(foo, '(a, *args, b=1, y=None, z=None, **kwargs)')
    [0m     
    



```

```
