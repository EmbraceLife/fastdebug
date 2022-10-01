# 00_quesolved_anno_dict


```
from fastcore.meta import *
from fastcore.test import *
import inspect
```

## `anno_dict` docs


```
inspect.getdoc(anno_dict)
```




    "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist"



I have to confess I don't undersatnd the docs statement very well. So, I look into the source code of `anno_dict` and `empty2none`.


```
print(inspect.getsource(anno_dict))
```

    def anno_dict(f):
        "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist"
        return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}
    



```
print(inspect.getsource(empty2none))
```

    def empty2none(p):
        "Replace `Parameter.empty` with `None`"
        return None if p==inspect.Parameter.empty else p
    


## Dive in

If a parameter's default value is `Parameter.empty`, then `empty2none` is to replace `Parameter.empty` with `None` . So, I think it is reasonable to assume `p` is primarily used as a parameter's default value. The cell below supports this assumption.


```
def foo(a, b:int=1): pass
sig = inspect.signature(foo)
for k,v in sig.parameters.items():
    print(f'{k} is a parameter {v}, whose default value is {v.default}, \
if apply empty2none to default value, then the default value is {empty2none(v.default)}')
    print(f'{k} is a parameter {v}, whose default value is {v.default}, \
if apply empty2none to parameter, then we get: {empty2none(v)}')
```

    a is a parameter a, whose default value is <class 'inspect._empty'>, if apply empty2none to default value, then the default value is None
    a is a parameter a, whose default value is <class 'inspect._empty'>, if apply empty2none to parameter, then we get: a
    b is a parameter b: int = 1, whose default value is 1, if apply empty2none to default value, then the default value is 1
    b is a parameter b: int = 1, whose default value is 1, if apply empty2none to parameter, then we get: b: int = 1


So, **what is odd** is that in `anno_dict`, `empty2none` is applied to `v` which is not parameter's default value, but mostly classes like `int`, `list` ect, as in `__annotations__`.

Then I experimented the section below and didn't find `anno_dict` doing anything new than `__annotations__`. 



## `anno_dict` seems not add anything new to `__annotations__`


```
def foo(a, b:int=1): pass
test_eq(foo.__annotations__, {'b': int})
test_eq(anno_dict(foo), {'b': int})
def foo(a:bool, b:int=1): pass
test_eq(foo.__annotations__, {'a': bool, 'b': int})
test_eq(anno_dict(foo), {'a': bool, 'b': int})
def foo(a, d:list, b:int=1, c:bool=True): pass
test_eq(foo.__annotations__, {'d': list, 'b': int, 'c': bool})
test_eq(anno_dict(foo), {'d': list, 'b': int, 'c': bool})
```


```
from fastcore.foundation import L
```


```
def foo(a, b): pass
test_eq(foo.__annotations__, {})
test_eq(anno_dict(foo), {})

def _f(a:int, b:L)->str: ...
test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})
```

**Question!** so far above anno_dict has done nothing new or more, so what am I missing here?

## use fastdebug to double check


```
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```


<style>.container { width:100% !important; }</style>



```
fdb = Fastdb(anno_dict)
fdb.eg = """
def foo(a, b): pass
test_eq(foo.__annotations__, {})
test_eq(anno_dict(foo), {})

from fastcore.foundation import L
def _f(a:int, b:L)->str: ...
test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})
"""
```


```
fdb.snoop(['empty2none(v)'])
```

    09:48:27.11 >>> Call to anno_dict in File "/tmp/anno_dict.py", line 3
    09:48:27.11 ...... f = <function foo>
    09:48:27.11    3 | def anno_dict(f):
    09:48:27.11    5 |     return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}
        09:48:27.11 Dict comprehension:
        09:48:27.11    5 |     return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}
        09:48:27.11 .......... Iterating over <dict_itemiterator object>
        09:48:27.11 Result: {}
    09:48:27.11    5 |     return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}
    09:48:27.12 <<< Return value from anno_dict: {}
    09:48:27.12 >>> Call to anno_dict in File "/tmp/anno_dict.py", line 3
    09:48:27.12 ...... f = <function _f>
    09:48:27.12    3 | def anno_dict(f):
    09:48:27.12    5 |     return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}
        09:48:27.12 Dict comprehension:
        09:48:27.12    5 |     return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}
        09:48:27.12 .......... Iterating over <dict_itemiterator object>
        09:48:27.12 .......... Values of k: 'a', 'b', 'return'
        09:48:27.12 .......... Values of v: <class 'int'>, <class 'fastcore.foundation.L'>, <class 'str'>
        09:48:27.12 .......... Values of empty2none(v): <class 'int'>, <class 'fastcore.foundation.L'>, <class 'str'>
        09:48:27.12 Result: {'a': <class 'int'>, 'b': <class 'fastcore.foundation.L'>, 'return': <class 'str'>}
    09:48:27.12    5 |     return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}
    09:48:27.12 <<< Return value from anno_dict: {'a': <class 'int'>, 'b': <class 'fastcore.foundation.L'>, 'return': <class 'str'>}


    ========================================================     Investigating [91;1manno_dict[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    def foo(a, b): pass
    test_eq(foo.__annotations__, {})
    test_eq(anno_dict(foo), {})
    
    from fastcore.foundation import L
    def _f(a:int, b:L)->str: ...
    test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
    test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})
    [0m     
    



```
fdb.docsrc(1, "empty2none works on paramter.default especially when the default is Parameter.empty; anno_dict works on the types \
of params, not the value of params; so it is odd to use empty2none in anno_dict;")
```

    ========================================================     Investigating [91;1manno_dict[0m     =========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
    def foo(a, b): pass
    test_eq(foo.__annotations__, {})
    test_eq(anno_dict(foo), {})
    
    from fastcore.foundation import L
    def _f(a:int, b:L)->str: ...
    test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
    test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def anno_dict(f):                                                                                                                                       (0)
        "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist"==========================================================(1)
    [91;1mempty2none works on paramter.default especially when the default is Parameter.empty; anno_dict works on the types of params, not the value of params; so it is odd to use empty2none in anno_dict;[0m
        return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}                                                                       (2)
                                                                                                                                                            (3)
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def anno_dict(f):=========================================================================(0)       
        "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist" # [93;1mempty2none works on paramter.default especially when the default is Parameter.empty[0m; [35;1manno_dict works on the types of params, not the value of params[0m; [93;1mso it is odd to use empty2none in anno_dict;[0m;  (1)
        return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}=========(2)       
                                                                                                                                                            (3)
                                                                                                                                         part No.1 out of 1 parts
    



```
fdb.print()
```

    ========================================================     Investigating [91;1manno_dict[0m     =========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
    def foo(a, b): pass
    test_eq(foo.__annotations__, {})
    test_eq(anno_dict(foo), {})
    
    from fastcore.foundation import L
    def _f(a:int, b:L)->str: ...
    test_eq(_f.__annotations__, {'a': int, 'b': L, 'return': str})
    test_eq(anno_dict(_f), {'a': int, 'b': L, 'return': str})
    [0m     
    
    def anno_dict(f):=========================================================================(0)       
        "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist" # [93;1mempty2none works on paramter.default especially when the default is Parameter.empty[0m; [37;1manno_dict works on the types of params, not the value of params[0m; [36;1mso it is odd to use empty2none in anno_dict;[0m;  (1)
        return {k:empty2none(v) for k,v in getattr(f, '__annotations__', {}).items()}=========(2)       
                                                                                                                                                            (3)


## Does fastcore want anno_dict to include params with no annos?

If so, I have written a lengthy `anno_dict_maybe` to do it. (can be shorter if needed)


```
def anno_dict_maybe(f):
    "`__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist"
    new_anno = {}
    for k, v in inspect.signature(f).parameters.items():
        if k not in f.__annotations__:
            new_anno[k] = None
        else: 
            new_anno[k] = f.__annotations__[k]
    if 'return' in f.__annotations__:
        new_anno['return'] = f.__annotations__['return']
#     if hasattr(f, '__annotations__'):
    if True in [bool(v) for k,v in new_anno.items()]:
        return new_anno
    else:
        return {}
```


```
def foo(a:int, b, c:bool=True)->str: pass
```


```
test_eq(foo.__annotations__, {'a': int, 'c': bool, 'return': str})
```


```
test_eq(anno_dict(foo), {'a': int, 'c': bool, 'return': str})
```


```
test_eq(anno_dict_maybe(foo), {'a': int, 'b': None, 'c': bool, 'return': str})
```


```
def foo(a, b, c): pass
```


```
test_eq(foo.__annotations__, {})
```


```
test_eq(anno_dict(foo), {})
```


```
test_eq(anno_dict_maybe(foo), {})
```

## Jeremy's response

A supportive and confirmative [response](https://forums.fast.ai/t/help-reading-fastcore-docs/100168/3?u=daniel) from Jeremy on this issue


```

```
