# 05_test_sig

## imports


```
from fastdebug.utils import *
from fastdebug.core import *
```


<style>.container { width:100% !important; }</style>



```
from fastcore.meta import *
import fastcore.meta as fm
```

## setups


```
whatinside(fm, dun=True)
```

    fastcore.meta has: 
    13 items in its __all__, and 
    43 user defined functions, 
    19 classes or class objects, 
    2 builtin funcs and methods, and
    74 callables.
    
    test_sig:            function    Test the signature of an object
    FixSigMeta:          metaclass, type    A metaclass that fixes the signature on classes that override `__new__`
    PrePostInitMeta:     metaclass, type    A metaclass that calls optional `__pre_init__` and `__post_init__` methods
    AutoInit:            class, PrePostInitMeta    Same as `object`, but no need for subclasses to call `super().__init__`
    NewChkMeta:          metaclass, type    Metaclass to avoid recreating object passed to constructor
    BypassNewMeta:       metaclass, type    Metaclass: casts `x` to this class if it's of type `cls._bypass_type`
    empty2none:          function    Replace `Parameter.empty` with `None`
    anno_dict:           function    `__annotation__ dictionary with `empty` cast to `None`, returning empty if doesn't exist
    use_kwargs_dict:     decorator, function    Decorator: replace `**kwargs` in signature with `names` params
    use_kwargs:          decorator, function    Decorator: replace `**kwargs` in signature with `names` params
    delegates:           decorator, function    Decorator: replace `**kwargs` in signature with params from `to`
    method:              function    Mark `f` as a method
    funcs_kwargs:        decorator, function    Replace methods in `cls._methods` with those from `kwargs`



```
g = locals()
fdb = Fastdb(test_sig, outloc=g)
```


```
fdb.print()
```

    =========================================================     Investigating [91;1mtest_sig[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    def test_sig(f, b):=======================================================================(0)       
        "Test the signature of an object"=====================================================(1)       
        test_eq(str(inspect.signature(f)), b)=================================================(2)       
                                                                                                                                                            (3)



```
fdb.eg = """
def func_1(h,i,j): pass
test_sig(func_1, '(h, i, j)')
"""

fdb.eg = """
class T:
    def __init__(self, a, b): pass
test_sig(T, '(a, b)')
"""
fdb.eg = """
def func_2(h,i=3, j=[5,6]): pass
test_sig(func_2, '(h, i=3, j=[5, 6])')
"""
```

## documents


```
fdb.docsrc(2, "test_sig is to test two strings with test_eq; how to turn a signature into a string;", "pprint(inspect.signature(f))", \
"inspect.signature(f)", "str(inspect.signature(f))")
```

    =========================================================     Investigating [91;1mtest_sig[0m     =========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
    =========================     with example [91;1m
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    [0m     =========================
    
    [93;1mprint selected srcline with expands below[0m--------
    def test_sig(f, b):                                                                                                                                     (0)
        "Test the signature of an object"                                                                                                                   (1)
        test_eq(str(inspect.signature(f)), b)===============================================================================================================(2)
                                                                             [91;1mtest_sig is to test two strings with test_eq; how to turn a signature into a string;[0m
                                                                                                                                                            (3)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
    <Signature (h, i=3, j=[5, 6])>
                                                                                              pprint(inspect.signature(f)) => pprint(inspect.signature(f)) : None
    
    
                                                                                                inspect.signature(f) => inspect.signature(f) : (h, i=3, j=[5, 6])
    
    
                                                                                      str(inspect.signature(f)) => str(inspect.signature(f)) : (h, i=3, j=[5, 6])
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def test_sig(f, b):=======================================================================(0)       
        "Test the signature of an object"=====================================================(1)       
        test_eq(str(inspect.signature(f)), b)=================================================(2) # [92;1mtest_sig is to test two strings with test_eq[0m; [36;1mhow to turn a signature into a string;[0m; 
                                                                                                                                                            (3)
                                                                                                                                         part No.1 out of 1 parts
    


## snoop


```
fdb.snoop()
```

    21:28:55.21 >>> Call to test_sig in File "/tmp/test_sig.py", line 3
    21:28:55.21 ...... f = <function func_2>
    21:28:55.21 ...... b = '(h, i=3, j=[5, 6])'
    21:28:55.21    3 | def test_sig(f, b):
    21:28:55.21    5 |     test_eq(str(inspect.signature(f)), b)
    21:28:55.21 <<< Return value from test_sig: None


    =========================================================     Investigating [91;1mtest_sig[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =========================     with example [91;1m
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    [0m     =========================
    



```
fdb.docsrc(1, "test_sig(f:FunctionType or ClassType, b:str); test_sig will get f's signature as a string; \
b is a signature in string provided by the user; in fact, test_sig is to compare two strings")
```

    =========================================================     Investigating [91;1mtest_sig[0m     =========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
    =========================     with example [91;1m
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    [0m     =========================
    
    [93;1mprint selected srcline with expands below[0m--------
    def test_sig(f, b):                                                                                                                                     (0)
        "Test the signature of an object"===================================================================================================================(1)
    [91;1mtest_sig(f:FunctionType or ClassType, b:str); test_sig will get f's signature as a string; b is a signature in string provided by the user; in fact, test_sig is to compare two strings[0m
        test_eq(str(inspect.signature(f)), b)                                                                                                               (2)
                                                                                                                                                            (3)



```
fdb.print()
```

    =========================================================     Investigating [91;1mtest_sig[0m     =========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
    =========================     with example [91;1m
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    [0m     =========================
    
    def test_sig(f, b):=======================================================================(0)       
        "Test the signature of an object"=====================================================(1) # [36;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [36;1mtest_sig will get f's signature as a string[0m; [93;1mb is a signature in string provided by the user[0m; [35;1min fact, test_sig is to compare two strings[0m; 
        test_eq(str(inspect.signature(f)), b)=================================================(2) # [37;1mtest_sig is to test two strings with test_eq[0m; [34;1mhow to turn a signature into a string;[0m; 
                                                                                                                                                            (3)



```

```
