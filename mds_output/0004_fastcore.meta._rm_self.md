# 04_rm_self

## imports


```
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```


<style>.container { width:100% !important; }</style>



```
from fastcore.meta import _rm_self
```

## set up


```
g = locals()
fdb = Fastdb(_rm_self, outloc = g)
```


```
fdb.print()
```

    =========================================================     Investigating [91;1m_rm_self[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =============================================================     with example [91;1m[0m     ==============================================================
    
    def _rm_self(sig):========================================================================(0)       
        sigd = dict(sig.parameters)===========================================================(1)       
        sigd.pop('self')======================================================================(2)       
        return sig.replace(parameters=sigd.values())==========================================(3)       
                                                                                                                                                            (4)



```
class Foo:
    def __init__(self, a, b:int=1): pass
pprint(inspect.signature(Foo.__init__))
pprint(_rm_self(inspect.signature(Foo.__init__)))
```

    <Signature (self, a, b: int = 1)>
    <Signature (a, b: int = 1)>



```
fdb.eg = """
class Foo:
    def __init__(self, a, b:int=1): pass
pprint(inspect.signature(Foo.__init__))
pprint(_rm_self(inspect.signature(Foo.__init__)))
"""
```

## document


```
fdb.docsrc(0, "remove parameter self from a signature which has self;")
fdb.docsrc(1, "how to access parameters from a signature; how is parameters stored in sig; how to turn parameters into a dict;", \
           "sig", "sig.parameters", "dict(sig.parameters)")
fdb.docsrc(2, "how to remove the self parameter from the dict of sig;")
fdb.docsrc(3, "how to update a sig using a updated dict of sig's parameters", "sigd", "sigd.values()")
```

    =========================================================     Investigating [91;1m_rm_self[0m     =========================================================
    ===============================================================     on line [91;1m0[0m     ================================================================
         with example [91;1m
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def _rm_self(sig):======================================================================================================================================(0)
                                                                                                           [91;1mremove parameter self from a signature which has self;[0m
        sigd = dict(sig.parameters)                                                                                                                         (1)
        sigd.pop('self')                                                                                                                                    (2)
    <Signature (self, a, b: int = 1)>
    <Signature (a, b: int = 1)>
    =========================================================     Investigating [91;1m_rm_self[0m     =========================================================
    ===============================================================     on line [91;1m1[0m     ================================================================
         with example [91;1m
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def _rm_self(sig):                                                                                                                                      (0)
        sigd = dict(sig.parameters)=========================================================================================================================(1)
                                                  [91;1mhow to access parameters from a signature; how is parameters stored in sig; how to turn parameters into a dict;[0m
        sigd.pop('self')                                                                                                                                    (2)
        return sig.replace(parameters=sigd.values())                                                                                                        (3)
    <Signature (self, a, b: int = 1)>
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                                                                               sig => sig : (self, a, b: int = 1)
    
    
                          sig.parameters => sig.parameters : OrderedDict([('self', <Parameter "self">), ('a', <Parameter "a">), ('b', <Parameter "b: int = 1">)])
    
    
                                 dict(sig.parameters) => dict(sig.parameters) : {'self': <Parameter "self">, 'a': <Parameter "a">, 'b': <Parameter "b: int = 1">}
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (a, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def _rm_self(sig):========================================================================(0) # [93;1mremove parameter self from a signature which has self;[0m; 
        sigd = dict(sig.parameters)===========================================================(1) # [91;1mhow to access parameters from a signature[0m; [36;1mhow is parameters stored in sig[0m; [36;1mhow to turn parameters into a dict;[0m; 
        sigd.pop('self')======================================================================(2)       
        return sig.replace(parameters=sigd.values())==========================================(3)       
                                                                                                                                                            (4)
                                                                                                                                         part No.1 out of 1 parts
    
    =========================================================     Investigating [91;1m_rm_self[0m     =========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def _rm_self(sig):                                                                                                                                      (0)
        sigd = dict(sig.parameters)                                                                                                                         (1)
        sigd.pop('self')====================================================================================================================================(2)
                                                                                                           [91;1mhow to remove the self parameter from the dict of sig;[0m
        return sig.replace(parameters=sigd.values())                                                                                                        (3)
                                                                                                                                                            (4)
    <Signature (self, a, b: int = 1)>
    <Signature (a, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def _rm_self(sig):========================================================================(0) # [92;1mremove parameter self from a signature which has self;[0m; 
        sigd = dict(sig.parameters)===========================================================(1) # [36;1mhow to access parameters from a signature[0m; [93;1mhow is parameters stored in sig[0m; [91;1mhow to turn parameters into a dict;[0m; 
        sigd.pop('self')======================================================================(2) # [92;1mhow to remove the self parameter from the dict of sig;[0m; 
        return sig.replace(parameters=sigd.values())==========================================(3)       
                                                                                                                                                            (4)
                                                                                                                                         part No.1 out of 1 parts
    
    =========================================================     Investigating [91;1m_rm_self[0m     =========================================================
    ===============================================================     on line [91;1m3[0m     ================================================================
         with example [91;1m
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        sigd = dict(sig.parameters)                                                                                                                         (1)
        sigd.pop('self')                                                                                                                                    (2)
        return sig.replace(parameters=sigd.values())========================================================================================================(3)
                                                                                                     [91;1mhow to update a sig using a updated dict of sig's parameters[0m
                                                                                                                                                            (4)
    <Signature (self, a, b: int = 1)>
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                                             sigd => sigd : {'a': <Parameter "a">, 'b': <Parameter "b: int = 1">}
    
    
                                                                        sigd.values() => sigd.values() : dict_values([<Parameter "a">, <Parameter "b: int = 1">])
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (a, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def _rm_self(sig):========================================================================(0) # [35;1mremove parameter self from a signature which has self;[0m; 
        sigd = dict(sig.parameters)===========================================================(1) # [91;1mhow to access parameters from a signature[0m; [36;1mhow is parameters stored in sig[0m; [36;1mhow to turn parameters into a dict;[0m; 
        sigd.pop('self')======================================================================(2) # [36;1mhow to remove the self parameter from the dict of sig;[0m; 
        return sig.replace(parameters=sigd.values())==========================================(3) # [91;1mhow to update a sig using a updated dict of sig's parameters[0m; 
                                                                                                                                                            (4)
                                                                                                                                         part No.1 out of 1 parts
    


## snoop


```
fdb.snoop()
```

    06:38:56.81 >>> Call to _rm_self in File "/tmp/_rm_self.py", line 3
    06:38:56.81 ...... sig = <Signature (self, a, b: int = 1)>
    06:38:56.81    3 | def _rm_self(sig):
    06:38:56.81    4 |     sigd = dict(sig.parameters)
    06:38:56.81 .......... sigd = {'self': <Parameter "self">, 'a': <Parameter "a">, 'b': <Parameter "b: int = 1">}
    06:38:56.81 .......... len(sigd) = 3
    06:38:56.81    5 |     sigd.pop('self')
    06:38:56.81 .......... sigd = {'a': <Parameter "a">, 'b': <Parameter "b: int = 1">}
    06:38:56.81 .......... len(sigd) = 2
    06:38:56.81    6 |     return sig.replace(parameters=sigd.values())
    06:38:56.81 <<< Return value from _rm_self: <Signature (a, b: int = 1)>


    =========================================================     Investigating [91;1m_rm_self[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    [0m     
    
    <Signature (self, a, b: int = 1)>
    <Signature (a, b: int = 1)>



```
fdb.print()
```

    =========================================================     Investigating [91;1m_rm_self[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    [0m     
    
    def _rm_self(sig):========================================================================(0) # [37;1mremove parameter self from a signature which has self;[0m; 
        sigd = dict(sig.parameters)===========================================================(1) # [34;1mhow to access parameters from a signature[0m; [92;1mhow is parameters stored in sig[0m; [36;1mhow to turn parameters into a dict;[0m; 
        sigd.pop('self')======================================================================(2) # [91;1mhow to remove the self parameter from the dict of sig;[0m; 
        return sig.replace(parameters=sigd.values())==========================================(3) # [36;1mhow to update a sig using a updated dict of sig's parameters[0m; 
                                                                                                                                                            (4)

