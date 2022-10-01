# 0001_Fastcore.meta.delegates


```
# from IPython.core.display import display, HTML # a depreciated import
from IPython.display import display, HTML 

display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>


## Import


```
from fastdebug.core import *
```


```
from fastcore.meta import delegates 
```

## Initiate Fastdb and example in str


```
g = locals() # this is a must
fdb = Fastdb(delegates, outloc=g)
```

## Example


```
def low(a, b=1): pass
@delegates(low) # this format is fine too
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
```

    <Signature (c, d=1, *, b=1)>



```
def low(a, b=1): pass
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(delegates(low)(mid))) 
```

    <Signature (c, d=1, *, b=1)>



```
fdb.eg = """
def low(a, b=1): pass
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(delegates(low)(mid)))
"""
```


```
fdb.eg = """
def low(a, b=1): pass
@delegates(low)
def mid(c, d=1, **kwargs): pass
pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
"""
```


```
fdb.eg = """
class Foo:
    @classmethod
    def clsm(a, b=1): pass
    
    @delegates(clsm)
    def instm(c, d=1, **kwargs): pass
f = Foo()
pprint(inspect.signature(f.instm)) # pprint and inspect is loaded from fastdebug
"""
```


```
fdb.eg = """
def low(a, b:int=1): pass
@delegates(low)
def mid(c, d:list=None, **kwargs): pass
pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
"""
```

## docsrc


```
fdb.eg
```




    '\ndef low(a, b:int=1): pass\n@delegates(low)\ndef mid(c, d:list=None, **kwargs): pass\npprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug\n'




```
fdb.print()
```

    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    def delegates(to:FunctionType=None, # Delegatee===========================================(0)       
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2)       
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6)       
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7)       
            from_f = getattr(from_f,'__func__',from_f)========================================(8)       
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10)      
            sig = inspect.signature(from_f)===================================================(11)      
            sigd = dict(sig.parameters)=======================================================(12)      
            k = sigd.pop('kwargs')============================================================(13)      
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items()                                    (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but}                                          (16)
            sigd.update(s2)===================================================================(17)      
            if keep: sigd['kwargs'] = k=======================================================(18)      
            else: from_f.__delwrap__ = to_f===================================================(19)      
            from_f.__signature__ = sig.replace(parameters=sigd.values())======================(20)      
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)========(21)      
            return f==========================================================================(22)      
        return _f=============================================================================(23)      
                                                                                                                                                            (24)



```
fdb.docsrc(21, "How to check whether a func has __annotations__; How add selected params' annotations from A to B's annotations;")

fdb.docsrc(14, "How to access a signature's parameters as a dict?; How to replace the kind of a parameter with a different kind?; \
how to check whether a parameter has a default value?; How to check whether a string is in a dict and a list?; \
how dict.items() and dict.values() differ", \
"inspect.signature(to_f).parameters.items()", "inspect.signature(to_f).parameters.values()", "inspect.Parameter.empty", \
"for k,v in inspect.signature(to_f).parameters.items():\\n\
    printright(f'v.kind: {v.kind}')\\n\
    printright(f'v.default: {v.default}')\\n\
    v1 = v.replace(kind=inspect.Parameter.KEYWORD_ONLY)\\n\
    printright(f'v1.kind: {v1.kind}')")

fdb.docsrc(16, "How to get A's __annotations__?; How to access it as a dict?; \
How to select annotations of the right params with names?; How to put them into a dict?; How to do it all in a single line", \
          "getattr(to_f, '__annotations__', {})", "getattr(from_f, '__annotations__', {})")
fdb.docsrc(17, "How to add the selected params from A's signature to B's signature; How to add items into a dict;")
fdb.docsrc(18, "How to add a new item into a dict;")
fdb.docsrc(19, "How to create a new attr for a function or obj;")
fdb.docsrc(20, "How to update a signature with a new set of parameters;", "sigd", "sigd.values()", "sig", \
           "sig.replace(parameters=sigd.values())")


fdb.docsrc(2, "how to make delegates(to, but) to have 'but' as list and default as None")
fdb.docsrc(0, "how to make delegates(to) to have to as FunctionType and default as None")
fdb.docsrc(6, "how to write 2 ifs and elses in 2 lines")
fdb.docsrc(7, "how to assign a,b together with if and else")
fdb.docsrc(8, "Is classmethod callable; does classmethod has __func__; can we do inspect.signature(clsmethod); \
how to use getattr(obj, attr, default)", \
           "from_f", "to_f", \
           "hasattr(from_f, '__func__')", "from_f = getattr(from_f,'__func__',from_f)", \
           "hasattr(to_f, '__func__')", "from_f = getattr(to_f,'__func__',to_f)", "callable(to_f)")

fdb.docsrc(10, "if B has __delwrap__, can we do delegates(A)(B) again?; hasattr(obj, '__delwrap__')")
fdb.docsrc(11, "how to get signature obj of B; what does a signature look like; what is the type", "inspect.signature(from_f)", "type(inspect.signature(from_f))")
fdb.docsrc(12, "How to access parameters of a signature?; How to turn parameters into a dict?", "sig.parameters", "dict(sig.parameters)")
fdb.docsrc(13, "How to remove an item from a dict?; How to get the removed item from a dict?; How to add the removed item back to the dict?; \
when writing expressions, as they share environment, so they may affect the following code", "sigd", "k = sigd.pop('kwargs')", "sigd", "sigd['kwargs'] = k")


# fdb.docsrc(3,"Personal Docs: delegates(A)(B), given B has kwargs in sig; delegating params of A to B; A and B can be funcs, methods, classes; \
# when A or B are classmethod or if they have __func__, then __func__ shall be used to get sig/params; \
# when A and B are classes, A.__init__ is actually used to get sig/params, B offers sig/params as a class; \
# B only take wanted and missing params with default values; B can keep its kwargs, if not, B will store A in __delwrap__ to prevent delegates(A)(B) again;\
# B will get __signature__ and store its new sig; if B has __annotations__, then add what B want from A's __annotations__")
```

    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m21[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            else: from_f.__delwrap__ = to_f                                                                                                                 (19)
            from_f.__signature__ = sig.replace(parameters=sigd.values())                                                                                    (20)
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)======================================================================(21)
                                                 [91;1mHow to check whether a func has __annotations__; How add selected params' annotations from A to B's annotations;[0m
            return f                                                                                                                                        (22)
        return _f                                                                                                                                           (23)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m14[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            sigd = dict(sig.parameters)                                                                                                                     (12)
            k = sigd.pop('kwargs')                                                                                                                          (13)
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items()====================================(14)
    [91;1mHow to access a signature's parameters as a dict?; How to replace the kind of a parameter with a different kind?; how to check whether a parameter has a default value?; How to check whether a string is in a dict and a list?; how dict.items() and dict.values() differ[0m
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}                                                               (15)
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but}                                          (16)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
    inspect.signature(to_f).parameters.items() => inspect.signature(to_f).parameters.items() : odict_items([('a', <Parameter "a">), ('b', <Parameter "b: int = 1">)])
    
    
           inspect.signature(to_f).parameters.values() => inspect.signature(to_f).parameters.values() : odict_values([<Parameter "a">, <Parameter "b: int = 1">])
    
    
                                                                                    inspect.Parameter.empty => inspect.Parameter.empty : <class 'inspect._empty'>
    
    
    for k,v in inspect.signature(to_f).parameters.items():
        printright(f'v.kind: {v.kind}')
        printright(f'v.default: {v.default}')
        v1 = v.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        printright(f'v1.kind: {v1.kind}')
    
    Running the code block above => ====================================================================
    
                                                                                                                                    v.kind: POSITIONAL_OR_KEYWORD
                                                                                                                              v.default: <class 'inspect._empty'>
                                                                                                                                            v1.kind: KEYWORD_ONLY
                                                                                                                                    v.kind: POSITIONAL_OR_KEYWORD
                                                                                                                                                     v.default: 1
                                                                                                                                            v1.kind: KEYWORD_ONLY
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (c, d: list = None, *, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def delegates(to:FunctionType=None, # Delegatee===========================================(0)       
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2)       
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6)       
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7)       
            from_f = getattr(from_f,'__func__',from_f)========================================(8)       
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10)      
            sig = inspect.signature(from_f)===================================================(11)      
            sigd = dict(sig.parameters)=======================================================(12)      
            k = sigd.pop('kwargs')============================================================(13)      
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [91;1mHow to access a signature's parameters as a dict?[0m; [93;1mHow to replace the kind of a parameter with a different kind?[0m; [93;1mhow to check whether a parameter has a default value?[0m; [91;1mHow to check whether a string is in a dict and a list?[0m; [36;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but}                                          (16)
            sigd.update(s2)===================================================================(17)      
            if keep: sigd['kwargs'] = k=======================================================(18)      
            else: from_f.__delwrap__ = to_f===================================================(19)      
                                                                                                                                         part No.1 out of 2 parts
    
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m16[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items()                                    (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}                                                               (15)
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but}==========================================(16)
    [91;1mHow to get A's __annotations__?; How to access it as a dict?; How to select annotations of the right params with names?; How to put them into a dict?; How to do it all in a single line[0m
            sigd.update(s2)                                                                                                                                 (17)
            if keep: sigd['kwargs'] = k                                                                                                                     (18)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                              getattr(to_f, '__annotations__', {}) => getattr(to_f, '__annotations__', {}) : {'b': <class 'int'>}
    
    
                                                         getattr(from_f, '__annotations__', {}) => getattr(from_f, '__annotations__', {}) : {'d': <class 'list'>}
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (c, d: list = None, *, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def delegates(to:FunctionType=None, # Delegatee===========================================(0)       
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2)       
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6)       
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7)       
            from_f = getattr(from_f,'__func__',from_f)========================================(8)       
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10)      
            sig = inspect.signature(from_f)===================================================(11)      
            sigd = dict(sig.parameters)=======================================================(12)      
            k = sigd.pop('kwargs')============================================================(13)      
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [34;1mHow to access a signature's parameters as a dict?[0m; [36;1mHow to replace the kind of a parameter with a different kind?[0m; [35;1mhow to check whether a parameter has a default value?[0m; [37;1mHow to check whether a string is in a dict and a list?[0m; [93;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [93;1mHow to get A's __annotations__?[0m; [93;1mHow to access it as a dict?[0m; [34;1mHow to select annotations of the right params with names?[0m; [35;1mHow to put them into a dict?[0m; [91;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17)      
            if keep: sigd['kwargs'] = k=======================================================(18)      
            else: from_f.__delwrap__ = to_f===================================================(19)      
                                                                                                                                         part No.1 out of 2 parts
    
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m17[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}                                                               (15)
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but}                                          (16)
            sigd.update(s2)=================================================================================================================================(17)
                                                                [91;1mHow to add the selected params from A's signature to B's signature; How to add items into a dict;[0m
            if keep: sigd['kwargs'] = k                                                                                                                     (18)
            else: from_f.__delwrap__ = to_f                                                                                                                 (19)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m18[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but}                                          (16)
            sigd.update(s2)                                                                                                                                 (17)
            if keep: sigd['kwargs'] = k=====================================================================================================================(18)
                                                                                                                               [91;1mHow to add a new item into a dict;[0m
            else: from_f.__delwrap__ = to_f                                                                                                                 (19)
            from_f.__signature__ = sig.replace(parameters=sigd.values())                                                                                    (20)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m19[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            sigd.update(s2)                                                                                                                                 (17)
            if keep: sigd['kwargs'] = k                                                                                                                     (18)
            else: from_f.__delwrap__ = to_f=================================================================================================================(19)
                                                                                                                  [91;1mHow to create a new attr for a function or obj;[0m
            from_f.__signature__ = sig.replace(parameters=sigd.values())                                                                                    (20)
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)                                                                      (21)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m20[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            if keep: sigd['kwargs'] = k                                                                                                                     (18)
            else: from_f.__delwrap__ = to_f                                                                                                                 (19)
            from_f.__signature__ = sig.replace(parameters=sigd.values())====================================================================================(20)
                                                                                                          [91;1mHow to update a signature with a new set of parameters;[0m
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)                                                                      (21)
            return f                                                                                                                                        (22)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                          sigd => sigd : {'c': <Parameter "c">, 'd': <Parameter "d: list = None">, 'b': <Parameter "b: int = 1">}
    
    
                                          sigd.values() => sigd.values() : dict_values([<Parameter "c">, <Parameter "d: list = None">, <Parameter "b: int = 1">])
    
    
                                                                                                                       sig => sig : (c, d: list = None, **kwargs)
    
    
                                               sig.replace(parameters=sigd.values()) => sig.replace(parameters=sigd.values()): (c, d: list = None, *, b: int = 1)
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (c, d: list = None, *, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
            from_f.__signature__ = sig.replace(parameters=sigd.values())======================(20) # [35;1mHow to update a signature with a new set of parameters;[0m; 
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)========(21) # [36;1mHow to check whether a func has __annotations__[0m; [91;1mHow add selected params' annotations from A to B's annotations;[0m; 
            return f==========================================================================(22)      
        return _f=============================================================================(23)      
                                                                                                                                                            (24)
                                                                                                                                         part No.2 out of 2 parts
    
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m2[0m     ================================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def delegates(to:FunctionType=None, # Delegatee                                                                                                         (0)
                  keep=False, # Keep `kwargs` in decorated function?                                                                                        (1)
                  but:list=None): # Exclude these parameters from signature=================================================================================(2)
                                                                                         [91;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m
        "Decorator: replace `**kwargs` in signature with params from `to`"                                                                                  (3)
        if but is None: but = []                                                                                                                            (4)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m0[0m     ================================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
    def delegates(to:FunctionType=None, # Delegatee=========================================================================================================(0)
                                                                                         [91;1mhow to make delegates(to) to have to as FunctionType and default as None[0m
                  keep=False, # Keep `kwargs` in decorated function?                                                                                        (1)
                  but:list=None): # Exclude these parameters from signature                                                                                 (2)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m6[0m     ================================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        if but is None: but = []                                                                                                                            (4)
        def _f(f):                                                                                                                                          (5)
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=====================================================================================(6)
                                                                                                                          [91;1mhow to write 2 ifs and elses in 2 lines[0m
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f                                                                       (7)
            from_f = getattr(from_f,'__func__',from_f)                                                                                                      (8)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m7[0m     ================================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
        def _f(f):                                                                                                                                          (5)
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__                                                                                     (6)
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=======================================================================(7)
                                                                                                                      [91;1mhow to assign a,b together with if and else[0m
            from_f = getattr(from_f,'__func__',from_f)                                                                                                      (8)
            to_f = getattr(to_f,'__func__',to_f)                                                                                                            (9)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m8[0m     ================================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__                                                                                     (6)
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f                                                                       (7)
            from_f = getattr(from_f,'__func__',from_f)======================================================================================================(8)
                           [91;1mIs classmethod callable; does classmethod has __func__; can we do inspect.signature(clsmethod); how to use getattr(obj, attr, default)[0m
            to_f = getattr(to_f,'__func__',to_f)                                                                                                            (9)
            if hasattr(from_f,'__delwrap__'): return f                                                                                                      (10)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                                                                 from_f => from_f : <function mid>
    
    
                                                                                                                     to_f => to_f : <function low>
    
    
                                                                                               hasattr(from_f, '__func__') => hasattr(from_f, '__func__') : False
    
    
                                                                              from_f = getattr(from_f,'__func__',from_f) => from_f: <function mid>
    
    
                                                                                                   hasattr(to_f, '__func__') => hasattr(to_f, '__func__') : False
    
    
                                                                                  from_f = getattr(to_f,'__func__',to_f) => from_f: <function low>
    
    
                                                                                                                          callable(to_f) => callable(to_f) : True
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (c, d: list = None, *, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [93;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [34;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [36;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [92;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [93;1mIs classmethod callable[0m; [34;1mdoes classmethod has __func__[0m; [93;1mcan we do inspect.signature(clsmethod)[0m; [92;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10)      
            sig = inspect.signature(from_f)===================================================(11)      
            sigd = dict(sig.parameters)=======================================================(12)      
            k = sigd.pop('kwargs')============================================================(13)      
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [36;1mHow to access a signature's parameters as a dict?[0m; [35;1mHow to replace the kind of a parameter with a different kind?[0m; [34;1mhow to check whether a parameter has a default value?[0m; [34;1mHow to check whether a string is in a dict and a list?[0m; [93;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [93;1mHow to get A's __annotations__?[0m; [36;1mHow to access it as a dict?[0m; [35;1mHow to select annotations of the right params with names?[0m; [34;1mHow to put them into a dict?[0m; [37;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [92;1mHow to add the selected params from A's signature to B's signature[0m; [34;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [37;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [34;1mHow to create a new attr for a function or obj;[0m; 
                                                                                                                                         part No.1 out of 2 parts
    
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m10[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            from_f = getattr(from_f,'__func__',from_f)                                                                                                      (8)
            to_f = getattr(to_f,'__func__',to_f)                                                                                                            (9)
            if hasattr(from_f,'__delwrap__'): return f======================================================================================================(10)
                                                                              [91;1mif B has __delwrap__, can we do delegates(A)(B) again?; hasattr(obj, '__delwrap__')[0m
            sig = inspect.signature(from_f)                                                                                                                 (11)
            sigd = dict(sig.parameters)                                                                                                                     (12)
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m11[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            to_f = getattr(to_f,'__func__',to_f)                                                                                                            (9)
            if hasattr(from_f,'__delwrap__'): return f                                                                                                      (10)
            sig = inspect.signature(from_f)=================================================================================================================(11)
                                                                                 [91;1mhow to get signature obj of B; what does a signature look like; what is the type[0m
            sigd = dict(sig.parameters)                                                                                                                     (12)
            k = sigd.pop('kwargs')                                                                                                                          (13)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                           inspect.signature(from_f) => inspect.signature(from_f) : (c, d: list = None, **kwargs)
    
    
                                                                 type(inspect.signature(from_f)) => type(inspect.signature(from_f)) : <class 'inspect.Signature'>
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (c, d: list = None, *, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [91;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [34;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [91;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [34;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [36;1mIs classmethod callable[0m; [34;1mdoes classmethod has __func__[0m; [36;1mcan we do inspect.signature(clsmethod)[0m; [92;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10) # [92;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [93;1mhasattr(obj, '__delwrap__')[0m; 
            sig = inspect.signature(from_f)===================================================(11) # [35;1mhow to get signature obj of B[0m; [37;1mwhat does a signature look like[0m; [91;1mwhat is the type[0m; 
            sigd = dict(sig.parameters)=======================================================(12)      
            k = sigd.pop('kwargs')============================================================(13)      
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [92;1mHow to access a signature's parameters as a dict?[0m; [92;1mHow to replace the kind of a parameter with a different kind?[0m; [35;1mhow to check whether a parameter has a default value?[0m; [36;1mHow to check whether a string is in a dict and a list?[0m; [36;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [35;1mHow to get A's __annotations__?[0m; [37;1mHow to access it as a dict?[0m; [92;1mHow to select annotations of the right params with names?[0m; [34;1mHow to put them into a dict?[0m; [37;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [34;1mHow to add the selected params from A's signature to B's signature[0m; [91;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [92;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [91;1mHow to create a new attr for a function or obj;[0m; 
                                                                                                                                         part No.1 out of 2 parts
    
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m12[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            if hasattr(from_f,'__delwrap__'): return f                                                                                                      (10)
            sig = inspect.signature(from_f)                                                                                                                 (11)
            sigd = dict(sig.parameters)=====================================================================================================================(12)
                                                                                    [91;1mHow to access parameters of a signature?; How to turn parameters into a dict?[0m
            k = sigd.pop('kwargs')                                                                                                                          (13)
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items()                                    (14)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                sig.parameters => sig.parameters : OrderedDict([('c', <Parameter "c">), ('d', <Parameter "d: list = None">), ('kwargs', <Parameter "**kwargs">)])
    
    
                       dict(sig.parameters) => dict(sig.parameters) : {'c': <Parameter "c">, 'd': <Parameter "d: list = None">, 'kwargs': <Parameter "**kwargs">}
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (c, d: list = None, *, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [91;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [92;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [36;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [91;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [36;1mIs classmethod callable[0m; [34;1mdoes classmethod has __func__[0m; [36;1mcan we do inspect.signature(clsmethod)[0m; [92;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10) # [93;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [36;1mhasattr(obj, '__delwrap__')[0m; 
            sig = inspect.signature(from_f)===================================================(11) # [91;1mhow to get signature obj of B[0m; [34;1mwhat does a signature look like[0m; [36;1mwhat is the type[0m; 
            sigd = dict(sig.parameters)=======================================================(12) # [35;1mHow to access parameters of a signature?[0m; [92;1mHow to turn parameters into a dict?[0m; 
            k = sigd.pop('kwargs')============================================================(13)      
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [34;1mHow to access a signature's parameters as a dict?[0m; [35;1mHow to replace the kind of a parameter with a different kind?[0m; [35;1mhow to check whether a parameter has a default value?[0m; [93;1mHow to check whether a string is in a dict and a list?[0m; [35;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [36;1mHow to get A's __annotations__?[0m; [93;1mHow to access it as a dict?[0m; [34;1mHow to select annotations of the right params with names?[0m; [37;1mHow to put them into a dict?[0m; [37;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [35;1mHow to add the selected params from A's signature to B's signature[0m; [37;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [93;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [93;1mHow to create a new attr for a function or obj;[0m; 
                                                                                                                                         part No.1 out of 2 parts
    
    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m13[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    [93;1mprint selected srcline with expands below[0m--------
            sig = inspect.signature(from_f)                                                                                                                 (11)
            sigd = dict(sig.parameters)                                                                                                                     (12)
            k = sigd.pop('kwargs')==========================================================================================================================(13)
    [91;1mHow to remove an item from a dict?; How to get the removed item from a dict?; How to add the removed item back to the dict?; when writing expressions, as they share environment, so they may affect the following code[0m
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items()                                    (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}                                                               (15)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                       sigd => sigd : {'c': <Parameter "c">, 'd': <Parameter "d: list = None">, 'kwargs': <Parameter "**kwargs">}
    
    
                                                                                                                            k = sigd.pop('kwargs') => k: **kwargs
    
    
                                                                                         sigd => sigd : {'c': <Parameter "c">, 'd': <Parameter "d: list = None">}
    
    
                                                                                                                   sigd['kwargs'] = k => sigd['kwargs']: **kwargs
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    <Signature (c, d: list = None, *, b: int = 1)>
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [34;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [92;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [91;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [37;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [34;1mIs classmethod callable[0m; [34;1mdoes classmethod has __func__[0m; [35;1mcan we do inspect.signature(clsmethod)[0m; [91;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10) # [34;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [91;1mhasattr(obj, '__delwrap__')[0m; 
            sig = inspect.signature(from_f)===================================================(11) # [36;1mhow to get signature obj of B[0m; [36;1mwhat does a signature look like[0m; [35;1mwhat is the type[0m; 
            sigd = dict(sig.parameters)=======================================================(12) # [91;1mHow to access parameters of a signature?[0m; [36;1mHow to turn parameters into a dict?[0m; 
            k = sigd.pop('kwargs')============================================================(13) # [36;1mHow to remove an item from a dict?[0m; [35;1mHow to get the removed item from a dict?[0m; [93;1mHow to add the removed item back to the dict?[0m; [37;1mwhen writing expressions, as they share environment, so they may affect the following code[0m; 
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [36;1mHow to access a signature's parameters as a dict?[0m; [34;1mHow to replace the kind of a parameter with a different kind?[0m; [91;1mhow to check whether a parameter has a default value?[0m; [34;1mHow to check whether a string is in a dict and a list?[0m; [93;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [93;1mHow to get A's __annotations__?[0m; [36;1mHow to access it as a dict?[0m; [93;1mHow to select annotations of the right params with names?[0m; [93;1mHow to put them into a dict?[0m; [36;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [36;1mHow to add the selected params from A's signature to B's signature[0m; [92;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [91;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [91;1mHow to create a new attr for a function or obj;[0m; 
                                                                                                                                         part No.1 out of 2 parts
    



```
fdb.print()
```

    ========================================================     Investigating [91;1mdelegates[0m     =========================================================
    ===============================================================     on line [91;1m13[0m     ===============================================================
         with example [91;1m
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    [0m     
    
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [91;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [37;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [37;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [36;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [36;1mIs classmethod callable[0m; [35;1mdoes classmethod has __func__[0m; [36;1mcan we do inspect.signature(clsmethod)[0m; [93;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10) # [34;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [34;1mhasattr(obj, '__delwrap__')[0m; 
            sig = inspect.signature(from_f)===================================================(11) # [93;1mhow to get signature obj of B[0m; [35;1mwhat does a signature look like[0m; [35;1mwhat is the type[0m; 
            sigd = dict(sig.parameters)=======================================================(12) # [93;1mHow to access parameters of a signature?[0m; [93;1mHow to turn parameters into a dict?[0m; 
            k = sigd.pop('kwargs')============================================================(13) # [35;1mHow to remove an item from a dict?[0m; [93;1mHow to get the removed item from a dict?[0m; [93;1mHow to add the removed item back to the dict?[0m; [34;1mwhen writing expressions, as they share environment, so they may affect the following code[0m; 
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [93;1mHow to access a signature's parameters as a dict?[0m; [93;1mHow to replace the kind of a parameter with a different kind?[0m; [92;1mhow to check whether a parameter has a default value?[0m; [34;1mHow to check whether a string is in a dict and a list?[0m; [35;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [35;1mHow to get A's __annotations__?[0m; [92;1mHow to access it as a dict?[0m; [93;1mHow to select annotations of the right params with names?[0m; [35;1mHow to put them into a dict?[0m; [34;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [35;1mHow to add the selected params from A's signature to B's signature[0m; [37;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [93;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [34;1mHow to create a new attr for a function or obj;[0m; 
            from_f.__signature__ = sig.replace(parameters=sigd.values())======================(20) # [91;1mHow to update a signature with a new set of parameters;[0m; 
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)========(21) # [36;1mHow to check whether a func has __annotations__[0m; [34;1mHow add selected params' annotations from A to B's annotations;[0m; 
            return f==========================================================================(22)      
        return _f=============================================================================(23)      
                                                                                                                                                            (24)


## Snoop


```
# fdb.snoop(deco=True) # both examples above works for Fastdb
```


```
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/Demos/fastcore_meta_delegates.ipynb
!mv /Users/Natsume/Documents/fastdebug/Demos/fastcore_meta_delegates.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/

!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```

    [jupytext] Reading /Users/Natsume/Documents/fastdebug/Demos/fastcore_meta_delegates.ipynb in format ipynb
    Traceback (most recent call last):
      File "/Users/Natsume/mambaforge/bin/jupytext", line 10, in <module>
        sys.exit(jupytext())
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/cli.py", line 488, in jupytext
        exit_code += jupytext_single_file(nb_file, args, log)
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/cli.py", line 552, in jupytext_single_file
        notebook = read(nb_file, fmt=fmt, config=config)
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/jupytext.py", line 411, in read
        with open(fp, encoding="utf-8") as stream:
    FileNotFoundError: [Errno 2] No such file or directory: '/Users/Natsume/Documents/fastdebug/Demos/fastcore_meta_delegates.ipynb'
    mv: rename /Users/Natsume/Documents/fastdebug/Demos/fastcore_meta_delegates.md to /Users/Natsume/Documents/divefastai/Debuggable/jupytext/fastcore_meta_delegates.md: No such file or directory
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/index.ipynb to markdown
    [NbConvertApp] Writing 58088 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/index.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/08_use_kwargs_dict.ipynb to markdown
    [NbConvertApp] Writing 56914 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/08_use_kwargs_dict.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/000_tour.ipynb to markdown
    [NbConvertApp] Writing 12625 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/000_tour.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/04_fastcore.meta._rm_self.ipynb to markdown
    [NbConvertApp] Writing 16058 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/04_fastcore.meta._rm_self.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/03_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb to markdown
    [NbConvertApp] Writing 89759 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/03_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/09_method_funcs_kwargs.ipynb to markdown
    [NbConvertApp] Writing 68929 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/09_method_funcs_kwargs.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0010_fastcore_meta_summary.ipynb to markdown
    [NbConvertApp] Writing 27864 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0010_fastcore_meta_summary.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/05_fastcore.meta.test_sig.ipynb to markdown
    [NbConvertApp] Writing 10361 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/05_fastcore.meta.test_sig.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/01_signature_from_callable_with_FixSigMeta.ipynb to markdown
    [NbConvertApp] Writing 47132 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/01_signature_from_callable_with_FixSigMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/07_fastcore.meta.BypassNewMeta.ipynb to markdown
    [NbConvertApp] Writing 30610 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/07_fastcore.meta.BypassNewMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/00_fastcore_meta_delegates.ipynb to markdown
    [NbConvertApp] Writing 75601 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_fastcore_meta_delegates.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/06_fastcore.meta.NewChkMeta.ipynb to markdown
    [NbConvertApp] Writing 30938 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/06_fastcore.meta.NewChkMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/questions/00_question_anno_dict.ipynb to markdown
    [NbConvertApp] Writing 11779 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_question_anno_dict.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/lib/utils.ipynb to markdown
    [NbConvertApp] Writing 20154 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/utils.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/lib/00_core.ipynb to markdown
    [NbConvertApp] Writing 410613 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_core.md



```

```
