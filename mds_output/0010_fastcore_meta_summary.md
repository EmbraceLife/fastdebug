# 0010_fastcore_meta_summary

## import


```
from fastdebug.utils import *
from fastdebug.core import *
```


<style>.container { width:100% !important; }</style>


## fastcore and fastcore.meta


```
import fastcore
```


```
whichversion("fastcore")
```

    fastcore: 1.5.27 
    Python supercharged for fastai development    
    Jeremy Howard and Sylvain Gugger 
    https://github.com/fastai/fastcore/     
    python_version: >=3.7     
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore



```
whatinside(fastcore, lib=True)
```

    The library has 21 modules
    ['_modidx',
     '_nbdev',
     'all',
     'basics',
     'dispatch',
     'docments',
     'docscrape',
     'foundation',
     'imports',
     'meta',
     'nb_imports',
     'net',
     'parallel',
     'script',
     'shutil',
     'style',
     'test',
     'transform',
     'utils',
     'xdg',
     'xtras']



```
from fastcore.meta import *
import fastcore.meta as fm
```

### What's inside fastcore.meta


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


## Review individual funcs and classes

### What is fastcore.meta all about? 

It is a submodule contains 4 metaclasses, 1 class built by a metaclass, 4 decorators and a few functions.    

Metaclasses give us the power to create new breeds of classes with new features.     

Decorators give us the power to add new features to existing funcions.    

We can find their basic info [above](#What's-inside-fastcore.meta)

### What can these metaclasses do for me?

We design/create classes to breed objects as we like.

We design/create metaclasses to breed classes as we like.

Before metaclasses, all classes are created by type and are born the same.

With metaclasses, e.g., FixSigMeta first uses type to its instance classes exactly like above, but then FixSigMeta immediately adds new features to them right before they are born.

#### FixSigMeta
can breed classes which are free of signature problems (or they can automatically fix signature problems).

#### PrePostInitMeta
inherited/evolved from `FixSigMeta` to breed classes which can initialize their objects using `__pre_init__`, 
`__init__`, `__post_init__` whichever is available (allow me to abbreviate it as triple_init).

#### AutoInit
is an instance class created by `PrePostInitMeta`, and together with its own defined `__pre_init__`, subclasses of `AutoInit` has to worry about running `super().__init__(...)` no more.

- As `AutoInit` is an instance class created by `PrePostInitMeta`, it can pass on both features (free of signature problem and triple_init) to its subclasses. 
- As it also defines its own `__pre_init__` function which calls its superclass `__init__` function, its subclasses will inherit this `__pre_init__` function too.
- When subclasses of `AutoInit` create and initialize object intances through `__call__` from `PrePostInitMeta`, `AutoInit`'s `__pre_init__` runs `super().__init__(...)`, so when we write `__init__` function of subclasses which inherits from `AutoInit`, we don't need to write `super().__init__(...)` any more.

#### NewChkMeta

is inherited from `FixSigMeta`, so any instance classes created by `NewChkMeta` can also pass on the no_signature_problem feature.

It defines its own `__call__` to enable all the instance objects e.g., `t` created by all the instance classes e.g., `T` created by `NewChkMeta` to do the following: 

- `T(t) is t if isinstance(t, T)` returns true
- when `T(t) is t if not isinstance(t, T)`, or when `T(t, 1) is t if isinstance(t, T)` or when `T(t, b=1) is t if isinstance(t, T)`, all return False

In other words, `NewChkMeta` creates a new breed of classes `T` as an example which won't recreate the same instance object `t` twice. But if `t` is not `T`'s instance object, or we choose to add more flavor to `t`, then `T(t)` or `T(t, 1)` will create a new instance object of `T`.

#### BypassNewMeta

is inherited from `FixSigMeta`, so it has the feature of free from signature problems.

It defines its own `__call__`, so that when its instance classes `_T` create and initialize objects with a param `t` which is an instance object of another class `_TestB`, they can do the following:

- If `_T` likes `_TestB` and prefers `t` as it is, then when we run `t2 = _T(t)`, and `t2 is t` will be True, and both are instances of `_T`.  
- If `_T` is not please with `t`, it could be that `_T` does not like `_TestB` any more, then `_T(t) is t` will be False
- or maybe `_T` still likes `_TestB`, but want to add some flavors to `t` by `_T(t, 1)` or `_T(t, b=1)` for example, in this case `_T(t) is t` will also be False.

In other words, `BypassNewMeta` creates a new breed of instance classes `_T` which don't need to create but make an object `t` of its own object instance, if `t` is an instance object of `_TestB` which is liked by `_T` and if `_T` likes `t` as it is.

### What can those decorators do for me?

A decorator is a function that takes in a function and returns a modified function.

A decorator allows us to modify the behavior of a function. 

#### use_kwargs_dict

allows us to replace an existing function's param `kwargs` with a number of params with default values.

The params with their default values are provided in a dictionary.

#### use_kwargs

allows us to replace an existing function's param `kwargs` with a number of params with None as their default values.

The params are provided as names in a list.

#### delegates

allows us to replace an existing function's param `kwargs` with a number of params with their default values from another existing function.

In fact, `delegates` can work on function, classes, and methods.

#### funcs_kwargs
is a decorator to classes. It can help classes to bring in existing functions as their methods. 

It can set the methods to use or not use `self` in the class.

### The remaining functions

`test_sig` and `method` are straightforward, their docs tell it all clearly.

`empty2none` and `anno_dict` are no in use much at all. see the thread [here](). 


```
fastview("FixSigMeta", nb=True)
```

    
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [36;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance[0m; [36;1mthen check whether its class instance has its own __init__;if so, remove self from the sig of __init__[0m; [91;1mthen assign this new sig to __signature__ for the class instance;[0m; 
        def __new__(cls, name, bases, dict):==================================================(2) # [36;1mhow does a metaclass create a class instance[0m; [35;1mwhat does super().__new__() do here;[0m; 
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [92;1mhow to remove self from a signature[0m; [36;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)



[Open `0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit` in Jupyter Notebook](http://localhost:8888/tree/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb)



```
%debug
```

    ERROR:root:No traceback has been produced, nothing to debug.



```
fastsrcs()
```

    test_sig.py
    BypassNewMeta.py
    snoop.py
    FixSigMeta.py
    fastnbs.py
    funcs_kwargs.py
    NewChkMeta.py
    printtitle.py
    AutoInit.py
    method.py
    _rm_self.py
    delegates.py
    create_explore_str.py
    PrePostInitMeta.py
    _funcs_kwargs.py
    whatinside.py



```
fastview(FixSigMeta)
```

    
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [36;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance[0m; [36;1mthen check whether its class instance has its own __init__;if so, remove self from the sig of __init__[0m; [91;1mthen assign this new sig to __signature__ for the class instance;[0m; 
        def __new__(cls, name, bases, dict):==================================================(2) # [36;1mhow does a metaclass create a class instance[0m; [35;1mwhat does super().__new__() do here;[0m; 
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [92;1mhow to remove self from a signature[0m; [36;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)



```
fastcodes("how to get signature's parameters", accu=0.8, nb=True, db=True)
```


keyword match is  <mark style="background-color: #ffff00">0.8</mark> , found a line: in  <mark style="background-color: #FFFF00">delegates.py</mark> 


            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [91;1mHow to access a signature's parameters as a dict?[0m; [36;1mHow to replace the kind of a parameter with a different kind?[0m; [91;1mhow to check whether a parameter has a default value?[0m; [35;1mHow to check whether a string is in a dict and a list?[0m; [34;1mhow dict.items() and dict.values() differ[0m;  (14)
    



the entire source code in  <mark style="background-color: #FFFF00">delegates.py</mark> 


    
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [37;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [93;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [37;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [34;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [37;1mIs classmethod callable[0m; [92;1mdoes classmethod has __func__[0m; [35;1mcan we do inspect.signature(clsmethod)[0m; [37;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10) # [36;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [37;1mhasattr(obj, '__delwrap__')[0m; 
            sig = inspect.signature(from_f)===================================================(11) # [91;1mhow to get signature obj of B[0m; [36;1mwhat does a signature look like[0m; [93;1mwhat is the type[0m; 
            sigd = dict(sig.parameters)=======================================================(12) # [34;1mHow to access parameters of a signature?[0m; [37;1mHow to turn parameters into a dict?[0m; 
            k = sigd.pop('kwargs')============================================================(13) # [93;1mHow to remove an item from a dict?[0m; [91;1mHow to get the removed item from a dict?[0m; [37;1mHow to add the removed item back to the dict?[0m; [34;1mwhen writing expressions, as they share environment, so they may affect the following code[0m; 
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [91;1mHow to access a signature's parameters as a dict?[0m; [36;1mHow to replace the kind of a parameter with a different kind?[0m; [91;1mhow to check whether a parameter has a default value?[0m; [35;1mHow to check whether a string is in a dict and a list?[0m; [34;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [35;1mHow to get A's __annotations__?[0m; [35;1mHow to access it as a dict?[0m; [92;1mHow to select annotations of the right params with names?[0m; [34;1mHow to put them into a dict?[0m; [93;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [34;1mHow to add the selected params from A's signature to B's signature[0m; [92;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [37;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [92;1mHow to create a new attr for a function or obj;[0m; 
            from_f.__signature__ = sig.replace(parameters=sigd.values())======================(20) # [91;1mHow to update a signature with a new set of parameters;[0m; 
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)========(21) # [35;1mHow to check whether a func has __annotations__[0m; [35;1mHow add selected params' annotations from A to B's annotations;[0m; 
            return f==========================================================================(22)      
        return _f=============================================================================(23)      
                                                                                                                                                            (24)
    



[Open `0001_fastcore_meta_delegates` in Jupyter Notebook](http://localhost:8888/tree/nbs/demos/0001_fastcore_meta_delegates.ipynb)



```
fastnotes("how make the most", folder="all")
```

    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">2022_part1/fastai-lecture-1.md</mark> 



do you want to know  <mark style="background-color: #ffff00">how</mark>  to  <mark style="background-color: #ffff00">make</mark>   <mark style="background-color: #ffff00">the</mark>   <mark style="background-color: #FFFF00">most</mark>  out of fastai? - do you know lesson 0 and  <mark style="background-color: #ffff00">the</mark>  book _meta learning_ by an alumni?



    



show  <mark style="background-color: #FFFF00">2</mark>  lines above and after in  <mark style="background-color: #ffff00"> <mark style="background-color: #FFFF00">2</mark> 0 <mark style="background-color: #FFFF00">2</mark>  <mark style="background-color: #FFFF00">2</mark> _part1/fastai-lecture-1.md</mark> :



Do Jeremy and fastai community take it very seriously in help beginners along the way?




[16:33](https://www.youtube.com/watch?v=RLvUfyLcT48&t=80s&loop=10&start=16:33&end=17:41) Make the most out of [fast.ai](http://fast.ai/)  




do you want to know  <mark style="background-color: #ffff00">how</mark>  to  <mark style="background-color: #ffff00">make</mark>   <mark style="background-color: #ffff00">the</mark>   <mark style="background-color: #FFFF00">most</mark>  out of fastai? - do you know lesson 0 and  <mark style="background-color: #ffff00">the</mark>  book _meta learning_ by an alumni?




[17:41](https://www.youtube.com/watch?v=RLvUfyLcT48&t=80s&loop=10&start=17:41&end=20:01) Learn in context  




Do you know people learn naturally (better) with context rather than by theoretical curriculum? - Do you want this course to make you a competent deep learning practitioner by context and practical knowledge? - If you want theory from ground up, should you go to part 2 fastai 2019?



    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">2022_part1/fastai-lecture-2.md</mark> 



-  <mark style="background-color: #ffff00">how</mark>  to  <mark style="background-color: #ffff00">make</mark>   <mark style="background-color: #ffff00">the</mark>   <mark style="background-color: #FFFF00">most</mark>  out of fastai forum?



    



show  <mark style="background-color: #FFFF00">2</mark>  lines above and after in  <mark style="background-color: #ffff00"> <mark style="background-color: #FFFF00">2</mark> 0 <mark style="background-color: #FFFF00">2</mark>  <mark style="background-color: #FFFF00">2</mark> _part1/fastai-lecture- <mark style="background-color: #FFFF00">2</mark> .md</mark> :



- Where can you get more quizzes of fastai and memorize them forever?




#### [02:38](https://www.youtube.com/watch?v=F4tvM4Vb3A0&loop=10&start=02:38&end=04:12) Introducing the forum  




-  <mark style="background-color: #ffff00">how</mark>  to  <mark style="background-color: #ffff00">make</mark>   <mark style="background-color: #ffff00">the</mark>   <mark style="background-color: #FFFF00">most</mark>  out of fastai forum?




#### [04:12](https://www.youtube.com/watch?v=F4tvM4Vb3A0&loop=10&start=04:12&end=05:58) Students' works after week 1




#### [06:08](https://www.youtube.com/watch?v=F4tvM4Vb3A0&loop=10&start=06:08&end=06:46) A Wow moment  



    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">2022_livecoding/live-coding-1.md</mark> 



- -  <mark style="background-color: #ffff00">how</mark>  to  <mark style="background-color: #ffff00">make</mark>   <mark style="background-color: #ffff00">the</mark>   <mark style="background-color: #FFFF00">most</mark>  out of `.bash_history` file? 



    



show  <mark style="background-color: #FFFF00">2</mark>  lines above and after in  <mark style="background-color: #ffff00"> <mark style="background-color: #FFFF00">2</mark> 0 <mark style="background-color: #FFFF00">2</mark>  <mark style="background-color: #FFFF00">2</mark> _livecoding/live-coding-1.md</mark> :



- - How to run jupyter lab without browser? `jupyter lab --no-browser` 




- - How to find and check the content of `.bash_history`? `cat .bash_history` [1:12:53](https://youtu.be/56sIyFjihEc?list=PLfYUBJiXbdtSLBPJ1GMx-sQWf6iNhb8mM&t=4373) 




- -  <mark style="background-color: #ffff00">how</mark>  to  <mark style="background-color: #ffff00">make</mark>   <mark style="background-color: #ffff00">the</mark>   <mark style="background-color: #FFFF00">most</mark>  out of `.bash_history` file? 




- - How to search the .bash_history commands? `ctrl + r` and use `delete` tab to clear the search, use `esc` tab to exit search 




- - How to run a command starting with `ju`? `!ju` 



## What is fastcore.meta all about


```

```


```

```


```

```


```

```
