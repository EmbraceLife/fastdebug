# Introducing fastdebug

> a little tool to help me explore fastai with joy and ease


```
%load_ext autoreload
%autoreload 2
```


```
from IPython.display import display, HTML 
```


```
display(HTML("<style>.container { width:100% !important; }</style>"))
```


<style>.container { width:100% !important; }</style>


## References I use when I explore

### fastai style

What is the fastai coding [style](https://docs.fast.ai/dev/style.html#style-guide)

How to do [abbreviation](https://docs.fast.ai/dev/abbr.html) the fastai way

A great example of how fastai libraries can make life more [comfortable](https://www.fast.ai/2019/08/06/delegation/)

## What is the motivation behind `fastdebug` library?

I have always wanted to explore and learn the fastai libraries thoroughly. However, reading source code is intimidating for beginners even for well-designed and written libraries like fastcore, fastai. So, I have relied on pdbpp to explore source code previously. To do fastai is to do exploratory coding with jupyter, but pdbpp is not working for jupyter at the moment and none of debugging tools I know exactly suit my needs. So, with the help of the amazing nbdev, I created this little library with 4 little tools to assist me explore source code and document my learning along the way.


Here are the four tools:
> Fastdb.snoop(): print out all the executed lines and local vars of the source code I am exploring

> Fastdb.explore(9): start pdb at source line 9 or any srcline at my choose

> Fastdb.print(10, 2): print out the 2nd part of source code, given the source code is divded into multi parts (each part has 10 lines)

> Fastdb.docsrc(10, "comments", "code expression1", "code expression2", "multi-line expressions"): to document the leanring of the srcline 10

As you should know now, this lib does two things: explore and document source code. Let's start with `Fastdb.explore` first on a simple example. If you would like to see it working on a more complex real world example, I have `fastcore.meta.FixSigMeta` [ready](./FixSigMeta.ipynb) for you.

If you find anything confusing or bug-like, please inform me in this forum [post](https://forums.fast.ai/t/hidden-docs-of-fastcore/98455).

## Fastdb.explore

### Why not explore with pure `ipdb.set_trace`?

`Fastdb.explore` is a wrapper around `ipdb.set_trace` and make my life easier when I am exploring because:

> I don't need to write `from ipdb import set_trace` for every notebook

> I don't need to manually open the source code file and scroll down the source code

> I don't need to insert `set_trace()` above every line of source code (srcline) I want to start exploring

> I don't need to remove `set_trace()` from the source code every time after exploration

### How to use `Fastdb.explore`?

Let's explore the source code of `whatinside` from `fastdebug.utils` using this tool.


```
from fastdebug.utils import * # for making an example 
from fastcore.test import * 
import inspect
```


<style>.container { width:100% !important; }</style>



```
import fastdebug.utils as fu
```


```
whatinside(fu) # this is the example we are going to explore whatinside with
```

    fastdebug.utils has: 
    8 items in its __all__, and 
    23 user defined functions, 
    1 classes or class objects, 
    0 builtin funcs and methods, and
    24 callables.
    



```
from fastdebug.core import * # Let's import Fastdb and its dependencies
```


```
# g = locals()
# fdb = Fastdb(whatinside, outloc=g) # first, create an object of Fastdb class, using `whatinside` as param
fdb = Fastdb(whatinside)
```


```
# 1. you can view source code in whole or in parts with the length you set, 
# and it gives you srcline idx so that you can set breakpoint with ease.
#| column: screen
fdb.print(20,1) 
```

    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9)       
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10)      
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11)      
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14)      
        if not lib:===========================================================================(15)      
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
        if hasattr(mo, "__all__") and dun: pprint(mo.__all__)=================================(17)      
        if func: =============================================================================(18)      
            print(f'The user defined functions are:')=========================================(19)      
                                                                                                                                         part No.1 out of 2 parts



```
#| column: screen
# 2. after viewing source code, choose a srcline idx to set breakpoint and write down why I want to explore this line
fdb.eg = """
import fastdebug.utils as fu
whatinside(fu)
"""
```


```
# fdb.explore(11) 
```


```
#| column: screen
# 2. you can set multiple breakpoints from the start if you like (but not necessary)
# fdb.explore([11, 16, 13]) 
```

## Fastdb.snoop

But more often I just want to have an overview of what srclines get run so that I know which lines to dive into and start documenting.

Note: I borrowed `snoop` from snoop library and automated it.


```
fdb.snoop()
```

    06:33:11.07 >>> Call to whatinside in File "/tmp/whatinside.py", line 3
    06:33:11.07 ...... mo = <module 'fastdebug.utils' from '/Users/Natsume/Documents/fastdebug/fastdebug/utils.py'>
    06:33:11.07 ...... dun = False
    06:33:11.07 ...... func = False
    06:33:11.07 ...... clas = False
    06:33:11.07 ...... bltin = False
    06:33:11.07 ...... lib = False
    06:33:11.07 ...... cal = False
    06:33:11.07    3 | def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here
    06:33:11.07   12 |     dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0
    06:33:11.07 .......... dun_all = 8
    06:33:11.07   13 |     funcs = inspect.getmembers(mo, inspect.isfunction)
    06:33:11.07 .......... funcs = [('distribution', <function distribution>), ('expandcell', <function expandcell>), ('inspect_class', <function inspect_class>), ..., ('version', <function version>), ('whatinside', <function whatinside>), ('whichversion', <function whichversion>)]
    06:33:11.07 .......... len(funcs) = 23
    06:33:11.07   14 |     classes = inspect.getmembers(mo, inspect.isclass)
    06:33:11.07 .......... classes = [('ExceptionExpected', <class 'fastcore.test.ExceptionExpected'>)]
    06:33:11.07 .......... len(classes) = 1
    06:33:11.07   15 |     builtins = inspect.getmembers(mo, inspect.isbuiltin)
    06:33:11.07 .......... builtins = []
    06:33:11.07   16 |     callables = inspect.getmembers(mo, callable)
    06:33:11.07 .......... callables = [('ExceptionExpected', <class 'fastcore.test.ExceptionExpected'>), ('distribution', <function distribution>), ('expandcell', <function expandcell>), ..., ('version', <function version>), ('whatinside', <function whatinside>), ('whichversion', <function whichversion>)]
    06:33:11.07 .......... len(callables) = 24
    06:33:11.07   17 |     pkgpath = os.path.dirname(mo.__file__)
    06:33:11.07 .......... pkgpath = '/Users/Natsume/Documents/fastdebug/fastdebug'
    06:33:11.07   18 |     if not lib:
    06:33:11.07   19 |         print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  
    06:33:11.07   20 |     if hasattr(mo, "__all__") and dun: pprint(mo.__all__)
    06:33:11.08   21 |     if func: 
    06:33:11.08   24 |     if clas: 
    06:33:11.08   27 |     if bltin: 
    06:33:11.08   30 |     if cal: 
    06:33:11.08   33 |     if lib: 
    06:33:11.08 <<< Return value from whatinside: None


    ========================================================     Investigating [91;1mwhatinside[0m     ========================================================
    ==============================================================     on line [91;1mNone[0m     ==============================================================
    =======================================     with example [91;1m
    import fastdebug.utils as fu
    whatinside(fu)
    [0m     =======================================
    
    fastdebug.utils has: 
    8 items in its __all__, and 
    23 user defined functions, 
    1 classes or class objects, 
    0 builtin funcs and methods, and
    24 callables.
    


## Fastdb.docsrc

After exploring and snooping, if I realize there is something new to learn and maybe want to come back for a second look, I find `ipdb` and the alike are not designed to document my learning. So, I created `docsrc` to make my life easier in the following ways:

> I won't need to scroll through a long cell output of pdb commands, src prints and results to find what I learnt during exploration

> I won't need to type all the expressions during last exploration to regenerate the findings for me

> I can choose any srclines to explore and write any sinlge or multi-line expressions to evaluate the srcline

> I can write down what I learn or what is new on any srcline as comment, and all comments are attached to the src code for review

> All expressions with results and comments for each srcline under exploration are documented for easy reviews

> Of course, no worry about your original source code, as it is untouched.


### Import


```
from fastdebug.core import * # to make Fastdb available
from fastdebug.utils import whatinside # for making an example 
```

### Initiating


```
g = locals()
fdb = Fastdb(whatinside, outloc=g) # use either fu.whatinside or whatinside is fine
```


```
#| column: screen
fdb.print(maxlines=20, part=1) # view the source code with idx 
```

    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9)       
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10)      
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11)      
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14)      
        if not lib:===========================================================================(15)      
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
        if hasattr(mo, "__all__") and dun: pprint(mo.__all__)=================================(17)      
        if func: =============================================================================(18)      
            print(f'The user defined functions are:')=========================================(19)      
                                                                                                                                         part No.1 out of 2 parts


### What does the first line do?


```
fdb.eg = "whatinside(fu)"
```


```
#| column: screen
fdb.docsrc(9, "how many items inside mo.__all__?", "mo", \
"if hasattr(mo, '__all__'):\\n\
    printright(f'mo: {mo}')\\n\
    printright(f'mo.__all__: {mo.__all__}')\\n\
    printright(f'len(mo.__all__): {len(mo.__all__)}')") 
```

    ========================================================     Investigating [91;1mwhatinside[0m     ========================================================
    ===============================================================     on line [91;1m9[0m     ================================================================
    ======================================================     with example [91;1mwhatinside(fu)[0m     =======================================================
    
    [93;1mprint selected srcline with expands below[0m--------
                 ):                                                                                                                                         (7)
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'                                                                (8)
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0==========================================================================================(9)
                                                                                                                                [91;1mhow many items inside mo.__all__?[0m
        funcs = inspect.getmembers(mo, inspect.isfunction)                                                                                                  (10)
        classes = inspect.getmembers(mo, inspect.isclass)                                                                                                   (11)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                               mo => mo : <module 'fastdebug.utils' from '/Users/Natsume/Documents/fastdebug/fastdebug/utils.py'>
    
    
    if hasattr(mo, '__all__'):
        printright(f'mo: {mo}')
        printright(f'mo.__all__: {mo.__all__}')
        printright(f'len(mo.__all__): {len(mo.__all__)}')     
    
    Running the code block above => ====================================================================
    
                                                                      mo: <module 'fastdebug.utils' from '/Users/Natsume/Documents/fastdebug/fastdebug/utils.py'>
                                         mo.__all__: ['expand', 'test_eq', 'test_is', 'expandcell', 'inspect_class', 'ismetaclass', 'whatinside', 'whichversion']
                                                                                                                                               len(mo.__all__): 8
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    fastdebug.utils has: 
    8 items in its __all__, and 
    23 user defined functions, 
    1 classes or class objects, 
    0 builtin funcs and methods, and
    24 callables.
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9) # [37;1mhow many items inside mo.__all__?[0m; 
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10)      
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11)      
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14)      
        if not lib:===========================================================================(15)      
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
        if hasattr(mo, "__all__") and dun: pprint(mo.__all__)=================================(17)      
        if func: =============================================================================(18)      
            print(f'The user defined functions are:')=========================================(19)      
                                                                                                                                         part No.1 out of 2 parts
    



```
#| column: screen
dbsrc = fdb.docsrc(10, "get all funcs of a module", "mo", "inspect.getdoc(inspect.isfunction)", \
            "inspect.getdoc(inspect.getmembers)", "funcs = inspect.getmembers(mo, inspect.isfunction)")
```

    ========================================================     Investigating [91;1mwhatinside[0m     ========================================================
    ===============================================================     on line [91;1m10[0m     ===============================================================
    ======================================================     with example [91;1mwhatinside(fu)[0m     =======================================================
    
    [93;1mprint selected srcline with expands below[0m--------
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'                                                                (8)
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0                                                                                          (9)
        funcs = inspect.getmembers(mo, inspect.isfunction)==================================================================================================(10)
                                                                                                                                        [91;1mget all funcs of a module[0m
        classes = inspect.getmembers(mo, inspect.isclass)                                                                                                   (11)
        builtins = inspect.getmembers(mo, inspect.isbuiltin)                                                                                                (12)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                               mo => mo : <module 'fastdebug.utils' from '/Users/Natsume/Documents/fastdebug/fastdebug/utils.py'>
    
    
    inspect.getdoc(inspect.isfunction) => inspect.getdoc(inspect.isfunction) : Return true if the object is a user-defined function.
    
    Function objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this function was defined
        __code__        code object containing compiled function bytecode
        __defaults__    tuple of any default values for arguments
        __globals__     global namespace in which this function was defined
        __annotations__ dict of parameter annotations
        __kwdefaults__  dict of keyword only parameters with defaults
    
    
    inspect.getdoc(inspect.getmembers) => inspect.getdoc(inspect.getmembers) : Return all members of an object as (name, value) pairs sorted by name.
    Optionally, only return members that satisfy a given predicate.
    
    
    funcs = inspect.getmembers(mo, inspect.isfunction) => funcs: [('distribution', <function distribution>), ('expandcell', <function expandcell>), ('inspect_class', <function inspect_class>), ('is_close', <function is_close>), ('ismetaclass', <function ismetaclass>), ('metadata', <function metadata>), ('nequals', <function nequals>), ('pprint', <function pprint>), ('python_version', <function python_version>), ('test', <function test>), ('test_close', <function test_close>), ('test_eq', <function test_eq>), ('test_eq_type', <function test_eq_type>), ('test_fail', <function test_fail>), ('test_fig_exists', <function test_fig_exists>), ('test_is', <function test_is>), ('test_ne', <function test_ne>), ('test_shuffled', <function test_shuffled>), ('test_stdout', <function test_stdout>), ('test_warns', <function test_warns>), ('version', <function version>), ('whatinside', <function whatinside>), ('whichversion', <function whichversion>)]
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    fastdebug.utils has: 
    8 items in its __all__, and 
    23 user defined functions, 
    1 classes or class objects, 
    0 builtin funcs and methods, and
    24 callables.
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9) # [92;1mhow many items inside mo.__all__?[0m; 
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10) # [34;1mget all funcs of a module[0m; 
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11)      
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14)      
        if not lib:===========================================================================(15)      
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
        if hasattr(mo, "__all__") and dun: pprint(mo.__all__)=================================(17)      
        if func: =============================================================================(18)      
            print(f'The user defined functions are:')=========================================(19)      
                                                                                                                                         part No.1 out of 2 parts
    


### If I find the src is too long, and I customize the print out of src the way I like


```
#| column: screen
fdb.print(maxlines=15, part=1)
```

    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9) # [36;1mhow many items inside mo.__all__?[0m; 
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10) # [92;1mget all funcs of a module[0m; 
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11)      
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14)      
                                                                                                                                         part No.1 out of 3 parts


### I can write a block of codes to evaluate


```
import fastcore.meta as core
```


```
#| column: screen
# fdb.takExample("whatinside(core)", whatinside=whatinside, core=core)
fdb.eg = "whatinside(core)"
dbsrc = fdb.docsrc(11, "get all classes from the module", \
"clas = inspect.getmembers(mo, inspect.isclass)\\n\
for c in clas:\\n\
    print(c)")
```

    ========================================================     Investigating [91;1mwhatinside[0m     ========================================================
    ===============================================================     on line [91;1m11[0m     ===============================================================
    =====================================================     with example [91;1mwhatinside(core)[0m     ======================================================
    
    [93;1mprint selected srcline with expands below[0m--------
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0                                                                                          (9)
        funcs = inspect.getmembers(mo, inspect.isfunction)                                                                                                  (10)
        classes = inspect.getmembers(mo, inspect.isclass)===================================================================================================(11)
                                                                                                                                  [91;1mget all classes from the module[0m
        builtins = inspect.getmembers(mo, inspect.isbuiltin)                                                                                                (12)
        callables = inspect.getmembers(mo, callable)                                                                                                        (13)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
    clas = inspect.getmembers(mo, inspect.isclass)
    for c in clas:
        print(c)                                                                                   
    
    Running the code block above => ====================================================================
    
    ('AutoInit', <class 'fastcore.meta.AutoInit'>)
    ('BuiltinFunctionType', <class 'builtin_function_or_method'>)
    ('BuiltinMethodType', <class 'builtin_function_or_method'>)
    ('BypassNewMeta', <class 'fastcore.meta.BypassNewMeta'>)
    ('ExceptionExpected', <class 'fastcore.test.ExceptionExpected'>)
    ('FixSigMeta', <class 'fastcore.meta.FixSigMeta'>)
    ('FunctionType', <class 'function'>)
    ('MethodDescriptorType', <class 'method_descriptor'>)
    ('MethodType', <class 'method'>)
    ('MethodWrapperType', <class 'method-wrapper'>)
    ('NewChkMeta', <class 'fastcore.meta.NewChkMeta'>)
    ('NoneType', <class 'NoneType'>)
    ('Path', <class 'pathlib.Path'>)
    ('PrePostInitMeta', <class 'fastcore.meta.PrePostInitMeta'>)
    ('SimpleNamespace', <class 'types.SimpleNamespace'>)
    ('WrapperDescriptorType', <class 'wrapper_descriptor'>)
    ('attrgetter', <class 'operator.attrgetter'>)
    ('itemgetter', <class 'operator.itemgetter'>)
    ('partial', <class 'functools.partial'>)
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    fastcore.meta has: 
    13 items in its __all__, and 
    43 user defined functions, 
    19 classes or class objects, 
    2 builtin funcs and methods, and
    74 callables.
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9) # [92;1mhow many items inside mo.__all__?[0m; 
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10) # [93;1mget all funcs of a module[0m; 
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11) # [35;1mget all classes from the module[0m; 
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14)      
        if not lib:===========================================================================(15)      
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
        if hasattr(mo, "__all__") and dun: pprint(mo.__all__)=================================(17)      
        if func: =============================================================================(18)      
            print(f'The user defined functions are:')=========================================(19)      
                                                                                                                                         part No.1 out of 2 parts
    



```
#| column: screen
dbsrc = fdb.docsrc(14, "get the file path of the module", "mo.__file__", "inspect.getdoc(os.path.dirname)", "pkgpath = os.path.dirname(mo.__file__)")
```

    ========================================================     Investigating [91;1mwhatinside[0m     ========================================================
    ===============================================================     on line [91;1m14[0m     ===============================================================
    =====================================================     with example [91;1mwhatinside(core)[0m     ======================================================
    
    [93;1mprint selected srcline with expands below[0m--------
        builtins = inspect.getmembers(mo, inspect.isbuiltin)                                                                                                (12)
        callables = inspect.getmembers(mo, callable)                                                                                                        (13)
        pkgpath = os.path.dirname(mo.__file__)==============================================================================================================(14)
                                                                                                                                  [91;1mget the file path of the module[0m
        if not lib:                                                                                                                                         (15)
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                              mo.__file__ => mo.__file__ : /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore/meta.py
    
    
                                               inspect.getdoc(os.path.dirname) => inspect.getdoc(os.path.dirname) : Returns the directory component of a pathname
    
    
                                                pkgpath = os.path.dirname(mo.__file__) => pkgpath: /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
    fastcore.meta has: 
    13 items in its __all__, and 
    43 user defined functions, 
    19 classes or class objects, 
    2 builtin funcs and methods, and
    74 callables.
    
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9) # [93;1mhow many items inside mo.__all__?[0m; 
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10) # [92;1mget all funcs of a module[0m; 
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11) # [93;1mget all classes from the module[0m; 
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14) # [93;1mget the file path of the module[0m; 
        if not lib:===========================================================================(15)      
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
        if hasattr(mo, "__all__") and dun: pprint(mo.__all__)=================================(17)      
        if func: =============================================================================(18)      
            print(f'The user defined functions are:')=========================================(19)      
                                                                                                                                         part No.1 out of 2 parts
    



```
#| column: screen
# fdb.takExample("whatinside(core, lib=True)", whatinside=whatinside, core=core)
fdb.eg = "whatinside(core, lib=True)"
dbsrc = fdb.docsrc(30, "get names of all modules of a lib", "pkgpath", "inspect.getdoc(pkgutil.iter_modules)", \
"for a, b, c in pkgutil.iter_modules([pkgpath]):\\n\
    printright(f'{a} ; {b}; {c}')", db=True)
```

    ========================================================     Investigating [91;1mwhatinside[0m     ========================================================
    ===============================================================     on line [91;1m30[0m     ===============================================================
    ================================================     with example [91;1mwhatinside(core, lib=True)[0m     =================================================
    
    [93;1mprint selected srcline with expands below[0m--------
            print(f'The callables are: ')                                                                                                                   (28)
            pprint([i[0] for i in callables])                                                                                                               (29)
        if lib: ============================================================================================================================================(30)
                                                                                                                                [91;1mget names of all modules of a lib[0m
            modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]                                                                              (31)
            print(f'The library has {len(modules)} modules')                                                                                                (32)
    
    ==================================================================================================================[91;1mStart of my srcline exploration:[0m
    
    
                                                                              pkgpath => pkgpath : /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore
    
    
    inspect.getdoc(pkgutil.iter_modules) => inspect.getdoc(pkgutil.iter_modules) : Yields ModuleInfo for all submodules on path,
    or, if path is None, all top-level modules on sys.path.
    
    'path' should be either None or a list of paths to look for
    modules in.
    
    'prefix' is a string to output on the front of every module name
    on output.
    
    
    for a, b, c in pkgutil.iter_modules([pkgpath]):
        printright(f'{a} ; {b}; {c}')                                                                            
    
    Running the code block above => ====================================================================
    
                                                                    FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; _modidx; False
                                                                     FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; _nbdev; False
                                                                        FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; all; False
                                                                     FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; basics; False
                                                                   FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; dispatch; False
                                                                   FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; docments; False
                                                                  FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; docscrape; False
                                                                 FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; foundation; False
                                                                    FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; imports; False
                                                                       FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; meta; False
                                                                 FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; nb_imports; False
                                                                        FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; net; False
                                                                   FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; parallel; False
                                                                     FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; script; False
                                                                     FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; shutil; False
                                                                      FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; style; False
                                                                       FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; test; False
                                                                  FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; transform; False
                                                                      FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; utils; False
                                                                        FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; xdg; False
                                                                      FileFinder('/Users/Natsume/mambaforge/lib/python3.9/site-packages/fastcore') ; xtras; False
    ====================================================================================================================[91;1mEnd of my srcline exploration:[0m
    
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
    
    [93;1mReview srcode with all comments added so far[0m======================================================================================================
            pprint([i[0] for i in funcs])=====================================================(20)      
        if clas: =============================================================================(21)      
            print(f'The class objects are:')==================================================(22)      
            pprint([i[0] for i in classes])===================================================(23)      
        if bltin: ============================================================================(24)      
            print(f'The builtin functions or methods are:')===================================(25)      
            pprint([i[0] for i in builtins])==================================================(26)      
        if cal: ==============================================================================(27)      
            print(f'The callables are: ')=====================================================(28)      
            pprint([i[0] for i in callables])=================================================(29)      
        if lib: ==============================================================================(30) # [36;1mget names of all modules of a lib[0m; 
            modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]================(31)      
            print(f'The library has {len(modules)} modules')==================================(32)      
            pprint(modules)===================================================================(33)      
                                                                                                                                                            (34)
                                                                                                                                         part No.2 out of 2 parts
    


### Print out the entire src with idx and comments, when I finish documenting


```
fdb.print()
```

    ========================================================     Investigating [91;1mwhatinside[0m     ========================================================
    ===============================================================     on line [91;1m30[0m     ===============================================================
    ================================================     with example [91;1mwhatinside(core, lib=True)[0m     =================================================
    
    def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
                   dun:bool=False, # print all items in __all__===============================(1)       
                   func:bool=False, # print all user defined functions========================(2)       
                   clas:bool=False, # print all class objects=================================(3)       
                   bltin:bool=False, # print all builtin funcs or methods=====================(4)       
                   lib:bool=False, # print all the modules of the library it belongs to=======(5)       
                   cal:bool=False # print all callables=======================================(6)       
                 ): ==========================================================================(7)       
        'Check what inside a module: `__all__`, functions, classes, builtins, and callables'==(8)       
        dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0============================(9) # [91;1mhow many items inside mo.__all__?[0m; 
        funcs = inspect.getmembers(mo, inspect.isfunction)====================================(10) # [91;1mget all funcs of a module[0m; 
        classes = inspect.getmembers(mo, inspect.isclass)=====================================(11) # [93;1mget all classes from the module[0m; 
        builtins = inspect.getmembers(mo, inspect.isbuiltin)==================================(12)      
        callables = inspect.getmembers(mo, callable)==========================================(13)      
        pkgpath = os.path.dirname(mo.__file__)================================================(14) # [37;1mget the file path of the module[0m; 
        if not lib:===========================================================================(15)      
            print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  (16)
        if hasattr(mo, "__all__") and dun: pprint(mo.__all__)=================================(17)      
        if func: =============================================================================(18)      
            print(f'The user defined functions are:')=========================================(19)      
            pprint([i[0] for i in funcs])=====================================================(20)      
        if clas: =============================================================================(21)      
            print(f'The class objects are:')==================================================(22)      
            pprint([i[0] for i in classes])===================================================(23)      
        if bltin: ============================================================================(24)      
            print(f'The builtin functions or methods are:')===================================(25)      
            pprint([i[0] for i in builtins])==================================================(26)      
        if cal: ==============================================================================(27)      
            print(f'The callables are: ')=====================================================(28)      
            pprint([i[0] for i in callables])=================================================(29)      
        if lib: ==============================================================================(30) # [35;1mget names of all modules of a lib[0m; 
            modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]================(31)      
            print(f'The library has {len(modules)} modules')==================================(32)      
            pprint(modules)===================================================================(33)      
                                                                                                                                                            (34)



```

```

### After running `.dbprint`, everything is back to normal automatically


```
inspect.getsourcefile(fu.whatinside)
```




    '/Users/Natsume/Documents/fastdebug/fastdebug/utils.py'




```
inspect.getsourcefile(whatinside)
```




    '/Users/Natsume/Documents/fastdebug/fastdebug/utils.py'




```

```


```

```

To check, when run `whatinside??` we should see the actually source code whereas the db version of `whatinside` does not have.

## Install

```sh
pip install fastdebug
```

## How to use

Fill me in please! Don't forget code examples:


```
1+1
```




    2



#|hide
## Send to Obsidian


```
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/index.ipynb
!mv /Users/Natsume/Documents/fastdebug/index.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/

!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```

    [jupytext] Reading /Users/Natsume/Documents/fastdebug/index.ipynb in format ipynb
    [jupytext] Writing /Users/Natsume/Documents/fastdebug/index.md
    [NbConvertApp] WARNING | pattern 'Users/Natsume/Documents/fastdebug/Demos/fastcore/*.ipynb' matched no files
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/utils.ipynb to markdown
    [NbConvertApp] Writing 14660 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/utils.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/00_core.ipynb to markdown
    [NbConvertApp] Writing 405737 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_core.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/index.ipynb to markdown
    [NbConvertApp] Writing 58037 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/index.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/08_use_kwargs_dict.ipynb to markdown
    [NbConvertApp] Writing 56920 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/08_use_kwargs_dict.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/04_fastcore.meta._rm_self.ipynb to markdown
    [NbConvertApp] Writing 15997 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/04_fastcore.meta._rm_self.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/03_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb to markdown
    [NbConvertApp] Writing 89716 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/03_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/05_fastcore.meta.test_sig.ipynb to markdown
    [NbConvertApp] Writing 10298 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/05_fastcore.meta.test_sig.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/01_signature_from_callable_with_FixSigMeta.ipynb to markdown
    [NbConvertApp] Writing 47166 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/01_signature_from_callable_with_FixSigMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/07_fastcore.meta.BypassNewMeta.ipynb to markdown
    [NbConvertApp] Writing 30628 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/07_fastcore.meta.BypassNewMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/00_fastcore_meta_delegates.ipynb to markdown
    [NbConvertApp] Writing 98471 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_fastcore_meta_delegates.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/Demos/06_fastcore.meta.NewChkMeta.ipynb to markdown
    [NbConvertApp] Writing 30935 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/06_fastcore.meta.NewChkMeta.md



```

```
