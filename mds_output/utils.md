# Utils

> little functions to tell you the basics of a module


```
#| default_exp utils
```


```
%load_ext autoreload
%autoreload 2
```

## Expand cells


```
#| export
def expandcell():
    "expand cells of the current notebook to its full width"
    from IPython.display import display, HTML 
    display(HTML("<style>.container { width:100% !important; }</style>"))
```


```
#| export
expand = expandcell()
```


<style>.container { width:100% !important; }</style>


## Import fastcore env


```
#| export
from fastcore.test import * # so that it automated
```


```
#| export
test_eq = test_eq
test_is = test_is
```


```
#| export 
from fastcore.imports import FunctionType, MethodType
```


```
#| export
FunctionType = FunctionType
MethodType = MethodType
```

## to inspect a class


```

def inspect_class(c):
    "examine the details of a class"
    try:
        print(inspect.getsource(c))
    except: 
        pass
    print()
    print(f'is {c.__name__} a metaclass: {ismetaclass(c)}')
    print(f'is {c.__name__} created by a metaclass: {False if c.__class__ == type else True}')
    if c.__class__ is not type:
        print(f'{c.__name__} is created by metaclass {c.__class__}')
    else:
        print(f'{c.__name__} is created by {c.__class__}')
    print(f'{c.__name__}.__new__ is object.__new__: {c.__new__ is object.__new__}')   
    print(f'{c.__name__}.__new__ is type.__new__: {c.__new__ is type.__new__}')       
    print(f'{c.__name__}.__new__: {c.__new__}')
    print(f'{c.__name__}.__init__ is object.__init__: {c.__init__ is object.__init__}')
    print(f'{c.__name__}.__init__ is type.__init__: {c.__init__ is type.__init__}')    
    print(f'{c.__name__}.__init__: {c.__init__}')
    print(f'{c.__name__}.__call__ is object.__call__: {c.__call__ is object.__call__}')
    print(f'{c.__name__}.__call__ is type.__call__: {c.__call__ is type.__call__}')    
    print(f'{c.__name__}.__call__: {c.__call__}')
    print(f'{c.__name__}.__class__: {c.__class__}')
    print(f'{c.__name__}.__bases__: {c.__bases__}')
    print(f'{c.__name__}.__mro__: {c.__mro__}')
    
    if c.__class__ is not type:
        print()
        print(f'{c.__name__}\'s metaclass {c.__class__}\'s function members are:')
        funcs = {item[0]: item[1] for item in inspect.getmembers(c.__class__) \
                 if inspect.isfunction(getattr(c.__class__, item[0], None))}   
        pprint(funcs)
        
    
    
    funcs = {item[0]: item[1] for item in inspect.getmembers(c) \
             if inspect.isfunction(getattr(c, item[0], None))}
    methods = {item[0]: item[1] for item in inspect.getmembers(c) \
             if inspect.ismethod(getattr(c, item[0], None))}
    classes = {item[0]: item[1] for item in inspect.getmembers(c) \
             if inspect.isclass(getattr(c, item[0], None))}
    
    print()
    print(f'{c.__name__}\'s function members are:')
    pprint(funcs)
    print()
    print(f'{c.__name__}\'s method members are:')
    pprint(methods)
    print()
    print(f'{c.__name__}\'s class members are:')
    pprint(classes)
    print()
    print(f'{c.__name__}\'s namespace are:')
    pprint(c.__dict__)
```

### get the docs for each function of a class


```

```


```
#| export 
def inspect_class(c, src=False):
    "examine the details of a class"
    if src:
        try:
            print(inspect.getsource(c))
        except: 
            pass
    print()
    print(f'is {c.__name__} a metaclass: {ismetaclass(c)}')
    print(f'is {c.__name__} created by a metaclass: {False if c.__class__ == type else True}')
    if c.__class__ is not type:
        print(f'{c.__name__} is created by metaclass {c.__class__}')
    else:
        print(f'{c.__name__} is created by {c.__class__}')
    print(f'{c.__name__}.__new__ is object.__new__: {c.__new__ is object.__new__}')   
    print(f'{c.__name__}.__new__ is type.__new__: {c.__new__ is type.__new__}')       
    print(f'{c.__name__}.__new__: {c.__new__}')
    print(f'{c.__name__}.__init__ is object.__init__: {c.__init__ is object.__init__}')
    print(f'{c.__name__}.__init__ is type.__init__: {c.__init__ is type.__init__}')    
    print(f'{c.__name__}.__init__: {c.__init__}')
    print(f'{c.__name__}.__call__ is object.__call__: {c.__call__ is object.__call__}')
    print(f'{c.__name__}.__call__ is type.__call__: {c.__call__ is type.__call__}')    
    print(f'{c.__name__}.__call__: {c.__call__}')
    print(f'{c.__name__}.__class__: {c.__class__}')
    print(f'{c.__name__}.__bases__: {c.__bases__}')
    print(f'{c.__name__}.__mro__: {c.__mro__}')
    
    if c.__class__ is not type:
        print()
        print(f'{c.__name__}\'s metaclass {c.__class__}\'s function members are:')
        funcs = {item[0]: item[1] for item in inspect.getmembers(c.__class__) \
                 if inspect.isfunction(getattr(c.__class__, item[0], None))}   
        pprint(funcs)
        
    funcs = {item[0]: item[1] for item in inspect.getmembers(c) \
             if inspect.isfunction(getattr(c, item[0], None))}
    methods = {item[0]: item[1] for item in inspect.getmembers(c) \
             if inspect.ismethod(getattr(c, item[0], None))}
    classes = {item[0]: item[1] for item in inspect.getmembers(c) \
             if inspect.isclass(getattr(c, item[0], None))}
    
    print()
    print(f'{c.__name__}\'s function members are:')
    # todos: print some space between k and v
    for k, v in funcs.items():
        print(f"{k}: {inspect.getdoc(v)}")
    print()
    print(f'{c.__name__}\'s method members are:')
    pprint(methods)
    print()
    print(f'{c.__name__}\'s class members are:')
    pprint(classes)
    print()
    print(f'{c.__name__}\'s namespace are:')
    pprint(c.__dict__)
```


```

```

## is it a metaclass?


```
#| export
def ismetaclass(mc): 
    "check whether a class is a metaclass or not"
    if inspect.isclass(mc):
        return type in mc.__mro__ 
```


```
from fastcore.meta import *
import inspect
```


```
ismetaclass(FixSigMeta)
```




    True




```
ismetaclass(PrePostInitMeta)
```




    True



## is it a decorator


```

def isdecorator(obj):
    if inspect.isfunction(obj):
        count = 0
        defretn = ""
        for l in inspect.getsource(obj).split('\n'):
            if "def " in l:
                if count >=1:
                    defretn = l
                count = count + 1
            if "return " in l and "partial(" in l:
                return True
            if "return " in l: 
                retn = l.split('return ')[1]
                if "(" not in retn:
                    if retn in defretn:
                        return True
                    try:
                        retneval = eval(retn, obj.__globals__)
                    except NameError:
                        return False
                    if type(retneval).__name__ == 'function':
                        return True
                
        return False

```

### handle all kinds of exceptions for evaluating retn 


```
#| export
def isdecorator(obj):
    "check whether a function is a decorator"
    if inspect.isfunction(obj):
        count = 0
        defretn = ""
        for l in inspect.getsource(obj).split('\n'):
            if "def " in l:
                if count >=1:
                    defretn = l
                count = count + 1
            if "return " in l and "partial(" in l:
                return True
            if "return " in l: 
                retn = l.split('return ')[1]
                if "(" not in retn:
                    if retn in defretn:
                        return True
                    try:
                        retneval = eval(retn, obj.__globals__)
                    except:
                        return False
                    if type(retneval).__name__ == 'function':
                        return True
                
        return False

```


```
test_eq(isdecorator(delegates), True)
```


```
test_eq(isdecorator(test_sig), False)
```

## whatinside a module of a library


```
#| export
# from inspect import getmembers, isfunction, isclass, isbuiltin, getsource
import os.path, pkgutil
from pprint import pprint
import inspect

```


```

def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here
               dun:bool=False, # print all items in __all__
               func:bool=False, # print all user defined functions
               clas:bool=False, # print all class objects
               bltin:bool=False, # print all builtin funcs or methods
               lib:bool=False, # print all the modules of the library it belongs to
               cal:bool=False # print all callables
             ): 
    'Check what inside a module: `__all__`, functions, classes, builtins, and callables'
    dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0
    funcs = inspect.getmembers(mo, inspect.isfunction)
    classes = inspect.getmembers(mo, inspect.isclass)
    builtins = inspect.getmembers(mo, inspect.isbuiltin)
    callables = inspect.getmembers(mo, callable)
    pkgpath = os.path.dirname(mo.__file__)
    if not lib:
        print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  
    if hasattr(mo, "__all__") and dun: pprint(mo.__all__)
    if func: 
        print(f'The user defined functions are:')
        pprint([i[0] for i in funcs])
    if clas: 
        print(f'The class objects are:')
        pprint([i[0] for i in classes])
    if bltin: 
        print(f'The builtin functions or methods are:')
        pprint([i[0] for i in builtins])
    if cal: 
        print(f'The callables are: ')
        pprint([i[0] for i in callables])
    if lib: 
        modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
        print(f'The library has {len(modules)} modules')
        pprint(modules)
```

### show the type of objects inside `__all__`


```

def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here
               dun:bool=False, # print all items in __all__
               func:bool=False, # print all user defined functions
               clas:bool=False, # print all class objects
               bltin:bool=False, # print all builtin funcs or methods
               lib:bool=False, # print all the modules of the library it belongs to
               cal:bool=False # print all callables
             ): 
    'Check what inside a module: `__all__`, functions, classes, builtins, and callables'
    dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0
    funcs = inspect.getmembers(mo, inspect.isfunction)
    classes = inspect.getmembers(mo, inspect.isclass)
    builtins = inspect.getmembers(mo, inspect.isbuiltin)
    callables = inspect.getmembers(mo, callable)
    pkgpath = os.path.dirname(mo.__file__)
    module_env = mo.__dict__
    kind = None # assignment first before reference
    if not lib:
        print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  
    if hasattr(mo, "__all__") and dun: 
        maxlen = max(map(lambda i : len(i) , mo.__all__ ))
        for i in mo.__all__:
            obj = eval(i, module_env)
            if ismetaclass(obj):
                kind = "metaclass" 
            elif inspect.isclass(obj):
                kind = "class"
            elif isdecorator(obj):
                kind = "decorator"
            elif inspect.isfunction(obj):
                kind = "function"
            tp = type(eval(i, module_env)).__name__
            startlen = len(i)
            if tp == kind: print(i + ":" + " "*(maxlen-startlen + 5) + kind + "    " + \
                                 inspect.getdoc(eval(i, module_env)))  
            elif tp != 'NoneType': print(i + ":" + " "*(maxlen-startlen+5) + kind + ", " + tp + "    " + \
                                 inspect.getdoc(eval(i, module_env)))
            else: print(i + ":" + tp)
    if func: 
        print(f'The user defined functions are:')
        maxlen = max(map(lambda i : len(i[0]) , funcs ))
        for i in funcs:
            if isdecorator(i[1]):
                kind = "decorator"
            elif inspect.isfunction(i[1]):
                kind = "function"
#             print(f"{i[0]}: {kind}")  
            startlen = len(i[0])
            print(i[0] + ":" + " "*(maxlen-startlen + 5) + kind + "    " + \
                                 str(inspect.signature(i[1])))               
    if clas: 
        print(f'The class objects are:')
        maxlen = max(map(lambda i : len(i[0]) , funcs ))
        for i in classes:
            if ismetaclass(i[1]):
                kind = "metaclass"
            elif inspect.isclass(i[1]):
                kind = "class"
#             print(f"{i[0]}: {kind}")  
            startlen = len(i[0])
            if not inspect.isbuiltin(i[1]):         
                print(i[0] + ":" + " "*(maxlen-startlen + 5) + kind)
    if bltin: 
        print(f'The builtin functions or methods are:')
        pprint([i[0] for i in builtins])
    if cal: 
        print(f'The callables are: ')
        pprint([i[0] for i in callables])
    if lib: 
        modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
        print(f'The library has {len(modules)} modules')
        pprint(modules)
```

### working for fastdebug.core


```

def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here
               dun:bool=False, # print all items in __all__
               func:bool=False, # print all user defined functions
               clas:bool=False, # print all class objects
               bltin:bool=False, # print all builtin funcs or methods
               lib:bool=False, # print all the modules of the library it belongs to
               cal:bool=False # print all callables
             ): 
    'Check what inside a module: `__all__`, functions, classes, builtins, and callables'
    dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0
    funcs = inspect.getmembers(mo, inspect.isfunction)
    classes = inspect.getmembers(mo, inspect.isclass)
    builtins = inspect.getmembers(mo, inspect.isbuiltin)
    callables = inspect.getmembers(mo, callable)
    pkgpath = os.path.dirname(mo.__file__)
    module_env = mo.__dict__
    kind = None # assignment first before reference
    if not lib:
        print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  
    if hasattr(mo, "__all__") and dun: 
        maxlen = max(map(lambda i : len(i) , mo.__all__ ))
        for i in mo.__all__:
            obj = eval(i, module_env)
            if ismetaclass(obj):
                kind = "metaclass" 
            elif inspect.isclass(obj):
                kind = "class"
            elif isdecorator(obj):
                kind = "decorator"
            elif inspect.isfunction(obj):
                kind = "function"
            tp = type(eval(i, module_env)).__name__
            startlen = len(i)
            if tp == kind: print(i + ":" + " "*(maxlen-startlen + 5) + kind + "    " + \
                                 inspect.getdoc(eval(i, module_env)))  
            elif kind != None and callable(eval(i, module_env)): print(i + ":" + " "*(maxlen-startlen+5) + kind + ", " + tp + "    " + \
                                 str(inspect.getdoc(eval(i, module_env))))
#             elif tp != 'NoneType': print(i + ":" + " "*(maxlen-startlen+5) + kind + ", " + tp + "    " + \
#                                  inspect.getdoc(eval(i, module_env)))
            else: print(i + ":" + tp)
    if func: 
        print(f'The user defined functions are:')
        maxlen = max(map(lambda i : len(i[0]) , funcs ))
        for i in funcs:
            if isdecorator(i[1]):
                kind = "decorator"
            elif inspect.isfunction(i[1]):
                kind = "function"
#             print(f"{i[0]}: {kind}")  
            startlen = len(i[0])
            print(i[0] + ":" + " "*(maxlen-startlen + 5) + kind + "    " + \
                                 str(inspect.signature(i[1])))               
    if clas: 
        print(f'The class objects are:')
        maxlen = max(map(lambda i : len(i[0]) , funcs ))
        for i in classes:
            if ismetaclass(i[1]):
                kind = "metaclass"
            elif inspect.isclass(i[1]):
                kind = "class"
#             print(f"{i[0]}: {kind}")  
            startlen = len(i[0])
            if not inspect.isbuiltin(i[1]):         
                print(i[0] + ":" + " "*(maxlen-startlen + 5) + kind)
    if bltin: 
        print(f'The builtin functions or methods are:')
        pprint([i[0] for i in builtins])
    if cal: 
        print(f'The callables are: ')
        pprint([i[0] for i in callables])
    if lib: 
        modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
        print(f'The library has {len(modules)} modules')
        pprint(modules)
```

### to show Fastdb methods


```
#| export
def whatinside(mo, # module, e.g., `import fastcore.all as fa`, use `fa` here
               dun:bool=False, # print all items in __all__
               func:bool=False, # print all user defined functions
               method:bool=False, 
               clas:bool=False, # print all class objects
               bltin:bool=False, # print all builtin funcs or methods
               lib:bool=False, # print all the modules of the library it belongs to
               cal:bool=False # print all callables
             ): 
    'Check what inside a module: `__all__`, functions, classes, builtins, and callables'
    dun_all = len(mo.__all__) if hasattr(mo, "__all__") else 0
    funcs = inspect.getmembers(mo, inspect.isfunction)
    methods = inspect.getmembers(mo, inspect.ismethod)    
    classes = inspect.getmembers(mo, inspect.isclass)
    builtins = inspect.getmembers(mo, inspect.isbuiltin)
    callables = inspect.getmembers(mo, callable)
    pkgpath = os.path.dirname(mo.__file__)
    module_env = mo.__dict__
    kind = None # assignment first before reference
    if not lib:
        print(f"{mo.__name__} has: \n{dun_all} items in its __all__, and \n{len(funcs)} user defined functions, \n{len(classes)} classes or class objects, \n{len(builtins)} builtin funcs and methods, and\n{len(callables)} callables.\n")  
    if hasattr(mo, "__all__") and dun: 
        maxlen = max(map(lambda i : len(i) , mo.__all__ ))
        for i in mo.__all__:
            obj = eval(i, module_env)
            if ismetaclass(obj):
                kind = "metaclass" 
            elif inspect.isclass(obj):
                kind = "class"
            elif isdecorator(obj):
                kind = "decorator"
            elif inspect.isfunction(obj):
                kind = "function"
            tp = type(eval(i, module_env)).__name__
            startlen = len(i)
            if tp == kind: print(i + ":" + " "*(maxlen-startlen + 5) + kind + "    " + \
                                 inspect.getdoc(eval(i, module_env)))  
            elif kind != None and callable(eval(i, module_env)): print(i + ":" + " "*(maxlen-startlen+5) + kind + ", " + tp + "    " + \
                                 str(inspect.getdoc(eval(i, module_env))))
#             elif tp != 'NoneType': print(i + ":" + " "*(maxlen-startlen+5) + kind + ", " + tp + "    " + \
#                                  inspect.getdoc(eval(i, module_env)))
            else: print(i + ":" + tp)
    if func: 
        print(f'The user defined functions are:')
        maxlen = max(map(lambda i : len(i[0]) , funcs ))
        for i in funcs:
            if isdecorator(i[1]):
                kind = "decorator"
            elif inspect.isfunction(i[1]):
                kind = "function"
#             print(f"{i[0]}: {kind}")  
            startlen = len(i[0])
            print(i[0] + ":" + " "*(maxlen-startlen + 5) + kind + "    " + \
                                 str(inspect.signature(i[1])))               
    if clas: 
        print(f'The class objects are:')
        maxlen = max(map(lambda i : len(i[0]) , funcs ))
        for i in classes:
            if ismetaclass(i[1]):
                kind = "metaclass"
            elif inspect.isclass(i[1]):
                kind = "class"
#             print(f"{i[0]}: {kind}")  
            startlen = len(i[0])
            if not inspect.isbuiltin(i[1]):         
                print(i[0] + ":" + " "*(maxlen-startlen + 5) + kind)
    if method: 
        print(f'The methods are:')
        pprint([i[0] for i in methods])
    if bltin: 
        print(f'The builtin functions or methods are:')
        pprint([i[0] for i in builtins])
    if cal: 
        print(f'The callables are: ')
        pprint([i[0] for i in callables])
    if lib: 
        modules = [name for _, name, _ in pkgutil.iter_modules([pkgpath])]
        print(f'The library has {len(modules)} modules')
        pprint(modules)
```


```

```


```
str(None)
```




    'None'




```

```

## whichversion of a library


```
#| export
from importlib.metadata import version, metadata, distribution
from platform import python_version 
```


```
#| export
def whichversion(libname:str, # library name not string
                req:bool=False, # print lib requirements 
                file:bool=False): # print all lib files
    "Give you library version and other basic info."
    if libname == "python":
        print(f"python: {python_version()}")
    else: 
        print(f"{metadata(libname)['Name']}: {version(libname)} \n{metadata(libname)['Summary']}\
    \n{metadata(libname)['Author']} \n{metadata(libname)['Home-page']} \
    \npython_version: {metadata(libname)['Requires-Python']} \
    \n{distribution(libname).locate_file(libname)}")

    if req: 
        print(f"\n{libname} requires: ")
        pprint(distribution(libname).requires)
    if file: 
        print(f"\n{libname} has: ")
        pprint(distribution(libname).files)
    
```


```
whichversion("python")
```

    python: 3.9.13



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
whichversion("fastai")
```

    fastai: 2.7.9 
    fastai simplifies training fast and accurate neural nets using modern best practices    
    Jeremy Howard, Sylvain Gugger, and contributors 
    https://github.com/fastai/fastai/tree/master/     
    python_version: >=3.7     
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastai



```
whichversion("snoop")
```

    snoop: 0.4.1 
    Powerful debugging tools for Python    
    Alex Hall 
    https://github.com/alexmojaki/snoop     
    python_version: None     
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/snoop



```
try:
    whichversion("inspect")
except: 
    print("inspect won't work here")
```

    inspect won't work here



```

def tstenv(outenv=globals()):
    print(f'out global env has {len(outenv.keys())} vars')
    print(f'inner global env has {len(globals().keys())} vars')
    print(f'inner local env has {len(globals().keys())} vars')
    lstout = list(outenv.keys())
    lstin = list(globals().keys())
    print(lstout[:10])
    print(lstin[:10])
    print(f"out env['__name__']: {outenv['__name__']}")
    print(f"inner env['__name__']: {globals()['__name__']}")
```


```
tstenv()
```

    out global env has 105 vars
    inner global env has 105 vars
    inner local env has 105 vars
    ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__builtin__', '__builtins__', '_ih', '_oh', '_dh']
    ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__builtin__', '__builtins__', '_ih', '_oh', '_dh']
    out env['__name__']: __main__
    inner env['__name__']: __main__



```
len(globals().keys())
```




    106



## fastview
display the commented source code


```
#| export
def fastview(name, # can be both object itself or str, e.g., delegates, FixSigMeta
            nb=False # add a link to the notebook where comments are added
            ): 
    "to view the commented src code in color print and with examples"
    if type(name) == str:
        file_name ='/Users/Natsume/Documents/fastdebug/learnings/' + name + '.py'
    else:
        file_name ='/Users/Natsume/Documents/fastdebug/learnings/' + name.__name__ + '.py' 

    with open(file_name, 'r') as f:
        # Read and print the entire file line by line
        for l in f:
            print(l, end='')
    if nb:
        openNB(name)    
```


```
# file_name ='/Users/Natsume/Documents/fastdebug/learnings/' + self.orisrc.__name__ + '.py' 
#     # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
# with open(file_name, 'w') as f:
# # with open(file_name, 'r') as f:    
#     f.write("this is me")

# with open('dog_breeds.txt', 'r') as reader:
#     # Read & print the entire file
#     print(reader.read())    
    
# with open(file_name, 'r') as f:
#     # Read and print the entire file line by line
#     for l in f:
#         print(l, end='')    
        
# with open(file_name, 'a') as f:
#     f.write('\nBeagle')
```


```
#| export
import os
```


```
# fastsrcs()
fastview("PrePostInitMeta")
```

    
    class _T(metaclass=PrePostInitMeta):
        def __pre_init__(self):  self.a  = 0; 
        def __init__(self,b=0):  self.b = self.a + 1; assert self.b==1
        def __post_init__(self): self.c = self.b + 2; assert self.c==3
    
    t = _T()
    test_eq(t.a, 0) # set with __pre_init__
    test_eq(t.b, 1) # set with __init__
    test_eq(t.c, 3) # set with __post_init__
    inspect.signature(_T)
    
    class PrePostInitMeta(FixSigMeta):========================================================(0)       
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"==========(1) # [37;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type)[0m; [92;1mnot from type, nor from object[0m; [91;1mPrePostInitMeta is itself a metaclass, which is used to create class instance not object instance[0m; [34;1mPrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m; 
        def __call__(cls, *args, **kwargs):===================================================(2)       
            res = cls.__new__(cls)============================================================(3) # [93;1mhow to create an object instance with a cls[0m; [36;1mhow to check the type of an object is cls[0m; [92;1mhow to run a function without knowing its params;[0m; 
            if type(res)==cls:================================================================(4)       
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)==============(5)       
                res.__init__(*args,**kwargs)==================================================(6) # [36;1mhow to run __init__ without knowing its params[0m; 
                if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)============(7)       
            return res========================================================================(8)       
                                                                                                                                                            (9)



```

```

## fastscrs


```
#| export
def fastsrcs():
    "to list all commented src files"
    folder ='/Users/Natsume/Documents/fastdebug/learnings/'
    for f in os.listdir(folder):
        if f.endswith(".py"):
            # Prints only text file present in My Folder
            print(f)
```


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


## getrootport


```

```


```
#| export
def getrootport():
    "get the local port and notebook dir"
    from notebook import notebookapp
    root_server = ""
    root_dir = ""
    for note in notebookapp.list_running_servers():
        if "fastdebug" in note['notebook_dir']:
            root_server = str(note['url']) + "tree/"
            root_dir = note['notebook_dir']
    return (root_server, root_dir)
```


```
from notebook import notebookapp
for note in notebookapp.list_running_servers():
    print(note)
```

    {'base_url': '/', 'hostname': 'localhost', 'notebook_dir': '/Users/Natsume/Documents/fastdebug', 'password': False, 'pid': 110, 'port': 8888, 'secure': False, 'sock': '', 'token': '973ad8ed13df3c4dafac59b8a936f6274757224a960a445b', 'url': 'http://localhost:8888/'}
    {'base_url': '/', 'hostname': 'localhost', 'notebook_dir': '/Users/Natsume/Documents/debuggable', 'password': False, 'pid': 186, 'port': 8891, 'secure': False, 'sock': '', 'token': 'c012783787faf5572be76a83057225b9183cf1d5f77407d0', 'url': 'http://localhost:8891/'}



```
getrootport()
```




    ('http://localhost:8888/tree/', '/Users/Natsume/Documents/fastdebug')



## jn_link


```
#| export
def jn_link(name, file_path):
    "Get a link to the notebook at `path` on Jupyter Notebook"
    from IPython.display import Markdown
    display(Markdown(f'[Open `{name}` in Jupyter Notebook]({file_path})'))                
```


```
jn_link("utils", "http://localhost:8888/notebooks/nbs/lib/utils.ipynb")
```


[Open `utils` in Jupyter Notebook](http://localhost:8888/notebooks/nbs/lib/utils.ipynb)


## get_all_nbs


```
def get_all_nbs(folder='/Users/Natsume/Documents/divefastai/Debuggable/jupytext/'):
    "return all nbs of subfolders of the `folder` into a list"
    all_nbs = []
    for i in os.listdir(folder):
        if "." not in i:
            all_nbs = all_nbs + [folder + i + "/" + j for j in os.listdir(folder + i) if j.endswith('.md')]
    return (all_nbs, folder)
```

### get all nbs path for both md and ipynb


```

def get_all_nbs():
    "return paths for all nbs both in md and ipynb format into lists"
#     md_folder = '/Users/Natsume/Documents/divefastai/Debuggable/jupytext/'
    md_folder = '/Users/Natsume/Documents/fastdebug/mds/'
    ipy_folder = '/Users/Natsume/Documents/fastdebug/nbs/'
    md_nbs = []
    for i in os.listdir(md_folder):
        if "." not in i:
            md_nbs = md_nbs + [md_folder + i + "/" + j for j in os.listdir(md_folder + i) if j.endswith('.md')]
            
    ipy_nbs = []
    for i in os.listdir(ipy_folder):
        if "." not in i:
            ipy_nbs = ipy_nbs + [ipy_folder + i + "/" + j for j in os.listdir(ipy_folder + i) if j.endswith('.ipynb')]
            
    return (md_nbs, md_folder, ipy_nbs, ipy_folder)
```


```

```


```

def get_all_nbs():
    "return paths for all nbs both in md and ipynb format into lists"
#     md_folder = '/Users/Natsume/Documents/divefastai/Debuggable/jupytext/'
    md_folder = '/Users/Natsume/Documents/fastdebug/mds/'
    md_output_folder = '/Users/Natsume/Documents/fastdebug/mds_output/'    
    ipy_folder = '/Users/Natsume/Documents/fastdebug/nbs/'
    md_nbs = []
    for i in os.listdir(md_folder):
        if "." not in i:
            md_nbs = md_nbs + [md_folder + i + "/" + j for j in os.listdir(md_folder + i) if j.endswith('.md')]

    md_output_nbs = [md_output_folder + i for i in os.listdir(md_output_folder) if ".md" in i]        
            
    ipy_nbs = []
    for i in os.listdir(ipy_folder):
        if "." not in i:
            ipy_nbs = ipy_nbs + [ipy_folder + i + "/" + j for j in os.listdir(ipy_folder + i) if j.endswith('.ipynb')]
            
    return (md_nbs, md_folder, ipy_nbs, ipy_folder, md_output_nbs, md_output_folder)
```

### add index.ipynb


```
#| export
def get_all_nbs():
    "return paths for all nbs both in md and ipynb format into lists"
#     md_folder = '/Users/Natsume/Documents/divefastai/Debuggable/jupytext/'
    md_folder = '/Users/Natsume/Documents/fastdebug/mds/'
    md_output_folder = '/Users/Natsume/Documents/fastdebug/mds_output/'    
    ipy_folder = '/Users/Natsume/Documents/fastdebug/nbs/'
    md_nbs = []
    for i in os.listdir(md_folder):
        if "." not in i:
            md_nbs = md_nbs + [md_folder + i + "/" + j for j in os.listdir(md_folder + i) if j.endswith('.md')]

    md_output_nbs = [md_output_folder + i for i in os.listdir(md_output_folder) if ".md" in i]        
            
    ipy_nbs = []
    for i in os.listdir(ipy_folder):
        if ".ipynb" in i: 
            ipy_nbs.append(ipy_folder + i)
        elif "." not in i:
            ipy_nbs = ipy_nbs + [ipy_folder + i + "/" + j for j in os.listdir(ipy_folder + i) if j.endswith('.ipynb')]

            
    return (md_nbs, md_folder, ipy_nbs, ipy_folder, md_output_nbs, md_output_folder)
```


```

```


```
nbs_md, fdmd, nbs_ipy, fdipy, md_out, md_out_fd = get_all_nbs()
for i in [nbs_md, fdmd, nbs_ipy, fdipy, md_out, md_out_fd]:
    pprint(i)
    print()
```

    ['/Users/Natsume/Documents/fastdebug/mds/2022part1/0001_is_it_a_bird.md',
     '/Users/Natsume/Documents/fastdebug/mds/lib/utils.md',
     '/Users/Natsume/Documents/fastdebug/mds/lib/00_core.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0001_fastcore_meta_delegates.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0004_fastcore.meta._rm_self.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0005_fastcore.meta.test_sig.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0007_fastcore.meta.BypassNewMeta.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0002_signature_from_callable.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0008_use_kwargs_dict.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0006_fastcore.meta.NewChkMeta.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0000_tour.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0012_fastcore_foundation_L.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0011_Fastdb.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0009_funcs_kwargs.md',
     '/Users/Natsume/Documents/fastdebug/mds/demos/0010_fastcore_meta_summary.md',
     '/Users/Natsume/Documents/fastdebug/mds/questions/00_question_anno_dict.md']
    
    '/Users/Natsume/Documents/fastdebug/mds/'
    
    ['/Users/Natsume/Documents/fastdebug/nbs/2022part1/0001_is_it_a_bird.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/lib/utils.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/lib/00_core.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/.ipynb_checkpoints',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0001_fastcore_meta_delegates.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0005_fastcore.meta.test_sig.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0000_tour.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0007_fastcore.meta.BypassNewMeta.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0008_use_kwargs_dict.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0012_fastcore_foundation_L.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0002_signature_from_callable.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0004_fastcore.meta._rm_self.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0009_funcs_kwargs.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0010_fastcore_meta_summary.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0006_fastcore.meta.NewChkMeta.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/demos/0011_Fastdb.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/index.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/questions/00_question_anno_dict.ipynb']
    
    '/Users/Natsume/Documents/fastdebug/nbs/'
    
    ['/Users/Natsume/Documents/fastdebug/mds_output/0001_fastcore_meta_delegates.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/00_question_anno_dict.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0004_fastcore.meta._rm_self.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/utils.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0005_fastcore.meta.test_sig.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0007_fastcore.meta.BypassNewMeta.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0002_signature_from_callable.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0008_use_kwargs_dict.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0001_is_it_a_bird.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0006_fastcore.meta.NewChkMeta.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/index.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/00_core.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0000_tour.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0012_fastcore_foundation_L.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0011_Fastdb.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0009_funcs_kwargs.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0010_fastcore_meta_summary.md']
    
    '/Users/Natsume/Documents/fastdebug/mds_output/'
    


## openNB


```

def openNB(name, folder='nbs/demos/', db=False):
    "Get a link to the notebook at by name locally"
#     root = "/Users/Natsume/Documents/fastdebug/"
#     root_server = "http://localhost:8888/tree/"
    root = getrootport()[1] + "/"
    root_server = getrootport()[0]
    path = root + folder
    path_server = root_server + folder
    for f in os.listdir(path):  
        if f.endswith(".ipynb"):
            if name in f: 
                file_name = path_server + f
                if db: print(f'file_name: {file_name}')
                jn_link(name, file_name)
```


```

def openNB(name, folder='nbs/demos/', db=False):
    "Get a link to the notebook at by name locally"
#     root = "/Users/Natsume/Documents/fastdebug/"
#     root_server = "http://localhost:8888/tree/"
    root = getrootport()[1] + "/"
    root_server = getrootport()[0]
    path = root + folder
    path_server = root_server + folder
    for f in os.listdir(path):  
        if f.endswith(".ipynb"):
            name = name.split('.md')[0]
            if db: print(f, name)
            if name in f: 
                file_name = path_server + f
                jn_link(name, file_name)
```


```

```


```
#| export
def openNB(name, db=False):
    "Get a link to the notebook at by searching keyword or notebook name"
    _, _, ipynbs, _, _, _= get_all_nbs()
    name = name.split(".md")[0]
    root = getrootport()[1]
    nb_path = ""
    for f in ipynbs:
        if name in f:
            nb_path = f
            name = f.split("/")[-1].split(".")[0]
            if db: print(f'nb_path:{nb_path}, name: {name}')
    root_server = getrootport()[0]
    folder_mid = nb_path.split(root)[1].split(name)[0]
    if db: print(f'root: {root}, root_server: {root_server}, name: {name}, folder_mid: {folder_mid}')
    path = root + folder_mid
    path_server = root_server[:-1] + folder_mid
    if db: print(f'path: {path}, path_server: {path_server}')
    for f in os.listdir(path):  
        if f.endswith(".ipynb"):
            if name in f: 
                file_name = path_server + f
                jn_link(name, file_name)
```


```
openNB("FixSigMeta", db=True)
```

    nb_path:/Users/Natsume/Documents/fastdebug/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb, name: 0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit
    root: /Users/Natsume/Documents/fastdebug, root_server: http://localhost:8888/tree/, name: 0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit, folder_mid: /nbs/demos/
    path: /Users/Natsume/Documents/fastdebug/nbs/demos/, path_server: http://localhost:8888/tree/nbs/demos/



[Open `0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit` in Jupyter Notebook](http://localhost:8888/tree/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb)



```

```

## highlight

<mark style="background-color: #FFFF00">text with highlighted background</mark>  


```

def highlight(question:str, line:str, db=False):
    "highlight a string with yellow background"
    questlst = question.split(' ')
    questlst_hl = [' <mark style="background-color: #FFFF00">' + q + '</mark> ' for q in questlst]
    for q, q_hl in zip(questlst, questlst_hl):
        if " " + q + " " in line.lower() or "[" + q + "]" in line.lower() and not '<mark style="background-color: #FFFF00">' + q in line.lower():
            line = line.lower().replace(" " + q + " ", q_hl)
            
    if db: print(f'line: {line}')
    return line
```


```

def highlight(question:str, line:str, db=False):
    "highlight a string with yellow background"
    questlst = question.split(' ')
    questlst_hl = [' <mark style="background-color: #FFFF00">' + q + '</mark> ' for q in questlst]
    for q, q_hl in zip(questlst, questlst_hl):
        if " " + q in line.lower(): # don't do anything to [q] or <>q<>. Using regex can be more accurate here
            line = line.lower().replace(q, q_hl)
            
    if db: print(f'line: {line}')
    return line
```


```
#| export
def highlight(question:str, line:str, db=False):
    "highlight a string with yellow background"
    questlst = question.split(' ')
    questlst_hl = [' <mark style="background-color: #FFFF00">' + q.lower() + '</mark> ' for q in questlst]
    for q, q_hl in zip(questlst, questlst_hl):
        if " " + q.lower() in line.lower(): # don't do anything to [q] or <>q<>. Using regex can be more accurate here
            line = line.lower().replace(q.lower(), q_hl)
            
    if db: print(f'line: {line}')
    return line
```


```
str(0.8).split(" ")
print(highlight("0.8", "this is 0.8"))

```

    this is  <mark style="background-color: #FFFF00">0.8</mark> 



```

```

## display_md


```
#| export
def display_md(text):
    "Get a link to the notebook at `path` on Jupyter Notebook"
    from IPython.display import Markdown
    display(Markdown(text))                
```


```
display_md("#### heading level 4")
```


#### heading level 4



```

```

## display_block


```

def display_block(line, file):
    "`line` is a section title, find all subsequent lines which belongs to the same section and display them together"
    from IPython.display import Markdown
    with open(file, 'r') as file:
        entire = file.read()
        belowline = entire.split(line)[1]
        head_no = line.count("#")
        lochead2 = belowline.find("##")
        lochead3 = belowline.find("###")
        lochead4 = belowline.find("####")
        loclst = [lochead2,lochead3, lochead4]
        loclst = [i for i in loclst if i != -1]
        num_hash = 0
        if bool(loclst):
            if lochead2 == min(loclst):
                num_hash = 2
            elif lochead3 == min(loclst):
                num_hash = 3
            elif lochead4 == min(loclst):
                num_hash = 4
        if num_hash == 0:
            section_content = belowline
        else:
            section_content = belowline.split("#"*num_hash)[0]
        entire_section = line + "\n" + section_content
        display(Markdown(entire_section))
```


```

```


```

def display_block(line, file, keywords=""):
    "`line` is a section title, find all subsequent lines which belongs to the same section and display them together"
    from IPython.display import Markdown
    with open(file, 'r') as file:
        entire = file.read()
        belowline = entire.split(line)[1]
        head_no = line.count("#")
        lochead2 = belowline.find("##")
        lochead3 = belowline.find("###")
        lochead4 = belowline.find("####")
        loclst = [lochead2,lochead3, lochead4]
        loclst = [i for i in loclst if i != -1]
        num_hash = 0
        if bool(loclst):
            if lochead2 == min(loclst):
                num_hash = 2
            elif lochead3 == min(loclst):
                num_hash = 3
            elif lochead4 == min(loclst):
                num_hash = 4
        if num_hash == 0:
            section_content = belowline
        else:
            section_content = belowline.split("#"*num_hash)[0]
#         entire_content = line + "\n" + section_content
#         display(Markdown(entire_content))        
        title_hl = highlight(keywords, line)
        display(Markdown(title_hl))
        display(Markdown(section_content))

```

### handle both file path and file content at the same time


```
#| export
def display_block(line, file, output=False, keywords=""):
    "`line` is a section title, find all subsequent lines which belongs to the same section and display them together"
    from IPython.display import Markdown
    entire = ""
    if file.endswith(".md") or file.endswith(".ipynb"):
        with open(file, 'r') as file:
            entire = file.read()
    else:
        entire = file
        
    belowline = entire.split(line)[1]
    head_no = line.count("#")
    lochead2 = belowline.find("##")
    lochead3 = belowline.find("###")
    lochead4 = belowline.find("####")
    loclst = [lochead2,lochead3, lochead4]
    loclst = [i for i in loclst if i != -1]
    num_hash = 0
    if bool(loclst):
        if lochead2 == min(loclst):
            num_hash = 2
        elif lochead3 == min(loclst):
            num_hash = 3
        elif lochead4 == min(loclst):
            num_hash = 4
    if num_hash == 0:
        section_content = belowline
    else:
        section_content = belowline.split("#"*num_hash)[0]
#         entire_content = line + "\n" + section_content
#         display(Markdown(entire_content))        
    title_hl = highlight(keywords, line)
    display(Markdown(title_hl))
    if not output: display(Markdown(section_content))
    else: print(section_content)

```


```
display_block("whichversion of a library", "/Users/Natsume/Documents/divefastai/Debuggable/jupytext/lib/utils.md", \
              keywords="whichversion library")
```


whichversion of a  <mark style="background-color: #FFFF00">library</mark> 





```python
#| export
from importlib.metadata import version, metadata, distribution
from platform import python_version 
```

```python
#| export
def whichversion(libname:str, # library name not string
                req:bool=False, # print lib requirements 
                file:bool=False): # print all lib files
    "Give you library version and other basic info."
    if libname == "python":
        print(f"python: {python_version()}")
    else: 
        print(f"{metadata(libname)['Name']}: {version(libname)} \n{metadata(libname)['Summary']}\
    \n{metadata(libname)['Author']} \n{metadata(libname)['Home-page']} \
    \npython_version: {metadata(libname)['Requires-Python']} \
    \n{distribution(libname).locate_file(libname)}")

    if req: 
        print(f"\n{libname} requires: ")
        pprint(distribution(libname).requires)
    if file: 
        print(f"\n{libname} has: ")
        pprint(distribution(libname).files)
    
```

```python
whichversion("python")
```

```python
whichversion("fastcore")
```

```python
whichversion("fastai")
```

```python
whichversion("snoop")
```

```python
try:
    whichversion("inspect")
except: 
    print("inspect won't work here")
```

```python

def tstenv(outenv=globals()):
    print(f'out global env has {len(outenv.keys())} vars')
    print(f'inner global env has {len(globals().keys())} vars')
    print(f'inner local env has {len(globals().keys())} vars')
    lstout = list(outenv.keys())
    lstin = list(globals().keys())
    print(lstout[:10])
    print(lstin[:10])
    print(f"out env['__name__']: {outenv['__name__']}")
    print(f"inner env['__name__']: {globals()['__name__']}")
```

```python
tstenv()
```

```python
len(globals().keys())
```





```

```


```

```

## fastnbs


```

def fastnbs(question:str, accu:float=0.8, n=10, nb=False, db=False):
    "using keywords to search learning points from my documented fastai notebooks"
    questlst = question.split(' ')
    folder ='/Users/Natsume/Documents/divefastai/Debuggable/jupytext/2022part1/'
    for f in os.listdir(folder):  
        if f.endswith(".md"):
            file_name =folder + f
            with open(file_name, 'r') as file:
                for count, l in enumerate(file):
#                 for l in file:
                    truelst = [q in l for q in questlst]
                    pct = sum(truelst)/len(truelst)

                    if pct >= accu:
                        head1 = f"keyword match is {pct}, Found a line: in {f}"
                        head1 = highlight(str(pct), head1)
                        head1 = highlight(f, head1)
                        display_md(head1)
                        l = highlight(question, l, db=db)                        
                        display_md(l)
                        print()
                        
                        head2 = f"Show {n} lines after in {f}:"
                        head2 = highlight(f, head2)
                        head2 = highlight(str(n), head2)
                        display_md(head2)                        
                        idx = count
                        with open(file_name, 'r') as file:
                            for count, l in enumerate(file):
                                if count >= idx and count <= idx + n:
                                    if count == idx: display_md(highlight(question, l))
                                    else: display_md(l)                        
                        if nb:
                            openNB(f, "nbs/2022part1/")
```


```
#| export
def fastnbs(question:str, output=False, accu:float=0.8, nb=True, db=False):
    "check with fastlistnbs() to find interesting things to search \
fastnbs() can use keywords to search learning points (a section title and a section itself) from my documented fastai notebooks"
    questlst = question.split(' ')
    mds_no_output, folder, ipynbs, ipyfolder, mds_output, output_fd = get_all_nbs()
    if not output: mds = mds_no_output
    else: mds = mds_output
        
    for file_fullname in mds:
        file_name = file_fullname.split('/')[-1]
        with open(file_fullname, 'r') as file:
            for count, l in enumerate(file):
                truelst = [q.lower() in l.lower() for q in questlst]
                pct = sum(truelst)/len(truelst)
                if pct >= accu and (l.startswith("##") or l.startswith("###") or l.startswith("####")):
                    if db: 
                        head1 = f"keyword match is {pct}, Found a section: in {file_name}"
                        head1 = highlight(str(pct), head1)
                        head1 = highlight(file_name, head1)
                        display_md(head1)
                        highlighted_line = highlight(question, l, db=db)                        
                        print()
                    display_block(l, file_fullname, output=output, keywords=question)
                    if nb: openNB(file_name, db=db)
```


```
fastnbs("Snoop them together in one go", output=True)
fastnbs("Snoop them together in one go")
fastnbs("what is fastai")
```


##  <mark style="background-color: #ffff00">snoop</mark>   <mark style="background-color: #ffff00">them</mark>   <mark style="background-color: #ffff00">together</mark>   <mark style="background-color: #ffff00">in</mark>   <mark style="background-color: #ffff00">one</mark>   <mark style="background-color: #FFFF00">go</mark> 



    
    
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
        
    
    
    



[Open `0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit` in Jupyter Notebook](http://localhost:8888/tree/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb)



##  <mark style="background-color: #ffff00">snoop</mark>   <mark style="background-color: #ffff00">them</mark>   <mark style="background-color: #ffff00">together</mark>   <mark style="background-color: #ffff00">in</mark>   <mark style="background-color: #ffff00">one</mark>   <mark style="background-color: #FFFF00">go</mark> 





```python

fdbF.snoop(watch=['res', 'type(res)', 'res.__class__', 'res.__dict__'])
```





[Open `0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit` in Jupyter Notebook](http://localhost:8888/tree/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb)



###  <mark style="background-color: #ffff00">what</mark>   <mark style="background-color: #ffff00">is</mark>   <mark style="background-color: #FFFF00">fastai</mark> 





```python
import fastai
```

```python
whichversion("fastai")
```

```python
whatinside(fastai, lib=True)
```

```python
import fastai.losses as fl
```

```python
whatinside(fl, dun=True)
```





[Open `0001_is_it_a_bird` in Jupyter Notebook](http://localhost:8888/tree/nbs/2022part1/0001_is_it_a_bird.ipynb)



```

```

## fastcodes


```
#| export
def fastcodes(question:str, accu:float=0.8, nb=False, db=False):
    "using keywords to search learning points from commented sources files"
    questlst = question.split(' ')
    # loop through all pyfile in learnings folder
    folder ='/Users/Natsume/Documents/fastdebug/learnings/'
    for f in os.listdir(folder):  
        if f.endswith(".py"):
            name = f.split('.py')[0]
            # open each pyfile and read each line
            file_name =folder + f
            with open(file_name, 'r') as file:
                for count, l in enumerate(file):                
                    # if search match >= 0.8, print the line and the pyfile
                    truelst = [q.lower() in l.lower() for q in questlst]
                    pct = sum(truelst)/len(truelst)
                    if pct >= accu:
                        head1 = f"keyword match is {pct}, Found a line: in {f}"
                        head1 = highlight(str(pct), head1)
                        head1 = highlight(f, head1)
                        display_md(head1)
                        print(l, end='')
                        print()                        
                        head2 = f"The entire source code in {f}"
                        head2 = highlight(f, head2)
                        display_md(head2)
                        fastview(name)
                        print()
                        if nb:
                            openNB(name)
```


```
fastcodes("how to remove self")
```


keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">fixsigmeta.py</mark> 


            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [93;1mhow to remove self from a signature[0m; [36;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
    



the entire source code in  <mark style="background-color: #FFFF00">fixsigmeta.py</mark> 


    
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [35;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance[0m; [92;1mthen check whether its class instance has its own __init__;if so, remove self from the sig of __init__[0m; [34;1mthen assign this new sig to __signature__ for the class instance;[0m; 
        def __new__(cls, name, bases, dict):==================================================(2) # [93;1mhow does a metaclass create a class instance[0m; [93;1mwhat does super().__new__() do here;[0m; 
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [93;1mhow to remove self from a signature[0m; [36;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)
    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">_rm_self.py</mark> 


        sigd.pop('self')======================================================================(2) # [93;1mhow to remove the self parameter from the dict of sig;[0m; 
    



the entire source code in  <mark style="background-color: #FFFF00">_rm_self.py</mark> 


    
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    
    def _rm_self(sig):========================================================================(0) # [91;1mremove parameter self from a signature which has self;[0m; 
        sigd = dict(sig.parameters)===========================================================(1) # [34;1mhow to access parameters from a signature[0m; [91;1mhow is parameters stored in sig[0m; [35;1mhow to turn parameters into a dict;[0m; 
        sigd.pop('self')======================================================================(2) # [93;1mhow to remove the self parameter from the dict of sig;[0m; 
        return sig.replace(parameters=sigd.values())==========================================(3) # [35;1mhow to update a sig using a updated dict of sig's parameters[0m; 
                                                                                                                                                            (4)
    



```

```

## fastnotes


```

def fastnotes(question:str, accu:float=0.8, n=2, db=False):
    "display found notes with key words search"
    questlst = question.split(' ')
    folder ='/Users/Natsume/Documents/divefastai/lectures/'
    for f in os.listdir(folder):  
        if f.endswith(".md"):
            name = f.split('.md')[0]
            file_name =folder + f
            with open(file_name, 'r') as file:
                for count, l in enumerate(file):
                    truelst = [q in l.lower() for q in questlst]
                    pct = sum(truelst)/len(truelst)
                    if pct >= accu:
                        print()
                        head1 = f"keyword match is {pct}, Found a line: in {f}"
                        head1 = highlight(str(pct), head1)
                        head1 = highlight(f, head1)
                        display_md(head1)
                        l = highlight(question, l, db=db)
                        display_md(l)
                        print()                        
                        print('{:=<157}'.format(f"Show {n} lines above and after in {f}:"))
                        head2 = f"Show {n} lines above and after in {f}:"
                        head2 = highlight(f, head2)
                        head2 = highlight(str(n), head2)
                        display_md(head2)                        
                        idx = count
                        with open(file_name, 'r') as file:
                            for count, l in enumerate(file):
                                if count >= idx - n and count <= idx + n:
                                    if count == idx: display_md(highlight(question, l))
                                    else: display_md(l)
```

### multiple folders


```
[1,3] + [5,6]
```




    [1, 3, 5, 6]




```
#| export
def fastnotes(question:str, accu:float=0.8, n=2, folder="lec", # folder: 'lec' or 'live' or 'all'
              db=False):
    "using key words to search notes and display the found line and lines surround it"
    questlst = question.split(' ')
    root = '/Users/Natsume/Documents/divefastai/'
    folder1 = '2022_part1/'
    folder2 = '2022_livecoding/'
    lectures = [root + folder1 + f for f in os.listdir(root + folder1)]
    livecodings = [root + folder2 + f for f in os.listdir(root + folder2)]
    all_notes = lectures + livecodings
    if folder == "lec": files = lectures
    elif folder == "live": files = livecodings
    else: files = all_notes
    for f in files:
#     for f in os.listdir(folder):  
        if f.endswith(".md"):
#             name = f.split('.md')[0]
#             file_name =folder + f
            with open(f, 'r') as file:
                for count, l in enumerate(file):
                    truelst = [q in l.lower() for q in questlst]
                    pct = sum(truelst)/len(truelst)
                    if pct >= accu:
                        print()
                        head1 = f"keyword match is {pct}, Found a line: in {f.split(root)[1]}"
                        head1 = highlight(str(pct), head1)
                        head1 = highlight(f.split(root)[1], head1)
                        display_md(head1)
                        l = highlight(question, l, db=db)
                        display_md(l)
                        print()                        
#                         print('{:=<157}'.format(f"Show {n} lines above and after in {f}:"))
                        head2 = f"Show {n} lines above and after in {f.split(root)[1]}:"
                        head2 = highlight(f.split(root)[1], head2)
                        head2 = highlight(str(n), head2)
                        display_md(head2)                        
                        idx = count
                        with open(f, 'r') as file:
                            for count, l in enumerate(file):
                                if count >= idx - n and count <= idx + n:
                                    if count == idx: display_md(highlight(question, l))
                                    else: display_md(l)
```


```
fastnotes("why read fastbook", n=2)
```

    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">2022_part1/fastai-lecture-1.md</mark> 



do you know that learning the same thing in different ways betters understanding? -  <mark style="background-color: #ffff00">why</mark>  should you  <mark style="background-color: #FFFF00">read</mark>  the [fastbook](https://github.com/fastai/fastbook)?



    



show  <mark style="background-color: #FFFF00">2</mark>  lines above and after in  <mark style="background-color: #ffff00"> <mark style="background-color: #FFFF00">2</mark> 0 <mark style="background-color: #FFFF00">2</mark>  <mark style="background-color: #FFFF00">2</mark> _part1/fastai-lecture-1.md</mark> :



Do you know people learn naturally (better) with context rather than by theoretical curriculum? - Do you want this course to make you a competent deep learning practitioner by context and practical knowledge? - If you want theory from ground up, should you go to part 2 fastai 2019?




[20:01](https://www.youtube.com/watch?v=RLvUfyLcT48&t=80s&loop=10&start=20:01&end=21:25) Fastbook  




do you know that learning the same thing in different ways betters understanding? -  <mark style="background-color: #ffff00">why</mark>  should you  <mark style="background-color: #FFFF00">read</mark>  the [fastbook](https://github.com/fastai/fastbook)?




[21:25](https://www.youtube.com/watch?v=RLvUfyLcT48&t=80s&loop=10&start=21:25&end=24:38) Take it seriously  




Why you must take this course very seriously?



    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">2022_part1/fastai-lecture-4.md</mark> 



- how to check what inside the dataset folder? -  <mark style="background-color: #ffff00">why</mark>  it is important to  <mark style="background-color: #ffff00">read</mark>  [competition data introduction](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data) which is often overlooked? - how to  <mark style="background-color: #ffff00">read</mark>  a csv file with pandas? [24:30](https://youtu.be/tougbqv1bt8?t=1470) - what are the key four libraries for data science in python? [24:46](https://youtu.be/tougbqv1bt8?t=1486) - what is the other [book](https://wesmckinney.com/book/) besides  <mark style="background-color: #FFFF00">fastbook</mark>  recommended by jeremy? [25:36](https://youtu.be/tougbqv1bt8?t=1536) -  <mark style="background-color: #ffff00">why</mark>  you must  <mark style="background-color: #ffff00">read</mark>  it too? - how to access and show the dataset in dataframe? [26:39](https://youtu.be/tougbqv1bt8?t=1599) - how to `describe` the dataset? what does it tell us in general? [27:10](https://youtu.be/tougbqv1bt8?t=1630) - what did the number of unique data samples mean to jeremy at first? [27:57](https://youtu.be/tougbqv1bt8?t=1677) - how to create a single string based on the model strategy? [28:26](https://youtu.be/tougbqv1bt8?t=1706) - how to refer to a column of a dataframe in  <mark style="background-color: #ffff00">read</mark> ing and writing a column data?



    



show  <mark style="background-color: #FFFF00">2</mark>  lines above and after in  <mark style="background-color: #ffff00"> <mark style="background-color: #FFFF00">2</mark> 0 <mark style="background-color: #FFFF00">2</mark>  <mark style="background-color: #FFFF00">2</mark> _part1/fastai-lecture-4.md</mark> :



- When and how to use a GPU on Kaggle? - Why Jeremy recommend Paperspace over Kaggle as your workstation? - How easy has Jeremy made it to download Kaggle dataset and work on Paperspace or locally? - How to do both python and bash in the same jupyter cell?




#### [23:33](https://youtu.be/toUgBQv1BT8?t=1413) Get raw dataset into documents  




- how to check what inside the dataset folder? -  <mark style="background-color: #ffff00">why</mark>  it is important to  <mark style="background-color: #ffff00">read</mark>  [competition data introduction](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/data) which is often overlooked? - how to  <mark style="background-color: #ffff00">read</mark>  a csv file with pandas? [24:30](https://youtu.be/tougbqv1bt8?t=1470) - what are the key four libraries for data science in python? [24:46](https://youtu.be/tougbqv1bt8?t=1486) - what is the other [book](https://wesmckinney.com/book/) besides  <mark style="background-color: #FFFF00">fastbook</mark>  recommended by jeremy? [25:36](https://youtu.be/tougbqv1bt8?t=1536) -  <mark style="background-color: #ffff00">why</mark>  you must  <mark style="background-color: #ffff00">read</mark>  it too? - how to access and show the dataset in dataframe? [26:39](https://youtu.be/tougbqv1bt8?t=1599) - how to `describe` the dataset? what does it tell us in general? [27:10](https://youtu.be/tougbqv1bt8?t=1630) - what did the number of unique data samples mean to jeremy at first? [27:57](https://youtu.be/tougbqv1bt8?t=1677) - how to create a single string based on the model strategy? [28:26](https://youtu.be/tougbqv1bt8?t=1706) - how to refer to a column of a dataframe in  <mark style="background-color: #ffff00">read</mark> ing and writing a column data?




#### [29:17](https://youtu.be/toUgBQv1BT8?t=1757) Tokenization: Intro  




- How to turn strings/documents into numbers for neuralnet? - Do we split the string into words first? - What's the problem with the Chinese language on words? - What are vocabularies compared with splitted words? - What to do with the vocabulary? - Why we want the vocabulary to be concise not too big? - What nowadays people prefer rather than words to be included in vocab?




```

```


```

```

## fastlistnbs


```

def fastlistnbs():
    "display all my commented notebooks subheadings in a long list"
    nbs, folder, _, _, _, _ = get_all_nbs()
    for nb in nbs:
        print("\n"+nb)
        with open(nb, 'r') as file:
            for idx, l in enumerate(file):
                if "##" in l:
                    print(l, end="") # no extra new line between each line printed
        
```


```
#| export
def fastlistnbs():
    "display all my commented notebooks subheadings in a long list. Best to work with fastnbs together."
    nbs, folder, _, _, _, _ = get_all_nbs()
    for nb in nbs:
        print("\n"+nb)
        with open(nb, 'r') as file:
            for idx, l in enumerate(file):
                if l.startswith("##"):
                    print(l, end="") # no extra new line between each line printed
        
```


```

```


```
fastlistnbs()
```

    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0001_is_it_a_bird.md
    ## Useful Course sites
    ## How to use autoreload
    ## How to install and update libraries
    ## Know a little about the libraries
    ### what is fastai
    ### what is duckduckgo
    ## How to use fastdebug with fastai notebooks
    ### what is fastdebug
    ### Did I document it in a notebook before?
    ### Did I document it in a src before?
    ## how to download images
    ### how to create folders using path; how to search and download images in folders; how to resize images 
    ## Train my model
    ### How to find and unlink images not properly downloaded
    ### How to create a DataLoaders with DataBlock; how to view data with it
    ### How to build my model with dataloaders and pretrained model; how to train my model
    ### How to predict with my model; how to avoid running cells in nbdev_prepare
    ## Check fastnbs and fastlistnbs
    ## search and display section input and output in notebooks
    
    /Users/Natsume/Documents/fastdebug/mds/lib/utils.md
    ## Expand cells
    ## Import fastcore env
    ## to inspect a class
    ### get the docs for each function of a class
    ## is it a metaclass?
    ## is it a decorator
    ### handle all kinds of exceptions for evaluating retn 
    ## whatinside a module of a library
    ### show the type of objects inside `__all__`
    ### working for fastdebug.core
    ### to show Fastdb methods
    ## whichversion of a library
    ## fastview
    ## fastscrs
    ## getrootport
    ## jn_link
    ## get_all_nbs
    ### get all nbs path for both md and ipynb
    ## openNB
    ## highlight
    ## display_md
    ## display_block
    ### handle both file path and file content at the same time
    ## add_py_block
    ## fastnbs
    ## fastcodes
    ## fastnotes
    ### multiple folders
    ## fastlistnbs
    ## fastlistsrcs
    ## Examples
    ## Export
    ## Send to Obsidian
    
    /Users/Natsume/Documents/fastdebug/mds/lib/00_core.md
    ## make life easier with defaults  
    ## globals() and locals()
    ## Execute strings
    ### new variable or updated variable by exec will only be accessible from locals()
    ### eval can override its own globals() and locals()
    ### when exec update existing functions
    ### when the func to be udpated involve other libraries
    ### inside a function, exec() allow won't give you necessary env from function namespace
    ### magic of `exec(b, globals().update(locals()))`
    ### Bring variables from a func namespace to the sideout world
    ### globals() in a cell vs globals() in a func
    ## make a colorful string
    ## align text to the most right
    ## printsrcwithidx
    ### print the entire source code with idx from 0
    ### print the whole src with idx or print them in parts
    ### use cmts from dbprint to print out src with comments
    ### no more update for printsrcwithidx, for the latest see Fastdb.print
    ## print out src code
    ### basic version
    ### print src with specific number of lines
    ### make the naming more sensible
    ### Allow a dbline occur more than once
    ### adding idx to the selected srclines
    #### printsrclinewithidx
    ### dblines can be string of code or idx number
    ### avoid multi-occurrance of the same srcline
    ## dbprint on expression
    ### basic version
    ### insert dbcode and make a new dbfunc
    ### Bring outside namespace variables into exec()
    ### Bring what inside the func namespace variables to the outside world
    ### Adding g = locals() to dbprintinsert to avoid adding env individually
    ### enable srclines to be either string or int 
    ### enable = to be used as assignment in codes
    ### avoid adding "env=g" for dbprintinsert
    ### collect cmt for later printsrcwithidx to print comments together
    ### make sure only one line with correct idx is debugged
    ### avoid typing "" when there is no codes
    ### no more update for dbprint, for the latest see Fastdb.dbprint
    ### use dbprint to override the original official code without changing its own pyfile
    ## dbprintinsert
    ### Run and display the inserted dbcodes 
    ### use locals() inside the dbsrc code to avoid adding env individually
    ### enable dbprintinsert to do exec on a block of code
    ## printrunsrclines() 
    ### Examples
    #### simple example
    #### complex example
    ### insert a line after each srcline to add idx
    ### add correct indentation to each inserted line
    #### count the indentation for each srcline
    ### indentation special case: if, else, for, def
    ### remove pure comments or docs from dbsrc
    ### print out the srclines which get run
    ### Make sure all if, else, for get printed
    ### Put all together into the function printrunsrclines()
    #### no more renaming of foo
    #### add example as a param into the function
    #### improve on search for `if`, else, for, def to avoid errors for more examples
    #### remove an empty line with indentation
    ### make it work
    ### more difficult examples to test printrunsrc()
    ## Make fastdebug a class
    ### improve on the line idx readability
    ### collect cmt from dbprint and print
    ### make sure only the line with correct idx is debugged
    ### having "" or "   " inside codes without causing error
    ### replace Fastdb.printsrcwithdix with Fastdb.print
    ### add idx to dbsrc when showdbsrc=True
    ### not load the inner locals() to outenv can prevent mysterious printing of previous db messages
    ### using @patch to enable docs for instance methods like `dbprint` and `print`
    ### move param env into `__init__`
    ### Add example to the obj
    ### Take not only function but also class
    ### To remove the necessity of self.takExample()
    ### Try to remove g = locals()
    ### Make sure `showdbsrc=True` give us the line starting with 'dbprintinsert'
    ### Make sure `showdbsrc=True` give us info on changes in g or outenv
    ### exit and print a warning message: idx has to be int
    ### handle errors by codes with trailing spaces 
    ### showdbsrc=True, check whether Fastdb.dbprint and fdb.dbprint are same object using `is`
    ### remove unnecessary db printout when showdbsrc=True and add printout to display sections
    ### raise TypeError when decode are not integer and showdbsrc=true working on both method and function
    ### when debugging dbprint, make sure dbsrc is printed with the same idx as original
    ### update dbsrc to the global env
    ### go back to normal before running dbprint again
    ### auto print src with cmt and idx as the ending part of dbprint
    ### to mark my explorations (expressions to evaluate) to stand out
    ### Add the print of src with idx and comments at the end of dbsrc
    ### embed example and autoprint to shorten the code to type
    ### Make title for dbprint
    ### Adding self.eg info and color group into dbprint and print
    #### todo: make the comments with same self.eg have the same color
    ### make dbsrc print idx right
    ### add self.eg to a dict with keys are idxsrc
    ### handle both function and class as src
    ### documenting on Fastdb.dbprint itself
    ## mk_dbsrc
    ## Turn mk_dbsrc into docsrc 
    ### handle when no codes are given
    ## create_dbsrc_from_string
    ## replaceWithDbsrc
    ### handle class and metaclass
    ### improve on handling function as decorator
    ### Handling `inspect._signature_from_callable` to become `self.dbsrc`
    ### handling usage of `@delegates`
    ### handling `@delegates` with indentation before it
    ### handling classes by inspect.isclass() rather than == type and add more class situations
    ### handling `class _T(_TestA, metaclass=BypassNewMeta): `
    ## run_example
    ### `exec(self.eg, globals().update(self.egEnv), locals())` works better than `...update(locals()), self.egEnv)
    ### no more env cells run before `fdb.eg` to make `fdb.run_example` work
    ## Autoprint
    ## Take an example and its env into Fastdb obj
    ## print src with idx and cmt in whole or parts
    ### print self.eg after each comment and colorize comments
    ### color examples and cmts separately and make the func simpler
    ### split each cmt and colorize parts randomly
    ### printcmts1 while saving into a file
    ## goback
    ## Fastdb.explore
    ### adding one breakpoint with comment
    ### Adding multiple breakpoints by multiple set_trace()
    ### Go back to normal before running explore again
    ### enable fdb.takExample("whatinside(fu), ...) without using `fu.whatinside`
    ### refactory explore
    ## snoop
    ### snoop on both function and class
    ### snoop on class and method and all???
    ### snoop
    ### simplify adding @snoop for both normal function and decorator
    ### handling classes
    ### add watch
    ## Snoop
    ### add watch
    ### use guide on Fastdb.dbprint
    ## reliveonce
    ## Fastdb.debug
    ## Export
    ## Send to Obsidian
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0001_fastcore_meta_delegates.md
    ## Import
    ## Initiate Fastdb and example in str
    ## Example
    ## docsrc
    ## Snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md
    ## Initialize fastdebug objects
    ## class FixSigMeta(type) vs class Foo(type)
    ## class Foo()
    ## class PrePostInitMeta(FixSigMeta)
    ## class Foo(metaclass=FixSigMeta)
    ## class AutoInit(metaclass=PrePostInitMeta)
    ## Prepare examples for FixSigMeta, PrePostInitMeta, AutoInit 
    ## Snoop them together in one go
    ### embed the dbsrc of FixSigMeta into PrePostInitMeta
    ### embed dbsrc of PrePostInitMeta into AutoInit
    ## Explore and Document on them together 
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0004_fastcore.meta._rm_self.md
    ## imports
    ## set up
    ## document
    ## snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0005_fastcore.meta.test_sig.md
    ## imports
    ## setups
    ## documents
    ## snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0007_fastcore.meta.BypassNewMeta.md
    ## Reading official docs
    ## Inspecting class
    ## Initiating with examples
    ## Snoop
    ## Document
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0002_signature_from_callable.md
    ## Expand cell
    ## Imports and initiate
    ## Examples
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0008_use_kwargs_dict.md
    ## Imports
    ## Reading official docs
    ## empty2none
    ## `_mk_param`
    ## use_kwargs_dict
    ### Reading docs
    ## use_kwargs
    ### Reading docs
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0006_fastcore.meta.NewChkMeta.md
    ## Import and Initalization
    ## Official docs
    ## Prepare Example
    ## Inspect classes
    ## Snoop
    ## Document
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0000_tour.md
    ### Documentation
    ### Testing
    ### Foundations
    ### L
    ### Transforms
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0012_fastcore_foundation_L.md
    ## Document `L` with fastdebug
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0011_Fastdb.md
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0009_funcs_kwargs.md
    ## fastcore.meta.method
    ### Reading Docs
    ### Running codes
    ### Document
    ### snoop
    ## funcs_kwargs
    ### Official docs
    ### snoop: from _funcs_kwargs to funcs_kwargs
    ### snoop only '_funcs_kwargs' by breaking up 'funcs_kwargs'
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0010_fastcore_meta_summary.md
    ## import
    ## fastcore and fastcore.meta
    ### What's inside fastcore.meta
    ## Review individual funcs and classes
    ### What is fastcore.meta all about? 
    ### What can these metaclasses do for me?
    #### FixSigMeta
    #### PrePostInitMeta
    #### AutoInit
    #### NewChkMeta
    #### BypassNewMeta
    ### What can those decorators do for me?
    #### use_kwargs_dict
    #### use_kwargs
    #### delegates
    #### funcs_kwargs
    ### The remaining functions
    ## What is fastcore.meta all about
    
    /Users/Natsume/Documents/fastdebug/mds/questions/00_question_anno_dict.md
    ## `anno_dict` docs
    ## Dive in
    ## `anno_dict` seems not add anything new to `__annotations__`
    ## use fastdebug to double check
    ## Does fastcore want anno_dict to include params with no annos?
    ## Jeremy's response


## fastlistsrcs

Todos: it should be retired by fastlistnbs() after all src notebooks are properly documented.


```
#| export
def fastlistsrcs():
    "display all my commented src codes learning comments in a long list"
    folder ='/Users/Natsume/Documents/fastdebug/learnings/'
    for f in os.listdir(folder):
        if f.endswith(".py"):
            path_f = folder + f
            with open(path_f, 'r') as file:
                for idx, l in enumerate(file):
                    if "#" in l:
                        print(l.split("#")[1], end="")
#                         print(l, end="") # no extra new line between each line printed
        
```


```
fastcodes("Test the signature", nb=True)
```


keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">test_sig.py</mark> 


        "Test the signature of an object"=====================================================(1) # [36;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [36;1mtest_sig will get f's signature as a string[0m; [92;1mb is a signature in string provided by the user[0m; [34;1min fact, test_sig is to compare two strings[0m; 
    



the entire source code in  <mark style="background-color: #FFFF00">test_sig.py</mark> 


    
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    
    def test_sig(f, b):=======================================================================(0)       
        "Test the signature of an object"=====================================================(1) # [36;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [36;1mtest_sig will get f's signature as a string[0m; [92;1mb is a signature in string provided by the user[0m; [34;1min fact, test_sig is to compare two strings[0m; 
        test_eq(str(inspect.signature(f)), b)=================================================(2) # [92;1mtest_sig is to test two strings with test_eq[0m; [92;1mhow to turn a signature into a string;[0m; 
                                                                                                                                                            (3)
    



[Open `0005_fastcore` in Jupyter Notebook](http://localhost:8888/tree/nbs/demos/0005_fastcore.meta.test_sig.ipynb)



```

```

## Best practice of fastdebug.utils


```
import fastdebug.utils as fu
import fastcore.meta as fm
```


<style>.container { width:100% !important; }</style>


**When looking for previous documented learning points**

Run `fastlistnbs()` to check and search for the interesting titles

Then run `fastnbs(...)` on the cell above `fastlistnbs()` to have a better view

Run `fastnbs(query, output=True)` to view the output with input together for notebooks on srcodes


```
fastnbs("snoop: from _funcs_kwargs to funcs_kwargs", output=True)
```


###  <mark style="background-color: #ffff00">snoop:</mark>   <mark style="background-color: #ffff00">from</mark>   <mark style="background-color: #ffff00">_ <mark style="background-color: #FFFF00">funcs_kwargs</mark> </mark>   <mark style="background-color: #ffff00">to</mark>   <mark style="background-color: #FFFF00">funcs_kwargs</mark> 



    
    how to snoop on two functions one wrap around another: `funcs_kwargs` is a wrapper around `_funcs_kwargs`, so I can first snoop on `_funcs_kwargs` and assign its snoop dbsrc to \
    `fm._funcs_kwargs` so that when I snoop on `funcs_kwargs`, it can use the snoop dbsrc of `_funcs_kwargs` and no example codes need to change.
    
    
    ```
    from fastcore.meta import _funcs_kwargs
    ```
    
    
    ```
    fdb_ = Fastdb(_funcs_kwargs)
    fdb_.eg = """
    @funcs_kwargs
    class T:
        _methods=['b'] # allows you to add method b upon instantiation
        def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
        def a(self): return 1
        def b(self): return 2
        
    t = T()
    test_eq(t.a(), 1)
    test_eq(t.b(), 2)
    
    test_sig(T, '(f=1, *, b=None)')
    inspect.signature(T)
    
    def _new_func(): return 5
    
    t = T(b = _new_func)
    test_eq(t.b(), 5)
    
    t = T(a = lambda:3)
    test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
    """
    ```
    
    
    ```
    #| column: screen
    # no snoop result, it is expected, because the example is not calling _funcs_kwargs, but funcs_kwargs
    fdb_.snoop(deco=True) # how to snoop decorator: _funcs_kwargs is a decorator, so set deco=True to see running codes in inner f
    ```
    
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ==============================================================     on line [91;1mNone[0m     ==============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        [0m     
        
    
    
    
    ```
    import fastcore.meta as fm
    ```
    
    
    ```
    fm._funcs_kwargs = fdb_.dbsrc # how to snoop on two functions one wrap around another
    ```
    
    
    ```
    fdb = Fastdb(funcs_kwargs)
    fdb.eg = """
    @funcs_kwargs
    class T:
        _methods=['b'] # allows you to add method b upon instantiation
        def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
        def a(self): return 1
        def b(self): return 2
        
    t = T()
    test_eq(t.a(), 1)
    test_eq(t.b(), 2)
    
    test_sig(T, '(f=1, *, b=None)')
    inspect.signature(T)
    
    def _new_func(): return 5
    
    t = T(b = _new_func)
    test_eq(t.b(), 5)
    
    t = T(a = lambda:3)
    test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
    
    def _f(self,a=1): return self.num + a # access the num attribute from the instance
    
    @funcs_kwargs(as_method=True)
    class T: 
        _methods=['b']
        num = 5
        
    t = T(b = _f) # adds method b
    test_eq(t.b(5), 10) # self.num + 5 = 10
    
    def _f(self,a=1): return self.num * a #multiply instead of add 
    
    class T2(T):
        def __init__(self,num):
            super().__init__(b = _f) # add method b from the super class
            self.num=num
            
    t = T2(num=3)
    test_eq(t.b(a=5), 15) # 3 * 5 = 15
    test_sig(T2, '(num)')
    """
    ```
    
    
    ```
    #| column: screen
    fdb.print()
    ```
    
        =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
        ==============================================================     on line [91;1mNone[0m     ==============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        def _f(self,a=1): return self.num + a # access the num attribute from the instance
        
        @funcs_kwargs(as_method=True)
        class T: 
            _methods=['b']
            num = 5
            
        t = T(b = _f) # adds method b
        test_eq(t.b(5), 10) # self.num + 5 = 10
        
        def _f(self,a=1): return self.num * a #multiply instead of add 
        
        class T2(T):
            def __init__(self,num):
                super().__init__(b = _f) # add method b from the super class
                self.num=num
                
        t = T2(num=3)
        test_eq(t.b(a=5), 15) # 3 * 5 = 15
        test_sig(T2, '(num)')
        [0m     
        
        def funcs_kwargs(as_method=False):========================================================(0)       
            "Replace methods in `cls._methods` with those from `kwargs`"==========================(1)       
            if callable(as_method): return _funcs_kwargs(as_method, False)========================(2)       
            return partial(_funcs_kwargs, as_method=as_method)====================================(3)       
                                                                                                                                                                (4)
    
    
    
    ```
    #| column: screen
    fdb_.print()
    ```
    
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ==============================================================     on line [91;1mNone[0m     ==============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        def _funcs_kwargs(cls, as_method):========================================================(0)       
            old_init = cls.__init__===============================================================(1)       
            def _init(self, *args, **kwargs):=====================================================(2)       
                for k in cls._methods:============================================================(3)       
                    arg = kwargs.pop(k,None)======================================================(4)       
                    if arg is not None:===========================================================(5)       
                        if as_method: arg = method(arg)===========================================(6)       
                        if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=======(7)       
                        setattr(self, k, arg)=====================================================(8)       
                old_init(self, *args, **kwargs)===================================================(9)       
            functools.update_wrapper(_init, old_init)=============================================(10)      
            cls.__init__ = use_kwargs(cls._methods)(_init)========================================(11)      
            if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
            return cls============================================================================(13)      
                                                                                                                                                                (14)
    
    
    
    ```
    #| column: screen
    fdb.docsrc(1, "how funcs_kwargs works; it is a wrapper around _funcs_kwargs; it offers two ways of running _funcs_kwargs; \
    the first, default way, is to add a func to a class without using self; second way is to add func to class enabling self use;")
    fdb.docsrc(2, "how to check whether an object is callable; how to return a result of running a func; ")
    fdb.docsrc(3, "how to custom the params of `_funcs_kwargs` for a particular use with partial")
    ```
    
        =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m1[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        def _f(self,a=1): return self.num + a # access the num attribute from the instance
        
        @funcs_kwargs(as_method=True)
        class T: 
            _methods=['b']
            num = 5
            
        t = T(b = _f) # adds method b
        test_eq(t.b(5), 10) # self.num + 5 = 10
        
        def _f(self,a=1): return self.num * a #multiply instead of add 
        
        class T2(T):
            def __init__(self,num):
                super().__init__(b = _f) # add method b from the super class
                self.num=num
                
        t = T2(num=3)
        test_eq(t.b(a=5), 15) # 3 * 5 = 15
        test_sig(T2, '(num)')
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
        def funcs_kwargs(as_method=False):                                                                                                                      (0)
            "Replace methods in `cls._methods` with those from `kwargs`"========================================================================================(1)
        [91;1mhow funcs_kwargs works; it is a wrapper around _funcs_kwargs; it offers two ways of running _funcs_kwargs; the first, default way, is to add a func to a class without using self; second way is to add func to class enabling self use;[0m
            if callable(as_method): return _funcs_kwargs(as_method, False)                                                                                      (2)
            return partial(_funcs_kwargs, as_method=as_method)                                                                                                  (3)
        =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m2[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        def _f(self,a=1): return self.num + a # access the num attribute from the instance
        
        @funcs_kwargs(as_method=True)
        class T: 
            _methods=['b']
            num = 5
            
        t = T(b = _f) # adds method b
        test_eq(t.b(5), 10) # self.num + 5 = 10
        
        def _f(self,a=1): return self.num * a #multiply instead of add 
        
        class T2(T):
            def __init__(self,num):
                super().__init__(b = _f) # add method b from the super class
                self.num=num
                
        t = T2(num=3)
        test_eq(t.b(a=5), 15) # 3 * 5 = 15
        test_sig(T2, '(num)')
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
        def funcs_kwargs(as_method=False):                                                                                                                      (0)
            "Replace methods in `cls._methods` with those from `kwargs`"                                                                                        (1)
            if callable(as_method): return _funcs_kwargs(as_method, False)======================================================================================(2)
                                                                               [91;1mhow to check whether an object is callable; how to return a result of running a func; [0m
            return partial(_funcs_kwargs, as_method=as_method)                                                                                                  (3)
                                                                                                                                                                (4)
        =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m3[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        def _f(self,a=1): return self.num + a # access the num attribute from the instance
        
        @funcs_kwargs(as_method=True)
        class T: 
            _methods=['b']
            num = 5
            
        t = T(b = _f) # adds method b
        test_eq(t.b(5), 10) # self.num + 5 = 10
        
        def _f(self,a=1): return self.num * a #multiply instead of add 
        
        class T2(T):
            def __init__(self,num):
                super().__init__(b = _f) # add method b from the super class
                self.num=num
                
        t = T2(num=3)
        test_eq(t.b(a=5), 15) # 3 * 5 = 15
        test_sig(T2, '(num)')
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
            "Replace methods in `cls._methods` with those from `kwargs`"                                                                                        (1)
            if callable(as_method): return _funcs_kwargs(as_method, False)                                                                                      (2)
            return partial(_funcs_kwargs, as_method=as_method)==================================================================================================(3)
                                                                                        [91;1mhow to custom the params of `_funcs_kwargs` for a particular use with partial[0m
                                                                                                                                                                (4)
    
    
    
    ```
    #| column: screen
    fdb_.print()
    ```
    
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ==============================================================     on line [91;1mNone[0m     ==============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        def _funcs_kwargs(cls, as_method):========================================================(0)       
            old_init = cls.__init__===============================================================(1)       
            def _init(self, *args, **kwargs):=====================================================(2)       
                for k in cls._methods:============================================================(3)       
                    arg = kwargs.pop(k,None)======================================================(4)       
                    if arg is not None:===========================================================(5)       
                        if as_method: arg = method(arg)===========================================(6)       
                        if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=======(7)       
                        setattr(self, k, arg)=====================================================(8)       
                old_init(self, *args, **kwargs)===================================================(9)       
            functools.update_wrapper(_init, old_init)=============================================(10)      
            cls.__init__ = use_kwargs(cls._methods)(_init)========================================(11)      
            if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
            return cls============================================================================(13)      
                                                                                                                                                                (14)
    
    
    
    ```
    #| column: screen
    fdb_.docsrc(0, "how does _funcs_kwargs work: _funcs_kwargs is a decorator; it helps class e.g., T to add more methods; \
    I need to give the method a name, \
    and put the name e.g., 'b' inside a list called _methods=['b'] inside class T; \
    then after writing a func e.g., _new_func, I can add it by T(b = _new_func); if I want the func added to class to use self, \
    I shall write @funcs_kwargs(as_method=True)")
    fdb_.docsrc(2, "how to define a method which can use self and accept any parameters")
    fdb_.docsrc(3, "how to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead; ")
    fdb_.docsrc(6, "how to turn a func into a method")
    fdb_.docsrc(7, "how to give a method a different instance, like self")
    fdb_.docsrc(8, "how to add a method to a class as an attribute")
    fdb_.docsrc(10, "how to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself")
    fdb_.docsrc(11, "how to add a list of names with None as default value to function `_init` to repalce its kwargs param")
    fdb_.docsrc(12, "how to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`")
    
    ```
    
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m0[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
        def _funcs_kwargs(cls, as_method):======================================================================================================================(0)
        [91;1mhow does _funcs_kwargs work: _funcs_kwargs is a decorator; it helps class e.g., T to add more methods; I need to give the method a name, and put the name e.g., 'b' inside a list called _methods=['b'] inside class T; then after writing a func e.g., _new_func, I can add it by T(b = _new_func); if I want the func added to class to use self, I shall write @funcs_kwargs(as_method=True)[0m
            old_init = cls.__init__                                                                                                                             (1)
            def _init(self, *args, **kwargs):                                                                                                                   (2)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m2[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
        def _funcs_kwargs(cls, as_method):                                                                                                                      (0)
            old_init = cls.__init__                                                                                                                             (1)
            def _init(self, *args, **kwargs):===================================================================================================================(2)
                                                                                                  [91;1mhow to define a method which can use self and accept any parameters[0m
                for k in cls._methods:                                                                                                                          (3)
                    arg = kwargs.pop(k,None)                                                                                                                    (4)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m3[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
            old_init = cls.__init__                                                                                                                             (1)
            def _init(self, *args, **kwargs):                                                                                                                   (2)
                for k in cls._methods:==========================================================================================================================(3)
                                      [91;1mhow to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead; [0m
                    arg = kwargs.pop(k,None)                                                                                                                    (4)
                    if arg is not None:                                                                                                                         (5)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m6[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
                    arg = kwargs.pop(k,None)                                                                                                                    (4)
                    if arg is not None:                                                                                                                         (5)
                        if as_method: arg = method(arg)=========================================================================================================(6)
                                                                                                                                     [91;1mhow to turn a func into a method[0m
                        if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)                                                                     (7)
                        setattr(self, k, arg)                                                                                                                   (8)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m7[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
                    if arg is not None:                                                                                                                         (5)
                        if as_method: arg = method(arg)                                                                                                         (6)
                        if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=====================================================================(7)
                                                                                                                 [91;1mhow to give a method a different instance, like self[0m
                        setattr(self, k, arg)                                                                                                                   (8)
                old_init(self, *args, **kwargs)                                                                                                                 (9)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m8[0m     ================================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
                        if as_method: arg = method(arg)                                                                                                         (6)
                        if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)                                                                     (7)
                        setattr(self, k, arg)===================================================================================================================(8)
                                                                                                                       [91;1mhow to add a method to a class as an attribute[0m
                old_init(self, *args, **kwargs)                                                                                                                 (9)
            functools.update_wrapper(_init, old_init)                                                                                                           (10)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m10[0m     ===============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
                        setattr(self, k, arg)                                                                                                                   (8)
                old_init(self, *args, **kwargs)                                                                                                                 (9)
            functools.update_wrapper(_init, old_init)===========================================================================================================(10)
                                                                              [91;1mhow to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself[0m
            cls.__init__ = use_kwargs(cls._methods)(_init)                                                                                                      (11)
            if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m11[0m     ===============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
                old_init(self, *args, **kwargs)                                                                                                                 (9)
            functools.update_wrapper(_init, old_init)                                                                                                           (10)
            cls.__init__ = use_kwargs(cls._methods)(_init)======================================================================================================(11)
                                                                [91;1mhow to add a list of names with None as default value to function `_init` to repalce its kwargs param[0m
            if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))                                                     (12)
            return cls                                                                                                                                          (13)
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m12[0m     ===============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        [93;1mprint selected srcline with expands below[0m--------
            functools.update_wrapper(_init, old_init)                                                                                                           (10)
            cls.__init__ = use_kwargs(cls._methods)(_init)                                                                                                      (11)
            if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))=====================================================(12)
                                                       [91;1mhow to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`[0m
            return cls                                                                                                                                          (13)
                                                                                                                                                                (14)
    
    
    
    ```
    #| column: screen
    fdb.snoop() # how to snoop together with docsrc: snoop first and docsrc above it
    ```
    
        22:28:59.99 >>> Call to funcs_kwargs in File "/tmp/funcs_kwargs.py", line 3
        22:28:59.99 ...... as_method = <class 'fastcore.meta.T'>
        22:28:59.99    3 | def funcs_kwargs(as_method=False):
        22:28:59.99    5 |     if callable(as_method): return _funcs_kwargs(as_method, False)
            22:28:59.99 >>> Call to _funcs_kwargs in File "/tmp/_funcs_kwargs.py", line 3
            22:28:59.99 ...... cls = <class 'fastcore.meta.T'>
            22:28:59.99 ...... as_method = False
            22:28:59.99    3 | def _funcs_kwargs(cls, as_method):
            22:28:59.99    4 |     old_init = cls.__init__
            22:28:59.99 .......... old_init = <function T.__init__>
            22:28:59.99    5 |     import snoop
            22:28:59.99 .......... snoop = <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>
            22:28:59.99    6 |     @snoop
            22:28:59.99    7 |     def _init(self, *args, **kwargs):
            22:28:59.99 .......... _init = <function _funcs_kwargs.<locals>._init>
            22:28:59.99   15 |     functools.update_wrapper(_init, old_init)
            22:28:59.99 .......... _init = <function T.__init__>
            22:28:59.99   16 |     cls.__init__ = use_kwargs(cls._methods)(_init)
            22:28:59.99   17 |     if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))
            22:28:59.99   18 |     return cls
            22:28:59.99 <<< Return value from _funcs_kwargs: <class 'fastcore.meta.T'>
        22:28:59.99    5 |     if callable(as_method): return _funcs_kwargs(as_method, False)
        22:28:59.99 <<< Return value from funcs_kwargs: <class 'fastcore.meta.T'>
        22:28:59.99 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
        22:28:59.99 .......... self = <fastcore.meta.T object>
        22:28:59.99 .......... args = ()
        22:28:59.99 .......... kwargs = {}
        22:28:59.99 .......... as_method = False
        22:28:59.99 .......... cls = <class 'fastcore.meta.T'>
        22:28:59.99 .......... old_init = <function T.__init__>
        22:28:59.99    7 |     def _init(self, *args, **kwargs):
        22:29:00.00    8 |         for k in cls._methods:
        22:29:00.00 .............. k = 'b'
        22:29:00.00    9 |             arg = kwargs.pop(k,None)
        22:29:00.00 .................. arg = None
        22:29:00.00   10 |             if arg is not None:
        22:29:00.00    8 |         for k in cls._methods:
        22:29:00.00   14 |         old_init(self, *args, **kwargs)
        22:29:00.00 <<< Return value from _funcs_kwargs.<locals>._init: None
        22:29:00.00 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
        22:29:00.00 .......... self = <fastcore.meta.T object>
        22:29:00.00 .......... args = ()
        22:29:00.00 .......... kwargs = {'b': <function _new_func>}
        22:29:00.00 .......... len(kwargs) = 1
        22:29:00.00 .......... as_method = False
        22:29:00.00 .......... cls = <class 'fastcore.meta.T'>
        22:29:00.00 .......... old_init = <function T.__init__>
        22:29:00.00    7 |     def _init(self, *args, **kwargs):
        22:29:00.00    8 |         for k in cls._methods:
        22:29:00.00 .............. k = 'b'
        22:29:00.00    9 |             arg = kwargs.pop(k,None)
        22:29:00.00 .................. kwargs = {}
        22:29:00.00 .................. arg = <function _new_func>
        22:29:00.00   10 |             if arg is not None:
        22:29:00.00   11 |                 if as_method: arg = method(arg)
        22:29:00.00   12 |                 if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
        22:29:00.00   13 |                 setattr(self, k, arg)
        22:29:00.00    8 |         for k in cls._methods:
        22:29:00.00   14 |         old_init(self, *args, **kwargs)
        22:29:00.00 <<< Return value from _funcs_kwargs.<locals>._init: None
        22:29:00.00 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
        22:29:00.00 .......... self = <fastcore.meta.T object>
        22:29:00.00 .......... args = ()
        22:29:00.00 .......... kwargs = {'a': <function <lambda>>}
        22:29:00.00 .......... len(kwargs) = 1
        22:29:00.00 .......... as_method = False
        22:29:00.00 .......... cls = <class 'fastcore.meta.T'>
        22:29:00.00 .......... old_init = <function T.__init__>
        22:29:00.00    7 |     def _init(self, *args, **kwargs):
        22:29:00.00    8 |         for k in cls._methods:
        22:29:00.00 .............. k = 'b'
        22:29:00.00    9 |             arg = kwargs.pop(k,None)
        22:29:00.00 .................. arg = None
        22:29:00.00   10 |             if arg is not None:
        22:29:00.00    8 |         for k in cls._methods:
        22:29:00.00   14 |         old_init(self, *args, **kwargs)
        22:29:00.00 <<< Return value from _funcs_kwargs.<locals>._init: None
        22:29:00.00 >>> Call to funcs_kwargs in File "/tmp/funcs_kwargs.py", line 3
        22:29:00.00 ...... as_method = True
        22:29:00.00    3 | def funcs_kwargs(as_method=False):
        22:29:00.00    5 |     if callable(as_method): return _funcs_kwargs(as_method, False)
        22:29:00.00    6 |     return partial(_funcs_kwargs, as_method=as_method)
        22:29:00.00 <<< Return value from funcs_kwargs: functools.partial(<function _funcs_kwargs>, as_method=True)
        22:29:00.00 >>> Call to _funcs_kwargs in File "/tmp/_funcs_kwargs.py", line 3
        22:29:00.00 ...... cls = <class 'fastcore.meta.T'>
        22:29:00.00 ...... as_method = True
        22:29:00.00    3 | def _funcs_kwargs(cls, as_method):
        22:29:00.00    4 |     old_init = cls.__init__
        22:29:00.00 .......... old_init = <slot wrapper '__init__' of 'object' objects>
        22:29:00.00    5 |     import snoop
        22:29:00.00 .......... snoop = <class 'snoop.configuration.Config.__init__.<locals>.ConfiguredTracer'>
        22:29:00.00    6 |     @snoop
        22:29:00.00    7 |     def _init(self, *args, **kwargs):
        22:29:00.00 .......... _init = <function _funcs_kwargs.<locals>._init>
        22:29:00.00   15 |     functools.update_wrapper(_init, old_init)
        22:29:00.00 .......... _init = <function object.__init__>
        22:29:00.00   16 |     cls.__init__ = use_kwargs(cls._methods)(_init)
        22:29:00.01   17 |     if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))
        22:29:00.01   18 |     return cls
        22:29:00.01 <<< Return value from _funcs_kwargs: <class 'fastcore.meta.T'>
        22:29:00.01 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
        22:29:00.01 .......... self = <fastcore.meta.T object>
        22:29:00.01 .......... args = ()
        22:29:00.01 .......... kwargs = {'b': <function _f>}
        22:29:00.01 .......... len(kwargs) = 1
        22:29:00.01 .......... as_method = True
        22:29:00.01 .......... cls = <class 'fastcore.meta.T'>
        22:29:00.01 .......... old_init = <slot wrapper '__init__' of 'object' objects>
        22:29:00.01    7 |     def _init(self, *args, **kwargs):
        22:29:00.01    8 |         for k in cls._methods:
        22:29:00.01 .............. k = 'b'
        22:29:00.01    9 |             arg = kwargs.pop(k,None)
        22:29:00.01 .................. kwargs = {}
        22:29:00.01 .................. arg = <function _f>
        22:29:00.01   10 |             if arg is not None:
        22:29:00.01   11 |                 if as_method: arg = method(arg)
        22:29:00.01 ...... arg = <bound method int._f of 1>
        22:29:00.01   12 |                 if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
        22:29:00.01 ...... arg = <bound method T._f of <fastcore.meta.T object>>
        22:29:00.01   13 |                 setattr(self, k, arg)
        22:29:00.01    8 |         for k in cls._methods:
        22:29:00.01   14 |         old_init(self, *args, **kwargs)
        22:29:00.01 <<< Return value from _funcs_kwargs.<locals>._init: None
        22:29:00.01 >>> Call to _funcs_kwargs.<locals>._init in File "/tmp/_funcs_kwargs.py", line 7
        22:29:00.01 .......... self = <fastcore.meta.T2 object>
        22:29:00.01 .......... args = ()
        22:29:00.01 .......... kwargs = {'b': <function _f>}
        22:29:00.01 .......... len(kwargs) = 1
        22:29:00.01 .......... as_method = True
        22:29:00.01 .......... cls = <class 'fastcore.meta.T'>
        22:29:00.01 .......... old_init = <slot wrapper '__init__' of 'object' objects>
        22:29:00.01    7 |     def _init(self, *args, **kwargs):
        22:29:00.01    8 |         for k in cls._methods:
        22:29:00.01 .............. k = 'b'
        22:29:00.01    9 |             arg = kwargs.pop(k,None)
        22:29:00.01 .................. kwargs = {}
        22:29:00.01 .................. arg = <function _f>
        22:29:00.01   10 |             if arg is not None:
        22:29:00.01   11 |                 if as_method: arg = method(arg)
        22:29:00.01 ...... arg = <bound method int._f of 1>
        22:29:00.01   12 |                 if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
        22:29:00.01 ...... arg = <bound method T2._f of <fastcore.meta.T2 object>>
        22:29:00.01   13 |                 setattr(self, k, arg)
        22:29:00.01    8 |         for k in cls._methods:
        22:29:00.01   14 |         old_init(self, *args, **kwargs)
        22:29:00.01 <<< Return value from _funcs_kwargs.<locals>._init: None
    
    
        =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
        ==============================================================     on line [91;1mNone[0m     ==============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        def _f(self,a=1): return self.num + a # access the num attribute from the instance
        
        @funcs_kwargs(as_method=True)
        class T: 
            _methods=['b']
            num = 5
            
        t = T(b = _f) # adds method b
        test_eq(t.b(5), 10) # self.num + 5 = 10
        
        def _f(self,a=1): return self.num * a #multiply instead of add 
        
        class T2(T):
            def __init__(self,num):
                super().__init__(b = _f) # add method b from the super class
                self.num=num
                
        t = T2(num=3)
        test_eq(t.b(a=5), 15) # 3 * 5 = 15
        test_sig(T2, '(num)')
        [0m     
        
    
    
    
    ```
    fdb_.debug()
    ```
    
        _funcs_kwargs's dbsrc code: ==============
        import snoop
        @snoop
        def _funcs_kwargs(cls, as_method):
            old_init = cls.__init__
            import snoop
            @snoop
            def _init(self, *args, **kwargs):
                for k in cls._methods:
                    arg = kwargs.pop(k,None)
                    if arg is not None:
                        if as_method: arg = method(arg)
                        if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)
                        setattr(self, k, arg)
                old_init(self, *args, **kwargs)
            functools.update_wrapper(_init, old_init)
            cls.__init__ = use_kwargs(cls._methods)(_init)
            if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__))
            return cls
        
        
        
        _funcs_kwargs's example processed with dbsrc: ===============
        
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        
    
    
    
    ```
    #| column: screen
    fdb.print()
    ```
    
        =======================================================     Investigating [91;1mfuncs_kwargs[0m     =======================================================
        ==============================================================     on line [91;1mNone[0m     ==============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        def _f(self,a=1): return self.num + a # access the num attribute from the instance
        
        @funcs_kwargs(as_method=True)
        class T: 
            _methods=['b']
            num = 5
            
        t = T(b = _f) # adds method b
        test_eq(t.b(5), 10) # self.num + 5 = 10
        
        def _f(self,a=1): return self.num * a #multiply instead of add 
        
        class T2(T):
            def __init__(self,num):
                super().__init__(b = _f) # add method b from the super class
                self.num=num
                
        t = T2(num=3)
        test_eq(t.b(a=5), 15) # 3 * 5 = 15
        test_sig(T2, '(num)')
        [0m     
        
        def funcs_kwargs(as_method=False):========================================================(0)       
            "Replace methods in `cls._methods` with those from `kwargs`"==========================(1) # [93;1mhow funcs_kwargs works[0m; [93;1mit is a wrapper around _funcs_kwargs[0m; [91;1mit offers two ways of running _funcs_kwargs[0m; [37;1mthe first, default way, is to add a func to a class without using self[0m; [91;1msecond way is to add func to class enabling self use;[0m; 
            if callable(as_method): return _funcs_kwargs(as_method, False)========================(2) # [37;1mhow to check whether an object is callable[0m; [93;1mhow to return a result of running a func[0m; [37;1m[0m; 
            return partial(_funcs_kwargs, as_method=as_method)====================================(3) # [92;1mhow to custom the params of `_funcs_kwargs` for a particular use with partial[0m; 
                                                                                                                                                                (4)
    
    
    
    ```
    #| column: screen
    fdb_.print()
    ```
    
        ======================================================     Investigating [91;1m_funcs_kwargs[0m     =======================================================
        ===============================================================     on line [91;1m12[0m     ===============================================================
             with example [91;1m
        @funcs_kwargs
        class T:
            _methods=['b'] # allows you to add method b upon instantiation
            def __init__(self, f=1, **kwargs): pass # don't forget to include **kwargs in __init__
            def a(self): return 1
            def b(self): return 2
            
        t = T()
        test_eq(t.a(), 1)
        test_eq(t.b(), 2)
        
        test_sig(T, '(f=1, *, b=None)')
        inspect.signature(T)
        
        def _new_func(): return 5
        
        t = T(b = _new_func)
        test_eq(t.b(), 5)
        
        t = T(a = lambda:3)
        test_eq(t.a(), 1) # the attempt to add a is ignored and uses the original method instead.
        
        [0m     
        
        def _funcs_kwargs(cls, as_method):========================================================(0) # [93;1mhow does _funcs_kwargs work: _funcs_kwargs is a decorator[0m; [35;1mit helps class e.g., T to add more methods[0m; [93;1mI need to give the method a name, and put the name e.g., 'b' inside a list called _methods=['b'] inside class T[0m; [36;1mthen after writing a func e.g., _new_func, I can add it by T(b = _new_func)[0m; [36;1mif I want the func added to class to use self, I shall write @funcs_kwargs(as_method=True)[0m; 
            old_init = cls.__init__===============================================================(1)       
            def _init(self, *args, **kwargs):=====================================================(2) # [37;1mhow to define a method which can use self and accept any parameters[0m; 
                for k in cls._methods:============================================================(3) # [35;1mhow to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead[0m; [91;1m[0m; 
                    arg = kwargs.pop(k,None)======================================================(4)       
                    if arg is not None:===========================================================(5)       
                        if as_method: arg = method(arg)===========================================(6) # [36;1mhow to turn a func into a method[0m; 
                        if isinstance(arg,MethodType): arg = MethodType(arg.__func__, self)=======(7) # [93;1mhow to give a method a different instance, like self[0m; 
                        setattr(self, k, arg)=====================================================(8) # [36;1mhow to add a method to a class as an attribute[0m; 
                old_init(self, *args, **kwargs)===================================================(9)       
            functools.update_wrapper(_init, old_init)=============================================(10) # [34;1mhow to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself[0m; 
            cls.__init__ = use_kwargs(cls._methods)(_init)========================================(11) # [34;1mhow to add a list of names with None as default value to function `_init` to repalce its kwargs param[0m; 
            if hasattr(cls, '__signature__'): cls.__signature__ = _rm_self(inspect.signature(cls.__init__)) # [35;1mhow to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`[0m;  (12)
            return cls============================================================================(13)      
                                                                                                                                                                (14)
    
    
    
    ```
    
    ```
    
    



[Open `0009_funcs_kwargs` in Jupyter Notebook](http://localhost:8888/tree/nbs/demos/0009_funcs_kwargs.ipynb)



```
fastlistnbs()
```

    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0001_is_it_a_bird.md
    ## Useful Course sites
    ## How to use autoreload
    ## How to install and update libraries
    ## Know a little about the libraries
    ### what is fastai
    ### what is duckduckgo
    ## How to use fastdebug with fastai notebooks
    ### what is fastdebug
    ### Did I document it in a notebook before?
    ### Did I document it in a src before?
    ## how to download images
    ### how to create folders using path; how to search and download images in folders; how to resize images 
    ## Train my model
    ### How to find and unlink images not properly downloaded
    ### How to create a DataLoaders with DataBlock; how to view data with it
    ### How to build my model with dataloaders and pretrained model; how to train my model
    ### How to predict with my model; how to avoid running cells in nbdev_prepare
    ## Check fastnbs and fastlistnbs
    ## search and display section input and output in notebooks
    
    /Users/Natsume/Documents/fastdebug/mds/lib/utils.md
    ## Expand cells
    ## Import fastcore env
    ## to inspect a class
    ### get the docs for each function of a class
    ## is it a metaclass?
    ## is it a decorator
    ### handle all kinds of exceptions for evaluating retn 
    ## whatinside a module of a library
    ### show the type of objects inside `__all__`
    ### working for fastdebug.core
    ### to show Fastdb methods
    ## whichversion of a library
    ## fastview
    ## fastscrs
    ## getrootport
    ## jn_link
    ## get_all_nbs
    ### get all nbs path for both md and ipynb
    ## openNB
    ## highlight
    ## display_md
    ## display_block
    ### handle both file path and file content at the same time
    ## add_py_block
    ## fastnbs
    ## fastcodes
    ## fastnotes
    ### multiple folders
    ## fastlistnbs
    ## fastlistsrcs
    ## Examples
    ## Export
    ## Send to Obsidian
    
    /Users/Natsume/Documents/fastdebug/mds/lib/00_core.md
    ## make life easier with defaults  
    ## globals() and locals()
    ## Execute strings
    ### new variable or updated variable by exec will only be accessible from locals()
    ### eval can override its own globals() and locals()
    ### when exec update existing functions
    ### when the func to be udpated involve other libraries
    ### inside a function, exec() allow won't give you necessary env from function namespace
    ### magic of `exec(b, globals().update(locals()))`
    ### Bring variables from a func namespace to the sideout world
    ### globals() in a cell vs globals() in a func
    ## make a colorful string
    ## align text to the most right
    ## printsrcwithidx
    ### print the entire source code with idx from 0
    ### print the whole src with idx or print them in parts
    ### use cmts from dbprint to print out src with comments
    ### no more update for printsrcwithidx, for the latest see Fastdb.print
    ## print out src code
    ### basic version
    ### print src with specific number of lines
    ### make the naming more sensible
    ### Allow a dbline occur more than once
    ### adding idx to the selected srclines
    #### printsrclinewithidx
    ### dblines can be string of code or idx number
    ### avoid multi-occurrance of the same srcline
    ## dbprint on expression
    ### basic version
    ### insert dbcode and make a new dbfunc
    ### Bring outside namespace variables into exec()
    ### Bring what inside the func namespace variables to the outside world
    ### Adding g = locals() to dbprintinsert to avoid adding env individually
    ### enable srclines to be either string or int 
    ### enable = to be used as assignment in codes
    ### avoid adding "env=g" for dbprintinsert
    ### collect cmt for later printsrcwithidx to print comments together
    ### make sure only one line with correct idx is debugged
    ### avoid typing "" when there is no codes
    ### no more update for dbprint, for the latest see Fastdb.dbprint
    ### use dbprint to override the original official code without changing its own pyfile
    ## dbprintinsert
    ### Run and display the inserted dbcodes 
    ### use locals() inside the dbsrc code to avoid adding env individually
    ### enable dbprintinsert to do exec on a block of code
    ## printrunsrclines() 
    ### Examples
    #### simple example
    #### complex example
    ### insert a line after each srcline to add idx
    ### add correct indentation to each inserted line
    #### count the indentation for each srcline
    ### indentation special case: if, else, for, def
    ### remove pure comments or docs from dbsrc
    ### print out the srclines which get run
    ### Make sure all if, else, for get printed
    ### Put all together into the function printrunsrclines()
    #### no more renaming of foo
    #### add example as a param into the function
    #### improve on search for `if`, else, for, def to avoid errors for more examples
    #### remove an empty line with indentation
    ### make it work
    ### more difficult examples to test printrunsrc()
    ## Make fastdebug a class
    ### improve on the line idx readability
    ### collect cmt from dbprint and print
    ### make sure only the line with correct idx is debugged
    ### having "" or "   " inside codes without causing error
    ### replace Fastdb.printsrcwithdix with Fastdb.print
    ### add idx to dbsrc when showdbsrc=True
    ### not load the inner locals() to outenv can prevent mysterious printing of previous db messages
    ### using @patch to enable docs for instance methods like `dbprint` and `print`
    ### move param env into `__init__`
    ### Add example to the obj
    ### Take not only function but also class
    ### To remove the necessity of self.takExample()
    ### Try to remove g = locals()
    ### Make sure `showdbsrc=True` give us the line starting with 'dbprintinsert'
    ### Make sure `showdbsrc=True` give us info on changes in g or outenv
    ### exit and print a warning message: idx has to be int
    ### handle errors by codes with trailing spaces 
    ### showdbsrc=True, check whether Fastdb.dbprint and fdb.dbprint are same object using `is`
    ### remove unnecessary db printout when showdbsrc=True and add printout to display sections
    ### raise TypeError when decode are not integer and showdbsrc=true working on both method and function
    ### when debugging dbprint, make sure dbsrc is printed with the same idx as original
    ### update dbsrc to the global env
    ### go back to normal before running dbprint again
    ### auto print src with cmt and idx as the ending part of dbprint
    ### to mark my explorations (expressions to evaluate) to stand out
    ### Add the print of src with idx and comments at the end of dbsrc
    ### embed example and autoprint to shorten the code to type
    ### Make title for dbprint
    ### Adding self.eg info and color group into dbprint and print
    #### todo: make the comments with same self.eg have the same color
    ### make dbsrc print idx right
    ### add self.eg to a dict with keys are idxsrc
    ### handle both function and class as src
    ### documenting on Fastdb.dbprint itself
    ## mk_dbsrc
    ## Turn mk_dbsrc into docsrc 
    ### handle when no codes are given
    ## create_dbsrc_from_string
    ## replaceWithDbsrc
    ### handle class and metaclass
    ### improve on handling function as decorator
    ### Handling `inspect._signature_from_callable` to become `self.dbsrc`
    ### handling usage of `@delegates`
    ### handling `@delegates` with indentation before it
    ### handling classes by inspect.isclass() rather than == type and add more class situations
    ### handling `class _T(_TestA, metaclass=BypassNewMeta): `
    ## run_example
    ### `exec(self.eg, globals().update(self.egEnv), locals())` works better than `...update(locals()), self.egEnv)
    ### no more env cells run before `fdb.eg` to make `fdb.run_example` work
    ## Autoprint
    ## Take an example and its env into Fastdb obj
    ## print src with idx and cmt in whole or parts
    ### print self.eg after each comment and colorize comments
    ### color examples and cmts separately and make the func simpler
    ### split each cmt and colorize parts randomly
    ### printcmts1 while saving into a file
    ## goback
    ## Fastdb.explore
    ### adding one breakpoint with comment
    ### Adding multiple breakpoints by multiple set_trace()
    ### Go back to normal before running explore again
    ### enable fdb.takExample("whatinside(fu), ...) without using `fu.whatinside`
    ### refactory explore
    ## snoop
    ### snoop on both function and class
    ### snoop on class and method and all???
    ### snoop
    ### simplify adding @snoop for both normal function and decorator
    ### handling classes
    ### add watch
    ## Snoop
    ### add watch
    ### use guide on Fastdb.dbprint
    ## reliveonce
    ## Fastdb.debug
    ## Export
    ## Send to Obsidian
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0001_fastcore_meta_delegates.md
    ## Import
    ## Initiate Fastdb and example in str
    ## Example
    ## docsrc
    ## Snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md
    ## Initialize fastdebug objects
    ## class FixSigMeta(type) vs class Foo(type)
    ## class Foo()
    ## class PrePostInitMeta(FixSigMeta)
    ## class Foo(metaclass=FixSigMeta)
    ## class AutoInit(metaclass=PrePostInitMeta)
    ## Prepare examples for FixSigMeta, PrePostInitMeta, AutoInit 
    ## Snoop them together in one go
    ### embed the dbsrc of FixSigMeta into PrePostInitMeta
    ### embed dbsrc of PrePostInitMeta into AutoInit
    ## Explore and Document on them together 
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0004_fastcore.meta._rm_self.md
    ## imports
    ## set up
    ## document
    ## snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0005_fastcore.meta.test_sig.md
    ## imports
    ## setups
    ## documents
    ## snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0007_fastcore.meta.BypassNewMeta.md
    ## Reading official docs
    ## Inspecting class
    ## Initiating with examples
    ## Snoop
    ## Document
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0002_signature_from_callable.md
    ## Expand cell
    ## Imports and initiate
    ## Examples
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0008_use_kwargs_dict.md
    ## Imports
    ## Reading official docs
    ## empty2none
    ## `_mk_param`
    ## use_kwargs_dict
    ### Reading docs
    ## use_kwargs
    ### Reading docs
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0006_fastcore.meta.NewChkMeta.md
    ## Import and Initalization
    ## Official docs
    ## Prepare Example
    ## Inspect classes
    ## Snoop
    ## Document
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0000_tour.md
    ### Documentation
    ### Testing
    ### Foundations
    ### L
    ### Transforms
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0012_fastcore_foundation_L.md
    ## Document `L` with fastdebug
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0011_Fastdb.md
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0009_funcs_kwargs.md
    ## fastcore.meta.method
    ### Reading Docs
    ### Running codes
    ### Document
    ### snoop
    ## funcs_kwargs
    ### Official docs
    ### snoop: from _funcs_kwargs to funcs_kwargs
    ### snoop only '_funcs_kwargs' by breaking up 'funcs_kwargs'
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0010_fastcore_meta_summary.md
    ## import
    ## fastcore and fastcore.meta
    ### What's inside fastcore.meta
    ## Review individual funcs and classes
    ### What is fastcore.meta all about? 
    ### What can these metaclasses do for me?
    #### FixSigMeta
    #### PrePostInitMeta
    #### AutoInit
    #### NewChkMeta
    #### BypassNewMeta
    ### What can those decorators do for me?
    #### use_kwargs_dict
    #### use_kwargs
    #### delegates
    #### funcs_kwargs
    ### The remaining functions
    ## What is fastcore.meta all about
    
    /Users/Natsume/Documents/fastdebug/mds/questions/00_question_anno_dict.md
    ## `anno_dict` docs
    ## Dive in
    ## `anno_dict` seems not add anything new to `__annotations__`
    ## use fastdebug to double check
    ## Does fastcore want anno_dict to include params with no annos?
    ## Jeremy's response


**When I just want to have a quick look of the commented source code**

Run `fastsrcs()` first to have the list of all commented srcodes files

Run `fastview(srcname)` on the cell above `fastsrcs()` to view the actual commented srcs with an example


```
fastview("test_sig")
```

    
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    
    def test_sig(f, b):=======================================================================(0)       
        "Test the signature of an object"=====================================================(1) # [36;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [36;1mtest_sig will get f's signature as a string[0m; [92;1mb is a signature in string provided by the user[0m; [34;1min fact, test_sig is to compare two strings[0m; 
        test_eq(str(inspect.signature(f)), b)=================================================(2) # [92;1mtest_sig is to test two strings with test_eq[0m; [92;1mhow to turn a signature into a string;[0m; 
                                                                                                                                                            (3)



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


#|hide
## Export


```
#| hide
from nbdev import nbdev_export
nbdev_export()
```

#|hide
## Send to Obsidian


```
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/utils.ipynb
!mv /Users/Natsume/Documents/fastdebug/utils.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/
```


<style>.container { width:100% !important; }</style>


    [jupytext] Reading /Users/Natsume/Documents/fastdebug/utils.ipynb in format ipynb
    Traceback (most recent call last):
      File "/Users/Natsume/mambaforge/bin/jupytext", line 10, in <module>
        sys.exit(jupytext())
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/cli.py", line 488, in jupytext
        exit_code += jupytext_single_file(nb_file, args, log)
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/cli.py", line 552, in jupytext_single_file
        notebook = read(nb_file, fmt=fmt, config=config)
      File "/Users/Natsume/mambaforge/lib/python3.9/site-packages/jupytext/jupytext.py", line 411, in read
        with open(fp, encoding="utf-8") as stream:
    FileNotFoundError: [Errno 2] No such file or directory: '/Users/Natsume/Documents/fastdebug/utils.ipynb'
    mv: rename /Users/Natsume/Documents/fastdebug/utils.md to /Users/Natsume/Documents/divefastai/Debuggable/jupytext/utils.md: No such file or directory



```
#| hide
!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```

    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/index.ipynb to markdown
    [NbConvertApp] Writing 58088 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/index.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0001_fastcore_meta_delegates.ipynb to markdown
    [NbConvertApp] Writing 75603 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0001_fastcore_meta_delegates.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0005_fastcore.meta.test_sig.ipynb to markdown
    [NbConvertApp] Writing 10316 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0005_fastcore.meta.test_sig.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0000_tour.ipynb to markdown
    [NbConvertApp] Writing 12617 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0000_tour.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0007_fastcore.meta.BypassNewMeta.ipynb to markdown
    [NbConvertApp] Writing 30610 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0007_fastcore.meta.BypassNewMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0008_use_kwargs_dict.ipynb to markdown
    [NbConvertApp] Writing 56914 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0008_use_kwargs_dict.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0012_fastcore_foundation_L.ipynb to markdown
    [NbConvertApp] Writing 8461 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0012_fastcore_foundation_L.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0002_signature_from_callable.ipynb to markdown
    [NbConvertApp] Writing 44237 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0002_signature_from_callable.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0004_fastcore.meta._rm_self.ipynb to markdown
    [NbConvertApp] Writing 16058 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0004_fastcore.meta._rm_self.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0009_funcs_kwargs.ipynb to markdown
    [NbConvertApp] Writing 68929 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0009_funcs_kwargs.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0010_fastcore_meta_summary.ipynb to markdown
    [NbConvertApp] Writing 22423 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0010_fastcore_meta_summary.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0006_fastcore.meta.NewChkMeta.ipynb to markdown
    [NbConvertApp] Writing 30938 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0006_fastcore.meta.NewChkMeta.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.ipynb to markdown
    [NbConvertApp] Writing 73586 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/demos/0011_Fastdb.ipynb to markdown
    [NbConvertApp] Writing 11924 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0011_Fastdb.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/questions/00_question_anno_dict.ipynb to markdown
    [NbConvertApp] Writing 11779 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_question_anno_dict.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/lib/utils.ipynb to markdown
    [NbConvertApp] Writing 124508 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/utils.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/lib/00_core.ipynb to markdown
    [NbConvertApp] Writing 411951 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/00_core.md
    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/2022part1/0001_is_it_a_bird.ipynb to markdown
    [NbConvertApp] Support files will be in 0001_is_it_a_bird_files/
    [NbConvertApp] Making directory /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0001_is_it_a_bird_files
    [NbConvertApp] Making directory /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0001_is_it_a_bird_files
    [NbConvertApp] Writing 56115 bytes to /Users/Natsume/Documents/divefastai/Debuggable/nbconvert/0001_is_it_a_bird.md



```

```
