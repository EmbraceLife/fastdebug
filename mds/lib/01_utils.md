---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Utils

> little functions to tell you the basics of a module


## multi_output
setup for exporting to a module

```python
#| default_exp utils
```

```python
#| export
import os
```

```python
#| exporti
def multi_output():
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
```

```python
#| exporti
multioutput = multi_output()
```

```python
a = [1,2,3]
b = [4,5,6]
a
b
```

## nb_url, nb_name, nb_path
how to get current notebook's name, path and url

```python

```

```python
#| export
def nb_url():
    "run this func to get nb_url of this current notebook"
    import ipyparams
    return eval("ipyparams.raw_url")
```

```python
nb_url()
```

```python
#| export
def nb_path():
    "run this func to get nb_path of this current notebook"
    import ipyparams
    return eval("os.path.join(os.getcwd(), ipyparams.notebook_name)")
```

```python
nb_path()
```

```python
#| export
def nb_name():
    "run this func to get nb_path of this current notebook"
    import ipyparams
    return eval("ipyparams.notebook_name")
```

```python
nb_name()
```

## ipy2md
how to convert ipynb to md automatically; how to run commands in python

```python
#| export
def ipy2md(db=True):
    "convert the current notebook to md"
    import ipyparams
    import os
    path = nb_path()
    name = nb_name()
    url = nb_url()
    obs_path = "/Users/Natsume/Documents/divefastai/Debuggable/jupytext"
    obs_last_folder = nb_path().split(nb_name())[0].split('/')[-2]
    obs_output_path = "/Users/Natsume/Documents/divefastai/Debuggable/nbconvert"    
    mds_path = path.replace("nbs", "mds").split(name)[0]
    mds_output = "/Users/Natsume/Documents/fastdebug/mds_output"
    # https://stackabuse.com/executing-shell-commands-with-python/
    os.system(f"jupytext --to md {path}")
    os.system(f"cp {path.split('.ipynb')[0]+'.md'} {obs_path + '/' + obs_last_folder}")
    if db: print(f'cp to : {obs_path + "/" + obs_last_folder}')
    os.system(f"mv {path.split('.ipynb')[0]+'.md'} {mds_path}")
    if db: print(f'move to : {mds_path}')
    os.system(f"jupyter nbconvert --to markdown {path}")
    os.system(f"cp {path.split('.ipynb')[0]+'.md'} {mds_output}")
    os.system(f"mv {path.split('.ipynb')[0]+'.md'} {obs_output_path}")
    if db: print(f'copy to : {mds_output}')
    if db: print(f'move to : {obs_output_path}')        
```

```python
# ipy2md()
```

```python
# #| export
# def ipy2md(db=False):
#     "convert the current notebook to md"
#     path, name, url = get_notebook_path()
#     obs_path = "/Users/Natsume/Documents/divefastai/Debuggable/jupytext"
#     mds_path = path.replace("nbs", "mds").split(name)[0]
#     mds_output = "/Users/Natsume/Documents/fastdebug/mds_output"
#     if db: 
#         print(f'path: {path}')
#         print(f'mds_path: {mds_path}')
#     eval(f'!jupytext --to md {path}')
#     eval(f'!cp {path.split(".ipynb")[0]+".md"} {obs_path}')
#     eval(f'!mv {path.split(".ipynb")[0]+".md"} {mds_path}')
#     eval(f'!jupyter nbconvert --to markdown {path}')
#     eval(f'!mv {path.split(".ipynb")[0]+".md"} {mds_output}')
```

```python
# ipy2md()
```

```python

```

## Autoreload plus matplotlib inline for every notebook

<!-- #region -->
As mentioned above, you need the autoreload extension. If you want it to automatically start every time you launch ipython, you need to add it to the ipython_config.py startup file:

It may be necessary to generate one first:
```python
ipython profile create
```
Then include these lines in ~/.ipython/profile_default/ipython_config.py:

```python
c.InteractiveShellApp.exec_lines = []
c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
c.InteractiveShellApp.exec_lines.append('%autoreload 2')
c.InteractiveShellApp.exec_lines.append('%matplotlib inline')
```

As well as an optional warning in case you need to take advantage of compiled Python code in .pyc files:
```python
c.InteractiveShellApp.exec_lines.append('print("Warning: disable autoreload in ipython_config.py to improve performance.")')
```
<!-- #endregion -->

### If individual notebook, I can just run the function below to setup autoreload

```python
#| exporti
def autoreload():
    from IPython.core.interactiveshell import InteractiveShell
    get_ipython().run_line_magic(magic_name="load_ext", line = "autoreload")
    get_ipython().run_line_magic(magic_name="autoreload", line = "2")
    get_ipython().run_line_magic(magic_name="matplotlib", line = "inline")
```

```python

```

## Expand cells

```python
#| exporti
def expandcell():
    "expand cells of the current notebook to its full width"
    from IPython.display import display, HTML 
    display(HTML("<style>.container { width:100% !important; }</style>"))
```

```python
#| exporti
expand = expandcell()
```

## Import fastcore env

```python
#| export
from fastcore.test import * # so that it automated
```

```python
#| export
test_eq = test_eq
test_is = test_is
```

```python
#| export 
from fastcore.imports import FunctionType, MethodType
```

```python
#| export
FunctionType = FunctionType
MethodType = MethodType
```

## to inspect a class

```python

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

```python

```

```python
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

```python

```

## is it a metaclass?

```python
#| export
def ismetaclass(mc): 
    "check whether a class is a metaclass or not"
    if inspect.isclass(mc):
        return type in mc.__mro__ 
    else: return False
```

```python
from fastcore.meta import *
import inspect
```

```python
ismetaclass(FixSigMeta)
```

```python
ismetaclass(PrePostInitMeta)
```

## is it a decorator

```python

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

```python
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
    else: return False

```

```python
test_eq(isdecorator(delegates), True)
```

```python
test_eq(isdecorator(test_sig), False)
```

## whatinside a module of a library

```python
#| export
# from inspect import getmembers, isfunction, isclass, isbuiltin, getsource
import os.path, pkgutil
from pprint import pprint
import inspect

```

```python

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

```python

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

```python

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

```python
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
        print(inspect.getdoc(mo))
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
                                 str(inspect.getdoc(eval(i, module_env))))  
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

```python

```

```python
str(None)
```

```python

```

## whichversion of a library

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

## fastview
display the commented source code

```python
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

```python
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

```python
#| export
import os
```

```python
# fastsrcs()
fastview("PrePostInitMeta")
```

```python

```

## fastscrs

```python
#| export
def fastsrcs():
    "to list all commented src files"
    folder ='/Users/Natsume/Documents/fastdebug/learnings/'
    for f in os.listdir(folder):
        if f.endswith(".py"):
            # Prints only text file present in My Folder
            print(f)
```

```python
fastsrcs()
```

## getrootport

```python

```

```python
#| exporti
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

```python
from notebook import notebookapp
for note in notebookapp.list_running_servers():
    print(note)
```

```python
getrootport()
```

## jn_link

```python
#| exporti
def jn_link(name, file_path, where="locally"):
    "Get a link to the notebook at `path` on Jupyter Notebook"
    from IPython.display import Markdown
    display(Markdown(f'[Open `{name}` in Jupyter Notebook {where}]({file_path})'))                
```

```python
jn_link("utils", "http://localhost:8888/notebooks/nbs/lib/utils.ipynb")
```

## get_all_nbs

```python
def get_all_nbs(folder='/Users/Natsume/Documents/divefastai/Debuggable/jupytext/'):
    "return all nbs of subfolders of the `folder` into a list"
    all_nbs = []
    for i in os.listdir(folder):
        if "." not in i:
            all_nbs = all_nbs + [folder + i + "/" + j for j in os.listdir(folder + i) if j.endswith('.md')]
    return (all_nbs, folder)
```

### get all nbs path for both md and ipynb

```python

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

```python

```

```python

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

```python
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

```python

```

```python
nbs_md, fdmd, nbs_ipy, fdipy, md_out, md_out_fd = get_all_nbs()
for i in [nbs_md, fdmd, nbs_ipy, fdipy, md_out, md_out_fd]:
    pprint(i)
    print()
```

## openNB

```python

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

```python

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

```python

```

```python
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

```python

```

```python
openNB("FixSigMeta", db=True)
```

## openNBKaggle

```python
"is_it_me".split("_")
```

```python
#| export
kagglenbs = [
    "https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work",
    "https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data",
    "https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework",
    "https://www.kaggle.com/code/jhoward/using-nbdev-export-in-a-kaggle-notebook",
    "https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning",
    "https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive",
    "https://www.kaggle.com/code/jhoward/multi-target-road-to-the-top-part-4",
    "https://www.kaggle.com/code/jhoward/scaling-up-road-to-the-top-part-3",
    "https://www.kaggle.com/code/jhoward/small-models-road-to-the-top-part-2",
    "https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1",
    "https://www.kaggle.com/code/jhoward/which-image-models-are-best",
    "https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch",
    "https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster",
    "https://www.kaggle.com/code/jhoward/how-random-forests-really-work",
    "https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners",
    "https://www.kaggle.com/code/jhoward/resize-images",
    "https://www.kaggle.com/code/jhoward/saving-a-basic-fastai-model",
    "https://www.kaggle.com/code/jhoward/practical-deep-learning-for-coders-chapter-1",
    "https://www.kaggle.com/code/jhoward/from-prototyping-to-submission-fastai",
    "https://www.kaggle.com/code/jhoward/cleaning-the-data-for-rapid-prototyping-fastai",
    "https://www.kaggle.com/code/jhoward/don-t-see-like-a-radiologist-fastai",
    "https://www.kaggle.com/code/jhoward/some-dicom-gotchas-to-be-aware-of-fastai",
    "https://www.kaggle.com/code/jhoward/creating-a-metadata-dataframe-fastai",
    "https://www.kaggle.com/code/jhoward/fastai-v2-pipeline-tutorial",
    "https://www.kaggle.com/code/jhoward/minimal-lstm-nb-svm-baseline-ensemble",
    "https://www.kaggle.com/code/jhoward/nb-svm-strong-linear-baseline",
    "https://www.kaggle.com/code/jhoward/improved-lstm-baseline-glove-dropout"
]
```

```python
len(kagglenbs)
```

```python
#| export
def openNBKaggle(filename_full, db=False):
    if 'fastai' in filename_full:
        # split by 'fastai' and take the later
        filename = filename_full
        filename = filename.split("fastai_")[1]
        # split by '.ipynb'
        filename = filename.split(".md")[0]
        # split by '_'
        nameparts = filename.split("_")
        if db: print(f'filename: {filename}')
        for f in kagglenbs: 
            truelst = [p.lower() in f.lower() for p in nameparts]
            pct = sum(truelst)/len(truelst)
            if pct >= 0.9:
                if db: print(f'pct is {pct}')
                jn_link(filename_full.split(".md")[0], f, where="on Kaggle") 
```

```python
# get_all_nbs()
openNBKaggle("0001_fastai_is_it_a_bird.md", db=True)
```

## highlight


<mark style="background-color: #FFFF00">text with highlighted background</mark>  

```python

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

```python

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

```python

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

```python

```

```python
#| exporti
def highlight(question:str, line:str, db=False):
    "highlight a string with yellow background"
    questlst = question.split(' ')
    questlst_hl = [' <mark style="background-color: #FFFF00">' + q.lower() + '</mark> ' for q in questlst]
    for q, q_hl in zip(questlst, questlst_hl):
        if " " + q.lower() in line.lower(): # don't do anything to [q] or <>q<>. Using regex can be more accurate here
            line = line.lower().replace(" "+ q.lower(), q_hl)
            
    if db: print(f'line: {line}')
    return line
```

```python
str(0.8).split(" ")
print(highlight("0.8 a", "this is a 0.8 of face"))

```

```python

```

## display_md

```python
#| exporti
def display_md(text):
    "Get a link to the notebook at `path` on Jupyter Notebook"
    from IPython.display import Markdown
    display(Markdown(text))                
```

```python
display_md("#### heading level 4")
```

```python

```

## display_block

```python

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

```python

```

```python

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

```python

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

```python

```

```python
"### this is head2\n### and this is head3\n#### this is head4\n## this is head2".find("\n## ")

```

```python
#| exporti
def display_block(line, file, output=False, keywords=""):
    "`line` is a section title, find all subsequent lines which belongs to the same section and display them together"
    from IPython.display import Markdown
    entire = ""
    if file.endswith(".md"): #or file.endswith(".ipynb"):
        with open(file, 'r') as file:
            entire = file.read()
            
    belowline = entire.split(line)[1]
    head_no = line.count("#")
    full_section = "" + f"This section contains only the current heading {head_no} and its subheadings"
    lst_belowline = belowline.split("\n")
    for idx, l in zip(range(len(lst_belowline)), lst_belowline):
        if l.strip().startswith("#"*(head_no-1)+" ") and not bool(lst_belowline[idx-1].strip()) \
        and not bool(lst_belowline[idx+1].strip()):
            full_section = full_section + f"start of heading {head_no-1}" + "\n" + l
            break
        elif l.strip().startswith("#"*head_no + " "):
            full_section = full_section + f"start of another heading {head_no}" + "\n" + l
            break
        else:  full_section = full_section + l + "\n"
    
    title_hl = highlight(keywords, line)
    display(Markdown(title_hl))
    if not output: display(Markdown(full_section))
    else: print(full_section)

```

```python
display_block("whichversion of a library", "/Users/Natsume/Documents/divefastai/Debuggable/jupytext/lib/utils.md", \
              keywords="whichversion library")
```

```python

```

```python

```

## fastnbs

```python

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

```python

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
                    if nb: 
                        openNB(file_name, db=db)
                        openNBKaggle(file_name, db=db)
```

```python
mds_no_output, folder, ipynbs, ipyfolder, mds_output, output_fd = get_all_nbs()
[file_path for file_path in mds_no_output if "_fastai_" in file_path and "_fastai_pt2_" not in file_path]
```

```python

def fastnbs(question:str, # query in string
            filter_folder="all", # options: all, fastai, part2
            output=False, # True for nice print of cell output
            accu:float=0.8, 
            nb=True, 
            db=False):
    "check with fastlistnbs() to find interesting things to search \
fastnbs() can use keywords to search learning points (a section title and a section itself) from my documented fastai notebooks"
    questlst = question.split(' ')
    mds_no_output, folder, ipynbs, ipyfolder, mds_output, output_fd = get_all_nbs()
    if not output: mds = mds_no_output
    else: mds = mds_output
        
    for file_path in mds:
        if filter_folder == "fastai" and "_fastai_" in file_path and not "_fastai_pt2_" in file_path:
            file_fullname = file_path
        elif filter_folder == "part2" and "_fastai_pt2_" in file_path:
            file_fullname = file_path
        elif filter_folder == "all": 
            file_fullname = file_path
        else: continue

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
#                         print()
                    display_block(l, file_fullname, output=output, keywords=question)
                    if nb: 
                        openNB(file_name, db=db)
                        openNBKaggle(file_name, db=db)
```

### optimize the search a little to speed up potentially

```python
#| export
def fastnbs(question:str, # query in string
            filter_folder="all", # options: all, fastai, part2
            output=False, # True for nice print of cell output
            accu:float=0.8, 
            nb=True, 
            db=False):
    "check with fastlistnbs() to skim through all the learning points as section titles; \
then use fastnotes() to find interesting lines which can be notes or codes, and finally \
use fastnbs() display the entire learning points section including notes and codes."
    questlst = question.split(' ')
    mds_no_output, folder, ipynbs, ipyfolder, mds_output, output_fd = get_all_nbs()
    if not output: mds = mds_no_output
    else: mds = mds_output
        
    for file_path in mds:
        if filter_folder == "fastai" and "_fastai_" in file_path and not "_fastai_pt2_" in file_path:
            file_fullname = file_path
        elif filter_folder == "part2" and "_fastai_pt2_" in file_path:
            file_fullname = file_path
        elif filter_folder == "all": 
            file_fullname = file_path
        else: continue

        file_name = file_fullname.split('/')[-1]
        with open(file_fullname, 'r') as file:
            for count, l in enumerate(file):
                if l.startswith("## ") or l.startswith("### ") or l.startswith("#### "):
                    truelst = [q.lower() in l.lower() for q in questlst]
                    pct = sum(truelst)/len(truelst)
                    if pct >= accu:
                        if db: 
                            head1 = f"keyword match is {pct}, Found a section: in {file_name}"
                            head1 = highlight(str(pct), head1)
                            head1 = highlight(file_name, head1)
                            display_md(head1)
                            highlighted_line = highlight(question, l, db=db)                        
    #                         print()
                        display_block(l, file_fullname, output=output, keywords=question)
                        if nb: 
                            openNB(file_name, db=db)
                            openNBKaggle(file_name, db=db)
```

```python
# fastnbs("Snoop them together in one go", output=True)
# fastnbs("how matrix multiplication")
# fastnbs("how matrix multiplication", folder="fastai")
fastnbs("how matrix multiplication", filter_folder="fastai")
```

```python

```

## fastcodes

```python
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

```python
fastcodes("how to remove self")
```

```python

```

## fastnotes

```python

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

```python
[1,3] + [5,6]
```

```python

def fastnotes(question:str, accu:float=0.8, n=2, folder="lec", # folder: 'lec' or 'live' or 'all'
              db=False):
    "using key words to search notes and display the found line and lines surround it"
    questlst = question.split(' ')
    root = '/Users/Natsume/Documents/divefastai/'
#     folder1 = '2022_part1/'
#     folder2 = '2022_livecoding/'
    folder1 = '2019_part2/'
    folder2 = '2019_walkthrus'
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
#                         l = highlight(question, l, db=db)
#                         display_md(l)
                        print()                        
#                         print('{:=<157}'.format(f"Show {n} lines above and after in {f}:"))
#                         head2 = f"Show {n} lines above and after in {f.split(root)[1]}:"
#                         head2 = highlight(f.split(root)[1], head2)
#                         head2 = highlight(str(n), head2)
#                         display_md(head2)                        
                        idx = count
                        with open(f, 'r') as file:
                            for count, l in enumerate(file):
                                if count >= idx - n and count <= idx + n:
                                    if count == idx: display_md(highlight(question, l))
                                    elif bool(l.strip()): display_md(l)
```

### handling python code block

```python

def fastnotes(question:str, accu:float=0.8, n=2, folder="lec", # folder: 'lec' or 'live' or 'all'
              db=False):
    "using key words to search notes and display the found line and lines surround it"
    questlst = question.split(' ')
    root = '/Users/Natsume/Documents/divefastai/'
#     folder1 = '2022_part1/'
#     folder2 = '2022_livecoding/'
    folder1 = '2019_part2/'
    folder2 = '2019_walkthrus'
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
#                         l = highlight(question, l, db=db)
#                         display_md(l)
                        print()                        
#                         print('{:=<157}'.format(f"Show {n} lines above and after in {f}:"))
#                         head2 = f"Show {n} lines above and after in {f.split(root)[1]}:"
#                         head2 = highlight(f.split(root)[1], head2)
#                         head2 = highlight(str(n), head2)
#                         display_md(head2)                        
                        idx = count
                        code = False
                        codeblock = ""
                        with open(f, 'r') as file:
                            for count, l in enumerate(file):
                                if count >= idx - n and count <= idx + n:
                                    if count == idx and bool(l.strip()): 
                                        display_md(highlight(question, l))
                                    elif bool(l.strip()) and "```python" == l.strip() and not code: 
                                        codeblock = codeblock + l
                                        code = True
                                    # make sure one elif won't be skipped/blocked by another elif above
                                    elif bool(l.strip()) and code and "```" != l.strip()and count < idx + n: 
                                        codeblock = codeblock + l
                                    elif bool(l.strip()) and code and "```" == l.strip():
                                        codeblock = codeblock + l                    
                                        code = False
                                        display_md(codeblock)
                                        codeblock = ""
                                    elif bool(l.strip()) and code and count == idx + n:
                                        codeblock = codeblock + l
                                        codeblock = codeblock + "# code block continues below" + "\n"                                        
                                        codeblock = codeblock + "```" + "\n"
                                        code = False
                                        display_md(codeblock)
                                        codeblock = ""
                                    elif bool(l.strip()): 
                                        display_md(l)

```

### handle codeblock even when the searched line is also code

```python

def fastnotes(question:str, 
              search_code:bool=False, # if search code, do check True for nicer printing
              accu:float=0.8, 
              n=2, 
              folder="lec", # folder: 'lec' or 'live' or 'all'
              db=False):
    "using key words to search notes and display the found line and lines surround it"
    questlst = question.split(' ')
    root = '/Users/Natsume/Documents/divefastai/'
#     root = '/Users/Natsume/Documents/fastdebug/mds/'
#     folder1 = '2022_part1/'
#     folder2 = '2022_livecoding/'
    folder1 = '2019_part2/'
    folder2 = '2019_walkthrus'
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
#                         l = highlight(question, l, db=db)
#                         display_md(l)
                        print()                        
#                         print('{:=<157}'.format(f"Show {n} lines above and after in {f}:"))
#                         head2 = f"Show {n} lines above and after in {f.split(root)[1]}:"
#                         head2 = highlight(f.split(root)[1], head2)
#                         head2 = highlight(str(n), head2)
#                         display_md(head2)                        
                        idx = count
                        code = search_code
                        codeblock = ""
                        with open(f, 'r') as file:
                            for count, l in enumerate(file):
                                if count >= idx - n and count <= idx + n:
                                    if count == idx and bool(l.strip()) and not code: 
                                        display_md(highlight(question, l))
                                    elif count == idx and bool(l.strip()) and code: 
                                        codeblock = codeblock + l
                                    elif count == idx-n and code and bool(l.strip()) and "```" != l.strip():
                                        codeblock = codeblock + "```python\n" + l
                                    elif bool(l.strip()) and "```python" == l.strip() and not code: 
                                        codeblock = codeblock + l
                                        code = True
                                    # make sure one elif won't be skipped/blocked by another elif above
                                    elif bool(l.strip()) and code and "```" != l.strip()and count < idx + n: 
                                        codeblock = codeblock + l
                                    elif bool(l.strip()) and code and "```" == l.strip():
                                        codeblock = codeblock + l                    
                                        code = False
                                        display_md(codeblock)
                                        codeblock = ""
                                    elif bool(l.strip()) and code and count == idx + n:
                                        codeblock = codeblock + l
                                        codeblock = codeblock + "# code block continues below" + "\n"                                        
                                        codeblock = codeblock + "```" + "\n"
                                        code = False
                                        display_md(codeblock)
                                        codeblock = ""
                                    elif bool(l.strip()) and not code: 
                                        display_md(l)

```

### to extract md files from all subfolders of fastdebug/mds/ directory

```python
#| export
def fastnotes(question:str, 
              search_code:bool=False, # if search code, do check True for nicer printing
              accu:float=0.8, 
              n=2, 
              folder="all", # folder: 'lec' or 'live' or 'all'
              nb=True, # to provide a link to open notebook
              db=False):
    "using key words to search every line of fastai notes (include codes) \
and display the found line and lines surround it. fastnotes can search both notes and codes\
to do primary search, and then use fastnbs to display the whole interested section instead\
of surrounding lines."
    questlst = question.split(' ')
    root = '/Users/Natsume/Documents/fastdebug/mds/'
    all_md_files = []
    for f in os.listdir(root):  
        if f.endswith(".md"): all_md_files.append(root + f)
        elif not "." in f: 
            for subf in os.listdir(root + f):
                if subf.endswith(".md"): all_md_files.append(root + f + "/" + subf)

    if db: pprint(all_md_files)
    for md in all_md_files:
        if folder == "part2" and "_fastai_pt2" in md: f = md
        elif folder == "part1" and "_fastai_" in md and not "_fastai_pt2" in md: f = md
        elif folder == "all": f = md
        else: 
            print("no folder is selected")
            return
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
                    print()                                              
                    idx = count
                    code = search_code
                    codeblock = ""
                    with open(f, 'r') as file: # must open the file again to read surrounding lines
                        for count, l in enumerate(file):
                            if count >= idx - n and count <= idx + n:
                                if count == idx and bool(l.strip()) and not code: 
                                    display_md(highlight(question, l))
                                elif count == idx and bool(l.strip()) and code: 
                                    codeblock = codeblock + l
                                elif count == idx-n and code and bool(l.strip()) and "```" != l.strip():
                                    codeblock = codeblock + "```python\n" + l
                                elif bool(l.strip()) and "```python" == l.strip() and not code: 
                                    codeblock = codeblock + l
                                    code = True
                                # make sure one elif won't be skipped/blocked by another elif above
                                elif bool(l.strip()) and code and "```" != l.strip()and count < idx + n: 
                                    codeblock = codeblock + l
                                elif bool(l.strip()) and code and "```" == l.strip():
                                    codeblock = codeblock + l                    
                                    code = False
                                    display_md(codeblock)
                                    codeblock = ""
                                elif bool(l.strip()) and code and count == idx + n:
                                    codeblock = codeblock + l
                                    codeblock = codeblock + "# code block continues below" + "\n"                                        
                                    codeblock = codeblock + "```" + "\n"
                                    code = False
                                    display_md(codeblock)
                                    codeblock = ""
                                elif bool(l.strip()) and not code: 
                                    display_md(l)
                    if nb:
                        file_name = f.split('/')[-1]
                        openNB(file_name, db=db)
                        openNBKaggle(file_name, db=db)
```

```python
fastnotes("how understand matrix multiplication", n=17, db=True)
```

```python

```

```python

```

## fastlistnbs

```python

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

```python

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

### add filter 

```python

def fastlistnbs(filter="fastai"):
    "display all my commented notebooks subheadings in a long list. Best to work with fastnbs together."
    nbs, folder, _, _, _, _ = get_all_nbs()
    nb_rt = ""
    for nb in nbs:
        if filter == "fastai" and "_fastai_" in nb: 
            nb_rt = nb
        elif filter == "all": 
            nb_rt = nb
        else: 
            continue
            
        print("\n"+nb_rt)
        with open(nb_rt, 'r') as file:
            for idx, l in enumerate(file):
                if "##" in l:
                    print(l, end="") # no extra new line between each line printed       
```

```python

```

```python
#| export
def fastlistnbs(flt_fd="fastai"): # other options: "part2", "all"
    "display all my commented notebooks subheadings in a long list. Best to work with fastnbs together."
    nbs, folder, _, _, _, _ = get_all_nbs()
    nb_rt = ""
    for nb in nbs:
        if flt_fd == "fastai" and "_fastai_" in nb and not "_fastai_pt2" in nb: 
            nb_rt = nb
        elif flt_fd == "part2" and "_fastai_pt2" in nb:
            nb_rt = nb
        elif flt_fd == "all": 
            nb_rt = nb
        else: 
            continue
            
        print("\n"+nb_rt)
        with open(nb_rt, 'r') as file:
            for idx, l in enumerate(file):
                if "##" in l:
                    print(l, end="") # no extra new line between each line printed       
```

```python

```

```python
fastlistnbs(flt_fd="part2")
```

## fastlistsrcs


Todos: it should be retired by fastlistnbs() after all src notebooks are properly documented.

```python
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

```python
fastcodes("Test the signature", nb=True)
```

```python

```

```python

```

## Best practice of fastdebug.core


1. test some examples 

2. import fastdebug and initialize fdb objects, put examples into fdb.eg

3. run fdb.snoop() to see all lines that get run

4. if error, run fdb.debug() to check for clues

5. run fdb.print() to see bare source

6. run fdb.docsrc() to run expression and document srcline

7. how snoop: from _funcs_kwargs to funcs_kwargs, see the learning point below
 


```python
fastnbs("snoop: from _funcs_kwargs to funcs_kwargs", output=True)
```

## Best practice of fastdebug.utils

```python
import fastdebug.utils as fu
import fastcore.meta as fm
```

**When looking for previous documented learning points**

Run `fastlistnbs()` to check and search for the interesting titles

Then run `fastnbs(...)` on the cell above `fastlistnbs()` to have a better view

Run `fastnbs(query, output=True)` to view the output with input together for notebooks on srcodes

```python
fastnbs("snoop: from _funcs_kwargs to funcs_kwargs", output=True)
```

```python
fastlistnbs()
```

**When I just want to have a quick look of the commented source code**

Run `fastsrcs()` first to have the list of all commented srcodes files

Run `fastview(srcname)` on the cell above `fastsrcs()` to view the actual commented srcs with an example

Run `fastcodes(query)` to search src comments for learning points

```python
fastcodes("how to turn a sig into string", accu=1)
```

```python
fastview("test_sig")
```

```python
fastsrcs()
```

#|hide
## Export

```python
#| hide
from nbdev import nbdev_export
nbdev_export()
```

```python

```