# Utils

> little functions to tell you the basics of a module

Todos: do I need whichversion? and fu?


```
#| default_exp utils
```


```
#| export
from __future__ import annotations
annotations = annotations
import inspect, torch
```

## Data

### L


```
#| export
from fastcore.foundation import L
```


```
#| export
L = L
```

### Path


```
#| export
from pathlib import *
```


```
#| export
Path = Path
```

### check_subfolders_img


```
#| export
# from fastai.vision.all import *
```


```
# #| export
# # @snoop
# def check_subfolders_img(path, db=False):
#     from pathlib import Path
#     for entry in path.iterdir():
#         if entry.is_file():
#             print(f'{str(entry.absolute())}')
#     for entry in path.iterdir():
#         if entry.is_dir() and not entry.name.startswith(".") and len(entry.ls(file_exts=image_extensions)) > 5:
#             print(f'{str(entry.parent.absolute())}: {len(entry.ls(file_exts=image_extensions))}  {entry.name}')
# #             print(entry.name, f': {len(entry.ls(file_exts=[".jpg", ".png", ".jpeg", ".JPG", ".jpg!d"]))}') # how to include both png and jpg
#             if db:
#                 for e in entry.ls(): # check any image file which has a different suffix from those above
#                     if e.is_file() and not e.name.startswith(".") and e.suffix not in image_extensions and e.suffix not in [".ipynb", ".py"]:
#     #                 if e.suffix not in [".jpg", ".png", ".jpeg", ".JPG", ".jpg!d"]:
#                         pp(e.suffix, e)
#                         try:
#                             pp(Image.open(e).width)
#                         except:
#                             print(f"{e} can't be opened")
#     #                     pp(Image.open(e).width if e.suffix in image_extensions)
#         elif entry.is_dir() and not entry.name.startswith("."): 
# #             with snoop:
#             count_files_in_subfolders(entry)
```

### randomdisplay(path, db=False)


```
#| export
from pathlib import *
```


```
# #| export
# # @snoop
# def randomdisplay(path, db=False):
# # https://www.geeksforgeeks.org/python-random-module/
#     import random
#     if type(path) == PosixPath:
#         rand = random.randint(0,len(path.ls())-1) # choose a random int between 0 and len(T-rex)-1
#         file = path.ls()[rand]
#     elif type(path) == L:
#         rand = random.randint(0,len(path)-1) # choose a random int between 0 and len(T-rex)-1
#         file = path[rand]
#     im = PILImage.create(file)
#     if db: pp(im.width, im.height, file)
#     pp(file)
#     return im
```


```

```


```

```

## Plotting
basic plotting [lines](https://www.geeksforgeeks.org/graph-plotting-python-set-3/), [animation](https://www.geeksforgeeks.org/graph-plotting-python-set-3/)

### single_func plot


```
#| export
def plot_func(x, y, label_x=None, label_y=None, title=None):
    import matplotlib.pyplot as plt
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel(label_x)
    # naming the y axis
    plt.ylabel(label_y)

    # giving a title to my graph
    plt.title(title)

    # function to show the plot
    plt.show()

```


```
# plot_func(x, y, 'x - axis', 'y - axis', 'My first graph!')
```

### multiple-line plot


```
#| export
# @snoop
def plot_funcs(*lines, # eg., (x1,y1, "line1"),(x2,y2, "line2"), (y1, y2, "line3")
               label_x=None, 
               label_y=None, 
               title=None, 
               ax_center=False):
    import matplotlib.pyplot as plt

    if ax_center:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center') 
        ax.spines['bottom'].set_position('center') 
    #     ax.spines['right'].set_position('center') # has no y ticks, so no use for us
    #     ax.spines['top'].set_position('center') # has no x ticks, so no use for us
        ax.spines['right'].set_color('none') # make the right side line (of the box) disappear
        ax.spines['top'].set_color('none') # make the left side line (of the box) disappear
    
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    
    plt.xlabel(label_x, loc='right')
    plt.ylabel(label_y, loc='bottom')
    
    xys_lst = [(l[0], l[1], l[2]) for l in lines]
    for idx, xys in enumerate(xys_lst):
        plt.plot(xys[0], xys[1], label=xys[2])

    plt.title(title)
    plt.legend()
    plt.show()
#     # add color and other features
#     plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,
#          marker='o', markerfacecolor='blue', markersize=12)
```


```
x1 = [1,2,3]
y1 = [2,4,1]
x2 = [1,2,3]
y2 = [4,1,3]
plot_funcs((x1,y1, "line1"),(x2,y2, "line2"), (y1, y2, "line3"), label_x='x - axis', label_y='y - axis', title='My first graph!')
plot_funcs((x1,y1, "line1"),(x2,y2, "line2"), (y1, y2, "line3"), label_x='x - axis', label_y='y - axis', title='My first graph!', ax_center=True)
```


    
![png](01_utils_files/01_utils_25_0.png)
    



    
![png](01_utils_files/01_utils_25_1.png)
    



```
len((1,2))
```




    2




```
#| export
# @snoop
def plot_fns(*funcs, # eg., (func1, ), (func2, ) or (func2, 'x**2'), (func1, 'log(x)')
               label_x=None, 
               label_y=None, 
               title=None, 
               ax_center=False):
    import matplotlib.pyplot as plt

    if ax_center:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center') 
        ax.spines['bottom'].set_position('center') 
        ax.spines['right'].set_color('none') # make the right side line (of the box) disappear
        ax.spines['top'].set_color('none') # make the left side line (of the box) disappear
    
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    plt.xlabel(label_x, loc='right')
    plt.ylabel(label_y, loc='bottom')
    
    fnlb_lst = [(f[0], f[1]) if len(f) > 1 else (f[0],) for f in funcs ]
    
    import torch, numpy as np
    for fnlb in fnlb_lst:
        if "np" in inspect.getsource(fnlb[0] ):
            x = np.linspace(-np.pi,np.pi,100)
        elif "torch" in inspect.getsource(fnlb[0] ):
            x = torch.linspace(-torch.pi, torch.pi, 100)

    from fastai.torch_core import ifnone
    for idx, fnlb in enumerate(fnlb_lst):
        plt.plot(x, fnlb[0](x), label=fnlb[1] if len(fnlb)>1 else "line {}".format(idx))

    plt.title(title)
    plt.legend() # plt.legend(loc='upper left')
    plt.show()
#     # add color and other features
#     plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,
#          marker='o', markerfacecolor='blue', markersize=12)
```


```
# inspect.getsource(func2)
```


```
func1 = lambda x: x**2
func2 = lambda x: torch.log(x) # torch.log(x)
plot_fns((func1, "x**2"), (func2, "log(x)"), label_x='x - axis', label_y='y - axis', title='My first graph!', ax_center=True)
plot_fns((func1, ), (func2, ), label_x='x - axis', label_y='y - axis', title='My first graph!', ax_center=True)
```


    
![png](01_utils_files/01_utils_29_0.png)
    



    
![png](01_utils_files/01_utils_29_1.png)
    


## doc from fastai.torch_core


```
#| export
from fastai.torch_core import doc
```


```
#| export
doc = doc
```


```
#| export
from nbdev.showdoc import show_doc
```


```
#| export
show_doc = show_doc
```

## snoop: pp, @snoop, doc_sig, pp.deep,%%snoop, watches


```
#| export
from snoop import snoop, pp
```


```
#| export
snoop = snoop
pp = pp
```


```
#| export 
def snoopon(): snoop.install(enabled=True)
```


```
#| export
def snoopoff(): snoop.install(enabled=False)
```


```
from fastcore.foundation import L
from fastai.torch_core import tensor
```


```
tensor(1,2,3).__len__()
L([1,2,3]).__len__()
```




    3




```
#| export
def chk(obj):
    "return obj's type, length and type if available"
    tp = type(obj)
    length = obj.__len__() if hasattr(obj, '__len__') else "no length"
    shape = obj.shape if hasattr(obj, 'shape') else "no shape"
    return tp, length, shape
```


```
#| export
def doc_sig(func):
    import inspect
    sig = inspect.signature(func) if callable(func) else "no signature"
    doc = inspect.getdoc(func) if inspect.getdoc(func) != None else "no doc"
    return  getattr(func, '__mro__', "no mro"), doc, sig
```


```
#| export
def src(func):
    try: 
        print(inspect.getsource(func))
    except: 
        print(f"can't get srcode from inspect.getsource")
```


```
src(doc_sig)
```

    def doc_sig(func):
        import inspect
        sig = inspect.signature(func) if callable(func) else "no signature"
        doc = inspect.getdoc(func) if inspect.getdoc(func) != None else "no doc"
        return  getattr(func, '__mro__', "no mro"), doc, sig
    



```
#| exporti
def doc_sig_complex(func):
    import inspect        
    if not inspect.isfunction(func) or not inspect.ismethod(func):
        func = getattr(func, '__class__', None)
        if func == None: 
            info = 'not a func, method, nor a class'
        else: 
            info = "it's a class"
    doc = inspect.getdoc(func) if hasattr(func, '__doc__') else "no doc"
    sig = inspect.signature(func) if callable(func) else "no signature"
    mro = getattr(func, '__mro__', "no mro")
    return info, doc, sig, mro
```


```
#| exporti
def type_watch(source, value):
    if value != None:
        return 'type({})'.format(source), type(value)
```


```
#| exporti
def sig_watch(source, value):
    if inspect.isfunction(value):
        return 'sig({})'.format(source), inspect.signature(value)
```


```
#| exporti
def view(data): return (data.mean(), data.std())
```


```
from torch import tensor
import torch
```


```
isinstance(tensor([1,2,3]), torch.Tensor)
```




    True




```
#| exporti
def stats_watch(source, value):
    if (isinstance(value, np.ndarray) or isinstance(value, torch.Tensor)): 
        return '{} stats: '.format(source), view(value)
```


```
[1, 2] + [2, 3]
```




    [1, 2, 2, 3]




```
#| exporti
def snoop_onoff(on=True):
    "activate or deactivate @snoop, pp, but not %%snoop in a cell which is activated by %load_ext snoop"
    import snoop
    from snoop.configuration import len_shape_watch
#     snoop.install(replace_watch_extras=[type_watch, len_shape_watch, sig_watch, stats_watch])
    snoop.install(replace_watch_extras=[]) # this is much simpler to read

```


```
#| exporti 
snoop_no_config = snoop_onoff() # # no import or config for using snoop now
```


```
snoop_config = """
# snoop_onoff()
# snoop.install(watch_extras=[type_watch, stats_watch])
from snoop.configuration import len_shape_watch
snoop.install(replace_watch_extras=[type_watch, len_shape_watch, sig_watch, stats_watch])
"""

```

## multi_output
setup for exporting to a module


```
#| export
import os
```


```
#| exporti
def multi_output():
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
```


```
#| exporti
multioutput = multi_output()
```


```
a = [1,2,3]
b = [4,5,6]
a
b
```




    [1, 2, 3]






    [4, 5, 6]



## nb_url, nb_name, nb_path
how to get current notebook's name, path and url


```

```


```
#| export
def nb_url():
    "run this func to get nb_url of this current notebook"
    import ipyparams
    return eval("ipyparams.raw_url")
```


```
nb_url()
```


    <IPython.core.display.Javascript object>





    ''




```
#| export
def nb_path():
    "run this func to get nb_path of this current notebook"
    import ipyparams
    return eval("os.path.join(os.getcwd(), ipyparams.notebook_name)")
```


```
nb_path()
```




    '/Users/Natsume/Documents/fastdebug/nbs/lib/'




```
#| export
def nb_name():
    "run this func to get nb_path of this current notebook"
    import ipyparams
    return eval("ipyparams.notebook_name")
```


```
nb_name()
```




    ''



## ipy2md
how to convert ipynb to md automatically; how to run commands in python


```
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


```
# ipy2md()
```


```
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


```
# ipy2md()
```


```

```

## Autoreload plus matplotlib inline for every notebook

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
c.InteractiveShellApp.exec_lines.append('%load_ext snoop')
```

As well as an optional warning in case you need to take advantage of compiled Python code in .pyc files:
```python
c.InteractiveShellApp.exec_lines.append('print("Warning: disable autoreload in ipython_config.py to improve performance.")')
```

### If individual notebook, I can just run the function below to setup autoreload


```
#| export
def automagics():
    from IPython.core.interactiveshell import InteractiveShell
    get_ipython().run_line_magic(magic_name="load_ext", line = "autoreload")
    get_ipython().run_line_magic(magic_name="autoreload", line = "2")
    get_ipython().run_line_magic(magic_name="matplotlib", line = "inline")
    get_ipython().run_line_magic(magic_name="load_ext", line = "snoop")
```


```

```

## Expand cells


```
#| exporti
def expandcell():
    "expand cells of the current notebook to its full width"
    from IPython.display import display, HTML 
    display(HTML("<style>.container { width:100% !important; }</style>"))
```


```
#| exporti
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
    else: return False
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
    else: return False

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
# from importlib.metadata import version, metadata, distribution
# from platform import python_version 
```


```
# #| export
# def whichversion(libname:str, # library name not string
#                 req:bool=False, # print lib requirements 
#                 file:bool=False): # print all lib files
#     "Give you library version and other basic info."
#     if libname == "python":
#         print(f"python: {python_version()}")
#     else: 
#         print(f"{metadata(libname)['Name']}: {version(libname)} \n{metadata(libname)['Summary']}\
#     \n{metadata(libname)['Author']} \n{metadata(libname)['Home-page']} \
#     \npython_version: {metadata(libname)['Requires-Python']} \
#     \n{distribution(libname).locate_file(libname)}")

#     if req: 
#         print(f"\n{libname} requires: ")
#         pprint(distribution(libname).requires)
#     if file: 
#         print(f"\n{libname} has: ")
#         pprint(distribution(libname).files)
    
```


```
# whichversion("python")
```


```
# whichversion("fastcore")
```


```
# whichversion("fastai")
```


```
# whichversion("snoop")
```


```
# try:
#     whichversion("inspect")
# except: 
#     print("inspect won't work here")
```


```

# def tstenv(outenv=globals()):
#     print(f'out global env has {len(outenv.keys())} vars')
#     print(f'inner global env has {len(globals().keys())} vars')
#     print(f'inner local env has {len(globals().keys())} vars')
#     lstout = list(outenv.keys())
#     lstin = list(globals().keys())
#     print(lstout[:10])
#     print(lstin[:10])
#     print(f"out env['__name__']: {outenv['__name__']}")
#     print(f"inner env['__name__']: {globals()['__name__']}")
```


```
# tstenv()
```


```
# len(globals().keys())
```

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
        "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"==========(1) # [92;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type)[0m; [93;1mnot from type, nor from object[0m; [35;1mPrePostInitMeta is itself a metaclass, which is used to create class instance not object instance[0m; [91;1mPrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m; 
        def __call__(cls, *args, **kwargs):===================================================(2)       
            res = cls.__new__(cls)============================================================(3) # [36;1mhow to create an object instance with a cls[0m; [35;1mhow to check the type of an object is cls[0m; [36;1mhow to run a function without knowing its params;[0m; 
            if type(res)==cls:================================================================(4)       
                if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)==============(5)       
                res.__init__(*args,**kwargs)==================================================(6) # [34;1mhow to run __init__ without knowing its params[0m; 
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

    anno_dict.py
    test_sig.py
    subplots.py
    show_titled_image.py
    DataBlock.py
    BypassNewMeta.py
    snoop.py
    FixSigMeta.py
    show_images.py
    fastnbs.py
    _fig_bounds.py
    funcs_kwargs.py
    __init__.py
    NewChkMeta.py
    printtitle.py
    show_image.py
    AutoInit.py
    get_image_files.py
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


```
from notebook import notebookapp
for note in notebookapp.list_running_servers():
    print(note)
```

    {'base_url': '/', 'hostname': 'localhost', 'notebook_dir': '/Users/Natsume/Documents/fastdebug', 'password': False, 'pid': 31933, 'port': 8888, 'secure': False, 'sock': '', 'token': '0ad534e0cda8455b4dcfac3905c184ca5a6921ee4fb70ebf', 'url': 'http://localhost:8888/'}



```
getrootport()
```




    ('http://localhost:8888/tree/', '/Users/Natsume/Documents/fastdebug')



## jn_link


```
#| exporti
def jn_link(name, file_path, where="locally"):
    "Get a link to the notebook at `path` on Jupyter Notebook"
    from IPython.display import Markdown
    display(Markdown(f'[Open `{name}` in Jupyter Notebook {where}]({file_path})'))                
```


```
jn_link("utils", "http://localhost:8888/notebooks/nbs/lib/utils.ipynb")
```


[Open `utils` in Jupyter Notebook locally](http://localhost:8888/notebooks/nbs/lib/utils.ipynb)


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
!push-code-new
```

    zsh:1: command not found: push-code-new



```
#| export
from pathlib import *
```


```
# fastnbs("Path")
```


```
list(map(str, list((Path.home()/"Documents/fastdebug/mds").ls())))
```




    ['/Users/Natsume/Documents/fastdebug/mds/.DS_Store',
     '/Users/Natsume/Documents/fastdebug/mds/index.md',
     '/Users/Natsume/Documents/fastdebug/mds/lib',
     '/Users/Natsume/Documents/fastdebug/mds/.ipynb_checkpoints',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks',
     '/Users/Natsume/Documents/fastdebug/mds/demos',
     '/Users/Natsume/Documents/fastdebug/mds/questions']




```
str(Path.home()/"Documents/fastdebug/mds") + "/"
```




    '/Users/Natsume/Documents/fastdebug/mds/'




```
#| export
def get_all_nbs():
    "return paths for all nbs both in md and ipynb format into lists"
#     md_folder = '/Users/Natsume/Documents/divefastai/Debuggable/jupytext/'
    md_folder = str(Path.home()/"Documents/fastdebug/mds") + "/" # '/Users/Natsume/Documents/fastdebug/mds/'
    md_output_folder = str(Path.home()/"Documents/fastdebug/mds_output") + "/" # '/Users/Natsume/Documents/fastdebug/mds_output/'    
    ipy_folder = str(Path.home()/"Documents/fastdebug/nbs") + "/" # '/Users/Natsume/Documents/fastdebug/nbs/'
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

    ['/Users/Natsume/Documents/fastdebug/mds/lib/00_core.md',
     '/Users/Natsume/Documents/fastdebug/mds/lib/01_utils.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0016_collaborative_filtering_deep_dive.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0022_fastai_pt2_2019_why_sqrt5.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0007_fastai_how_random_forests_really_work.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0004_fastai_how_neuralnet_work.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0001_fastai_is_it_a_bird.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0014_iterate_like_grandmaster.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0006_fastai_why_should_use_framework.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0011_fastai_multi_target_road_to_top_part_4.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0010_fastai_scaling_up_road_to_top_part_3.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/regex.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0017_fastai_pt2_2019_matmul.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0021_fastai_pt2_2019_fully_connected.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0009_fastai_small_models_road_to_the_top_part_2.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0005_fastai_linear_neuralnet_scratch.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0015_getting_started_with_nlp_for_absolute_beginner.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0013_best_vision_models_for_fine_tuning.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0012_fastai_using_nbdev_export_in_kaggle_notebook.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0020_fastai_pt2_2019_source_explained.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0003_fastai_which_image_model_best.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0018_fastai_pt2_2019_exports.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0019_fastai_pt2_2019_lecture1_intro.md']
    
    '/Users/Natsume/Documents/fastdebug/mds/'
    
    ['/Users/Natsume/Documents/fastdebug/nbs/Interesting_fastai/0001_The_origin_of_APL '
     '.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/Interesting_fastai/Interesting_things_fastai.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/Math/math_0002_calculus.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/Math/math_0001_highschool.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/Math/sympy1.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/lib/00_core.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/lib/01_utils.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/.ipynb_checkpoints',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0009_fastai_small_models_road_to_the_top_part_2.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0007_fastai_how_random_forests_really_work.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0001_fastai_is_it_a_bird.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0005_fastai_linear_neuralnet_scratch.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0003_fastai_which_image_model_best.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0014_iterate_like_grandmaster.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0006_fastai_why_should_use_framework.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/regex.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0013_best_vision_models_for_fine_tuning.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0015_getting_started_with_nlp_for_absolute_beginner.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0017_fastai_pt2_2019_matmul.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0019_fastai_pt2_2019_lecture1_intro.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0012_fastai_using_nbdev_export_in_kaggle_notebook.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0021_fastai_pt2_2019_fully_connected.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0010_fastai_scaling_up_road_to_top_part_3.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0022_fastai_pt2_2019_why_sqrt5.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0018_fastai_pt2_2019_exports.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0016_collaborative_filtering_deep_dive.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0011_fastai_multi_target_road_to_top_part_4.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0020_fastai_pt2_2019_source_explained.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0004_fastai_how_neuralnet_work.ipynb',
     '/Users/Natsume/Documents/fastdebug/nbs/index.ipynb']
    
    '/Users/Natsume/Documents/fastdebug/nbs/'
    
    ['/Users/Natsume/Documents/fastdebug/mds_output/0016_collaborative_filtering_deep_dive.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0022_fastai_pt2_2019_why_sqrt5.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0007_fastai_how_random_forests_really_work.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0004_fastai_how_neuralnet_work.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0001_fastai_is_it_a_bird.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0014_iterate_like_grandmaster.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0006_fastai_why_should_use_framework.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0011_fastai_multi_target_road_to_top_part_4.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0010_fastai_scaling_up_road_to_top_part_3.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/regex.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0017_fastai_pt2_2019_matmul.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0008_fastai_first_steps_road_to_top_part_1.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0021_fastai_pt2_2019_fully_connected.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0009_fastai_small_models_road_to_the_top_part_2.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0002_fastai_saving_a_basic_fastai_model.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/index.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/00_core.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0005_fastai_linear_neuralnet_scratch.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0015_getting_started_with_nlp_for_absolute_beginner.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0013_best_vision_models_for_fine_tuning.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0012_fastai_using_nbdev_export_in_kaggle_notebook.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0020_fastai_pt2_2019_source_explained.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0003_fastai_which_image_model_best.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0018_fastai_pt2_2019_exports.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/01_utils.md',
     '/Users/Natsume/Documents/fastdebug/mds_output/0019_fastai_pt2_2019_lecture1_intro.md']
    
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
#| export
def openNB(name, heading=None, db=False):
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
                file_name = path_server + f + "#" + heading if bool(heading) else path_server + f
                jn_link(name, file_name)
```


```
bool(None)
bool("head")
```




    False






    True




```
# openNB("FixSigMeta", db=True)
```

## openNBKaggle


```
"is_it_me".split("_")
```




    ['is', 'it', 'me']




```
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


```
len(kagglenbs)
```




    27




```
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


```
# get_all_nbs()
openNBKaggle("0001_fastai_is_it_a_bird.md", db=True)
```

    filename: is_it_a_bird
    pct is 1.0



[Open `0001_fastai_is_it_a_bird` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data)


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

```


```
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


```
str(0.8).split(" ")
print(highlight("0.8 a", "this is a 0.8 of face"))

```




    ['0.8']



    this is <mark style="background-color: #FFFF00">a</mark>  <mark style="background-color: #ffff00">0.8</mark>  of face



```

```

## display_md


```
#| exporti
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

```


```
"### this is head2\n### and this is head3\n#### this is head4\n## this is head2".find("\n## ")

```




    58




```
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
    full_section = "" + f"The current section is heading {head_no}." + "\n\n"
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


```
display_block("whichversion of a library", "/Users/Natsume/Documents/divefastai/Debuggable/jupytext/lib/utils.md", \
              keywords="whichversion library")
```


whichversion of a <mark style="background-color: #FFFF00">library</mark> 



The current section is heading 0.



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
display_block("




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


```
mds_no_output, folder, ipynbs, ipyfolder, mds_output, output_fd = get_all_nbs()
[file_path for file_path in mds_no_output if "_fastai_" in file_path and "_fastai_pt2_" not in file_path]
```




    ['/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0007_fastai_how_random_forests_really_work.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0004_fastai_how_neuralnet_work.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0001_fastai_is_it_a_bird.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0006_fastai_why_should_use_framework.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0011_fastai_multi_target_road_to_top_part_4.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0010_fastai_scaling_up_road_to_top_part_3.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0009_fastai_small_models_road_to_the_top_part_2.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0005_fastai_linear_neuralnet_scratch.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0012_fastai_using_nbdev_export_in_kaggle_notebook.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0003_fastai_which_image_model_best.md']




```

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


```
#| export
# @snoop
def fastnbs(question:str, # query options, "doc: ImageDataLoaders", "src: DataBlock", "ht: git", "jn: help others is the way"
            filter_folder="all", # options: src, all,
            strict=False, # loose search keyword, not as the first query word
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
        elif filter_folder == "src" and "fast" in file_path:
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
                    ctn = l.split("# ```")[1] if "# ```" in l else l.split("# ")[1] if "# " in l else l.split("# `")
                    if strict:
                        if pct >= accu and ctn.startswith(questlst[0]): # make sure the heading start with the exact quest word
                            if db: 
                                head1 = f"keyword match is {pct}, Found a section: in {file_name}"
                                head1 = highlight(str(pct), head1)
                                head1 = highlight(file_name, head1)
                                display_md(head1)
                                highlighted_line = highlight(question, l, db=db)                        
        #                         print()
                            display_block(l, file_fullname, output=output, keywords=question)
                            if nb: # to link a notebook with specific heading
                                if "# ```" in l: openNB(file_name, l.split("```")[1].replace(" ", "-"), db=db)
                                else: openNB(file_name, l.split("# ")[1].replace(" ", "-"), db=db)

                                openNBKaggle(file_name, db=db)
                    else: 
                        if pct >= accu: # make sure the heading start with the exact quest word
                            if db: 
                                head1 = f"keyword match is {pct}, Found a section: in {file_name}"
                                head1 = highlight(str(pct), head1)
                                head1 = highlight(file_name, head1)
                                display_md(head1)
                                highlighted_line = highlight(question, l, db=db)                        
        #                         print()
                            display_block(l, file_fullname, output=output, keywords=question)
                            if nb: # to link a notebook with specific heading
                                if "# ```" in l: openNB(file_name, l.split("```")[1].replace(" ", "-"), db=db)
                                else: openNB(file_name, l.split("# ")[1].replace(" ", "-"), db=db)

                                openNBKaggle(file_name, db=db)
```


```
"# this is me".split("# ")[1].replace(" ", "-")
"# thisisme".split("# ")[1].replace(" ", "-")
```




    'this-is-me'






    'thisisme'




```
# fastnbs("Snoop them together in one go", output=True)
# fastnbs("how matrix multiplication")
# fastnbs("how matrix multiplication", folder="fastai")
# fastnbs("show_image", "src")
# fastnbs("module", "src")
# fastnbs("module", "src", False)
# fastnbs("module")
# fastnbs("apply")
# fastnbs("get all nbs")
```


```
"### ```show_image(b, a, c)```".split("```")[1].replace(" ", "-")
```




    'show_image(b,-a,-c)'



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


keyword match is <mark style="background-color: #ffff00">1.0</mark> , found a line: in <mark style="background-color: #FFFF00">fixsigmeta.py</mark> 


            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [37;1mhow to remove self from a signature[0m; [35;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
    



the entire source code in <mark style="background-color: #FFFF00">fixsigmeta.py</mark> 


    
    class BaseMeta(FixSigMeta): 
        # using __new__ of  FixSigMeta instead of type
        def __call__(cls, *args, **kwargs): pass
    
    class Foo_call_fix(metaclass=BaseMeta): # Base
        def __init__(self, d, e, f): pass
    
    pprint(inspect._signature_from_callable(Foo_call_fix, sigcls=inspect.Signature))    
    
    class FixSigMeta(type):===================================================================(0)       
        "A metaclass that fixes the signature on classes that override `__new__`"=============(1) # [37;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance[0m; [37;1mthen check whether its class instance has its own __init__;if so, remove self from the sig of __init__[0m; [93;1mthen assign this new sig to __signature__ for the class instance;[0m; 
        def __new__(cls, name, bases, dict):==================================================(2) # [91;1mhow does a metaclass create a class instance[0m; [92;1mwhat does super().__new__() do here;[0m; 
            res = super().__new__(cls, name, bases, dict)=====================================(3)       
            if res.__init__ is not object.__init__: res.__signature__ = _rm_self(inspect.signature(res.__init__)) # [37;1mhow to remove self from a signature[0m; [35;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
            return res========================================================================(5)       
                                                                                                                                                            (6)
    



keyword match is <mark style="background-color: #ffff00">1.0</mark> , found a line: in <mark style="background-color: #FFFF00">_rm_self.py</mark> 


        sigd.pop('self')======================================================================(2) # [34;1mhow to remove the self parameter from the dict of sig;[0m; 
    



the entire source code in <mark style="background-color: #FFFF00">_rm_self.py</mark> 


    
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    
    def _rm_self(sig):========================================================================(0) # [92;1mremove parameter self from a signature which has self;[0m; 
        sigd = dict(sig.parameters)===========================================================(1) # [93;1mhow to access parameters from a signature[0m; [35;1mhow is parameters stored in sig[0m; [34;1mhow to turn parameters into a dict;[0m; 
        sigd.pop('self')======================================================================(2) # [34;1mhow to remove the self parameter from the dict of sig;[0m; 
        return sig.replace(parameters=sigd.values())==========================================(3) # [36;1mhow to update a sig using a updated dict of sig's parameters[0m; 
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


```

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


```

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


```
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


```
fastnotes("how understand matrix multiplication", n=17, db=True)
```

    ['/Users/Natsume/Documents/fastdebug/mds/index.md',
     '/Users/Natsume/Documents/fastdebug/mds/lib/00_core.md',
     '/Users/Natsume/Documents/fastdebug/mds/lib/01_utils.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0016_collaborative_filtering_deep_dive.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0022_fastai_pt2_2019_why_sqrt5.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0007_fastai_how_random_forests_really_work.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0004_fastai_how_neuralnet_work.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0001_fastai_is_it_a_bird.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0014_iterate_like_grandmaster.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0006_fastai_why_should_use_framework.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0011_fastai_multi_target_road_to_top_part_4.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0010_fastai_scaling_up_road_to_top_part_3.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/regex.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0017_fastai_pt2_2019_matmul.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0021_fastai_pt2_2019_fully_connected.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0009_fastai_small_models_road_to_the_top_part_2.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0005_fastai_linear_neuralnet_scratch.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0015_getting_started_with_nlp_for_absolute_beginner.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0013_best_vision_models_for_fine_tuning.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0012_fastai_using_nbdev_export_in_kaggle_notebook.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0020_fastai_pt2_2019_source_explained.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0003_fastai_which_image_model_best.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0018_fastai_pt2_2019_exports.md',
     '/Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0019_fastai_pt2_2019_lecture1_intro.md']
    



keyword match is <mark style="background-color: #ffff00">1.0</mark> , found a line: in <mark style="background-color: #FFFF00">lib/01_utils.md</mark> 


    



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
# code block continues below
```



    nb_path:/Users/Natsume/Documents/fastdebug/nbs/lib/01_utils.ipynb, name: 01_utils
    root: /Users/Natsume/Documents/fastdebug, root_server: http://localhost:8888/tree/, name: 01_utils, folder_mid: /nbs/lib/
    path: /Users/Natsume/Documents/fastdebug/nbs/lib/, path_server: http://localhost:8888/tree/nbs/lib/



[Open `01_utils` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/01_utils.ipynb)


    



keyword match is <mark style="background-color: #ffff00">1.0</mark> , found a line: in <mark style="background-color: #FFFF00">fastai_notebooks/0017_fastai_pt2_2019_matmul.md</mark> 


    



 [Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=2342)




 [Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=2342)




### [39:04](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2344) - If we are to build a simplest linear model for mnist dataset, how to create the weights and biases for the model using `weights = torch.randn(784,10)` and `bias = torch.zeros(10)`. check the [notebook](https://nbviewer.org/github/fastai/course-v3/blob/7fceebfd14d4f3bc7e0ec649834309b8cb786e40/nbs/dl2/01_matmul.ipynb#Initial-python-model) 




```python
weights = torch.randn(784,10)
```




```python
bias = torch.zeros(10)
```




## Matrix multiplication




### [39:49](https://youtu.be/4u8fxnedueg?list=plfyubjixbdttidte1u8qgyxo4jy2y91uj&t=2389) - <mark style="background-color: #ffff00">how</mark>  to <mark style="background-color: #ffff00">understand</mark>  the <mark style="background-color: #ffff00">matrix</mark>  <mark style="background-color: #FFFF00">multiplication</mark>  calculation process (see [animation](http://matrixmultiplication.xyz/)); <mark style="background-color: #ffff00">how</mark>  to implement the <mark style="background-color: #ffff00">matrix</mark>  <mark style="background-color: #FFFF00">multiplication</mark>  with 3 loops (see src code below); imagine an input <mark style="background-color: #ffff00">matrix</mark>  `rows=5, cols=28*28` and output <mark style="background-color: #ffff00">matrix</mark>  `rows=5, cols=10`, what would the weights <mark style="background-color: #ffff00">matrix</mark>  be? `(rows=28*28, cols=10)` in the src below, `a` would be the input <mark style="background-color: #ffff00">matrix</mark>  and `b` be the weights, we want to find out about the output <mark style="background-color: #ffff00">matrix</mark>  `c`. <mark style="background-color: #ffff00">how</mark>  to use `assert` (i found a useful link [here](https://www.programiz.com/python-programming/assert-statement))




```python
def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                c[i,j] += a[i,k] * b[k,j]
    return c
```




### [42:57](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=2577) - run an example on `matmul` and test it and check how long does it take to calc a matrix of 5 rows; python is 1000 times slower than pytorch



    nb_path:/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0017_fastai_pt2_2019_matmul.ipynb, name: 0017_fastai_pt2_2019_matmul
    root: /Users/Natsume/Documents/fastdebug, root_server: http://localhost:8888/tree/, name: 0017_fastai_pt2_2019_matmul, folder_mid: /nbs/fastai_notebooks/
    path: /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/, path_server: http://localhost:8888/tree/nbs/fastai_notebooks/



[Open `0017_fastai_pt2_2019_matmul` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0017_fastai_pt2_2019_matmul.ipynb)


    filename: pt2_2019_matmul
    



keyword match is <mark style="background-color: #ffff00">1.0</mark> , found a line: in <mark style="background-color: #FFFF00">fastai_notebooks/0017_fastai_pt2_2019_matmul.md</mark> 


    



- they are equal, or




- one of them is 1, in which case that dimension is broadcasted to make it the same size




Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:




    Image  (3d array): 256 x 256 x 3




    Scale  (1d array):             3




    Result (3d array): 256 x 256 x 3




The [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) includes several examples of what dimensions can and can not be broadcast together.




### [1:10:02](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=4202) - why the broadcasting trick is the most important technique in creating fastai from scratch




## Einstein summation




### [1:10:43](https://youtu.be/4u8fxnedueg?list=plfyubjixbdttidte1u8qgyxo4jy2y91uj&t=4243) - <mark style="background-color: #ffff00">how</mark>  to <mark style="background-color: #ffff00">understand</mark>  einstein summation and <mark style="background-color: #ffff00">how</mark>  to use `torch.einsum` to do <mark style="background-color: #ffff00">matrix</mark>  <mark style="background-color: #FFFF00">multiplication</mark>  with no loop at all and speed up 16000 times faster than pure python 3-for loop version; <mark style="background-color: #ffff00">how</mark>  to trick `torch.einsum` to do batch <mark style="background-color: #ffff00">matrix</mark>  <mark style="background-color: #FFFF00">multiplication</mark>  and even more transforms and tweaks




Einstein summation (`einsum`) is a compact representation for combining products and sums in a general way. From the numpy docs:




"The subscripts string is a comma-separated list of subscript labels, where each label refers to a dimension of the corresponding operand. Whenever a label is repeated it is summed, so `np.einsum('i,i', a, b)` is equivalent to `np.inner(a,b)`. If a label appears only once, it is not summed, so `np.einsum('i', a)` produces a view of a with no changes."




[Jump_to lesson 8 video](https://course19.fast.ai/videos/?lesson=8&t=4280)




```python
# c[i,j] += a[i,k] * b[k,j]
# c[i,j] = (a[i,:] * b[:,j]).sum()
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
def batch_matmul(a,b): return torch.einsum('bik,bkj->bij', a, b)
```



    nb_path:/Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/0017_fastai_pt2_2019_matmul.ipynb, name: 0017_fastai_pt2_2019_matmul
    root: /Users/Natsume/Documents/fastdebug, root_server: http://localhost:8888/tree/, name: 0017_fastai_pt2_2019_matmul, folder_mid: /nbs/fastai_notebooks/
    path: /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/, path_server: http://localhost:8888/tree/nbs/fastai_notebooks/



[Open `0017_fastai_pt2_2019_matmul` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0017_fastai_pt2_2019_matmul.ipynb)


    filename: pt2_2019_matmul



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


```

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


```

```


```

def fastlistnbs(query="all", # howto, srcode, journey, or all
                flt_fd="src"): # other options: "groundup", "part2", "all"
    "display section headings of notebooks, filter options: fastai, part2, groundup, src_fastai,\
src_fastcore, all"
    nbs, folder, _, _, _, _ = get_all_nbs()
    nb_rt = ""
    for nb in nbs:
        if flt_fd == "fastai" and "_fastai_" in nb.split("/")[-1] and not "_fastai_pt2" in nb.split("/")[-1]: 
            nb_rt = nb
        elif flt_fd == "part2" and "_fastai_pt2" in nb.split("/")[-1]:
            nb_rt = nb
        elif flt_fd == "groundup" and "groundup_" in nb.split("/")[-1]:            
            nb_rt = nb
        elif flt_fd == "src" and "fast" in nb.split("/")[-1]:
            nb_rt = nb            
        elif flt_fd == "all": 
            nb_rt = nb
        else: 
            continue
            
#         print("\n"+nb_rt)
        with open(nb_rt, 'r') as file:
            found = False
            for idx, l in enumerate(file):
                if "##" in l:
                    if query == "howto" and "ht:" in l:
                        if l.count("#") == 2: print()
                        print(l, end="") # no extra new line between each line printed   
                        found = True
                    elif query == "srcode" and "src:" in l:
                        if l.count("#") == 2: print()                        
                        print(l, end="") # no extra new line between each line printed    
                        found = True
                    elif query == "doc" and "doc:" in l:
                        if l.count("#") == 2: print()                        
                        print(l, end="") # no extra new line between each line printed    
                        found = True                        
                    elif query == "journey" and "jn:" in l:
                        if l.count("#") == 2: print()                        
                        print(l, end="") # no extra new line between each line printed   
                        found = True                        
                    elif query == "all": 
                        if l.count("#") == 2: print()                        
                        print(l, end="") # no extra new line between each line printed
                        found = True                        
            if found: print(nb_rt + "\n")
```

### make howto splittable


```
#| export 
import pandas as pd
```


```
#| export
hts = pd.Series(list(map(lambda x: "ht: " + x, "imports, data_download, data_access, data_prep, data_loaders, cbs_tfms, learner, fit, pred, fu".split(", "))))
```


```
hts
```




    0          ht: imports
    1    ht: data_download
    2      ht: data_access
    3        ht: data_prep
    4     ht: data_loaders
    5         ht: cbs_tfms
    6          ht: learner
    7              ht: fit
    8             ht: pred
    9               ht: fu
    dtype: object




```
#| export
def fastlistnbs(query="all", # howto, srcode, journey, question, doc, or all
                flt_fd="src"): # other options: "groundup", "part2", "all"
    "display section headings of notebooks, filter options: fastai, part2, groundup, src_fastai,\
src_fastcore, all"
    nbs, folder, _, _, _, _ = get_all_nbs()
    nb_rt = ""
    nbs_fd = []
    for nb in nbs:
        if flt_fd == "fastai" and "_fastai_" in nb.split("/")[-1] and not "_fastai_pt2" in nb.split("/")[-1]: 
            nbs_fd.append(nb)
        elif flt_fd == "part2" and "_fastai_pt2" in nb.split("/")[-1]:
            nbs_fd.append(nb)
        elif flt_fd == "groundup" and "groundup_" in nb.split("/")[-1]:            
            nbs_fd.append(nb)
        elif flt_fd == "src" and "fast" in nb.split("/")[-1]:
            nbs_fd.append(nb)
        elif flt_fd == "all": 
            nbs_fd.append(nb)
        else: 
            continue      

    if query != "howto":
        for nb_rt in nbs_fd:
            with open(nb_rt, 'r') as file:
                found = False
                for idx, l in enumerate(file):
                    if "##" in l:
                        if query == "howto" and "ht:" in l:
                            if l.count("#") == 2: print()
                            print(l, end="") # no extra new line between each line printed   
                            found = True
                        elif query == "srcode" and "src:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") # no extra new line between each line printed    
                            found = True
                        elif query == "doc" and "doc:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") # no extra new line between each line printed    
                            found = True                        
                        elif query == "journey" and "jn:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") # no extra new line between each line printed   
                            found = True
                        elif query == "question" and "qt:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") # no extra new line between each line printed   
                            found = True                            
                        elif query == "all": 
                            if l.count("#") == 2: print()                        
                            print(l, end="") # no extra new line between each line printed
                            found = True                        
                if found: print(nb_rt + "\n")
    else:
        for idx, o in enumerate(hts):
            print('{:=<157}'.format(f"step {idx}: {o}"))
            for nb_rt in nbs_fd:
                with open(nb_rt, 'r') as file:
                    found = False
                    for idx, l in enumerate(file):
                        if "##" in l:
                            if o in l:
                                if l.count("#") == 2: print()
                                print(l, end="") # no extra new line between each line printed   
                                found = True                   
                    if found: print(nb_rt + "\n")
```


```
# for i, o in enumerate("imports, data-download, data-access, data-prep, data-loaders, cbs-tfms, learner, fit, pred".split(", ")):
#     i, o
```


```
fastlistnbs("howto")
# fastlistnbs("doc")
# fastlistnbs("srcode")
# fastlistnbs("journey")

```

    step 0: ht: imports==========================================================================================================================================
    
    ## ht: imports - vision
    ### ht: imports - fastkaggle 
    ### ht: imports - use mylib in kaggle
    ### ht: imports - fastkaggle - push libs to kaggle
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 1: ht: data_download====================================================================================================================================
    
    ## ht: data_download - kaggle competition dataset
    ### ht: data_download - kaggle set up
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 2: ht: data_access======================================================================================================================================
    step 3: ht: data_prep========================================================================================================================================
    step 4: ht: data_loaders=====================================================================================================================================
    step 5: ht: cbs_tfms=========================================================================================================================================
    step 6: ht: learner==========================================================================================================================================
    step 7: ht: fit==============================================================================================================================================
    step 8: ht: pred=============================================================================================================================================
    step 9: ht: fu===============================================================================================================================================
    ### ht: fu - whatinside, show_doc, fastlistnbs, fastnbs
    ### ht: fu - git - when a commit takes too long
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    


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
# fastcodes("Test the signature", nb=True)
```


```

```


```

```

## Best practice of fastdebug.core

1. test some examples 

2. import fastdebug and initialize fdb objects, put examples into fdb.eg

3. run fdb.snoop() to see all lines that get run

4. if error, run fdb.debug() to check for clues

5. run fdb.print() to see bare source

6. run fdb.docsrc() to run expression and document srcline

7. how snoop: from _funcs_kwargs to funcs_kwargs, see the learning point below
 



```
fastnbs("snoop: from _funcs_kwargs to funcs_kwargs", output=True)
```


### <mark style="background-color: #ffff00">snoop:</mark>  <mark style="background-color: #ffff00">from</mark>  <mark style="background-color: #ffff00">_funcs_kwargs</mark>  <mark style="background-color: #ffff00">to</mark>  <mark style="background-color: #FFFF00">funcs_kwargs</mark> 



    The current section is heading 8.
    
    
    
    
        The current section is heading 3.
        
        
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
        
        start of another heading 3
        ### snoop only '_funcs_kwargs' by breaking up 'funcs_kwargs'
    
    
    
    [Open `0009_funcs_kwargs` in Jupyter Notebook locally](http://localhost:8889/tree/nbs/demos/0009_funcs_kwargs.ipynb#snoop:-from-_funcs_kwargs-to-funcs_kwargs
    )
    
    
    
    
    



[Open `01_utils` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/01_utils.ipynb#<mark-style="background-color:-#ffff00">snoop:</mark>--<mark-style="background-color:-#ffff00">from</mark>--<mark-style="background-color:-#ffff00">_funcs_kwargs</mark>--<mark-style="background-color:-#ffff00">to</mark>--<mark-style="background-color:-#FFFF00">funcs_kwargs</mark>-
)



### <mark style="background-color: #ffff00">snoop:</mark>  <mark style="background-color: #ffff00">from</mark>  <mark style="background-color: #ffff00">_funcs_kwargs</mark>  <mark style="background-color: #ffff00">to</mark>  <mark style="background-color: #FFFF00">funcs_kwargs</mark> 



    The current section is heading 8.
    
    
    
    
        The current section is heading 3.
        
        
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
        
        start of another heading 3
        ### snoop only '_funcs_kwargs' by breaking up 'funcs_kwargs'
    
    
    
    [Open `0009_funcs_kwargs` in Jupyter Notebook locally](http://localhost:8889/tree/nbs/demos/0009_funcs_kwargs.ipynb#snoop:-from-_funcs_kwargs-to-funcs_kwargs
    )
    
    
    
    
    



[Open `01_utils` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/01_utils.ipynb#<mark-style="background-color:-#ffff00">snoop:</mark>--<mark-style="background-color:-#ffff00">from</mark>--<mark-style="background-color:-#ffff00">_funcs_kwargs</mark>--<mark-style="background-color:-#ffff00">to</mark>--<mark-style="background-color:-#FFFF00">funcs_kwargs</mark>-
)


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
# fastnbs("snoop: from _funcs_kwargs to funcs_kwargs", output=True)
```


```
# fastlistnbs()
```

**When I just want to have a quick look of the commented source code**

Run `fastsrcs()` first to have the list of all commented srcodes files

Run `fastview(srcname)` on the cell above `fastsrcs()` to view the actual commented srcs with an example

Run `fastcodes(query)` to search src comments for learning points


```
fastcodes("how to turn a sig into string", accu=1)
```


keyword match is <mark style="background-color: #ffff00">1.0</mark> , found a line: in <mark style="background-color: #FFFF00">test_sig.py</mark> 


        test_eq(str(inspect.signature(f)), b)=================================================(2) # [93;1mtest_sig is to test two strings with test_eq[0m; [91;1mhow to turn a signature into a string;[0m; 
    



the entire source code in <mark style="background-color: #FFFF00">test_sig.py</mark> 


    
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    
    def test_sig(f, b):=======================================================================(0)       
        "Test the signature of an object"=====================================================(1) # [92;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [91;1mtest_sig will get f's signature as a string[0m; [34;1mb is a signature in string provided by the user[0m; [35;1min fact, test_sig is to compare two strings[0m; 
        test_eq(str(inspect.signature(f)), b)=================================================(2) # [93;1mtest_sig is to test two strings with test_eq[0m; [91;1mhow to turn a signature into a string;[0m; 
                                                                                                                                                            (3)
    



```
fastview("test_sig")
```

    
    def func_2(h,i=3, j=[5,6]): pass
    test_sig(func_2, '(h, i=3, j=[5, 6])')
    
    def test_sig(f, b):=======================================================================(0)       
        "Test the signature of an object"=====================================================(1) # [92;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [91;1mtest_sig will get f's signature as a string[0m; [34;1mb is a signature in string provided by the user[0m; [35;1min fact, test_sig is to compare two strings[0m; 
        test_eq(str(inspect.signature(f)), b)=================================================(2) # [93;1mtest_sig is to test two strings with test_eq[0m; [91;1mhow to turn a signature into a string;[0m; 
                                                                                                                                                            (3)



```
fastsrcs()
```

    anno_dict.py
    test_sig.py
    subplots.py
    show_titled_image.py
    DataBlock.py
    BypassNewMeta.py
    snoop.py
    FixSigMeta.py
    show_images.py
    fastnbs.py
    _fig_bounds.py
    funcs_kwargs.py
    __init__.py
    NewChkMeta.py
    printtitle.py
    show_image.py
    AutoInit.py
    get_image_files.py
    method.py
    _rm_self.py
    delegates.py
    create_explore_str.py
    PrePostInitMeta.py
    _funcs_kwargs.py
    whatinside.py


### import fastdebug.utils as fu


```
# #| export
# import fastdebug.utils as fu
```


```
# #| export 
# fu = fu
```

#|hide
## Export


```
#| hide
from nbdev import nbdev_export
nbdev_export()
```


```

```
