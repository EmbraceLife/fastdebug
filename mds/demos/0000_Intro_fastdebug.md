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

# Introducing fastdebug

> a little tool to help me explore fastai with joy and ease

```python
%load_ext autoreload
%autoreload 2
```

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

```python
from fastdebug.utils import * # for making an example 
from fastcore.test import * 
import inspect
```

```python
import fastdebug.utils as fu
```

```python
whatinside(fu) # this is the example we are going to explore whatinside with
```

```python
from fastdebug.core import * # Let's import Fastdb and its dependencies
```

```python
# g = locals()
# fdb = Fastdb(whatinside, outloc=g) # first, create an object of Fastdb class, using `whatinside` as param
fdb = Fastdb(whatinside)
```

```python
# 1. you can view source code in whole or in parts with the length you set, 
# and it gives you srcline idx so that you can set breakpoint with ease.
#| column: screen
fdb.print(20,1) 
```

```python
#| column: screen
# 2. after viewing source code, choose a srcline idx to set breakpoint and write down why I want to explore this line
fdb.eg = """
import fastdebug.utils as fu
whatinside(fu)
"""
```

```python
# fdb.explore(11) 
```

```python
#| column: screen
# 2. you can set multiple breakpoints from the start if you like (but not necessary)
# fdb.explore([11, 16, 13]) 
```

## Fastdb.snoop

But more often I just want to have an overview of what srclines get run so that I know which lines to dive into and start documenting.

Note: I borrowed `snoop` from snoop library and automated it.

```python
fdb.snoop()
```

## Fastdb.docsrc


After exploring and snooping, if I realize there is something new to learn and maybe want to come back for a second look, I find `ipdb` and the alike are not designed to document my learning. So, I created `docsrc` to make my life easier in the following ways:

> I won't need to scroll through a long cell output of pdb commands, src prints and results to find what I learnt during exploration

> I won't need to type all the expressions during last exploration to regenerate the findings for me

> I can choose any srclines to explore and write any sinlge or multi-line expressions to evaluate the srcline

> I can write down what I learn or what is new on any srcline as comment, and all comments are attached to the src code for review

> All expressions with results and comments for each srcline under exploration are documented for easy reviews

> Of course, no worry about your original source code, as it is untouched.



### Import

```python
from fastdebug.core import * # to make Fastdb available
from fastdebug.utils import whatinside # for making an example 
```

### Initiating

```python
g = locals()
fdb = Fastdb(whatinside, outloc=g) # use either fu.whatinside or whatinside is fine
```

```python
#| column: screen
fdb.print(maxlines=20, part=1) # view the source code with idx 
```

### What does the first line do?

```python
fdb.eg = "whatinside(fu)"
```

```python
#| column: screen
fdb.docsrc(9, "how many items inside mo.__all__?", "mo", \
"if hasattr(mo, '__all__'):\\n\
    printright(f'mo: {mo}')\\n\
    printright(f'mo.__all__: {mo.__all__}')\\n\
    printright(f'len(mo.__all__): {len(mo.__all__)}')") 
```

```python
#| column: screen
dbsrc = fdb.docsrc(10, "get all funcs of a module", "mo", "inspect.getdoc(inspect.isfunction)", \
            "inspect.getdoc(inspect.getmembers)", "funcs = inspect.getmembers(mo, inspect.isfunction)")
```

### If I find the src is too long, and I customize the print out of src the way I like

```python
#| column: screen
fdb.print(maxlines=15, part=1)
```

### I can write a block of codes to evaluate

```python
import fastcore.meta as core
```

```python
#| column: screen
# fdb.takExample("whatinside(core)", whatinside=whatinside, core=core)
fdb.eg = "whatinside(core)"
dbsrc = fdb.docsrc(11, "get all classes from the module", \
"clas = inspect.getmembers(mo, inspect.isclass)\\n\
for c in clas:\\n\
    print(c)")
```

```python
#| column: screen
dbsrc = fdb.docsrc(14, "get the file path of the module", "mo.__file__", "inspect.getdoc(os.path.dirname)", "pkgpath = os.path.dirname(mo.__file__)")
```

```python
#| column: screen
# fdb.takExample("whatinside(core, lib=True)", whatinside=whatinside, core=core)
fdb.eg = "whatinside(core, lib=True)"
dbsrc = fdb.docsrc(30, "get names of all modules of a lib", "pkgpath", "inspect.getdoc(pkgutil.iter_modules)", \
"for a, b, c in pkgutil.iter_modules([pkgpath]):\\n\
    printright(f'{a} ; {b}; {c}')", db=True)
```

### Print out the entire src with idx and comments, when I finish documenting

```python
fdb.print()
```

```python

```

### After running `.dbprint`, everything is back to normal automatically

```python
inspect.getsourcefile(fu.whatinside)
```

```python
inspect.getsourcefile(whatinside)
```

```python

```

```python

```

To check, when run `whatinside??` we should see the actually source code whereas the db version of `whatinside` does not have.


## Install

<!-- #region -->
```sh
pip install fastdebug
```
<!-- #endregion -->

## How to use


Fill me in please! Don't forget code examples:

```python
1+1
```

#|hide
## Send to Obsidian

```python
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/index.ipynb
!mv /Users/Natsume/Documents/fastdebug/index.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/

!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```

```python

```
