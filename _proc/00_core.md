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

```python

```

```python

```

# core

> the core functionalities of fastdebug

```python
from fastdebug.utils import expand
```

This notebook records the history of how I built this library from scratch. Well, not exactly from the scratch, because there is a [first version](https://github.com/EmbraceLife/debuggable) of this library where many difficulties have been already handled.

If you want to see the latest functions of this module, just `cmd/ctrl + f` to search '#| export' and check them out this way.

```python
#| default_exp core
```

```python
#| hide
from nbdev.showdoc import *
```

## make life easier with defaults  

```python
#| exports
defaults = type('defaults', (object,), {'margin': 157, # align to the right by 157
                                        'orisrc': None, # keep a copy of original official src code
                                        'outenv': globals(), # outside global env
                                        'cmts': {} # a dict to store idx and cmt
                                        # 'eg': None, # examples
                                        # 'src': None, # official src
                                       }) 
```

## globals() and locals()

Interesting behavior of [locals()](https://stackoverflow.com/questions/7969949/whats-the-difference-between-globals-locals-and-vars)

```python
locals() == globals()
```

```python
"functools" in locals()
```

```python
"inspect" in locals()
```

```python
"eval" in locals()
```

```python
"defaults" in locals()
```

```python
"whatinside" in locals()
```

```python
"whichversion" in locals()
```

```python
from fastdebug.utils import *
```

```python
"whatinside" in locals()
```

```python
"whichversion" in globals()
```

## Execute strings

```python
eval?
```

```python
exec?
```

```python
#| export
import pprint
```

```python
#| export
pprint = pprint.pprint # so that pprint is included inside __all__
```

### new variable or updated variable by exec will only be accessible from locals()

```python
x = 1
def test():
    a = "1+x"
    pprint(f'locals: {locals()}', width=157) # x is not in locals() but in globals()
    print(eval(a)) 
test()
```

```python
x = 1
def test():
    a = "b = 1+x"
    pprint(locals())
    print(f'exec(a): {exec(a)}') 
    pprint(locals())
    print(b) # b can't be accessed directly and not use the key b

try:
    test()
except NameError as e:
    print(e)
    
```

```python
x = 1
def test():
    a = "b = 1+x"
    pprint(locals())
    print(f'exec(a): {exec(a)}') 
    pprint(locals())
    b = locals()['b'] # you can't even assign the value to b, otherwise b:value won't be available
    print(b)

# test()
try:
    test()
except KeyError as e:
    print("KeyError: 'b' does not exist")
```

```python
x = 1
def test():
    a = "b = 1+x"
    pprint(locals())
    print(f'exec(a): {exec(a)}') 
    pprint(locals())
    c = locals()['b'] # if assign to a different name, then b is available
    print(c)
    pprint(locals())

# test()
try:
    test()
except KeyError as e:
    print("KeyError: 'b' does not exist")
```

```python
x = 1
def test():
    a = "b = 1+x"
    pprint(locals())
    print(f'exec(a): {exec(a)}') 
    pprint(locals())
    c = locals()['b'] # if assign to a different name, then b is available
    print(f'c = locals()["b"]; c: {c}')
    pprint(locals())
    print(f'exec(c = c + 1): {exec("c = c + 1")}') # update c won't change anything
    pprint(locals())

test()
```

```python
x = 1
def test():
    a = "b = 1+x"
    pprint(locals())
    print(f'exec(a): {exec(a)}') 
    pprint(locals())
    c = locals()['b'] # if assign to a different name, then b is available
    print(f'c = locals()["b"]; c: {c}')
    pprint(locals())
    print(f'exec(d = c + 1): {exec("d = c + 1")}') # must assign to a different variable, not c anymore
    pprint(locals())

test()
```

### eval can override its own globals() and locals()

```python
def test():
    a = 1
    b = "a+1"
    pprint(f'locals: {locals()}', width=157)
    print(f'b: {eval(b, {}, {"a": 2})}') # globals() put to empty and locals() to include 'a' with a different value
    pprint(f'locals: {locals()}', width=157) 

test()
```

### when exec update existing functions

```python
#| export
import inspect
```

```python
#| export
inspect = inspect # so that inspect is included inside __all__
```

```python
def foo(x, y): return x + y
def test(func):
    a = 1
    b = inspect.getsource(func)
    newb = ""
    for l in b.split('\n'):
        if bool(l):
            newb = newb + l
    newb = newb + " + 3\n"
    pprint(f'locals: {locals()}', width=157) # foo is not available in locals()
    print(f'exec(newb): {exec(newb)}') 
    pprint(f'locals: {locals()}', width=157) # a foo is available in locals(), but which one is it
    newfoo = locals()["foo"] # to use it, must assign to a different name
    print(newfoo(1,9))
    print(locals()["foo"](1,9))
    print(func(1,9))
    print(foo(1,9))

test(foo)    
```

### when the func to be udpated involve other libraries

```python
import functools
```

```python
def foo(x, y): 
    print(inspect.signature(foo))
    return x + y
def test(func):
    a = 1
    b = inspect.getsource(func)
    newb = ""
    for l in b.split('\n'):
        if bool(l) and "return" not in l :
            newb = newb + l + '\n'
        elif bool(l):
            newb = newb + l
    newb = newb + " + 3\n"
    pprint(f'locals: {locals()}', width=157) # foo is not available in locals()
    print(f'exec(newb): {exec(newb)}') 
    pprint(f'locals: {locals()}', width=157) # a foo is available in locals(), but which one is it
    newfoo = locals()["foo"]
    print(newfoo(1,9))
    print(locals()["foo"](1,9))
    print(func(1,9))
    print(foo(1,9))

test(foo)    
```

### inside a function, exec() allow won't give you necessary env from function namespace

```python
def add(x, y): return 1

def test(func):
    a = 1
    lst = []
    b = "def add(x, y):\n    lst.append(x)\n    return x + y"
    pprint(f'locals: {locals()}', width=157)
    exec(b) # create the new add in locals
    pprint(f'locals: {locals()}', width=157)
    print(f'add(5,6): {add(5,6)}')
    add1 = locals()['add'] # assign a different name, add1
    
    print(f'add1(5,6): {add1(5,6)}') # error: lst is not defined, even though lst=[] is right above
    
    pprint(f'locals: {locals()}', width=157)

try:
    test(add)
except NameError as e:
    print(e)
```

### magic of `exec(b, globals().update(locals()))`


What about `exec(b, globals().update(globals()))`

```python
def add(x, y): return 1

def test(func):
    a = 1
    lst = []
    b = "def add(x, y):\n    lst.append(x)\n    return x + y"
    pprint(f'locals: {locals()}', width=157)
    
    exec(b, globals().update(globals())) # update(globals()) won't give us lst above
    
    pprint(f'locals: {locals()}', width=157)
    add1 = locals()['add'] 
    print(add1(5,6))
    pprint(f'locals: {locals()}', width=157)



try:
    test(add)
except: 
    print("exec(b, globals().update(globals())) won't give us lst in the func namespace")
```

```python
def add(x, y): return 1

def test(func):
    a = 1
    lst = []
    b = "def add(x, y):\n    lst.append(x)\n    return x + y"
    pprint(f'locals: {locals()}', width=157)
    
    exec(b, globals().update(locals())) # make sure b can access lst from above
    
    pprint(f'locals: {locals()}', width=157)
    add1 = locals()['add'] 
    print(add1(5,6))
    pprint(f'locals: {locals()}', width=157)

test(add)
print(add(5,6))
try:
    print(add1(5,6))
except: 
    print("you can't bring add1 from a function namespace to the outside world")
```

### Bring variables from a func namespace to the sideout world

```python
def add(x, y): return 1

def test(func):
    a = 1
    lst = []
    b = "def add(x, y):\n    lst.append(x)\n    return x + y"
    pprint(f'locals: {locals()}', width=157)    
    
    exec(b, globals().update(locals())) # make sure b can access lst from above

    add1 = locals()['add'] 
    print(add1(5,6))
    add1(5,6)
    pprint(f'locals: {locals()}', width=157)

    # bring variables inside a func to the outside
    globals().update(locals())

test(add)
pprint(add(5,6)) # the original add is override by the add from the function's locals()
pprint(add1(5,6))
print(lst)
```

### globals() in a cell vs globals() in a func

```python
from fastdebug.utils import tstenv
```

```python
len(globals().keys())
```

```python
globals()['__name__']
```

```python
tstenv??
```

```python
tstenv()
```

## make a colorful string

```python
#| export
from fastcore.basics import *
```

```python
#| export
class dbcolors: # removing ;1 will return to normal color, with ;1 is bright color
    g = '\033[92;1m' #GREEN
    y = '\033[93;1m' #YELLOW
    r = '\033[91;1m' #RED
    m = '\u001b[35;1m' # megenta
    c = '\u001b[36;1m' # cyan
    b = '\u001b[34;1m' # blue
    w = '\u001b[37;1m' # white
    reset = '\033[0m' #RESET COLOR
```

```python
#| export
import random
```

```python
random.randint(0,1)
```

```python
#| export
def randomColor():
    colst = ['b', 'c', 'm', 'r', 'y', 'w', 'g']
    dictNumColor = {i:v for i, v in zip(range(len(colst)), colst)}

    return dictNumColor[random.randint(0, 6)]
```

```python
randomColor()
```

```python
#| export
def colorize(cmt, color:str=None):
    if type(cmt) != str:
        cmt = str(cmt)
    if color == "g":
        return dbcolors.g + cmt + dbcolors.reset
    elif color == "y":
        return dbcolors.y + cmt + dbcolors.reset
    elif color == "r":
        return dbcolors.r + cmt + dbcolors.reset
    elif color == "b":
        return dbcolors.b + cmt + dbcolors.reset
    elif color == "c":
        return dbcolors.c + cmt + dbcolors.reset    
    elif color == "m":
        return dbcolors.m + cmt + dbcolors.reset    
    elif color == "w":
        return dbcolors.w + cmt + dbcolors.reset    
    else: 
        return cmt
```

```python
colorize("this is me", "r")
```

```python
print(colorize("this is me", "r"))
```

## align text to the most right

```python
#| export
import re
```

```python
#| export
def strip_ansi(source):
    return re.sub(r'\033\[(\d|;)+?m', '', source)
```

```python

def alignright(blocks):
    lst = blocks.split('\n')
    maxlen = max(map(lambda l : len(strip_ansi(l)) , lst ))
    indent = defaults.margin - maxlen
    for l in lst:
        print(' '*indent + format(l))
```

```python

def alignright(blocks, margin:int=157):
    lst = blocks.split('\n')
    maxlen = max(map(lambda l : len(strip_ansi(l)) , lst ))
    indent = margin - maxlen
    for l in lst:
        print(' '*indent + format(l))
```

```python
alignright("this is me")
```

```python
#| export
def printright(blocks, margin:int=157):
    lst = blocks.split('\n')
    maxlen = max(map(lambda l : len(strip_ansi(l)) , lst ))
    indent = margin - maxlen
    for l in lst:
        print(' '*indent + format(l))
```

```python

```

```python

```

## printsrcwithidx


### print the entire source code with idx from 0

```python

def printsrcwithidx(src):
    totallen = 157
    lenidx = 5
    lstsrc = inspect.getsource(src).split('\n')
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        lenl = len(l)

        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
```

### print the whole src with idx or print them in parts

```python

def printsrcwithidx(src, 
                    maxlines:int=33, # maximum num of lines per page
                    part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    totallen = 157
    lenidx = 5
    lstsrc = inspect.getsource(src).split('\n')
    numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
    
    if part == 0: 
        for idx, l in zip(range(len(lstsrc)), lstsrc):
            lenl = len(l)
            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return
```

### use cmts from dbprint to print out src with comments

```python

def printsrcwithidx(src, 
                    maxlines:int=33, # maximum num of lines per page
                    part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(src).split('\n')
    numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
    # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
    cmts = defaults.cmts
    if part == 0: 
        for idx, l in zip(range(len(lstsrc)), lstsrc):
            lenl = len(l)

            if not bool(l.strip()):
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            elif lenl + lspace >= 100:
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                    else:
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                else: 
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            else:


                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                else:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 

    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                else:

                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return
```

```python
# printsrcwithidx(foo)
```

```python
# printsrcwithidx(pprint)
```

### no more update for printsrcwithidx, for the latest see Fastdb.print


see a [complex example](./examples/dbprint.ipynb#test-printsrcwithidx) on printsrcwithidx


## print out src code

```python
#| export
import inspect
```

### basic version

```python

def printsrc(src, srclines, cmt):
# print out the title
    print('\n')
    print('{:#^157}'.format(" srcline under investigation "))
    print('\n')


    # convert the source code of the function into a list of strings splitted by '\n'
    lst = inspect.getsource(src).split('\n')

    ccount = 0
    for l in lst:
        if bool(l) and l.strip() in srclines:# print out the srcline under investigation
            print('{:=<157}'.format(l))
            ccount = ccount + 1

            if bool(cmt): # print out comment at the end of the srclines under investigation
                numsrclines = len(srclines.split("\n"))
                if ccount == numsrclines:
                    colcmt = colorize(cmt, "r") # colorize the comment
                    alignright(colcmt) # put the comment to the most right

        else: 
            print('{:<157}'.format(l)) # print out the rest of source code
```

```python
def foo():     
    a = 1 + 1
    b = a + 1
    pass
```

```python
printsrc(foo, "a = 1 + 1", "this is comment")
```

```python
printsrc(foo, "    a = 1 + 1\n    b = a + 1", "this is comment")
```

### print src with specific number of lines

```python
len("    this is a code\nthis is another code".split('\n'))
list(map(lambda x: bool(x.strip()), "    this is a code\n    this is another code\n   \n   ".split('\n'))).count(True)
```

```python

def printsrc(src, srclines, cmt, expand:int=2):

    # convert the source code of the function into a list of strings splitted by '\n'
    lst = inspect.getsource(src).split('\n')
    
    # find out the idx of start srclines and end srclines
    startidx = 0
    numsrclines = list(map(lambda x: bool(x.strip()), srclines.split('\n'))).count(True)
    for idx, l in zip(range(len(lst)), lst):
        if bool(l) and l.strip() in srclines:
            startidx = idx
    endidx = startidx + numsrclines - 1
    
    ccount = 0
    for idx, l in zip(range(len(lst)), lst):
        if bool(l) and l.strip() in srclines:# print out the srcline under investigation with # as padding
            print('{:=<157}'.format(l))
            ccount = ccount + 1

            if bool(cmt): # print out comment at the end of the srclines under investigation
                # numsrclines = len(srclines.split("\n"))
                if ccount == numsrclines:
                    colcmt = colorize(cmt, "r") # colorize the comment
                    alignright(colcmt) # put the comment to the most right

        elif idx < startidx and idx >= startidx - expand:
            print('{:<157}'.format(l)) # print out the rest of source code to the most left

        elif idx <= endidx + expand and idx > endidx:
            print('{:<157}'.format(l)) # print out the rest of source code to the most left
    
```

### make the naming more sensible

```python

def printsrc(src, dblines, cmt, expand:int=2):

    lstsrc = inspect.getsource(src).split('\n')

    numdblines = list(map(lambda x: bool(x.strip()), dblines.split('\n'))).count(True)
    
    startdbidx = 0
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            startdbidx = idx
    enddbidx = startdbidx + numdblines - 1
    
    dbcount = 0
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            print('{:=<157}'.format(l))
            dbcount = dbcount + 1

            if bool(cmt): 
                if dbcount == numdblines:
                    colcmt = colorize(cmt, "r")
                    alignright(colcmt) # also print the comment

        elif idx < startdbidx and idx >= startdbidx - expand:
            print('{:<157}'.format(l)) 

        elif idx <= enddbidx + expand and idx > enddbidx:
            print('{:<157}'.format(l)) 
    
```

### Allow a dbline occur more than once

```python

def printsrc(src, dblines, cmt, expand:int=2):

    lstsrc = inspect.getsource(src).split('\n')

    numdblines = list(map(lambda x: bool(x.strip()), dblines.split('\n'))).count(True)
    
    dblineidx = []
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            # print(f'idx: {idx}; and l: {l}') # debugging
            dblineidx.append(idx)
    # enddbidx = startdbidx + numdblines - 1
    
    # dbcount = 0
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            # print(f'idx: {idx}; and l: {l}') # debugging
            print('{:=<157}'.format(l))
            # dbcount = dbcount + 1

            if bool(cmt): 
                # if dbcount == numdblines:
                    colcmt = colorize(cmt, "r")
                    alignright(colcmt) # also print the comment

        for o, dbidx in zip(range(len(dblineidx)), dblineidx):
            if idx >= dbidx - expand and idx < dbidx: 
                print('{:<157}'.format(l)) 
            elif idx <= dbidx + expand and idx > dbidx: 
                print('{:<157}'.format(l)) 
                
                if idx == dbidx + expand and len(dblineidx)>=2:
                    print("\n")
                    print('{:=^157}'.format(f" The occurance {o} "))
                    print("\n")                
```

### adding idx to the selected srclines


#### printsrclinewithidx

```python
#| export
def printsrclinewithidx(idx, l, fill=" "):
    totallen = 157
    lenidx = 5
    lenl = len(l)
    print(l + fill*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
```

```python

def printsrc(src, # name of src code such as foo, or delegates
             dblines,
             cmt,
             expand:int=2): # expand the codes around the srcline under investigation
    
    lstsrc = inspect.getsource(src).split('\n')
    numdblines = list(map(lambda x: bool(x.strip()), dblines.split('\n'))).count(True)
    
    dblineidx = []
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            dblineidx.append(idx)

    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            printsrclinewithidx(idx, l, fill="=")

            if bool(cmt):
                colcmt = colorize(cmt, "r")
                alignright(colcmt) # also print the comment

        for o, dbidx in zip(range(len(dblineidx)), dblineidx):
            if idx >= dbidx - expand and idx < dbidx:
                printsrclinewithidx(idx, l)
            elif idx <= dbidx + expand and idx > dbidx:
                printsrclinewithidx(idx, l)
                
                if idx == dbidx + expand and len(dblineidx)>=2:
                    print("\n")
                    print('{:=^157}'.format(f" The occurance {o} "))
                    print("\n")
```

### dblines can be string of code or idx number

```python
type(int("120"))
```

```python
a = 120
type(a) == int
```

```python

def printsrc(src, # name of src code such as foo, or delegates
             dbcode, # string of codes or int of code idx number
             cmt,
             expand:int=2): # expand the codes around the srcline under investigation
    
    lstsrc = inspect.getsource(src).split('\n')
    
    dblines = ""
    if type(dbcode) == int:
        dblines = lstsrc[dbcode]
    else:
        dblines = dbcode
        numdblines = list(map(lambda x: bool(x.strip()), dblines.split('\n'))).count(True)
    
    dblineidx = []
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            dblineidx.append(idx)

    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            printsrclinewithidx(idx, l, fill="=")

            if bool(cmt):
                colcmt = colorize(cmt, "r")
                alignright(colcmt) # also print the comment

        for o, dbidx in zip(range(len(dblineidx)), dblineidx):
            if idx >= dbidx - expand and idx < dbidx:
                printsrclinewithidx(idx, l)
            elif idx <= dbidx + expand and idx > dbidx:
                printsrclinewithidx(idx, l)
                
                if idx == dbidx + expand and len(dblineidx)>=2:
                    print("\n")
                    print('{:=^157}'.format(f" The occurance {o} "))
                    print("\n")
```

### avoid multi-occurrance of the same srcline

```python
#| export
def printsrc(src, # name of src code such as foo, or delegates
             dbcode, # string of codes or int of code idx number
             cmt,
             expand:int=2): # expand the codes around the srcline under investigation
    "print the seleted srcline with comment, idx and specified num of expanding srclines"
    lstsrc = inspect.getsource(src).split('\n')
    
    dblines = ""
    if type(dbcode) == int:
        dblines = lstsrc[dbcode]
    else:
        dblines = dbcode
        numdblines = list(map(lambda x: bool(x.strip()), dblines.split('\n'))).count(True)
    
    
    dblineidx = []
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        if bool(l) and l.strip() in dblines:
            dblineidx.append(idx)

    for idx, l in zip(range(len(lstsrc)), lstsrc):
        
        srcidx = dbcode if type(dbcode) == int else dblineidx[0]
        
        if bool(l) and l.strip() in dblines and idx == srcidx:
            printsrclinewithidx(idx, l, fill="=")

            if bool(cmt):
                colcmt = colorize(cmt, "r")
#                 alignright(colcmt) # also print the comment
                printright(colcmt) # also print the comment                

        if idx >= srcidx - expand and idx < srcidx:
            printsrclinewithidx(idx, l)
        elif idx <= srcidx + expand and idx > srcidx:
            printsrclinewithidx(idx, l)

```

```python
def foo(a):
    "this is docs"
    # this is a pure comment
    if a > 1:
        for i in range(3):
            a = i + 1
    else:
        "this is docs"
        b = a + 1
    "this is docs"
    return a
```

```python
printsrc(foo, "else:", "this is comment", expand=4)
```

```python
printsrc(foo, "return a", "this is comment", expand=10)
```

```python
printsrc(foo, 3, "this is comment")
```

more [complex example](./examples/printsrc.ipynb) on printsrc


## dbprint on expression


### basic version

```python

def dbprint(src, # the src func name, e.g., foo
            srclines:str, # the srclines under investigation
            cmt:str, # comment
            *code,   # a number of codes to run, each code is in str, e.g., "a + b", "c = a - b"
            **env
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "debug a srcline with one or more expressions with src printed."
    
    # print out src code: the basic version
    printsrc(src, srclines, cmt)
    
    for c in code:
    # print(f"{c} => {c} : {eval(c, globals().update(env))}") 
        output = f"{c} => {c} : {eval(c, globals().update(env))}"
        print('{:>157}'.format(output))   
```

```python
def foo():     
    a = 1 + 1
    b = a + 1
    pass
```

```python
def foo(): 
    dbprint(foo, "    a = 1 + 1", "this is a test", "1+2", "str(1+2)")
    a = 1 + 1
    pass
```

```python
foo()
```

```python

```

### insert dbcode and make a new dbfunc


when bringing back splitted lines back, we need add '\n' back to them

```python
back = ""
for l in inspect.getsource(foo).split('\n'):
    back = back + l
pprint(back, width=157)
```

```python

def dbprint(src, # the src func name, e.g., foo
            srclines:str, # the srclines under investigation
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2 # span 2 lines of srcode up and down from the srcline investigated
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"
    
    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
    
    
    # print out src code: the basic version
    printsrc(src, srclines, cmt, expand)
    
    # insert the dbcodes from *code into the original official srcode
    dbsrc = ""
    indent = 4
    onedbprint = False
    
    # make sure the last line which is "" is removed from lst
    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]
    
    # express and insert the dbcode after the srcline under investigation
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:
            
            # get current l's indentation is enough here, as dbcode is above l
            numindent = len(l) - len(l.strip())
            # attach dbcode above the l under investigation
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ")"
                elif count == len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ")"
                elif count != len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ","
                elif count != len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            # make sure dbprint only written once for multi-srclines under investigation
            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n' # don't forget to add the srcline below dbprint
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'
        
        elif bool(l.strip()) and idx + 1 == len(lst): # handle the last line of srcode
            dbsrc = dbsrc + l
            
        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
                

    # print out the new srcode
    # for l in dbsrc.split('\n'):
    #     print(l)
    
    # exec the dbsrc to replace the official source code
    exec(dbsrc) # created new foo and saved inside locals()

    
    # check to see whether the new srcode is created
    # print(locals())
    
    # move this new foo into globals, so that outside the cell we can still use it
    globals().update(locals())
    
    return locals()[defaults.orisrc.__name__]
    
```

### Bring outside namespace variables into exec()

```python

def dbprint(src, # the src func name, e.g., foo
            srclines:str, # the srclines under investigation
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env = globals() # outer env
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"
    
    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
    
    
    # print out src code: the basic version
    printsrc(src, srclines, cmt, expand)
    
    # insert the dbcodes from *code into the original official srcode
    dbsrc = ""
    indent = 4
    onedbprint = False
    
    # make sure the last line which is "" is removed from lst
    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]
    
    # express and insert the dbcode after the srcline under investigation
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:
            
            # get current l's indentation is enough here, as dbcode is above l
            numindent = len(l) - len(l.strip())
            # attach dbcode above the l under investigation
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ")"
                elif count == len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ")"
                elif count != len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ","
                elif count != len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            # make sure dbprint only written once for multi-srclines under investigation
            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n' # don't forget to add the srcline below dbprint
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'
        
        elif bool(l.strip()) and idx + 1 == len(lst): # handle the last line of srcode
            dbsrc = dbsrc + l
            
        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
                

    # print out the new srcode
    # for l in dbsrc.split('\n'):
    #     print(l)
    
    # exec the dbsrc to replace the official source code
    # exec(dbsrc) # created new foo and saved inside locals()
    # exec(dbsrc, globals().update(locals())) # make sure b can access lst from above
    exec(dbsrc, globals().update(env)) # make sure b can access lst from above
    
    # check to see whether the new srcode is created
    # print(f'locals()["src"]: {locals()["src"]}')
    # print(f'locals()["{src.__name__}"]: {locals()[src.__name__]}')
    
    # move this new foo into globals, so that outside the cell we can still use it
    globals().update(locals())
    
    return locals()[defaults.orisrc.__name__]
    
```

### Bring what inside the func namespace variables to the outside world

```python

def dbprint(src, # the src func name, e.g., foo
            srclines:str, # the srclines under investigation
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env = globals() # outer env
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"
    
    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
    
    
    # print out src code: the basic version
    printsrc(src, srclines, cmt, expand)
    
    # insert the dbcodes from *code into the original official srcode
    dbsrc = ""
    indent = 4
    onedbprint = False
    
    # make sure the last line which is "" is removed from lst
    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]
    
    # express and insert the dbcode after the srcline under investigation
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:
            
            # get current l's indentation is enough here, as dbcode is above l
            numindent = len(l) - len(l.strip())
            # attach dbcode above the l under investigation
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ")"
                elif count == len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ")"
                elif count != len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ","
                elif count != len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            # make sure dbprint only written once for multi-srclines under investigation
            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n' # don't forget to add the srcline below dbprint
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'
        
        elif bool(l.strip()) and idx + 1 == len(lst): # handle the last line of srcode
            dbsrc = dbsrc + l
            
        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
                

    exec(dbsrc, globals().update(env)) # make sure b can access lst from above
    
    # check to see whether the new srcode is created
    # print(f'locals()["src"]: {locals()["src"]}')
    # print(f'locals()["{src.__name__}"]: {locals()[src.__name__]}')
    
    # this is crucial to bring what inside a func namespace into the outside world
    env.update(locals())
    
    return locals()[defaults.orisrc.__name__]
    
```

see a [complex example](./examples/dbprint.ipynb) on dbprint


### Adding g = locals() to dbprintinsert to avoid adding env individually

```python

def dbprint(src, # the src func name, e.g., foo
            srclines:str, # the srclines under investigation
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env = {} # outer env
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"
    
    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
    
    
    # print out src code: the basic version
    printsrc(src, srclines, cmt, expand)
    
    # insert the dbcodes from *code into the original official srcode
    dbsrc = ""
    indent = 4
    onedbprint = False
    
    # make sure the last line which is "" is removed from lst
    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]
    
    # express and insert the dbcode after the srcline under investigation
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:
            
            # get current l's indentation is enough here, as dbcode is above l
            numindent = len(l) - len(l.strip())
            # attach dbcode above the l under investigation
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ")"
                elif count == len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ")"
                elif count != len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ","
                elif count != len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            # make sure dbprint only written once for multi-srclines under investigation
            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n' # adding this line above dbprint line
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n' # don't forget to add the srcline below dbprint
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'
        
        elif bool(l.strip()) and idx + 1 == len(lst): # handle the last line of srcode
            dbsrc = dbsrc + l
            
        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
    
    # print out the new srcode
    # for l in dbsrc.split('\n'):
    #     print(l)

    exec(dbsrc, globals().update(env)) # make sure b can access lst from above
    
    # check to see whether the new srcode is created
    # print(f'locals()["src"]: {locals()["src"]}')
    # print(f'locals()["{src.__name__}"]: {locals()[src.__name__]}')
    
    # this is crucial to bring what inside a func namespace into the outside world
    env.update(locals())
    
    return locals()[defaults.orisrc.__name__]
    
```

### enable srclines to be either string or int 

```python

def dbprint(src, # the src func name, e.g., foo
            dbcode, # the srclines under investigation, can be either string or int
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env = {} # outer env
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"
    
    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
    
    
    # print out src code: the basic version
    printsrc(src, dbcode, cmt, expand)
    
    # insert the dbcodes from *code into the original official srcode
    dbsrc = ""
    indent = 4
    onedbprint = False
    
    # make sure the last line which is "" is removed from lst
    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]
    
    # dbprint is enabled to accept both string or int
    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode
    
    # express and insert the dbcode after the srcline under investigation
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:
            
            # get current l's indentation is enough here, as dbcode is above l
            numindent = len(l) - len(l.strip())
            # attach dbcode above the l under investigation
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ")"
                elif count == len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ")"
                elif count != len(codes) and "=" in c:
                    dbcodes = dbcodes + c + ","
                elif count != len(codes) and "=" not in c:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            # make sure dbprint only written once for multi-srclines under investigation
            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n' # adding this line above dbprint line
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n' # don't forget to add the srcline below dbprint
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'
        
        elif bool(l.strip()) and idx + 1 == len(lst): # handle the last line of srcode
            dbsrc = dbsrc + l
            
        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
    
    # print out the new srcode
    # for l in dbsrc.split('\n'):
    #     print(l)

    exec(dbsrc, globals().update(env)) # make sure b can access lst from above
    
    # check to see whether the new srcode is created
    # print(f'locals()["src"]: {locals()["src"]}')
    # print(f'locals()["{src.__name__}"]: {locals()[src.__name__]}')
    
    # this is crucial to bring what inside a func namespace into the outside world
    env.update(locals())
    
    return locals()[defaults.orisrc.__name__]
    
```

### enable = to be used as assignment in codes

```python

def dbprint(src, # the src func name, e.g., foo
            dbcode, # the srclines under investigation, can be either string or int
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env = {} # outer env
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"
    
    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
    
    
    # print out src code: the basic version
    printsrc(src, dbcode, cmt, expand)
    
    # insert the dbcodes from *code into the original official srcode
    dbsrc = ""
    indent = 4
    onedbprint = False
    
    # make sure the last line which is "" is removed from lst
    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]
    
    # dbprint is enabled to accept both string or int
    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode
    
    # express and insert the dbcode after the srcline under investigation
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:
            
            # get current l's indentation is enough here, as dbcode is above l
            numindent = len(l) - len(l.strip())
            # attach dbcode above the l under investigation
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes) and "env=" in c:
                    dbcodes = dbcodes + c + ")"
                # elif count == len(codes) and "=" not in c:
                #     dbcodes = dbcodes + '"' + c + '"' + ")"
                # elif count != len(codes) and "=" in c:
                #     dbcodes = dbcodes + c + ","
                # elif count != len(codes) and "=" not in c:
                else:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            # make sure dbprint only written once for multi-srclines under investigation
            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n' # adding this line above dbprint line
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n' # don't forget to add the srcline below dbprint
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'
        
        elif bool(l.strip()) and idx + 1 == len(lst): # handle the last line of srcode
            dbsrc = dbsrc + l
            
        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
    
    # print out the new srcode
    # for l in dbsrc.split('\n'):
    #     print(l)

    exec(dbsrc, globals().update(env)) # make sure b can access lst from above
    
    # check to see whether the new srcode is created
    # print(f'locals()["src"]: {locals()["src"]}')
    # print(f'locals()["{src.__name__}"]: {locals()[src.__name__]}')
    
    # this is crucial to bring what inside a func namespace into the outside world
    env.update(locals())
    
    return locals()[defaults.orisrc.__name__]
    
```

### avoid adding "env=g" for dbprintinsert

```python

def dbprint(src, # the src func name, e.g., foo
            dbcode, # the srclines under investigation, can be either string or int
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env = {} # outer env
           ):  # a number of stuff needed to run the code, e.g. var1 = var1, func1 = func1
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"
    
    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
    
    
    # print out src code: the basic version
    printsrc(src, dbcode, cmt, expand)
    
    # insert the dbcodes from *code into the original official srcode
    dbsrc = ""
    indent = 4
    onedbprint = False
    
    # make sure the last line which is "" is removed from lst
    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]
    
    # dbprint is enabled to accept both string or int
    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode
    
    # express and insert the dbcode after the srcline under investigation
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:
            
            # get current l's indentation is enough here, as dbcode is above l
            numindent = len(l) - len(l.strip())
            # attach dbcode above the l under investigation
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes):
                    dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                else:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            # make sure dbprint only written once for multi-srclines under investigation
            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n' # adding this line above dbprint line
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n' # don't forget to add the srcline below dbprint
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'
        
        elif bool(l.strip()) and idx + 1 == len(lst): # handle the last line of srcode
            dbsrc = dbsrc + l
            
        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
    
    # print out the new srcode
    # for l in dbsrc.split('\n'):
    #     print(l)

    exec(dbsrc, globals().update(env)) # make sure b can access lst from above
    
    # check to see whether the new srcode is created
    # print(f'locals()["src"]: {locals()["src"]}')
    # print(f'locals()["{src.__name__}"]: {locals()[src.__name__]}')
    
    # this is crucial to bring what inside a func namespace into the outside world
    env.update(locals())
    
    return locals()[defaults.orisrc.__name__]
```

### collect cmt for later printsrcwithidx to print comments together

```python

def dbprint(src, # the src func name, e.g., foo
            dbcode, # the srclines under investigation, can be either string or int
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env={}): # out environment
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
        
    if type(dbcode) == int: defaults.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode

    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines:

            numindent = len(l) - len(l.strip())
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes):
                    dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                else:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            if onedbprint == False:
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'
                onedbprint = True
            else:
                dbsrc = dbsrc + l + '\n'

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    exec(dbsrc, globals().update(env)) # make sure b can access lst from above

    env.update(locals())

    return locals()[defaults.orisrc.__name__]

```

### make sure only one line with correct idx is debugged

```python

def dbprint(src, # the src func name, e.g., foo
            dbcode, # the srclines under investigation, can be either string or int
            cmt:str, # comment
            *codes, # a list of dbcodes
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            env={}, # out environment
            showdbsrc=False): # print out dbsrc or not
    "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

    # make sure the original official src is kept safe and used whenever dbprint is used
    if defaults.orisrc == None:
        defaults.orisrc = src
    else: 
        src = defaults.orisrc
        
    if type(dbcode) == int: defaults.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode

    for idx, l in zip(range(len(lst)), lst):
        # make sure the line with correct idx is debugged
        if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

            numindent = len(l) - len(l.strip())
            dbcodes = "dbprintinsert("
            count = 1
            for c in codes:
                if count == len(codes):
                    dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                else:
                    dbcodes = dbcodes + '"' + c + '"' + ","
                count = count + 1

            dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'  

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'
  

    if showdbsrc: # added to debug
        for l in dbsrc.split('\n'):
            print(l)
    
    exec(dbsrc, globals().update(env)) # make sure b can access lst from above

    env.update(locals())

    return locals()[defaults.orisrc.__name__]

```

### avoid typing "" when there is no codes

```python

# def dbprint(src, # the src func name, e.g., foo
#             dbcode, # the srclines under investigation, can be either string or int
#             cmt:str, # comment
#             *codes, # a list of dbcodes
#             expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
#             env={}, # out environment
#             showdbsrc=False): # print out dbsrc or not
#     "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

#     # make sure the original official src is kept safe and used whenever dbprint is used
#     if defaults.orisrc == None:
#         defaults.orisrc = src
#     else: 
#         src = defaults.orisrc
        
#     if type(dbcode) == int: defaults.cmts.update({dbcode: cmt})

#     printsrc(src, dbcode, cmt, expand)

#     dbsrc = ""
#     indent = 4
#     onedbprint = False

#     lst = inspect.getsource(src).split('\n')
#     if not bool(lst[-1]): lst = lst[:-1]

#     srclines = ""
#     if type(dbcode) == int:
#         srclines = lst[dbcode]
#     else:
#         srclines = dbcode

#     for idx, l in zip(range(len(lst)), lst):
#         if bool(l.strip()) and l.strip() in srclines:

#             numindent = len(l) - len(l.strip())
#             dbcodes = "dbprintinsert("
#             count = 1
#             for c in codes:
#                 if count == len(codes):
#                     dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
#                 else:
#                     dbcodes = dbcodes + '"' + c + '"' + ","
#                 count = count + 1

#             if onedbprint == False:
#                 dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
#                 dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
#                 dbsrc = dbsrc + l + '\n'
#                 onedbprint = True
#             else:
#                 dbsrc = dbsrc + l + '\n'

#         elif bool(l.strip()) and idx + 1 == len(lst):
#             dbsrc = dbsrc + l

#         elif bool(l.strip()): # make sure pure indentation + \n is ignored
#             dbsrc = dbsrc + l + '\n'

#     if showdbsrc: # added to debug
#         for l in dbsrc.split('\n'):
#             print(l)
    
#     exec(dbsrc, globals().update(env)) # make sure b can access lst from above

#     env.update(locals())

#     return locals()[defaults.orisrc.__name__]

```

### no more update for dbprint, for the latest see Fastdb.dbprint


### use dbprint to override the original official code without changing its own pyfile

<!-- #region -->
see the example [here](./examples/dbprint.ipynb#make-inspect.signature-to-run-our-dbsrc-code)

```python
sfc = dbprint(_signature_from_callable, "if isinstance(obj, type):", "this is comment", \
              "locals()", "isinstance(obj, type)", "env=g", \
              expand=1, env=g)

# overriding the original official source with our dbsrc, even though rewriting _signature_from_callable inside inspect.py ######################
inspect._signature_from_callable = _signature_from_callable
inspect.signature(Foo) 

```
<!-- #endregion -->

## dbprintinsert


### Run and display the inserted dbcodes 
for each srcline under investigation, used inside dbprint

```python

def dbprintinsert(*codes, **env): 
    for c in codes:
    # print(f"{c} => {c} : {eval(c, globals().update(env))}") 
        output = f"{c} => {c} : {eval(c, globals().update(env))}"
        print('{:>157}'.format(output))   
        
```

```python
def foo(a):
    a = a*2
    b = a + 1
    c = a * b
    return c
```

```python
bool("   ".strip())
```

```python
g = globals()
```

```python
# dbprint(foo, "b = a + 1", "comment", "a", "a + 1", env=g)
```

```python
# dbprint(foo, "b = a + 1", "comment", "a", "a + 1", env=g)
```

```python
foo(3) # 
```

```python
# dbprint(foo, "c = a * b", "comment", "b", "b * a", expand=3)
```

### use locals() inside the dbsrc code to avoid adding env individually

```python

def dbprintinsert(*codes, env={}): 
    for c in codes:
    # print(f"{c} => {c} : {eval(c, globals().update(env))}") 
        output = f"{c} => {c} : {eval(c, globals().update(env))}"
        print('{:>157}'.format(output))   
        
```

### enable dbprintinsert to do exec on a block of code

```python
#| export
import ast
```

```python
#| export
def dbprintinsert(*codes, env={}): 

        
    # trial and error version for real code, still not quite why globals vs locals work in exec and eval
    for c in codes:
        print("\n")
        
        # handle a block of code
        if "\n" in c: 
            output = f"Running the code block above => "
            print('{:<157}'.format(c))
            print()
            print('{:=<100}'.format(output)) 
            print()
            block = ast.parse(c, mode='exec')
            exec(compile(block, '<string>', mode='exec'), globals().update(env))
        
        # handle assignment:  1. when no if only =; 2. when = occur before if;
        elif ("=" in c and "if" not in c) or ("=" in c and c.find("=") < c.find("if")): 
            
            # print('k' in locals())
            exec(c, globals().update(env)) 
            # print('k' in locals())
            variable = c.partition(" = ")[0]
            # print(f"{c} => {variable}: {eval(variable)}")
            output = f"{c} => {variable}: {eval(variable)}"
            print('{:>157}'.format(output))       
            
        # handle if statement
        # Note: do insert code like this : `if abc == def: print(abc)`, print is a must
        elif "if" in c: 
            cond = re.search('if (.*?):', c).group(1)
            
            # when code in string is like 'if abc == def:'
            if c.endswith(':'):
                
                # print ... 
                # print(f"{c} => {cond}: {eval(cond)}")      
                output = f"{c} => {cond}: {eval(cond)}"
                print('{:>157}'.format(output))
                
            # when code in string is like 'if abc == def: print(...)'
            else: 
                # if the cond is true, then print ...
                if eval(cond):
                    
                    # "if abc == def: print(abc)".split(': ', 2)[1] to get 'print(abc)'
                    printc = c.split(': ', 1)[1]
                    # print(f"{c} => {printc} : ")
                    output = f"{c} => {printc} : "
                    print('{:>157}'.format(output))      
                    exec(c, globals().update(env))
                    
                # if cond is false, then print ...
                else: 
                    # print(f"{c} => {cond}: {eval(cond)}")
                    output = f"{c} => {cond}: {eval(cond)}"
                    print('{:>157}'.format(output))   
                
                
        # handle for in statement
        elif "for " in c and " in " in c:           
            
            # in example like 'for k, v in abc:' or `for k, v in abc: print(...)`, if abc is empty
            # step 1: access abc
            # get the substring between 'in ' and ':', which is like 'abc'
            abc = re.search('in (.*?):', c).group(1)
            # if abc is empty dict or list: print and pass
            if not bool(eval(abc)): 
                # print(f'{c} => {abc} is an emtpy {type(eval(abc))}')
                output = f'{c} => {abc} is an emtpy {type(eval(abc))}'
                print('{:>157}'.format(output))   
                continue 
                # The break statement can be used if you need to break out of a for or while loop and move onto the next section of code.
                # The continue statement can be used if you need to skip the current iteration of a for or while loop and move onto the next iteration.
            
            # if the code in string is like 'for k, v in abc:', there is no more code after `:`
            if c.endswith(':'):
                
                # get the substring between 'for ' and ' in', which is like 'k, v'
                variables = re.search('for (.*?) in', c).group(1)
                
                # if variables has a substring like ', ' inside
                if (',') in variables: 
                    
                    # split it by ', ' into a list of substrings
                    vl = variables.split(',')
                    key = vl[0]
                    value = vl[1]
                    
                    # make sure key and value will get evaluated first before exec run
                    # printc is for exec to run
                    printc = "print(f'{key}:{eval(key)}, {type(eval(key))} ; {value}:{eval(value)}, {type(eval(value))}')" 
                    # printmsg is for reader to understand with ease
                    printmsg = "print(f'key: {key}, {type(key)} ; value: {value}, {type(value)}')"
                    c1 = c + " " + printc
                    # print(f"{c} => {printmsg} : ")      
                    output = f"{c} => {printmsg} : "
                    print('{:>157}'.format(output))   
                    exec(c1, globals().update(env))
                
                else:
                    printc = "print(f'{variables} : {eval(variables)}')"
                    printmsg = "print(f'i : {variables}')"
                    c1 = c + " " + printc
                    # print(f"{c} => {printmsg} : ")     
                    output = f"{c} => {printmsg} : "
                    print('{:>157}'.format(output))   
                    exec(c1, globals().update(env))
                    
            # if the code in string is like 'for k, v in abc: print(abc)'
            else:                 
                # "for k, v in abc: print(k)".split(': ', 1)[1] to get 'print(k)'
                printc = c.split(': ', 1)[1]
                # print(f"{c} => {printc} : ")
                output = f"{c} => {printc} : "
                print('{:>157}'.format(output))   
                exec(c, globals().update(env)) # we can't use eval to run `for in` loop, but exec can.
            ### Note: we shall not use the expression like `for k, v in abc print(abc)`
            ### Note: we shall not use the expression like `for k, v in abc if k == def`
        
        
        # handle evaluation
        else: 
            print('{:>157}'.format(f"{c} => {c} : {eval(c, globals().update(env))}"))   
            
        # the benefit of using global().update(env) is 
        # to ensure we don't need to include the same env fo
```

```python
g = globals()
```

```python
# foo1 = dbprint(foo, "b = a + 1", "comment", "a", "a + 1")
# foo1(3)
```

```python
# foo1 = dbprint(foo, "b = a + 1", "comment", "a", "a + 1", "pprint(a)", "if a > 2:\\n    print('I am here')")
# foo1(3)
```

```python
# foo1 = dbprint(foo, "b = a + 1", "comment", "for i in range(3):\\n    print(f'I am {i}')")
# foo1(3)
```

```python
# foo1 = dbprint(foo, "b = a + 1", "comment", "for i in range(3): print(f'I am {i}')")
# foo1(3)
```

```python
# foo1 = dbprint(foo, "b = a + 1", "comment", "pprint(locals())")
# foo1(3)
```

```python
# foo1 = dbprint(foo, "b = a + 1", "comment", "pprint(locals(), width=157)") # width=157 can cause error
# foo1(3)
```

```python
from fastcore.meta import delegates
```

```python
ls = inspect.getsource(delegates).split('\n')
ls = ls[:-1]
ls
```

## printrunsrclines() 

It can print out only srclines which actually ran


### Examples


#### simple example

```python
# def foo(a):
#     if a > 1:
#         a = 1 + 1
#     else:
#         b = a + 1
```

#### complex example

```python
def foo(a):
    "this is docs"
    # this is a pure comment
    if a > 1:
        for i in range(3):
            a = i + 1
    else:
        "this is docs"
        b = a + 1
    "this is docs"
    return a
        
```

```python
foo(3)
```

### insert a line after each srcline to add idx

```python
srclines = inspect.getsource(foo).split('\n')
dbsrc = ""

for idx, l in zip(range(len(srclines)), srclines):
    # if "if" in l or "else" in l or "for" in l:
        
    dbsrc = dbsrc + l + f"\n    srcidx.append({idx})\n" # add srcidx.append(idx) to each line
```

```python
for l in dbsrc.split('\n'):
    print(l)
```

### add correct indentation to each inserted line


#### count the indentation for each srcline

```python
len("    a = 1") - len("    a = 1".strip())
```

```python
srclines = inspect.getsource(foo).split('\n')
dbsrc = ""

for idx, l in zip(range(len(srclines)), srclines):
    numindent = len(l) - len(l.strip())
    addline = f"srcidx.append({idx})"
    dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"   # add srcidx.append(idx) to each line
```

```python
for l in dbsrc.split('\n'):
    print(l)
```

### indentation special case: if, else, for, def

```python
srclines = inspect.getsource(foo).split('\n')
dbsrc = ""
indent = 4

for idx, l in zip(range(len(srclines)), srclines):
    numindent = len(l) - len(l.strip())
    addline = f"srcidx.append({idx})"

    if "if" in l or "else" in l or "for" in l or "def" in l:
        numindent = numindent + indent
    
    dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line
```

```python
for l in dbsrc.split('\n'):
    print(l)
```

### remove pure comments or docs from dbsrc
Do not insert line for pure comment or pure "\n"

```python
from pprint import pprint
```

```python
for l in srclines:
    pprint(l)
```

```python
"# this is a comment".startswith("#")
```

```python
"a = 1 # this is comment".startswith("#")
```

```python
srclines = inspect.getsource(foo).split('\n')
dbsrc = ""
indent = 4

for idx, l in zip(range(len(srclines)), srclines):
    numindent = len(l) - len(l.strip())
    addline = f"srcidx.append({idx})"

    if "if" in l or "else" in l or "for" in l or "def" in l:
        numindent = numindent + indent
    
    if bool(l): # ignore pure '\n'
        dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line
```

```python
for l in dbsrc.split('\n'):
    print(l)
```

```python
srclines = inspect.getsource(foo).split('\n')
dbsrc = ""
indent = 4

for idx, l in zip(range(len(srclines)), srclines):
    numindent = len(l) - len(l.strip())
    addline = f"srcidx.append({idx})"

    if "if" in l or "else" in l or "for" in l or "def" in l:
        numindent = numindent + indent
    
    if bool(l) and not l.strip().startswith('#') and not (l.strip().startswith('"') and l.strip().endswith('"')): # ignore/remove pure quotations or docs
        dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line
```

```python
for l in dbsrc.split('\n'): # now the dbsrc has no pure comment and pure docs
    print(l)
```

```python
foo??
```

```python
exec(dbsrc) # give life to dbsrc
```

```python
foo??
```

```python
srcidx = [] #used outside the srcode
```

```python
foo(3) # run the example using dbsrc
# foo(-1) # run the example using dbsrc
srcidx # Now it should have all the idx whose srclines have run
```

### print out the srclines which get run

```python
for idx, l in zip(range(len(srclines)), srclines):
    if idx in srcidx:
        print(l)
```

### Make sure all if, else, for get printed

```python
for idx, l in zip(range(len(srclines)), srclines):
    if idx in srcidx or "for" in l or "if" in l or "else" in l:
        print(l)
```

### Put all together into the function printrunsrclines()

```python
def foo(a):
    "this is docs"
    # this is a pure comment
    if a > 1:
        for i in range(3):
            a = i + 1
    else:
        "this is docs"
        b = a + 1
    "this is docs"
    return a
        
```

```python
def printrunsrclines(func):
    srclines = inspect.getsource(func).split('\n')
    dbsrc = ""
    indent = 4

    for idx, l in zip(range(len(srclines)), srclines):
        numindent = len(l) - len(l.strip())
        addline = f"srcidx.append({idx})"

        if "if" in l or "else" in l or "for" in l or "def" in l:
            numindent = numindent + indent

        if bool(l) and not l.strip().startswith('#') and not (l.strip().startswith('"') and l.strip().endswith('"')): # ignore/remove pure quotations or docs
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line
    
    srcidx = [] 
    exec(dbsrc, globals().update(locals()))
    fool = locals()['foo']
    pprint(fool(3))
    pprint(locals())

     
    # run = "foo(3)"
    exec("fool(3)")
    print(srcidx)

    for idx, l in zip(range(len(srclines)), srclines):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)
```

```python
printrunsrclines(foo)
```

#### no more renaming of foo

```python
def printrunsrclines(func):
    srclines = inspect.getsource(func).split('\n')
    dbsrc = ""
    indent = 4

    for idx, l in zip(range(len(srclines)), srclines):
        numindent = len(l) - len(l.strip())
        addline = f"srcidx.append({idx})"

        if "if" in l or "else" in l or "for" in l or "def" in l:
            numindent = numindent + indent

        if bool(l) and not l.strip().startswith('#') and not (l.strip().startswith('"') and l.strip().endswith('"')): # ignore/remove pure quotations or docs
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line
    
    srcidx = [] 
    exec(dbsrc, globals().update(locals()))    
    exec("foo(3)") # now we can use foo as the new foo 
    print(srcidx)

    for idx, l in zip(range(len(srclines)), srclines):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)
```

```python
def foo(a):

    if a > 1:
        for i in range(3):
            a = i + 1
    else:
        b = a + 1
    return a
```

```python
printrunsrclines(foo)
```

```python

```

#### add example as a param into the function

```python

def printrunsrclines(func, example):
    srclines = inspect.getsource(func).split('\n')
    dbsrc = ""
    indent = 4

    for idx, l in zip(range(len(srclines)), srclines):
        numindent = len(l) - len(l.strip())
        addline = f"srcidx.append({idx})"

        if "if" in l or "else" in l or "for" in l or "def" in l:
            numindent = numindent + indent

        if bool(l) and not l.strip().startswith('#') and not (l.strip().startswith('"') and l.strip().endswith('"')): # ignore/remove pure quotations or docs
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line

    pprint(dbsrc)
    srcidx = [] 
    exec(dbsrc, globals().update(locals()))    
    exec(example) # now we can use foo as the new foo 
    print(srcidx)

    for idx, l in zip(range(len(srclines)), srclines):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)
```

```python
printrunsrclines(foo, "foo(-1)")
```

```python
printrunsrclines(foo, "foo(2)")
```

#### improve on search for `if`, else, for, def to avoid errors for more examples

```python

def printrunsrclines(func, example):
    srclines = inspect.getsource(func).split('\n')
    dbsrc = ""
    indent = 4

    for idx, l in zip(range(len(srclines)), srclines):
        numindent = len(l) - len(l.strip())
        addline = f"srcidx.append({idx})"

        if "if " in l or "else:" in l or "for " in l or "def " in l:
            numindent = numindent + indent

        if bool(l) and not l.strip().startswith('#') \
        and not (l.strip().startswith('"') and l.strip().endswith('"')): # ignore/remove pure quotations or docs
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line

    pprint(dbsrc)
    srcidx = [] 
    exec(dbsrc, globals().update(locals()))    
    exec(example) # now we can use foo as the new foo 
    print(srcidx)

    for idx, l in zip(range(len(srclines)), srclines):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)
```

```python
printrunsrclines(alignright, 'alignright("this is me")')
```

#### remove an empty line with indentation

```python
lst = """
this is code\n\
     \n\
this is code
""".split('\n')
print(lst)
for l in lst:
    print(bool(l.strip()))
```

```python

def printrunsrclines(func, example):
    srclines = inspect.getsource(func).split('\n')
    dbsrc = ""
    indent = 4

    for idx, l in zip(range(len(srclines)), srclines):
        numindent = len(l) - len(l.strip()) # how to strip only the left not the right?????
        addline = f"srcidx.append({idx})"

        if "if " in l or "else:" in l or "for " in l or "def " in l:
            numindent = numindent + indent

        if bool(l.strip()) and not l.strip().startswith('#') \
        and not (l.strip().startswith('"') and l.strip().endswith('"')): 
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line

    pprint(dbsrc)
    srcidx = [] 
    exec(dbsrc, globals().update(locals()))    
    exec(example) # now we can use foo as the new foo 
    print(srcidx)

    for idx, l in zip(range(len(srclines)), srclines):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)
```

```python
printrunsrclines(alignright, 'alignright("this is me")')
```

```python
pprint(inspect.getsource(printsrc))
```

```python
", " in "    env, ".strip()
```

### make it work

```python
a=[]
a.append([1,2])
a
```

```python

def printrunsrclines(src, example, env):
    srclst = inspect.getsource(src).split('\n')
    dbsrc = ""
    indent = 4
    bracketidx = []
    bracketindent = 0
    ifelseidx = []
    ifelseindent = 0
    
    for idx, l in zip(range(len(srclst)), srclst):
        numindent = len(l) - len(l.strip()) # how to strip only the left not the right?????
        addline = f"srcidx.append({idx})"

        if ("if " in l and l.strip().endswith(':')) or l.strip().endswith("else:") or ("elif " in l and l.strip().endswith(':')) \
        or  (l.strip().endswith(':') and "for " in l) or ("def " in l and l.strip().endswith(':')):
            numindent = numindent + indent

            
        
        if l.strip().startswith('"""'):
            dbsrc = dbsrc + l + '\n'
        elif srclst[idx - 1].strip().startswith('"""') and '"""' not in l.strip():
            dbsrc = dbsrc + l + '\n'
        elif idx <= len(srclst) - 2 and srclst[idx + 1].strip().startswith('"""') and '"""' not in l.strip():
            dbsrc = dbsrc + l + '\n'
            
        elif "{" in l and "}" not in l:
            bracketidx.append(idx)
            bracketindent = len(l) - len(l.strip())
            dbsrc = dbsrc + l + '\n'
        elif "}" in l and "{" not in l: 
            bracketidx.append(idx)
            addup = ""
            for i in bracketidx:
                line = f"srcidx.append({i})"
                addup = addup + " "*bracketindent + line + "\n"
            dbsrc = dbsrc + l + "\n" + addup
            
        elif (l.strip().startswith("if") or l.strip().startswith("elif")) and ":" in l and not l.strip().endswith(":") and ": #" not in l \
        and ("elif" in srclst[idx + 1] or "else" in srclst[idx + 1]):
            ifelseidx.append(idx)
            ifelseindent = len(l) - len(l.strip())
            dbsrc = dbsrc + l + '\n'
        elif l.strip().startswith("else") and ":" in l and not l.strip().endswith(":") and ": #" not in l:
            ifelseidx.append(idx)
            addup = ""
            for i in ifelseidx:
                line = f"srcidx.append({i})"
                addup = addup + " "*ifelseindent + line + "\n"
            dbsrc = dbsrc + l + "\n" + addup            
            
            
        elif bool(l.strip()) and not l.strip().startswith('#') \
        and not (l.strip().startswith('"') and l.strip().endswith('"')) \
        and not (l.strip().endswith(',') or ', #' in l or '): #' in l): 
            dbsrc = dbsrc + l + "\n" + " "*numindent + addline + "\n"  # add srcidx.append(idx) to each line
        else: 
            dbsrc = dbsrc + l + '\n'            

    pprint(dbsrc, width=157)
    srcidx = [] 
    exec(dbsrc, globals().update(env), locals())    
    exec(example) # now we can use foo as the new foo 
    print(srcidx)

    # pprint(srclines)
    for idx, l in zip(range(len(srclst)), srclst):
        if idx in srcidx or "for" in l or "if" in l or "else" in l:
            print(l)
```

```python
from fastcore.meta import *
from fastcore.imports import *
```

```python
g = globals()
```

```python
printrunsrclines(delegates, '', env=g) # make sure to use \\n not \n
```

```python
# printsrcwithidx(delegates)
```

### more difficult examples to test printrunsrc()

```python

```

## Make fastdebug a class

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2): # span 2 lines of srcode up and down from the srcline investigated
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):
            if bool(l.strip()) and l.strip() in srclines:

                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes) and "env=" in c:
                        dbcodes = dbcodes + c + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                if onedbprint == False:
                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                    dbsrc = dbsrc + l + '\n'
                    onedbprint = True
                else:
                    dbsrc = dbsrc + l + '\n'

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'

        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def printsrcwithidx(self, 
                        maxlines:int=33, # maximum num of lines per page
                        part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33

        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2): # span 2 lines of srcode up and down from the srcline investigated
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):
            if bool(l.strip()) and l.strip() in srclines:

                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                if onedbprint == False:
                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                    dbsrc = dbsrc + l + '\n'
                    onedbprint = True
                else:
                    dbsrc = dbsrc + l + '\n'

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'

        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def printsrcwithidx(self, 
                        maxlines:int=33, # maximum num of lines per page
                        part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33

        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### improve on the line idx readability

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2): # span 2 lines of srcode up and down from the srcline investigated
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):
            if bool(l.strip()) and l.strip() in srclines:

                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                if onedbprint == False:
                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                    dbsrc = dbsrc + l + '\n'
                    onedbprint = True
                else:
                    dbsrc = dbsrc + l + '\n'

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'

        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def printsrcwithidx(self, 
                        maxlines:int=33, # maximum num of lines per page
                        part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33

        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                else:
                    print('{:=<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    if not bool(l.strip()):
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    elif lenl + lspace >= 100:
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else:
                        print('{:=<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### collect cmt from dbprint and print

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2): # span 2 lines of srcode up and down from the srcline investigated
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):
            if bool(l.strip()) and l.strip() in srclines:

                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                if onedbprint == False:
                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                    dbsrc = dbsrc + l + '\n'
                    onedbprint = True
                else:
                    dbsrc = dbsrc + l + '\n'

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'

                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def printsrcwithidx(self, 
                        maxlines:int=33, # maximum num of lines per page
                        part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
        cmts = self.cmts
        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        
                else:


                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    if not bool(l.strip()):
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    elif lenl + lspace >= 100:
                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                            else:
                                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        else: 
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                    else:

                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                            else:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          
                            
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### make sure only the line with correct idx is debugged

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
                showdbsrc=False): # display dbsrc or not
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):
            # make sure the line with correct idx is debugged
            if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                # if onedbprint == False:
                #     dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                #     dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                #     dbsrc = dbsrc + l + '\n'
                #     onedbprint = True
                # else:
                #     dbsrc = dbsrc + l + '\n'
                
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'                

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'
                
        if showdbsrc: # added to debug
            for l in dbsrc.split('\n'):
                print(l)
                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def printsrcwithidx(self, 
                        maxlines:int=33, # maximum num of lines per page
                        part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
        cmts = self.cmts
        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        
                else:


                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    if not bool(l.strip()):
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    elif lenl + lspace >= 100:
                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                            else:
                                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        else: 
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                    else:

                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                            else:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          
                            
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### having "" or "   " inside codes without causing error

```python
bool(["   "][0].strip())
```

```python
bool("   ".strip())
```

```python
lst = ["...", "", "  ", ""]
newlst = []
for i in lst: 
    if bool(i.strip()): newlst.append(i)
newlst        
```

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
                showdbsrc=False): # display dbsrc or not
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]
        
        newlst = []
        for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
            if bool(i.strip()): newlst.append(i)
        codes = newlst

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):

            if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

                if len(codes) > 0: # if the new codes is not empty
                    numindent = len(l) - len(l.strip())
                    dbcodes = "dbprintinsert("
                    count = 1
                    for c in codes:
                        if count == len(codes):
                            dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                        else:
                            dbcodes = dbcodes + '"' + c + '"' + ","
                        count = count + 1

                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'

                else:
                    dbsrc = dbsrc + l + '\n'                

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'
                
        if showdbsrc: # added to debug
            for l in dbsrc.split('\n'):
                print(l)
                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def printsrcwithidx(self, 
                        maxlines:int=33, # maximum num of lines per page
                        part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
        cmts = self.cmts
        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        
                else:


                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    if not bool(l.strip()):
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    elif lenl + lspace >= 100:
                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                            else:
                                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        else: 
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                    else:

                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                            else:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          
                            
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### replace Fastdb.printsrcwithdix with Fastdb.print

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
                showdbsrc=False): # display dbsrc or not
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]
        
        newlst = []
        for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
            if bool(i.strip()): newlst.append(i)
        codes = newlst

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):

            if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

                if len(codes) > 0: # if the new codes is not empty
                    numindent = len(l) - len(l.strip())
                    dbcodes = "dbprintinsert("
                    count = 1
                    for c in codes:
                        if count == len(codes):
                            dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                        else:
                            dbcodes = dbcodes + '"' + c + '"' + ","
                        count = count + 1

                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                    dbsrc = dbsrc + l + '\n'     
                else:
                    dbsrc = dbsrc + l + '\n'                

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'
                
        if showdbsrc: # added to debug
            for l in dbsrc.split('\n'):
                print(l)
                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def print(self, 
                maxlines:int=33, # maximum num of lines per page
                part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
        cmts = self.cmts
        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        
                else:


                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    if not bool(l.strip()):
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    elif lenl + lspace >= 100:
                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                            else:
                                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        else: 
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                    else:

                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                            else:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          
                            
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### add idx to dbsrc when showdbsrc=True

```python

class Fastdb():
    
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode, # the srclines under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of dbcodes
                expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
                showdbsrc=False): # display dbsrc or not
        "Insert dbcodes under srclines under investigation, and create a new dbsrc function to replace the official one"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]
        
        newlst = []
        for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
            if bool(i.strip()): newlst.append(i)
        codes = newlst

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):

            if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

                if len(codes) > 0: # if the new codes is not empty
                    numindent = len(l) - len(l.strip())
                    dbcodes = "dbprintinsert("
                    count = 1
                    for c in codes:
                        if count == len(codes):
                            dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                        else:
                            dbcodes = dbcodes + '"' + c + '"' + ","
                        count = count + 1

                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                    dbsrc = dbsrc + l + '\n'     
                else:
                    dbsrc = dbsrc + l + '\n'                

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'
                
        if showdbsrc: # added to debug
            totallen = 157
            lenidx = 5
            dblst = dbsrc.split('\n')
            for idx, l in zip(range(len(dblst)), dblst):
                lenl = len(l)
                if "dbprintinsert" in l: 
                    print(l + "="*(totallen-lenl-lenidx) + "(db)")
                else:
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        self.outenv.update(locals())

        return locals()[self.orisrc.__name__]
    
    
    def print(self, 
                maxlines:int=33, # maximum num of lines per page
                part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
        cmts = self.cmts
        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        
                else:


                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    if not bool(l.strip()):
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    elif lenl + lspace >= 100:
                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                            else:
                                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        else: 
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                    else:

                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                            else:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          
                            
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### not load the inner locals() to outenv can prevent mysterious printing of previous db messages

```python

class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code
                 env): # env = g, as g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}

        
    def dbprint(self, 
                # src, # the src func name, e.g., foo
                dbcode:int, # a srcline under investigation, can be either string or int
                cmt:str, # comment
                *codes, # a list of expressions (str) you write to be evaluated above the srcline
                expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
                showdbsrc:bool=False): # display dbsrc
        "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code you are investigating"

        src = self.orisrc
        if type(dbcode) == int: self.cmts.update({dbcode: cmt})

        printsrc(src, dbcode, cmt, expand)

        dbsrc = ""
        indent = 4
        onedbprint = False

        lst = inspect.getsource(src).split('\n')
        if not bool(lst[-1]): lst = lst[:-1]
        
        newlst = []
        for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
            if bool(i.strip()): newlst.append(i)
        codes = newlst

        srclines = ""
        if type(dbcode) == int:
            srclines = lst[dbcode]
        else:
            srclines = dbcode

        for idx, l in zip(range(len(lst)), lst):

            if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

                if len(codes) > 0: # if the new codes is not empty
                    numindent = len(l) - len(l.strip())
                    dbcodes = "dbprintinsert("
                    count = 1
                    for c in codes:
                        if count == len(codes):
                            dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                        else:
                            dbcodes = dbcodes + '"' + c + '"' + ","
                        count = count + 1

                    dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                    dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                    dbsrc = dbsrc + l + '\n'     
                else:
                    dbsrc = dbsrc + l + '\n'                

            elif bool(l.strip()) and idx + 1 == len(lst):
                dbsrc = dbsrc + l

            elif bool(l.strip()): # make sure pure indentation + \n is ignored
                dbsrc = dbsrc + l + '\n'
                
        if showdbsrc: # added to debug
            totallen = 157
            lenidx = 5
            dblst = dbsrc.split('\n')
            for idx, l in zip(range(len(dblst)), dblst):
                lenl = len(l)
                if "dbprintinsert" in l: 
                    print(l + "="*(totallen-lenl-lenidx) + "(db)")
                else:
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

        # self.outenv.update(locals())
        # self.outenv.update({self.orisrc.__name__: locals()[self.orisrc.__name__]})

        return locals()[self.orisrc.__name__]
    
    
    def print(self, 
                maxlines:int=33, # maximum num of lines per page
                part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
        "Print the source code in whole or parts with idx and comments you added with dbprint along the way."
        
        totallen = 157
        lenidx = 5
        lspace = 10
        lstsrc = inspect.getsource(self.orisrc).split('\n')
        numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
        # cmts = {5:"this is me", 111:"this is me", 14:"this is you this is you this is you this is you this is you this is you this is you this is you "}
        cmts = self.cmts
        if part == 0: 
            for idx, l in zip(range(len(lstsrc)), lstsrc):
                lenl = len(l)
                
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        
                else:


                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 
        
        for p in range(numparts):
            for idx, l in zip(range(len(lstsrc)), lstsrc):

                if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                    lenl = len(l)
                    if not bool(l.strip()):
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    elif lenl + lspace >= 100:
                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                            else:
                                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                        else: 
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                    else:

                        if bool(cmts):
                            cmtidx = [cmt[0] for cmt in list(cmts.items())]
                            if idx in cmtidx:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                            else:
                                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          
                            
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

                if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                    print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                    return
```

### using @patch to enable docs for instance methods like `dbprint` and `print`

```python

class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code you are exploring
                 env): # env variables needed for exploring the source code, e.g., g = globals()
        self.orisrc = src
        self.margin = 157
        self.outenv = env
        self.cmts = {}
```

### move param env into `__init__`

```python

class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code you are exploring
                 db=False): # db=True will run some debugging prints
        self.orisrc = src
        self.margin = 157
        self.outenv = src.__globals__
        self.cmts = {}
        if db:
            print(f"self.orisrc: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
```

### Add example to the obj

```python

class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code you are exploring
                 db=False): # db=True will run some debugging prints
        self.orisrc = src # important: it is making a real copy
        self.idxsrc = None # the idx of srcline under investigation
        self.margin = 157
        self.outenv = src.__globals__
        self.cmts = {}
        self.egsidx = {}
        self.eg = None # add example in string format
        self.egEnv = None # add example env in dict
        if db:
            print(f"self.orisrc: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
```

```python
# pprint(type(reliveonce))
```

### Take not only function but also class

```python
inspect.isfunction
```

```python
class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code you are exploring
                 db=False, # db=True will run some debugging prints
                 **env): # adding env variables
        self.orisrc = src # important: it is making a real copy
        self.dbsrc = None # store dbsrc func
        self.dbsrcstr = None # store dbsrc string
        self.idxsrc = None # the idx of srcline under investigation
        self.margin = 157
        if inspect.isfunction(src):
            self.outenv = src.__globals__
        elif type(src) == type:
            exec(f"import {src.__module__}")
            self.outenv = eval(src.__module__ + ".__dict__")
#             self.outenv = env # this approach works
        self.cmts = {}
        self.egsidx = {}
        self.eg = None # add example in string format
        self.egEnv = None # add example env in dict
        if db:
            print(f"self.orisrc: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
```

### To remove the necessity of self.takExample()

```python
#| export
class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code you are exploring
                 db=False, # db=True will run some debugging prints
                 outloc=None): # outloc = g; g = locals() from the outer cell
        self.orisrc = src # important: it is making a real copy
        self.dbsrc = None # store dbsrc func
        self.dbsrcstr = None # store dbsrc string
        self.idxsrc = None # the idx of srcline under investigation
        self.margin = 157
        if inspect.isfunction(src):
            self.outenv = src.__globals__
#         elif type(src) == type:
        elif inspect.isclass(src):
            exec(f"import {src.__module__}")
            self.outenv = eval(src.__module__ + ".__dict__")
#             self.outenv = env # this approach works
        self.cmts = {}
        self.egsidx = {}
        self.orieg = ""
        self.eg = "" # add example in string format
        self.egEnv = outloc # no need to use self.takExample()
        if db:
            print(f"self.orisrc: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
```

```python

```

```python

class Fastdb():
    "Create a Fastdebug class which has two functionalities: dbprint and print."
    def __init__(self, 
                 src, # name of src code you are exploring
                 db=False, # db=True will run some debugging prints
                 outloc=None): # outloc = g; g = locals() from the outer cell
        self.orisrc = src # important: it is making a real copy
        self.dbsrc = None # store dbsrc func
        self.dbsrcstr = None # store dbsrc string
        self.idxsrc = None # the idx of srcline under investigation
        self.margin = 157
        if inspect.isfunction(src):
            self.outenv = src.__globals__
#         elif type(src) == type:
        elif inspect.isclass(src):
            if inspect.isfunction(src.__new__):
                self.outenv = src.__new__.__globals__
            elif inspect.isfunction(src.__init__):
                self.outenv = src.__init__.__globals__
            elif inspect.isfunction(src.__call__):
                self.outenv = src.__call__.__globals__                
#             self.outenv = env # this approach works
        self.cmts = {}
        self.egsidx = {}
        self.orieg = ""
        self.eg = "" # add example in string format
        self.egEnv = outloc # no need to use self.takExample()
        if db:
            print(f"self.orisrc: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
```

```python
from fastcore.meta import *
import inspect
```

```python
hasattr(FixSigMeta, '__new__')
```

```python
inspect.isfunction(PrePostInitMeta.__call__)
```

```python
 
@patch
def dbprint(self:Fastdb, 
            dbcode:int, # a srcline under investigation, can be either string or int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code you are investigating"

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode:

            if len(codes) > 0: # if the new codes is not empty
                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
            if "dbprintinsert" in l:
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above

    return locals()[self.orisrc.__name__]

```

### Make sure `showdbsrc=True` give us the line starting with 'dbprintinsert'

```python
     
@patch
def dbprint(self:Fastdb, 
            dbcode:int, # a srcline under investigation, can be either string or int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code you are investigating"

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

            if len(codes) > 0: # if the new codes is not empty
                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"before exec, locals(): {list(locals().keys())}")

    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    if showdbsrc: 
        print(f"after exec, locals(): {list(locals().keys())}")
        print(f"type(locals()[self.orisrc.__name__]): {type(locals()[self.orisrc.__name__])}")
    return locals()[self.orisrc.__name__]

```

### Make sure `showdbsrc=True` give us info on changes in g or outenv

```python

@patch
def dbprint(self:Fastdb, 
            dbcode:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code you are investigating"

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        srclines = dbcode

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode: 

            if len(codes) > 0: # if the new codes is not empty
                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"before exec, locals(): {list(locals().keys())}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
        print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint}")
    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    if showdbsrc: 
        print(f"after exec, locals(): {list(locals().keys())}")
        print(f"type(locals()[self.orisrc.__name__]): {type(locals()[self.orisrc.__name__])}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
        print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint}")

    return locals()[self.orisrc.__name__]

```

### exit and print a warning message: idx has to be int

```python
    
@patch
def dbprint(self:Fastdb, 
            dbcode:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code you are investigating"

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        colwarn = colorize("Warning!", color="r")
        colmsg = colorize(" param decode has to be an int as idx.", color="y")
        print(colwarn + colmsg)
#         srclines = dbcode
        return

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode:

            if len(codes) > 0: # if the new codes is not empty
                numindent = len(l) - len(l.strip())
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"before exec, locals(): {list(locals().keys())}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
        print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint}")
    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    if showdbsrc: 
        print(f"after exec, locals(): {list(locals().keys())}")
        print(f"type(locals()[self.orisrc.__name__]): {type(locals()[self.orisrc.__name__])}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
        print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint}")

    return locals()[self.orisrc.__name__]

```

### handle errors by codes with trailing spaces 

```python
    
@patch
def dbprint(self:Fastdb, 
            dbcode:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code you are investigating"

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        colwarn = colorize("Warning!", color="r")
        colmsg = colorize(" param decode has to be an int as idx.", color="y")
        print(colwarn + colmsg)
#         srclines = dbcode
        return

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"before exec, locals(): {list(locals().keys())}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
        print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint}")
    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    if showdbsrc: 
        print(f"after exec, locals(): {list(locals().keys())}")
        print(f"type(locals()[self.orisrc.__name__]): {type(locals()[self.orisrc.__name__])}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
        print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint}")

    return locals()[self.orisrc.__name__]

```

### showdbsrc=True, check whether Fastdb.dbprint and fdb.dbprint are same object using `is`

```python

@patch
def dbprint(self:Fastdb, 
            dbcode:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code you are investigating"

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)

    dbsrc = ""
    indent = 4
    onedbprint = False

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        colwarn = colorize("Warning!", color="r")
        colmsg = colorize(" param decode has to be an int as idx.", color="y")
        print(colwarn + colmsg)
#         srclines = dbcode
        return

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"before exec, locals(): {list(locals().keys())}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
#         print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint is Fastdb.dbprint}")
    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    if showdbsrc: 
        print(f"after exec, locals(): {list(locals().keys())}")
        print(f"type(locals()[self.orisrc.__name__]): {type(locals()[self.orisrc.__name__])}")
        print(f"Fastdb.dbprint has __code__?: {hasattr(Fastdb.dbprint, '__code__')}")
        print(f"does Fastdb.dbprint has source available?: {not inspect.getsourcefile(Fastdb.dbprint) == '<string>'}")
#         print(f"outenv[self.orisrc.__qualname__.split('.')[0]].dbprint == Fastdb.dbprint: {self.outenv[self.orisrc.__qualname__.split('.')[0]].dbprint is Fastdb.dbprint}")

    print(f'self.orisrc.__name__: {self.orisrc.__name__}')
    print(f'locals()[self.orisrc.__name__]: {locals()[self.orisrc.__name__]}')
    return locals()[self.orisrc.__name__]

```

### remove unnecessary db printout when showdbsrc=True and add printout to display sections

```python
     
@patch
def dbprint(self:Fastdb, 
            dbcode:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)
    print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        colwarn = colorize("Warning!", color="r")
        colmsg = colorize(" param decode has to be an int as idx.", color="y")
        print(colwarn + colmsg)
#         srclines = dbcode
        return

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        names = self.orisrc.__qualname__.split('.')
        clsname = names[0]
        methodname = names[1]
        print(f"before exec, is {methodname} in locals(): {methodname in locals()}")
        print(f"before exec, is {clsname} in locals(): {clsname in locals()}")
        print(f"before exec, is {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
        print(f"before exec, is {methodname} in self.outenv: {methodname in self.outenv}")
        print(f"before exec, is {clsname} in self.outenv: {clsname in self.outenv}")
        print(f"before exec, is {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
        expr = "self.outenv[" + "'" + clsname + "']." + methodname
        expr1 = "self.outenv[" + "'" + methodname + "']"
        print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
        print(f"self.outenv[{methodname}]: {eval(expr1)}")
    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        print(f"after exec, is {methodname} in locals(): {methodname in locals()}")
        print(f"after exec, is {clsname} in locals(): {clsname in locals()}")
        print(f"after exec, is {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
        print(f"after exec, is {methodname} in self.outenv: {methodname in self.outenv}")
        print(f"after exec, is {clsname} in self.outenv: {clsname in self.outenv}")
        print(f"after exec, is {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
#         print(f"after exec, are {methodname} and {clsname} and {self.orisrc.__qualname__} in locals(): {[i in list(locals().keys()) for i in [self.orisrc.__name__, clsname, self.orisrc.__qualname__]]}")
#         print(f"after exec, are {methodname} and {clsname} and {self.orisrc.__qualname__} in self.outenv(): {[i in self.outenv for i in [methodname, clsname, self.orisrc.__qualname__]]}")
        print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
        print(f"self.outenv[{methodname}]: {eval(expr1)}")
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f'locals()[self.orisrc.__name__]: {locals()[self.orisrc.__name__]}')
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))
        
    return locals()[self.orisrc.__name__]

```

### raise TypeError when decode are not integer and showdbsrc=true working on both method and function

```python

@patch
def dbprint(self:Fastdb, 
            dbcode:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)
    print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l

        elif bool(l.strip()): # make sure pure indentation + \n is ignored
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            else:
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        names = self.orisrc.__qualname__.split('.')
        if len(names) == 2:
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv[{methodname}]: {eval(expr1)}")
    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print(f"after exec, is {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
    #         print(f"after exec, are {methodname} and {clsname} and {self.orisrc.__qualname__} in locals(): {[i in list(locals().keys()) for i in [self.orisrc.__name__, clsname, self.orisrc.__qualname__]]}")
    #         print(f"after exec, are {methodname} and {clsname} and {self.orisrc.__qualname__} in self.outenv(): {[i in self.outenv for i in [methodname, clsname, self.orisrc.__qualname__]]}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv[{methodname}]: {eval(expr1)}")
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f'locals()[self.orisrc.__name__]: {locals()[self.orisrc.__name__]}')
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))
        
    return locals()[self.orisrc.__name__]

```

### when debugging dbprint, make sure dbsrc is printed with the same idx as original

```python

@patch
def dbprint(self:Fastdb, 
            dbcode:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."

    src = self.orisrc
    if type(dbcode) == int: self.cmts.update({dbcode: cmt})

    printsrc(src, dbcode, cmt, expand)
    print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(dbcode) == int:
        srclines = lst[dbcode]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == dbcode:

            if len(codes) > 0: # no codes, no dbprintinsert
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
            elif not bool(l.strip()):
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        names = self.orisrc.__qualname__.split('.')
        if len(names) == 2:
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv[{methodname}]: {eval(expr1)}")
            print(f"{self.orisrc} is {expr}: {self.orisrc is eval(expr)}")
    exec(dbsrc, globals().update(self.outenv)) # make sure b can access lst from above
    print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print(f"after exec, is {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
    #         print(f"after exec, are {methodname} and {clsname} and {self.orisrc.__qualname__} in locals(): {[i in list(locals().keys()) for i in [self.orisrc.__name__, clsname, self.orisrc.__qualname__]]}")
    #         print(f"after exec, are {methodname} and {clsname} and {self.orisrc.__qualname__} in self.outenv(): {[i in self.outenv for i in [methodname, clsname, self.orisrc.__qualname__]]}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv[{methodname}]: {eval(expr1)}")
            print(f"{self.orisrc} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f'locals()[self.orisrc.__name__]: {locals()[self.orisrc.__name__]}')
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))
        
    return locals()[self.orisrc.__name__]

```

### update dbsrc to the global env


### go back to normal before running dbprint again

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
    self.goback() # refresh 
    src = self.orisrc
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: # no codes, no dbprintinsert
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx            
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

        
```

### auto print src with cmt and idx as the ending part of dbprint

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
    self.goback() # refresh 
    src = self.orisrc
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: # no codes, no dbprintinsert
                numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx            
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments
    totalines = len(inspect.getsource(self.orisrc).split('\n'))
    maxpcell = 33
    pt = idxsrc // maxpcell

    if idx > maxpcell and idx % maxpcell != 0:
        self.print(maxpcell, pt + 1)
    elif idx % maxpcell == 0:
        self.print(maxpcell, pt + 1)
    else:
        self.print(maxpcell, 1)
    
```

### to mark my explorations (expressions to evaluate) to stand out

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
    self.goback() # refresh 
    src = self.orisrc
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx            
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments
    totalines = len(inspect.getsource(self.orisrc).split('\n'))
    maxpcell = 33
    pt = idxsrc // maxpcell

    print()
    print('{:=<157}'.format(colorize("Review srcode with all comments added so far", color="y")))
    if idx > maxpcell and idx % maxpcell != 0:
        self.print(maxpcell, pt + 1)
    elif idx % maxpcell == 0:
        self.print(maxpcell, pt + 1)
    else:
        self.print(maxpcell, 1)
    print()
    
```

### Add the print of src with idx and comments at the end of dbsrc

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
    self.goback() # refresh 
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx            
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments using self.autoprint()
#     print(f"self: {self}")
    
```

### embed example and autoprint to shorten the code to type

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx            
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments using self.autoprint() and enable using whatinside without fu in front of it
    if bool(self.eg):
        self.egEnv[self.orisrc.__name__] = locals()[self.orisrc.__name__]        
#         exec(self.eg, {}, self.egEnv)
        exec("pprint(" + self.eg + ")", globals(), self.egEnv) # use globals() so that pprint can be used       
        self.autoprint()
    
    self.goback() # refresh 
```

```python

```

### Make title for dbprint

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) 
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))
    print('{:=^157}'.format(f"     with example {colorize(self.eg, color='r')}     ")) 
    print()
        
    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("dbprintinsert"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx            
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments using self.autoprint() and enable using whatinside without fu in front of it
    if bool(self.eg):
        self.egEnv[self.orisrc.__name__] = locals()[self.orisrc.__name__]        
#         exec(self.eg, {}, self.egEnv)
        exec("pprint(" + self.eg + ")", globals(), self.egEnv) # use globals() so that pprint can be used       
        self.autoprint()
    
    self.goback() # refresh 
```

### Adding self.eg info and color group into dbprint and print


#### todo: make the comments with same self.eg have the same color


### make dbsrc print idx right

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: self.cmts.update({idxsrc: cmt})

    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) 
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))
    print('{:=^157}'.format(f"     with example {colorize(self.eg, color='r')}     ")) 
    print()
        
    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            if l.strip().startswith("dbprintinsert"): 
                idxsrcline = idx           
                
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
#             if l.strip().startswith("dbprintinsert"): 
            if idx >= idxsrcline - 3 and idx <= idxsrcline + 2:
                print(l + "="*(totallen-lenl-lenidx) + "(db)")   
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline + 2:
#                     idx = idx - 1
                    idx = idx - 6
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline + 2:
                    idx = idx - 6                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments using self.autoprint() and enable using whatinside without fu in front of it
    if bool(self.eg):
        self.egEnv[self.orisrc.__name__] = locals()[self.orisrc.__name__]        
#         exec(self.eg, {}, self.egEnv)
        exec("pprint(" + self.eg + ")", globals(), self.egEnv) # use globals() so that pprint can be used       
        self.autoprint()
    
    self.goback() # refresh 
```

### add self.eg to a dict with keys are idxsrc

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: 
        self.cmts.update({idxsrc: cmt})
        self.egsidx.update({idxsrc: self.eg}) # add up idxsrc: self.eg

    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) 
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))
    print('{:=^157}'.format(f"     with example {colorize(self.eg, color='r')}     ")) 
    print()
        
    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            if l.strip().startswith("dbprintinsert"): 
                idxsrcline = idx           
                
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
#             if l.strip().startswith("dbprintinsert"): 
            if idx >= idxsrcline - 3 and idx <= idxsrcline + 2:
                print(l + "="*(totallen-lenl-lenidx) + "(db)")   
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline + 2:
#                     idx = idx - 1
                    idx = idx - 6
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline + 2:
                    idx = idx - 6                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
            
    exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments using self.autoprint() and enable using whatinside without fu in front of it
    if bool(self.eg):
        self.egEnv[self.orisrc.__name__] = locals()[self.orisrc.__name__]        
#         exec(self.eg, {}, self.egEnv)
        example = self.takeoutExample()
        exec("pprint(" + example + ")", globals(), self.egEnv) # use globals() so that pprint can be used       
        self.autoprint()
    
    self.goback() # refresh 
```

### handle both function and class as src

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: 
        self.cmts.update({idxsrc: cmt})
        self.egsidx.update({idxsrc: self.eg}) # add up idxsrc: self.eg

    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) 
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))
    print('{:=^157}'.format(f"     with example {colorize(self.eg, color='r')}     ")) 
    print()
        
    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'
            
    self.dbsrc = dbsrc # make it available to the outside

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            if l.strip().startswith("dbprintinsert"): 
                idxsrcline = idx           
                
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
#             if l.strip().startswith("dbprintinsert"): 
            if idx >= idxsrcline - 3 and idx <= idxsrcline + 2:
                print(l + "="*(totallen-lenl-lenidx) + "(db)")   
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline + 2:
#                     idx = idx - 1
                    idx = idx - 6
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline + 2:
                    idx = idx - 6                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
#         print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
#     if type(self.orisrc) == type:
#         self.outenv.pop(self.orisrc.__name__)
        
        
#     exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    
    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    
#     if type(self.orisrc) == type:    
#         print(f"inspect.getsourcefile(locals()['{self.orisrc.__name__})): {inspect.getsourcefile(locals()[self.orisrc.__name__])}")     
#         print(f"inspect.getsource(locals()['{self.orisrc.__name__})): {inspect.getsource(locals()[self.orisrc.__name__])}")     
        
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")
        print(f"inspect.getsourcefile(locals()['{self.orisrc.__name__})): {inspect.getsourcefile(locals()[self.orisrc.__name__])}")
        
    if showdbsrc:
#         print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
#         print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments using self.autoprint() and enable using whatinside without fu in front of it
    if bool(self.eg):
        self.egEnv[self.orisrc.__name__] = locals()[self.orisrc.__name__] 
        if type(self.orisrc) == type:
#             exec(self.eg, {}, self.egEnv) # working, but not including pprint from globals()
            exec(self.eg, globals(), self.egEnv)  # working great
#             exec(self.eg, globals(), self.egEnv.update(locals()))     # not working, not sure why!!!     
        elif inspect.isfunction(self.orisrc):
            example = self.takeoutExample()
            exec("pprint(" + example + ")", globals(), self.egEnv) # use globals() so that pprint can be used       
        self.autoprint()
        self.goback() # if no self.eg executed, then there should be no self.goback() get called 
    
    else: # to run the fdb.dbprint(....) cell again to dbprint or document on itself as an example
        self.autoprint()
        self.goback()
    return locals()[self.orisrc.__name__]
```

### documenting on Fastdb.dbprint itself

```python

@patch
def dbprint(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "Add comment and evaluate custom (single or multi lines) expressions to any srcline of the source code \
you are investigating. Run exec on the entire srcode with added expressions (dbsrc), so that dbsrc is callable."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: 
        self.cmts.update({idxsrc: cmt})
        self.egsidx.update({idxsrc: self.eg}) # add up idxsrc: self.eg

    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) 
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))
    print('{:=^157}'.format(f"     with example {colorize(self.eg, color='r')}     ")) 
    print()
        
    printsrc(src, idxsrc, cmt, expand)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    newlst = []
    for i in codes: # no matter whether there is "" or "  " in the front or in the middle of codes
        if bool(i.strip()): newlst.append(i)
    codes = newlst

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'
            
    self.dbsrc = dbsrc # make it available to the outside

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("showdbsrc=Start", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            if l.strip().startswith("dbprintinsert"): 
                idxsrcline = idx           
                
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
#             if l.strip().startswith("dbprintinsert"): 
            if idx >= idxsrcline - 3 and idx <= idxsrcline + 2:
                print(l + "="*(totallen-lenl-lenidx) + "(db)")   
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline + 2:
#                     idx = idx - 1
                    idx = idx - 6
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline + 2:
                    idx = idx - 6                
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                
        print(f"locals() keys: {list(locals().keys())}")
#         print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")
            
#     if type(self.orisrc) == type:
#         self.outenv.pop(self.orisrc.__name__)
        
        
#     exec(dbsrc, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    
    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    
#     if type(self.orisrc) == type:    
#         print(f"inspect.getsourcefile(locals()['{self.orisrc.__name__})): {inspect.getsourcefile(locals()[self.orisrc.__name__])}")     
#         print(f"inspect.getsource(locals()['{self.orisrc.__name__})): {inspect.getsource(locals()[self.orisrc.__name__])}")     
        
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")
        print(f"inspect.getsourcefile(locals()['{self.orisrc.__name__})): {inspect.getsourcefile(locals()[self.orisrc.__name__])}")
        
    if showdbsrc:
#         print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
#         print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

# print out the srcode with comments using self.autoprint() and enable using whatinside without fu in front of it
    if bool(self.eg):
        self.egEnv[self.orisrc.__name__] = locals()[self.orisrc.__name__] 
        if type(self.orisrc) == type:
#             exec(self.eg, {}, self.egEnv) # working, but not including pprint from globals()
            exec(self.eg, globals(), self.egEnv)  # working great
#             exec(self.eg, globals(), self.egEnv.update(locals()))     # not working, not sure why!!!     
        elif inspect.isfunction(self.orisrc):
            example = self.takeoutExample()
            exec("pprint(" + example + ")", globals(), self.egEnv) # use globals() so that pprint can be used       
        self.autoprint()
        self.goback() # if no self.eg executed, then there should be no self.goback() get called 
    # using Fastdb.dbprint itsel as example, without using fdb.takExample()
    else: # to run the fdb.dbprint(....) cell again to dbprint or document on itself as an example
        self.autoprint()
#         self.goback()
    return locals()[self.orisrc.__name__]
```

## mk_dbsrc

```python

@patch
def printtitle(self:Fastdb):
    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) 
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))
    print('{:=^157}'.format(f"     with example {colorize(self.eg, color='r')}     ")) 
    print()
```

```python
#| export
@patch
def printtitle(self:Fastdb):

    if 'self.dbsrc' not in self.eg:
        self.orieg = self.eg  # make sure self.orieg has no self inside
    print('{:=^157}'.format(f"     Investigating {colorize(self.orisrc.__name__, color='r')}     ")) 
    print('{:=^157}'.format(f"     on line {colorize(str(self.idxsrc), color='r')}     "))
    print('{:=^157}'.format(f"     with example {colorize(self.orieg, color='r')}     ")) 
    print()
```

```python

```

```python

```

```python
# This is the working version (but it needs further split out into smaller functions)
@patch
def mk_dbsrc(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "create dbsrc the string and turn the string into actual dbsrc function, we have self.dbsrcstr and self.dbsrc available from now on."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: 
        self.cmts.update({idxsrc: cmt})
        self.egsidx.update({idxsrc: self.eg}) # add up idxsrc: self.eg

    self.printtitle()    
    print('{:-<60}'.format(colorize("print selected srcline with expands below", color="y")))    
    printsrc(src, idxsrc, cmt, expand)
    
    
    # create dbsrc the string
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    codes = [i  for i in codes if bool(i.strip())]

    if type(idxsrc) == int: srclines = lst[idxsrc]  
    else: raise TypeError("decode must be an integer.")

    # writing up dbsrc in string
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize('Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'
            
    # store dbsrc in string to the Fastdb obj
    self.dbsrcstr = dbsrc 
        
    # creating dbsrc as function from a string
    file_name ='/tmp/' + self.orisrc.__name__ + '.py' 
    # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    
    # store dbsrc func inside Fastdb obj
    self.dbsrc = locals()[self.orisrc.__name__]

```

```python

```

```python

@patch
def mk_dbsrc(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            showdbsrc:bool=False): # display dbsrc
    "create dbsrc the string and turn the string into actual dbsrc function, we have self.dbsrcstr and self.dbsrc available from now on."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: 
        self.cmts.update({idxsrc: cmt})
        self.egsidx.update({idxsrc: self.eg}) # add up idxsrc: self.eg

    self.printtitle()    
    print('{:-<60}'.format(colorize("print selected srcline with expands below", color="y")))    
    printsrc(src, idxsrc, cmt, expand)
    
    # create dbsrc the string
    self.create_dbsrc_string(idxsrc, *codes)
    
    # creating dbsrc as function from a string
    self.create_dbsrc_from_string()

```

## Turn mk_dbsrc into docsrc 

```python
#| export
@patch
def docsrc(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment added to the srcline
            *codes, # a list of expressions (str) you write to be evaluated above the srcline
            expand:int=2, # span 2 lines of srcode up and down from the srcline investigated
            db:bool=False): # display debugging print
    "create dbsrc the string and turn the string into actual dbsrc function, we have self.dbsrcstr and self.dbsrc available from now on."
#     self.goback() # refresh, but put it in the front will cause multiple reprints of dbcodes outputs
    src = self.orisrc
    self.idxsrc = idxsrc
    
    if type(idxsrc) == int: 
        self.cmts.update({idxsrc: cmt})
        self.egsidx.update({idxsrc: self.eg}) # add up idxsrc: self.eg

    self.printtitle()    
    print('{:-<60}'.format(colorize("print selected srcline with expands below", color="y")))    
    printsrc(src, idxsrc, cmt, expand)
    
    # create dbsrc the string
    self.create_dbsrc_string(idxsrc, *codes)
    
    # creating dbsrc as function from a string
    self.create_dbsrc_from_string()
    
    # run example with dbsrc
    self.run_example(db=db)

```

```python
compile?
```

## create_dbsrc_from_string

```python

@patch
def create_dbsrc_from_string(self:Fastdb):
    file_name ='/tmp/' + self.orisrc.__name__ + '.py' 
    # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(self.dbsrcstr)
    code = compile(self.dbsrcstr, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class

    # store dbsrc func inside Fastdb obj
    self.dbsrc = locals()[self.orisrc.__name__]

```

```python
#| export
@patch
def create_dbsrc_from_string(self:Fastdb):
    file_name ='/tmp/' + self.orisrc.__name__ + '.py' 
    # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(self.dbsrcstr)
    
    code = compile(self.dbsrcstr, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    
    # store dbsrc func inside Fastdb obj
    self.dbsrc = locals()[self.orisrc.__name__]

    # replace original srcode with self.
#     self.egEnv[self.orisrc.__name__] = self.dbsrc

#     print(f'create_dbsrc_from_string, locals(): {locals()}')
#     print(f'locals()[self.orisrc.__name__] src is {inspect.getsource(locals()[self.orisrc.__name__])}')
#     print(f'create_dbsrc_from_string, self.dbsrcstr: {self.dbsrcstr}')
#     print(f'create_dbsrc_from_string, inspect.getsource(self.dbsrc): {inspect.getsource(self.dbsrc)}')
```

```python

```

```python

```

```python

```

```python
#| export
@patch
def create_dbsrc_string(self:Fastdb, idxsrc, *codes): 
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(self.orisrc).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    codes = [i  for i in codes if bool(i.strip())]

    if type(idxsrc) == int: srclines = lst[idxsrc]  
    else: raise TypeError("decode must be an integer.")

    # writing up dbsrc in string
    for idx, l in zip(range(len(lst)), lst):
        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:

            if len(codes) > 0: 
                numindent = len(l) - len(l.lstrip()) 
                dbcodes = "dbprintinsert("
                count = 1
                for c in codes:
                    if count == len(codes):
                        dbcodes = dbcodes + '"' + c + '"' + "," + "env=g" + ")"
                    else:
                        dbcodes = dbcodes + '"' + c + '"' + ","
                    count = count + 1

                emptyline = "print()"
                exploreStart = "print('{:=>157}'.format(colorize(f'Start of my srcline exploration:', color='r')))"
                exploreEnd = "print('{:=>157}'.format(colorize('End of my srcline exploration:', color='r')))"
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + " "*numindent + exploreStart + '\n'   
                dbsrc = dbsrc + " "*numindent + "g = locals()" + '\n'
                dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
                dbsrc = dbsrc + " "*numindent + exploreEnd + '\n'
                dbsrc = dbsrc + " "*numindent + emptyline + '\n'                   
                dbsrc = dbsrc + l + '\n'     
            else:
                dbsrc = dbsrc + l + '\n'                

        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    # store dbsrc in string to the Fastdb obj
    self.dbsrcstr = dbsrc 

```

## replaceWithDbsrc

```python

@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and return this new self.eg"
    oldname = None
    new_eg = ""
    for l in self.eg.split('\n'):
        if self.orisrc.__name__ in l:
            oldname = l.split('(')[0]
            rest = l.split('(')[1]
            new_eg = new_eg + "self.dbsrc" + "(" + rest
        else:
            new_eg = new_eg + l + "\n"
    if db:
        print(f"old name: {oldname}")
    self.eg = new_eg
```

### handle class and metaclass

```python
from fastcore.meta import FixSigMeta
```

```python
type(FixSigMeta)
```

```python

@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg"
    oldname = None
    new_eg = ""
    if type(self.orisrc) == type: # as class
        for l in self.eg.split('\n'):
            if "(" + self.orisrc.__name__ + ")" in l or "(metaclass=" + self.orisrc.__name__ + ")" in l: # (target) or (metaclass=targe)
                lst = l.split(f'{self.orisrc.__name__ + ")"}')
                new_eg = new_eg + lst[0] + "self.dbsrc)" + lst[1] + "\n"
            else:
                new_eg = new_eg + l + "\n"

    else: # as function
        for l in self.eg.split('\n'):
            if self.orisrc.__name__ in l:
                oldname = l.split('(')[0]
                rest = l.split('(')[1]
                new_eg = new_eg + "self.dbsrc" + "(" + rest
            else:
                new_eg = new_eg + l + "\n"

    self.eg = new_eg
```

### improve on handling function as decorator

```python

@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg"
    oldname = None
    new_eg = ""
    if type(self.orisrc) == type: # as class
        for l in self.eg.split('\n'):
            if "(" + self.orisrc.__name__ + ")" in l or "(metaclass=" + self.orisrc.__name__ + ")" in l: # (target) or (metaclass=targe)
                lst = l.split(f'{self.orisrc.__name__ + ")"}')
                new_eg = new_eg + lst[0] + "self.dbsrc)" + lst[1] + "\n"
            else:
                new_eg = new_eg + l + "\n"

    else: # as function
        for l in self.eg.split('\n'):
            if self.orisrc.__name__ in l:
                lst = l.split(self.orisrc.__name__ + '(')
                start = lst[0]
                rest = lst[1]
                new_eg = new_eg + start + "self.dbsrc(" + rest
            else:
                new_eg = new_eg + l + "\n"

    self.eg = new_eg
```

### Handling `inspect._signature_from_callable` to become `self.dbsrc`

```python
"abc(print(inspect.".split('(')[:-1]
```

```python

@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg"
    new_eg = ""
    if type(self.orisrc) == type: # as class
        for l in self.eg.split('\n'):
            if "(" + self.orisrc.__name__ + ")" in l or "(metaclass=" + self.orisrc.__name__ + ")" in l: # (target) or (metaclass=targe)
                lst = l.split(f'{self.orisrc.__name__ + ")"}')
                new_eg = new_eg + lst[0] + "self.dbsrc)" + lst[1] + "\n"
            else:
                new_eg = new_eg + l + "\n"

    else: # as function
        for l in self.eg.split('\n'):
            if self.orisrc.__name__ in l:
                lst = l.split(self.orisrc.__name__ + '(') 
                start = ""
                if lst[0].endswith(".") and "(" not in lst[0]: # if lst[0] = inspect.
                    start = "" # then remove 'inspect.'
                elif lst[0].endswith(".") and "(" in lst[0]: # if lst[0] = abc(print(inspect.
                    start = ""
                    for l in lst[0].split('(')[:-1]: # remove the last item, 'inspect.'
                        start = start + l + '('
                else:
                    start = lst[0]
                rest = lst[1]
                new_eg = new_eg + start + "self.dbsrc(" + rest
            else:
                new_eg = new_eg + l + "\n"

    self.eg = new_eg
```

### handling usage of `@delegates`

```python

@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg"
    new_eg = ""
    if type(self.orisrc) == type: # as class
        for l in self.eg.split('\n'):
            if "(" + self.orisrc.__name__ + ")" in l or "(metaclass=" + self.orisrc.__name__ + ")" in l: # (target) or (metaclass=targe)
                lst = l.split(f'{self.orisrc.__name__ + ")"}')
                new_eg = new_eg + lst[0] + "self.dbsrc)" + lst[1] + "\n"
            else:
                new_eg = new_eg + l + "\n"
    
    else: # as function
        for l in self.eg.split('\n'):
            if "@" + self.orisrc.__name__ in l: # handling @delegates
                lst = l.split("@" + self.orisrc.__name__)
                new_eg = new_eg + "@self.dbsrc" + lst[1] + "\n"
            elif self.orisrc.__name__ in l:
                lst = l.split(self.orisrc.__name__ + '(') 
                start = ""
                if lst[0].endswith(".") and "(" not in lst[0]: # if lst[0] = inspect.
                    start = "" # then remove 'inspect.'
                elif lst[0].endswith(".") and "(" in lst[0]: # if lst[0] = abc(print(inspect.
                    start = ""
                    for l in lst[0].split('(')[:-1]: # remove the last item, 'inspect.'
                        start = start + l + '('
                else:
                    start = lst[0]
                rest = lst[1]
                new_eg = new_eg + start + "self.dbsrc(" + rest
            else:
                new_eg = new_eg + l + "\n"

    self.eg = new_eg
```

### handling `@delegates` with indentation before it

```python

@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg"
    new_eg = ""
    if type(self.orisrc) == type: # as class
        for l in self.eg.split('\n'):
            if "(" + self.orisrc.__name__ + ")" in l or "(metaclass=" + self.orisrc.__name__ + ")" in l: # (target) or (metaclass=targe)
                lst = l.split(f'{self.orisrc.__name__ + ")"}')
                new_eg = new_eg + lst[0] + "self.dbsrc)" + lst[1] + "\n"
            else:
                new_eg = new_eg + l + "\n"
    
    else: # as function
        for l in self.eg.split('\n'):
            if "@" + self.orisrc.__name__ in l: # handling @delegates with indentation
                indent = len(l) - len(l.lstrip())
                lst = l.split("@" + self.orisrc.__name__)
                new_eg = new_eg + " "*indent + "@self.dbsrc" + lst[1] + "\n"
            elif self.orisrc.__name__ in l:
                lst = l.split(self.orisrc.__name__ + '(') 
                start = ""
                if lst[0].endswith(".") and "(" not in lst[0]: # if lst[0] = inspect.
                    start = "" # then remove 'inspect.'
                elif lst[0].endswith(".") and "(" in lst[0]: # if lst[0] = abc(print(inspect.
                    start = ""
                    for l in lst[0].split('(')[:-1]: # remove the last item, 'inspect.'
                        start = start + l + '('
                else:
                    start = lst[0]
                rest = lst[1]
                new_eg = new_eg + start + "self.dbsrc(" + rest
            else:
                new_eg = new_eg + l + "\n"

    self.eg = new_eg
```

### handling classes by inspect.isclass() rather than == type and add more class situations

```python

@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg"
    new_eg = ""
#     if type(self.orisrc) == type: # as class
    if inspect.isclass(self.orisrc):
        for l in self.eg.split('\n'):
            # 1. class Foo(TargetCalss) ; 2. class Foo(metaclass=TargetClass); 
            if "(" + self.orisrc.__name__ + ")" in l or "(metaclass=" + self.orisrc.__name__ + ")" in l: 
                lst = l.split(f'{self.orisrc.__name__ + ")"}')
                new_eg = new_eg + lst[0] + "self.dbsrc)" + lst[1] + "\n"
            elif "(" + self.orisrc.__name__ + "," in l: # 3. class Foo(TargetClass, OtherClass)
                lst = l.split(f'{self.orisrc.__name__ + ","}')
                new_eg = new_eg + lst[0] + "self.dbsrc," + lst[1] + "\n"
            else:
                new_eg = new_eg + l + "\n"
    
    else: # as function
        for l in self.eg.split('\n'):
            if "@" + self.orisrc.__name__ in l: # handling @delegates with indentation
                indent = len(l) - len(l.lstrip())
                lst = l.split("@" + self.orisrc.__name__)
                new_eg = new_eg + " "*indent + "@self.dbsrc" + lst[1] + "\n"
            elif self.orisrc.__name__ in l:
                lst = l.split(self.orisrc.__name__ + '(') 
                start = ""
                if lst[0].endswith(".") and "(" not in lst[0]: # if lst[0] = inspect.
                    start = "" # then remove 'inspect.'
                elif lst[0].endswith(".") and "(" in lst[0]: # if lst[0] = abc(print(inspect.
                    start = ""
                    for l in lst[0].split('(')[:-1]: # remove the last item, 'inspect.'
                        start = start + l + '('
                else:
                    start = lst[0]
                rest = lst[1]
                new_eg = new_eg + start + "self.dbsrc(" + rest
            else:
                new_eg = new_eg + l + "\n"

    self.eg = new_eg
```

### handling `class _T(_TestA, metaclass=BypassNewMeta): `

```python
#| export
@patch
def replaceWithDbsrc(self:Fastdb, db=False):
    "to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg"
    new_eg = ""
#     if type(self.orisrc) == type: # as class
    if inspect.isclass(self.orisrc):
        for l in self.eg.split('\n'):
            # 1. class Foo(TargetCalss): ; 2. class Foo(metaclass=TargetClass): ; 4: class _T(_TestA, metaclass=BypassNewMeta):
            if "(" + self.orisrc.__name__ + ")" in l or "(metaclass=" + self.orisrc.__name__ + ")" in l \
            or ", metaclass=" + self.orisrc.__name__ + ")" in l: 
                lst = l.split(f'{self.orisrc.__name__ + ")"}')
                new_eg = new_eg + lst[0] + "self.dbsrc)" + lst[1] + "\n"
            elif "(" + self.orisrc.__name__ + "," in l: # 3. class Foo(TargetClass, OtherClass)
                lst = l.split(f'{self.orisrc.__name__ + ","}')
                new_eg = new_eg + lst[0] + "self.dbsrc," + lst[1] + "\n"
            else:
                new_eg = new_eg + l + "\n"
    
    else: # as function
        for l in self.eg.split('\n'):
            if "@" + self.orisrc.__name__ in l: # handling @delegates with indentation
                indent = len(l) - len(l.lstrip())
                lst = l.split("@" + self.orisrc.__name__)
                new_eg = new_eg + " "*indent + "@self.dbsrc" + lst[1] + "\n"
            elif self.orisrc.__name__ in l:
                lst = l.split(self.orisrc.__name__ + '(') 
                start = ""
                if lst[0].endswith(".") and "(" not in lst[0]: # if lst[0] = inspect.
                    start = "" # then remove 'inspect.'
                elif lst[0].endswith(".") and "(" in lst[0]: # if lst[0] = abc(print(inspect.
                    start = ""
                    for l in lst[0].split('(')[:-1]: # remove the last item, 'inspect.'
                        start = start + l + '('
                else:
                    start = lst[0]
                rest = lst[1]
                new_eg = new_eg + start + "self.dbsrc(" + rest
            else:
                new_eg = new_eg + l + "\n"

    self.eg = new_eg
```

```python

```

```python

```

```python

```

```python
"adb(this is me)".split('adb(')
```

```python

```

```python

```

```python

```

## run_example

```python

@patch
def run_example(self:Fastdb, db=False):
    
    self.replaceWithDbsrc()
    # use locals() as global env to bring in self or obj to enable self.dbsrc to run
    # use globals() to see whether pprint can be brought in 
    # so combine them both can have them both, as globals() do not contain locals() fully  
    exec(self.eg, globals().update(locals()), self.egEnv) # locals() gives me self, # globals() gives fastdebug env
    self.autoprint()
    
#     print(f"run_example: globals(): {list(globals().keys())}")
#     print(f"run_example: locals(): {locals()}") 
#     print(f"run_example: locals()['self'].dbsrc: {locals()['self'].dbsrc}") 
#     print(f'get the source of self.dbsrc: {inspect.getsource(self.dbsrc)}')
#     print(f'get the source of self.dbsrc: {inspect.getsource(locals()["self"].dbsrc)}')  
```

### `exec(self.eg, globals().update(self.egEnv), locals())` works better than `...update(locals()), self.egEnv)

```python

@patch
def run_example(self:Fastdb, db=False):
    
    self.replaceWithDbsrc()
    # use locals() as global env to bring in self or obj to enable self.dbsrc to run
    # use globals() to see whether pprint can be brought in 
    # so combine them both can have them both, as globals() do not contain locals() fully  
    exec(self.eg, globals().update(self.egEnv), locals()) # locals() gives me self, # globals() gives fastdebug env
    self.autoprint()
      
```

### no more env cells run before `fdb.eg` to make `fdb.run_example` work

```python
#| export
@patch
def run_example(self:Fastdb, db=False):
    
    self.replaceWithDbsrc()
    # use locals() as global env to bring in self or obj to enable self.dbsrc to run
    # use globals() to see whether pprint can be brought in 
    # so combine them both can have them both, as globals() do not contain locals() fully  
    globals().update(self.egEnv)
    globals().update(locals())
    exec(self.eg, globals()) # this is totally different from  =>   exec(self.eg, globals(), {})
    # leaving the local param as default, is using funcs and classes defined inside self.eg
    
    self.autoprint()
      
```

```python

```

```python

```

## Autoprint

```python

```

```python
#| export
@patch
def autoprint(self:Fastdb, maxpcell=20):
    totalines = len(inspect.getsource(self.orisrc).split('\n'))
    idx = self.idxsrc
    if bool(idx): 
        pt = idx // maxpcell 
    else:
        return
    
    print()
    print('{:=<157}'.format(colorize("Review srcode with all comments added so far", color="y")))
    if idx > maxpcell and idx % maxpcell != 0:
        self.print(maxpcell, pt + 1)
    elif idx % maxpcell == 0:
        self.print(maxpcell, pt + 1)
    else:
        self.print(maxpcell, 1)
    print()
```

## Take an example and its env into Fastdb obj

```python

@patch
def takExample(self:Fastdb,
               eg, 
               **env):
    self.eg = eg
    self.egEnv = env
```

```python

@patch
def example(self:Fastdb,
               eg, 
               **env):
    self.eg = eg
    self.egEnv = env
```

```python

```

## print src with idx and cmt in whole or parts

```python
dct = {1:"a", 3:"b", 2:"c"}
dct1 = {k: idx for (k, v), idx in zip(dct.items(), range(len(list(dct))))}
dct1


```

```python

@patch
def print(self:Fastdb, 
            maxlines:int=33, # maximum num of lines per page
            part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    "Print the source code in whole or parts with idx and comments you added with dbprint along the way."

    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(self.orisrc).split('\n')
    numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
    cmts = self.cmts
    
    if part == 0: 
        for idx, l in zip(range(len(lstsrc)), lstsrc):
            lenl = len(l)

            if not bool(l.strip()):
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            elif lenl + lspace >= 100:
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                    else:
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                else: 
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            else:


                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                else:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 

    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                else:

                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return
```

```python

@patch
def print(self:Fastdb, 
            maxlines:int=33, # maximum num of lines per page
            part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    "Print the source code in whole or parts with idx and comments you added with dbprint along the way."

    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(self.orisrc).split('\n')
    numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
    cmts = self.cmts
    idxcmts = {k: idx for (k, v), idx in zip(cmts.items(), range(len(list(cmts))))} # order of cmts correspond to idxsrc

    
    if part == 0: 
        for idx, l in zip(range(len(lstsrc)), lstsrc):
            lenl = len(l)

            if not bool(l.strip()):
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            elif lenl + lspace >= 100:
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print(l + " # " + "step "+ str(idxcmts[idx]) + ": " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                    else:
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                else: 
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            else:


                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + "step "+ str(idxcmts[idx]) + ": " + cmts[idx]))
                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                else:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 

    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                else:

                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + cmts[idx]))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return
```

### print self.eg after each comment and colorize comments

```python

@patch
def print(self:Fastdb, 
            maxlines:int=33, # maximum num of lines per page
            part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    "Print the source code in whole or parts with idx and comments you added with dbprint along the way."

    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(self.orisrc).split('\n')
    numparts = len(lstsrc) // 33 + 1 if len(lstsrc) % 33 != 0 else len(lstsrc) // 33
    cmts = self.cmts
    idxcmts = {k: idx for (k, v), idx in zip(cmts.items(), range(len(list(cmts))))} # order of cmts correspond to idxsrc

    randCol = randomColor()
    
    if part == 0: 
        self.printtitle()
        
        for idx, l in zip(range(len(lstsrc)), lstsrc):
            lenl = len(l)

            if not bool(l.strip()):
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            elif lenl + lspace >= 100:
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print(l + " # " + "spot "+ str(idxcmts[idx]) + ": " + colorize(cmts[idx], color=randCol) + \
                              " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + " (" + str(idx) + ")" + " => " + self.egsidx[idx])
                    else:
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                else: 
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            else:                
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + "spot "+ str(idxcmts[idx]) + ": " + \
                                               colorize(cmts[idx], color=randCol)))
                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                else:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 

    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + cmts[idx] + " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + "(" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                else:

                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + "spot "+ str(idxcmts[idx]) + ": " + \
                                               colorize(cmts[idx], color=randCol)))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return
```

### color examples and cmts separately and make the func simpler

```python

@patch
def print(self:Fastdb, 
            maxlines:int=33, # maximum num of lines per page
            part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    "Print the source code in whole or parts with idx and comments you added with dbprint along the way."

    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(self.orisrc).split('\n')
    numparts = len(lstsrc) // maxlines + 1 if len(lstsrc) % maxlines != 0 else len(lstsrc) // maxlines
    cmts = self.cmts
    idxcmts = {k: idx for (k, v), idx in zip(cmts.items(), range(len(list(cmts))))} # order of cmts correspond to idxsrc

    randCol1 = randomColor()
    randCol2 = randomColor()
    
    if part == 0: 
        self.printtitle()
        
        for idx, l in zip(range(len(lstsrc)), lstsrc):
            lenl = len(l)

            if not bool(l.strip()):
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            elif lenl + lspace >= 100:
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print(l + " # " + "spot "+ str(idxcmts[idx]) + ": " + colorize(cmts[idx], color=randCol1) + \
                              " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + " (" + str(idx) + ")")
                    else:
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                else: 
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

            else:                
                if bool(cmts):
                    cmtidx = [cmt[0] for cmt in list(cmts.items())]
                    if idx in cmtidx:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + "spot "+ str(idxcmts[idx]) + ": " + \
                                               colorize(cmts[idx], color=randCol1)))
                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

                else:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                 

    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + "spot "+ str(idxcmts[idx]) + ": " + colorize(cmts[idx], color=randCol1) + \
                              " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + " (" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")


                else:

                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + "spot "+ str(idxcmts[idx]) + ": " + \
                                               colorize(cmts[idx], color=randCol1)))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return
```

### split each cmt and colorize parts randomly

```python
#| export
@patch
def printcmts1(self:Fastdb, maxlines):
    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(self.orisrc).split('\n')
    numparts = len(lstsrc) // maxlines + 1 if len(lstsrc) % maxlines != 0 else len(lstsrc) // maxlines
    cmts = self.cmts
    idxcmts = {k: idx for (k, v), idx in zip(cmts.items(), range(len(list(cmts))))} # order of cmts correspond to idxsrc

    randCol1 = randomColor()
    randCol2 = randomColor()            
    for idx, l in zip(range(len(lstsrc)), lstsrc):
        lenl = len(l)

        if not bool(l.strip()):
            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

        elif lenl + lspace >= 100:
            if bool(cmts):
                cmtidx = [cmt[0] for cmt in list(cmts.items())]
                if idx in cmtidx:
                    print(l + " # " + randomize_cmtparts_color(cmts[idx]) + \
                          #colorize(cmts[idx], color=randCol1) + \
                          " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + " (" + str(idx) + ")")
                else:
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else: 
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

        else:                
            if bool(cmts):
                cmtidx = [cmt[0] for cmt in list(cmts.items())]
                if idx in cmtidx:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + \
                                           randomize_cmtparts_color(cmts[idx])))#colorize(cmts[idx], color=randCol1)))
                else:
                    print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                      

            else:
                print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))      

```

```python
#| export
@patch
def printcmts2(self:Fastdb, maxlines, part):
    totallen = 157
    lenidx = 5
    lspace = 10
    lstsrc = inspect.getsource(self.orisrc).split('\n')
    numparts = len(lstsrc) // maxlines + 1 if len(lstsrc) % maxlines != 0 else len(lstsrc) // maxlines
    cmts = self.cmts
    idxcmts = {k: idx for (k, v), idx in zip(cmts.items(), range(len(list(cmts))))} # order of cmts correspond to idxsrc

    randCol1 = randomColor()
    randCol2 = randomColor()    
    for p in range(numparts):
        for idx, l in zip(range(len(lstsrc)), lstsrc):

            if (maxlines*p <= idx < maxlines*(p+1) and p+1 == part):
                lenl = len(l)
                if not bool(l.strip()):
                    print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                elif lenl + lspace >= 100:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print(l + " # " + randomize_cmtparts_color(cmts[idx]) + \
                                  #colorize(cmts[idx], color=randCol1) + \
                              " "*(totallen-lenl-lenidx-len(cmts[idx])-3) + " (" + str(idx) + ")")
                        else:
                            print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
                    else: 
                        print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")

                else:
                    if bool(cmts):
                        cmtidx = [cmt[0] for cmt in list(cmts.items())]
                        if idx in cmtidx:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})" + " # " + \
                                               randomize_cmtparts_color(cmts[idx])))#colorize(cmts[idx], color=randCol1)))
                        else:
                            print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                                                          

                    else:
                        print('{:<100}'.format(l + "="*(100-lenl-lspace) + f"({idx})"))                      

            if (idx == maxlines*(p+1) or idx == len(lstsrc) - 1) and p+1 == part:
                print('{:>157}'.format(f"part No.{p+1} out of {numparts} parts"))
                return
```

```python

```

```python
#| export
def randomize_cmtparts_color(cmt):
    newcmt = ""
    for p in cmt.split('; '):
        col = randomColor()
        newcmt = newcmt + colorize(p, color=col) + "; "
    return newcmt
```

```python

```

```python
#| exports
@patch
def print(self:Fastdb, 
            maxlines:int=33, # maximum num of lines per page
            part:int=0): # if the src is more than 33 lines, then divide the src by 33 into a few parts
    "Print the source code in whole or parts with idx and comments you added with dbprint along the way."


    
    if part == 0: 
        self.printtitle()
        self.printcmts1(maxlines=maxlines)
        
           
    self.printcmts2(maxlines=maxlines, part=part)


```

```python

```

```python

```

```python

```

```python

```

## goback

```python
#| export
@patch
def goback(self:Fastdb):
    "Return src back to original state."
    self.outenv[self.orisrc.__name__] = self.orisrc
```

## Fastdb.explore


### adding one breakpoint with comment

```python

@patch
def explore(self:Fastdb, 
            idxsrc:int, # idx of a srcline under investigation, can only be int
            cmt:str, # comment
            showdbsrc:bool=False): # display dbsrc
    "insert 'import ipdb; ipdb.set_trace()' above srcline of idx to create dbsrc, and exec on dbsrc"
    src = self.orisrc

    printsrc(src, idxsrc, cmt)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = ""
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    else:
        raise TypeError("decode must be an integer.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and l.strip() in srclines and idx == idxsrc:
            numindent = len(l) - len(l.lstrip()) # make sure indent not messed up by trailing spaces
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'     
        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("print out dbsrc", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("import ipdb"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - 1
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")

    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

        
```

### Adding multiple breakpoints by multiple set_trace()

```python

@patch
def explore(self:Fastdb, 
            idxsrc:int, # idxsrc can be an int or a list of int
            cmt:str, # comment can be a string or a list of strings
            showdbsrc:bool=False): # display dbsrc
    "insert 'import ipdb; ipdb.set_trace()' above srcline of idx to create dbsrc, and exec on dbsrc"
    src = self.orisrc

#     printsrc(src, idxsrc, cmt)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = None
    idxlst = None
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    elif type(idxsrc) == list:
        idxlst = idxsrc
    else:
        raise TypeError("decode must be an integer or a list.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and type(idxsrc) == int and idx == idxsrc:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'     
        elif type(idxsrc) == list and idx in idxlst:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'  
            idxlst.remove(idx)
        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("print out dbsrc", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        idxcount = 0
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("import ipdb"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx
                idxcount = idxcount + 1
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")

    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

        
```

### Go back to normal before running explore again

```python

@patch
def explore(self:Fastdb, 
            idxsrc:int, # idxsrc can be an int or a list of int
            cmt:str, # comment can be a string or a list of strings
            showdbsrc:bool=False): # display dbsrc
    "insert 'import ipdb; ipdb.set_trace()' above srcline of idx to create dbsrc, and exec on dbsrc"
    self.goback()
    src = self.orisrc

#     printsrc(src, idxsrc, cmt)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = None
    idxlst = None
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    elif type(idxsrc) == list:
        idxlst = idxsrc
    else:
        raise TypeError("decode must be an integer or a list.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and type(idxsrc) == int and idx == idxsrc:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'     
        elif type(idxsrc) == list and idx in idxlst:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'  
            idxlst.remove(idx)
        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("print out dbsrc", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        idxcount = 0
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("import ipdb"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx
                idxcount = idxcount + 1
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")

    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))

        
```

### enable fdb.takExample("whatinside(fu), ...) without using `fu.whatinside`

```python
#| export
import ipdb 
# this handles the partial import error
```

```python

@patch
def explore(self:Fastdb, 
            idxsrc:int, # idxsrc can be an int or a list of int
            cmt:str, # comment can be a string or a list of strings
            showdbsrc:bool=False): # display dbsrc
    "insert 'import ipdb; ipdb.set_trace()' above srcline of idx to create dbsrc, and exec on dbsrc"
#     self.goback()
    src = self.orisrc

#     printsrc(src, idxsrc, cmt)
    if showdbsrc:
        print('{:-<60}'.format(colorize("print selected srcline with expands above", color="y")))
    
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(src).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = None
    idxlst = None
    if type(idxsrc) == int:
        srclines = lst[idxsrc]
    elif type(idxsrc) == list:
        idxlst = idxsrc
    else:
        raise TypeError("decode must be an integer or a list.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and type(idxsrc) == int and idx == idxsrc:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'     
        elif type(idxsrc) == list and idx in idxlst:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'  
            idxlst.remove(idx)
        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    if showdbsrc: # added to debug
        print('{:-<60}'.format(colorize("print out dbsrc", color="y")))
        totallen = 157
        lenidx = 5
        dblst = dbsrc.split('\n')
        idxsrcline = None
        idxcount = 0
        for idx, l in zip(range(len(dblst)), dblst):
            lenl = len(l)
#             if "dbprintinsert" in l: 
            if l.strip().startswith("import ipdb"): 
                print(l + "="*(totallen-lenl-lenidx) + "(db)")
                idxsrcline = idx
                idxcount = idxcount + 1
            elif not bool(l.strip()):
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + " "*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
            else:
                if bool(idxsrcline) and idx > idxsrcline:
                    idx = idx - idxcount
                print(l + "-"*(totallen-lenl-lenidx) + "(" + str(idx) + ")")
        
        print(f"locals() keys: {list(locals().keys())}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"before exec, self.orisrc.__name__: {self.orisrc.__name__} is : {self.orisrc}")
        names = self.orisrc.__qualname__.split('.')        
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))
            clsname = names[0]
            methodname = names[1]
            print(f"before exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"before exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"before exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"before exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"before exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            expr = "self.outenv[" + "'" + clsname + "']." + methodname
            print(f"expr: {expr}")
            expr1 = "self.outenv[" + "'" + methodname + "']"
            print(f"expr1: {expr1}")
            print(f"inspect.getsourcefile('{expr}') == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")

    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(dbsrc)
    code = compile(dbsrc, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    if showdbsrc:
        print('{:-<60}'.format(colorize("exec on dbsrc above", color="y")))
    
    
    if showdbsrc: 
        print(f"locals() keys: {list(locals().keys())}")
        if len(names) == 2:
            print('{:-<60}'.format(colorize("the src is a method of a class", color="y")))            
            print(f"after exec, is methodname: {methodname} in locals(): {methodname in locals()}")
            print(f"after exec, is clsname: {clsname} in locals(): {clsname in locals()}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in locals(): {self.orisrc.__qualname__ in locals()}")
            print(f"after exec, is methodname: {methodname} in self.outenv: {methodname in self.outenv}")
            print(f"after exec, is clsname: {clsname} in self.outenv: {clsname in self.outenv}")
            print(f"after exec, is self.orisrc.__qualname__: {self.orisrc.__qualname__} in self.outenv: {self.orisrc.__qualname__ in self.outenv}")
            print(f"inspect.getsourcefile({expr}) == '<string>': {True if inspect.getsourcefile(eval(expr)) == '<string>' else inspect.getsourcefile(eval(expr))}")
            print(f"self.outenv['{methodname}']: {eval(expr1)}")
            print(f"self.orisrc.__name__: {self.orisrc.__name__} is {expr}: {self.orisrc is eval(expr)}")            
        print(f'self.orisrc.__name__: {self.orisrc.__name__}')
        print(f"locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__]}")

    
    if showdbsrc:
        print(f"after exec and before self.outenv update, self.outenv['{self.orisrc.__name__}'] is locals()['{self.orisrc.__name__}']: {locals()[self.orisrc.__name__] is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        print(f"after exec and before self.outenv update, self.orisrc.__name__: {self.orisrc.__name__} is locals()['{self.orisrc.__name__}']: {self.orisrc is locals()[self.orisrc.__name__]}")
    # Important! update dbsrc to module.func, fu.whatinside, inspect._signature_from_callable
    self.outenv.update(locals()) 
#     self.outenv[self.orisrc.__name__] = locals()[self.orisrc.__name__]
#     return locals()[self.orisrc.__name__]
    if showdbsrc:
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is self.outenv['{self.orisrc.__name__}']: {self.orisrc is self.outenv[self.orisrc.__name__]}")
        exec("import " + self.orisrc.__module__.split('.')[0])
        print(f"after update self.outenv, self.orisrc.__name__: {self.orisrc.__name__} is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.orisrc is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")
        print(f"after update self.outenv, self.outenv['{self.orisrc.__name__}'] is {self.orisrc.__module__}.{self.orisrc.__name__}: {self.outenv[self.orisrc.__name__] is eval(self.orisrc.__module__ + '.' + self.orisrc.__name__, {}, locals())}")        
        print(f"Therefore, to use dbsrc we must use self.outenv['{self.orisrc.__name__}'], or {self.orisrc.__module__}.{self.orisrc.__name__}")
        print('{:-<60}'.format(colorize("showdbsrc=End", color="y")))
        
#     self.orisrc = locals()[self.orisrc.__name__] # see whether whatinside in the outside updated too?

# print out the srcode with comments using self.autoprint()
    if bool(self.eg):
        self.egEnv[self.orisrc.__name__] = locals()[self.orisrc.__name__]        
        exec(self.eg, {}, self.egEnv)
        self.autoprint()
        
    self.goback() # at the end will avoid some multi print of dbcodes
```

### refactory explore
- create_explore_str
- create_explore_from_string
- run_example


```python
#| export
@patch
def create_explore_str(self:Fastdb):
    dbsrc = ""
    indent = 4

    lst = inspect.getsource(self.orisrc).split('\n')
    if not bool(lst[-1]): lst = lst[:-1]

    srclines = None
    idxlst = None
    if type(self.idxsrc) == int:
        srclines = lst[self.idxsrc]
    elif type(self.idxsrc) == list:
        idxlst = self.idxsrc
    else:
        raise TypeError("decode must be an integer or a list.")

    for idx, l in zip(range(len(lst)), lst):

        if bool(l.strip()) and type(self.idxsrc) == int and idx == self.idxsrc:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'     
        elif type(self.idxsrc) == list and idx in idxlst:
            numindent = len(l) - len(l.lstrip()) 
            dbcodes = "import ipdb; ipdb.set_trace()"
            dbsrc = dbsrc + " "*numindent + dbcodes + '\n'
            dbsrc = dbsrc + l + '\n'  
            idxlst.remove(idx)
        elif bool(l.strip()) and idx + 1 == len(lst):
            dbsrc = dbsrc + l
        else: # make sure this printout is identical to the printsrc output
            dbsrc = dbsrc + l + '\n'

    self.dbsrcstr = dbsrc

```

```python
#| export
@patch
def create_explore_from_string(self:Fastdb):
    file_name ='/tmp/' + self.orisrc.__name__ + '.py' # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    with open(file_name, 'w') as f:
        f.write(self.dbsrcstr)
    code = compile(self.dbsrcstr, file_name, 'exec')
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class

    self.dbsrc = locals()[self.orisrc.__name__]
```

```python
#| export
@patch
def explore(self:Fastdb, 
            idxsrc:int, # idxsrc can be an int or a list of int
            db:bool=False): # display dbsrc
    "insert 'import ipdb; ipdb.set_trace()' above srcline of idx to create dbsrc, and exec on dbsrc"

    self.idxsrc = idxsrc
    self.create_explore_str()
    self.create_explore_from_string()
    self.run_example()
            
    
 
```

```python

```

```python

```

## snoop

```python
#| export
import snoop
```

```python
#| export
@patch
def takeoutExample(self:Fastdb):
    example = ""
    for l in self.eg.split('\n'):
        if self.orisrc.__name__ in l:
            example = l
    return example
```

```python

@patch
def snoop(self:Fastdb):
#     self.eg = "inspect._signature_from_callable(whatinside, sigcls=inspect.Signature)"

    example = self.takeoutExample()
    lst = example.split('(')
    snp = "snoop.snoop(depth=1)(" + lst[0] + ")(" + lst[1]
    exec("import snoop")
    eval(snp, locals(), self.egEnv)
    


```

### snoop on both function and class

```python

@patch
def snoop(self:Fastdb, db=False):

    if bool(self.eg):
        if inspect.isfunction(self.orisrc):
            example = self.takeoutExample()
            lst = example.split('(')
            snp = "snoop.snoop(depth=1)(" + lst[0] + ")(" + lst[1]
            exec("import snoop")
            eval(snp, locals(), self.egEnv)
        elif type(self.orisrc) == type:
#             dbsrc="import snoop\n"
            dbsrc=""
            for l in inspect.getsource(self.orisrc).split('\n'):
                if "def __new__" in l:
                    indent = len(l) - len(l.lstrip())
                    dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                    dbsrc = dbsrc + " "*indent + "@snoop\n"
                    dbsrc = dbsrc + l + '\n'
                else:
                    dbsrc = dbsrc + l + '\n'
            if db:
                pprint(dbsrc)
            
            # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
            file_name ='/tmp/' + self.orisrc.__name__ + '.py' 
            with open(file_name, 'w') as f:
                f.write(dbsrc)
            code = compile(dbsrc, file_name, 'exec')
            
            if db: 
                print(f"before exec on snoop FixSigMeta, locals(): {locals()}")
                      
            exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
                      
            if db:
                print(f"after exec on snoop FixSigMeta, locals(): {locals()}")
                print(f"self.egEnv: {self.egEnv}")
                      
            exec(self.eg, self.egEnv.update(locals()), self.egEnv) # working nicely, not sure why?????
#             exec(self.eg, globals(), self.egEnv.update(locals()))       # not working, not sure why??????     

```

### snoop on class and method and all???

```python
# from fastdebug.core import Fastdb
```

```python
# type(Fastdb.dbprint)
```

```python

@patch
def snoop(self:Fastdb, db=False):

    if bool(self.eg):
        if inspect.isfunction(self.orisrc):
            example = self.takeoutExample()
            lst = example.split('(')
            snp = "snoop.snoop(depth=4)(" + lst[0] + ")(" + lst[1]
            exec("import snoop")
            # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
            file_name ='/tmp/' + self.orisrc.__name__ + '.py' 
            with open(file_name, 'w') as f:
                f.write(snp)
            code = compile(snp, file_name, 'exec')
            exec(snp, locals(), self.egEnv)
            
        elif type(self.orisrc) == type:
            dbsrc=""
            for l in inspect.getsource(self.orisrc).split('\n'):
                if "def " in l:
                    indent = len(l) - len(l.lstrip())
                    dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                    dbsrc = dbsrc + " "*indent + "@snoop\n"
                    dbsrc = dbsrc + l + '\n'
                else:
                    dbsrc = dbsrc + l + '\n'
            if db:
                pprint(dbsrc)

            # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
            file_name ='/tmp/' + self.orisrc.__name__ + '.py' 
            with open(file_name, 'w') as f:
                f.write(dbsrc)
            code = compile(dbsrc, file_name, 'exec')

            if db: 
                print(f"before exec on snoop FixSigMeta, locals(): {locals()}")

            exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class

            if db:
                print(f"after exec on snoop FixSigMeta, locals(): {locals()}")
                print(f"self.egEnv: {self.egEnv}")

            exec(self.eg, self.egEnv.update(locals()), self.egEnv) # working nicely, not sure why?????
    #             exec(self.eg, globals(), self.egEnv.update(locals()))       # not working, not sure why??????     
```

### snoop

- create_snoop_str
- create_snoop_from_string
- run_example

```python

@patch
def create_snoop_str(self:Fastdb, deco=False, db=False):
    dbsrc=""
    if not deco:
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                dbsrc = dbsrc + " "*indent + "@snoop\n"
                dbsrc = dbsrc + l + '\n'
            else:
                dbsrc = dbsrc + l + '\n'

    if deco: # if self.orisrc is a decorator
        countdef = 0
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " in l:
                indent = len(l) - len(l.lstrip())
                if countdef == 0:
                    dbsrc = dbsrc + " "*indent + "import snoop\n"
                    countdef = countdef + 1
                elif countdef == 1:
                    dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                    dbsrc = dbsrc + " "*indent + "@snoop\n"
                dbsrc = dbsrc + l + '\n'
            else:
                dbsrc = dbsrc + l + '\n'            
    self.dbsrcstr = dbsrc
```

```python
lst = [1, 2, 3, 4]
new = lst[2:]
new.insert(0, lst[0])
new
```

### simplify adding @snoop for both normal function and decorator

```python

@patch
def create_snoop_str(self:Fastdb, 
                     deco=False, # whether it is a decorator or a normal func
                     db=False):
    dbsrc=""
    if not deco:
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " + self.orisrc.__name__ in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                dbsrc = dbsrc + " "*indent + "@snoop\n"
                dbsrc = dbsrc + l + '\n'
            else:
                dbsrc = dbsrc + l + '\n'

    if deco: # if self.orisrc is a decorator
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " + self.orisrc.__name__ in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"    
                dbsrc = dbsrc + " "*indent + "@snoop\n"         
                dbsrc = dbsrc + l + '\n'
            elif "def " in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                dbsrc = dbsrc + " "*indent + "@snoop\n"                
                dbsrc = dbsrc + l + '\n'                
            else:
                dbsrc = dbsrc + l + '\n'          
    self.dbsrcstr = dbsrc
```

### handling classes

```python

@patch
def create_snoop_str(self:Fastdb, 
                     deco=False, # whether it is a decorator or a normal func
                     db=False):
    dbsrc=""
    if not deco:
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " + self.orisrc.__name__ in l or "def __" in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                dbsrc = dbsrc + " "*indent + "@snoop\n"
                dbsrc = dbsrc + l + '\n'
            else:
                dbsrc = dbsrc + l + '\n'

    if deco: # if self.orisrc is a decorator
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " + self.orisrc.__name__ in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"    
                dbsrc = dbsrc + " "*indent + "@snoop\n"         
                dbsrc = dbsrc + l + '\n'
            elif "def " in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                dbsrc = dbsrc + " "*indent + "@snoop\n"                
                dbsrc = dbsrc + l + '\n'                
            else:
                dbsrc = dbsrc + l + '\n'          
    self.dbsrcstr = dbsrc
```

### add watch

```python
#| export
@patch
def create_snoop_str(self:Fastdb, 
                     watch:list=None,
                     deco=False, # whether it is a decorator or a normal func
                     db=False):
    dbsrc=""
    if not deco:
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " + self.orisrc.__name__ in l or "def __" in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"
                if bool(watch):
                    spreadlst = ''
                    for idx, i in zip(range(len(watch)), watch):
                        if idx < len(watch)-1:
                            spreadlst = spreadlst + '"' + i + '"' + ','
                        else:
                            spreadlst = spreadlst + '"' + i + '"'
                    dbsrc = dbsrc + " "*indent + f"@snoop(watch=({spreadlst}))\n"
                else:
                    dbsrc = dbsrc + " "*indent + "@snoop\n"
                dbsrc = dbsrc + l + '\n'
            else:
                dbsrc = dbsrc + l + '\n'

    if deco: # if self.orisrc is a decorator
        for l in inspect.getsource(self.orisrc).split('\n'):
            if "def " + self.orisrc.__name__ in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"    
                dbsrc = dbsrc + " "*indent + "@snoop\n"         
                dbsrc = dbsrc + l + '\n'
            elif "def " in l:
                indent = len(l) - len(l.lstrip())
                dbsrc = dbsrc + " "*indent + "import snoop\n"                    
                dbsrc = dbsrc + " "*indent + "@snoop\n"                
                dbsrc = dbsrc + l + '\n'                
            else:
                dbsrc = dbsrc + l + '\n'          
    self.dbsrcstr = dbsrc
```

```python

```

```python

```

```python
#| export
@patch
def create_snoop_from_string(self:Fastdb, db=False):
    # learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm
    file_name ='/tmp/' + self.orisrc.__name__ + '.py' 
    with open(file_name, 'w') as f:
        f.write(self.dbsrcstr)
    code = compile(self.dbsrcstr, file_name, 'exec')
#             exec(dbsrc, locals(), self.egEnv)                
#     exec(code, globals().update(self.outenv), locals()) # when dbsrc is a method, it will update as part of a class
    exec(code, globals().update(self.outenv)) # when dbsrc is a method, it will update as part of a class
    # store dbsrc func inside Fastdb obj
    self.dbsrc = locals()[self.orisrc.__name__]

```

```python

```

```python

@patch
def snoop(self:Fastdb, deco=False, db=False):

    self.idxsrc = None # so that autoprint won't print src at all for snoop
    if bool(self.eg):
        self.create_snoop_str(deco=deco, db=db)
        self.create_snoop_from_string(db=db)
        self.run_example()

```

## Snoop

```python

@patch
def snoop(self:Fastdb, deco=False, db=False):

    self.idxsrc = None # so that autoprint won't print src at all for snoop
    self.printtitle() # maybe at some point, I should use await to make the output of printtitle appear first
    if bool(self.eg):
        self.create_snoop_str(deco=deco, db=db)
        self.create_snoop_from_string(db=db)
        self.run_example()

```

### add watch

```python
#| export
@patch
def snoop(self:Fastdb, watch:list=None, deco=False, db=False):

    self.idxsrc = None # so that autoprint won't print src at all for snoop
    self.printtitle() # maybe at some point, I should use await to make the output of printtitle appear first
    if bool(self.eg):
        self.create_snoop_str(watch=watch, deco=deco, db=db)
        self.create_snoop_from_string(db=db)
        self.run_example()

```

```python

```

```python

```

### use guide on Fastdb.dbprint


1. don't use for the line start with `elif`, as putting `dbprintinsert` above `elif` without indentation will cause syntax error. I am not sure whether I need to fix it now.


see example [here](./examples/Fastdb.ipynb)


test it with example [here](./examples/print.ipynb)


## reliveonce

```python
#| export
def reliveonce(func, # the current func
               oldfunc:str, # the old version of func in string
               alive:bool=True, # True to bring old to live, False to return back to normal
               db=False): # True, to print for debugging
    "Replace current version of srcode with older version, and back to normal"
    if alive:
        safety = func
        block = ast.parse(oldfunc, mode='exec')
        exec(compile(block, '<string>', mode='exec'), globals().update(func.__globals__))
        if db:
            print(f"after exec: list(locals().keys()): {list(locals().keys())}")
            print(f"before update: inspect.getsourcefile(func.__globals__[func.__name__]): {inspect.getsourcefile(func.__globals__[func.__name__])}")
        
        # update the old version of func from locals() into func.__globals__, so that outside this reliveonce function, the old func can be used
        func.__globals__.update(locals())
        if db:
            print(f"after update: inspect.getsourcefile(func.__globals__[func.__name__]): {inspect.getsourcefile(func.__globals__[func.__name__])}")
    else:
        func.__globals__[func.__name__] = func.__globals__['safety']
```

see example [here](./Demos/Debug_FixSigMeta_signature.ipynb#Using-old-_signature_from_callable)


## Fastdb.debug

```python
#| export
@patch
def debug(self:Fastdb):
    print(f"{self.orisrc.__name__}\'s dbsrc code: ==============")
    address = "/tmp/BypassNewMeta.py"
    dbsrc = open(address, "r+")
    print(dbsrc.read())
    print()
    print(f"{self.orisrc.__name__}\'s example processed with dbsrc: ===============")
    print(self.eg)
```

```python

```

```python

```

#|hide
## Export

```python
#| hide
from nbdev import nbdev_export
nbdev_export()
```

#|hide
## Send to Obsidian

```python
#| hide
!jupytext --to md /Users/Natsume/Documents/fastdebug/00_core.ipynb
!mv /Users/Natsume/Documents/fastdebug/00_core.md \
/Users/Natsume/Documents/divefastai/Debuggable/jupytext/

!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```

```python

```
