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

# core

> the core functionalities of fastdebug

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
from pprint import pprint
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
    newfoo = locals()["foo"]
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
tstenv()
```

## make a colorful string

```python
#| export
from fastcore.basics import *
```

```python
#|export
class dbcolors:
    g = '\033[92m' #GREEN
    y = '\033[93m' #YELLOW
    r = '\033[91m' #RED
    reset = '\033[0m' #RESET COLOR
```

```python
#|export
def colorize(cmt, color:str=None):
    if color == "g":
        return dbcolors.g + cmt + dbcolors.reset
    elif color == "y":
        return dbcolors.y + cmt + dbcolors.reset
    elif color == "r":
        return dbcolors.r + cmt + dbcolors.reset
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
#| export
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
#| export
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
                alignright(colcmt) # also print the comment

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
#| export
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
            output = f"Running your code block => "
            print('{:<157}'.format(c))       
            print('{:>157}'.format(output))  
            print('The code block printout => : ')
            block = ast.parse(c, mode='exec')
            exec(compile(block, '<string>', mode='exec'), globals().update(env))
        
        # handle assignment: 2. when = occur before if; 1. when no if only =
        elif ("=" in c and "if" not in c) or ("=" in c and c.find("=") < c.find("if")): # make sure assignment and !== and == are differentiated
            
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
            # print(f"{c} => {c} : {eval(c, globals().update(env))}") 
            output = f"{c} => {c} : {eval(c, globals().update(env))}"
            print('{:>157}'.format(output))   
            
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
#| export
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
#| export
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
#| exports
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

```python
#| exports        
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

```python
#| exports
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

### use dbprint to override the original official code without changing its own pyfile

<!-- #region -->
see the example [here](./examples/dbprint.ipynb#make-inspect.signature-to-run-our-dbsrc-code)

```python
dbsig = sig.dbprint(29, "why has to unwrap?", "hasattr(obj, '__signature__')")
inspect._signature_from_callable = dbsig
pprint(inspect.signature(Foo))
sig.print(part=1)
```
<!-- #endregion -->

### use guide on Fastdb.dbprint


1. don't use for the line start with `elif`, as putting `dbprintinsert` above `elif` without indentation will cause syntax error. I am not sure whether I need to fix it now.


see example [here](./examples/Fastdb.ipynb)


test it with example [here](./examples/print.ipynb)


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
```

```python
#| hide
!jupyter nbconvert --config /Users/Natsume/Documents/mynbcfg.py --to markdown \
--output-dir /Users/Natsume/Documents/divefastai/Debuggable/nbconvert
```
