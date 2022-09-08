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

# Explore Fastdb class

```python
from IPython.display import display, HTML 
```

```python
display(HTML("<style>.container { width:90% !important; }</style>"))
```

## Why


So far I have created two demos, a simple one on `fastdebug.utils.whatinside` and a real world one on `fastcore.meta.FixSigMeta`. They should be enough as demos to show the usefulness of fastdebug library. 


However, I want to be thorough and really prove to myself that it can help me to conquer unknowns and clear doubts with ease and joy when exploring source code. 


Fastdb class and the funcs it uses contains all the tricks I learnt and difficulties I conquered which I don't always remember how and why. In fact I don't want to remember them. I want to just run all cells here and skim through the notebook and everything makes sense to me. 


Can fastdebug help me achieve it? Let's give a try!


## How to do it?


I need a few examples first. Maybe I could just use the simple demo as example for `Fastdb`.


## A example of using `Fastdb.dbprint`


### Get env 

```python
from fastdebug.utils import whatinside
```

```python
gw = {}
gw.update(whatinside.__globals__)
len(gw)
```

### Actual example

```python
from fastdebug.core import *
```

```python
fdbw = Fastdb(whatinside, gw)
```

```python
fdbw.print(10,1)
```

```python
import fastdebug.core as core
```

```python
dbw = fdbw.dbprint(9, "what is mo?", "mo", "type(mo)", "type(mo.__all__)", showdbsrc=True)
whatinside = dbw
whatinside(core)
```

## Explore `fdbw.dbprintt` with the example above


### Get env for `fdbw.dbprint`

```python
import inspect
```

```python
from pprint import pprint
```

```python
g = {}
g.update(fdbw.dbprint.__globals__)
len(g)
```

```python
# import fastdebug.core as core
# g = {}
# g.update(core.__dict__)
# len(g)
```

```python
"fdbw" in g
```

```python
'dbprint' in g
```

```python
type(g['dbprint'])
```

```python
'Fastdb' in g
```

```python
type(g['Fastdb'].dbprint)
```

```python
inspect.signature(g['Fastdb'].dbprint)
```

```python
pprint(inspect.getsource(g['Fastdb'].dbprint), width=157)
```

### fdbw.dbprint vs Fastdb.dbprint

```python
fdbw.dbprint == Fastdb.dbprint
```

```python
type(fdbw.dbprint)
```

```python
type(Fastdb.dbprint)
```

```python
inspect.signature(Fastdb.dbprint)
```

```python
inspect.signature(fdbw.dbprint)
```

```python
pprint(inspect.getsource(fdbw.dbprint), width=157)
```

### Create a dbsrc for `fdbw.dbprint`

```python
fdb = Fastdb(fdbw.dbprint, g)
```

```python
pprint(inspect.getsource(fdbw.dbprint), width=157)
```

```python
fdb.orisrc.__name__
```

```python
type(fdb.orisrc)
```

```python
fdb.print(15, 1)
```

```python
dbsrc = fdb.dbprint(9, "keep orisrc safe", "self.orisrc", showdbsrc=True)
```

```python
# pprint(inspect.getsource(fdbw.dbprint), width=157) # no source
```

```python
"Fastdb" in list(g.keys())
```

```python
"dbprint" in list(g.keys())
```

```python
g['dbprint']
```

```python
type(g['Fastdb'].dbprint)
```

```python
inspect.signature(g['Fastdb'].dbprint)
```

```python
g['Fastdb'].dbprint
```

```python
fdbw.dbprint = g['Fastdb'].dbprint
```

### Will the example in the section above use our db version of `Fastdb.dbprint`?

```python
try:
    dbw = fdbw.dbprint(9, "what is mo?", "mo", "type(mo)", "type(mo.__all__)")
except AttributeError as e:
    print(e)
whatinside = dbw
whatinside(core)
```

## Dot it in a more natural and ordered way


### Create Fastdb object on Fastdb.dbprint

```python
g = {}
g.update(Fastdb.dbprint.__globals__)
len(g)
```

```python
'Fastdb.dbprint' in g
```

```python
'Fastdb' in g
```

```python
'dbprint' in g
```

```python
g['dbprint']
```

```python
g['Fastdb'].dbprint
```

```python
inspect.getsource(g['Fastdb'].dbprint)
```

```python

```
