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


## `whatinside` and `gw['whatinside']` are the same

```python
from fastdebug.utils import *
import inspect
```

```python
gw = {}
gw.update(whatinside.__globals__)
len(gw)
```

```python
whatinside.__code__.__repr__()
```

```python
gw['whatinside'].__code__
```

```python
inspect.getsource(gw['whatinside'])
```

```python
gw['whatinside'] == whatinside
```

```python

```

```python

```

## Dot it in a more natural and ordered way

```python
from fastdebug.core import *
```

```python
import inspect
```

### Prepare env for Fastdb.dbprint

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
g['dbprint'] # has nothing, this is probably come from the notebook 00_core
```

```python
g['Fastdb'].dbprint
```

```python
inspect.getdoc(g['Fastdb'].dbprint)
```

```python
Fastdb.dbprint == g['Fastdb'].dbprint
```

```python
inspect.getsource(Fastdb.dbprint)
```

### Create Fastdb object for `Fastdb.dbprint`

```python
fdb = Fastdb(Fastdb.dbprint, g)
```

```python
fdb.print(10, 1)
```

After running the following line, `Fastdb.dbprint` is updated by exec and source code can't be seen anymore.

```python
dbsrc = fdb.dbprint(9, "keep original src safe", "self.orisrc", showdbsrc=True)
```

As updating `Fastdb.dbprint` with exec will send the updated src to the class `Fastdb`, dbsrc is actually NoneType.


Luckily, we still got `fdb.orisrc` which keep the original `Fastdb.dbprint` for us.


So, we can run `Fastdb.dbprint = fdb.orisrc` to get `Fastdb.dbprint` back to normal


### Make an example with new `Fastdb.dbprint`

```python
from fastdebug.utils import *
```

```python
import fastdebug.core as core
```

```python
gw = {}
gw.update(whatinside.__globals__)
len(gw)
```

```python
fdbw = Fastdb(whatinside, gw) # we are using the updated Fastdb whose dbprint has dbprintinsert() ready
```

```python
fdbw.print(10,1)
```

#### Important lines to run

```python
new = fdbw.dbprint(9, "count num of funcs in __all__", "type(mo)") # the updated dbprint will run here.
whatinside = new
whatinside(core)
whatinside = fdbw.orisrc # Important! remember to bring whatinside back to normal after each srcline exploration
```

### explore the second srcline of dbprint

```python
fdb.print(20, 1) # we are using fdb.orisrc to print, so no dbprintinsert() will be seen.
```

```python
Fastdb.dbprint = fdb.orisrc
fdb.dbprint(10, "collect and organize cmt by idx", "self.cmts", "dbcode", "cmt", showdbsrc=True)
```

```python
# whatinside = fdbw.orisrc
fdbw = Fastdb(whatinside, gw)
fdbw.print(10,1)
```

```python
new = fdbw.dbprint(9, "count num of funcs in __all__", "type(mo)")
whatinside = new
whatinside(core)
```

```python
Fastdb.dbprint.__qualname__
```

```python
Fastdb.dbprint.__qualname__ in g
```

```python
Fastdb.dbprint.__qualname__.split('.')[0] in g
```

```python
fdb.orisrc.__qualname__.split('.')[0]
```

```python

```
