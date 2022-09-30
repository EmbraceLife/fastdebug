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

# 04_rm_self


## imports

```python
from fastdebug.utils import *
from fastdebug.core import *
from fastcore.meta import *
```

```python
from fastcore.meta import _rm_self
```

## set up

```python
g = locals()
fdb = Fastdb(_rm_self, outloc = g)
```

```python
fdb.print()
```

```python
class Foo:
    def __init__(self, a, b:int=1): pass
pprint(inspect.signature(Foo.__init__))
pprint(_rm_self(inspect.signature(Foo.__init__)))
```

```python
fdb.eg = """
class Foo:
    def __init__(self, a, b:int=1): pass
pprint(inspect.signature(Foo.__init__))
pprint(_rm_self(inspect.signature(Foo.__init__)))
"""
```

## document

```python
fdb.docsrc(0, "remove parameter self from a signature which has self;")
fdb.docsrc(1, "how to access parameters from a signature; how is parameters stored in sig; how to turn parameters into a dict;", \
           "sig", "sig.parameters", "dict(sig.parameters)")
fdb.docsrc(2, "how to remove the self parameter from the dict of sig;")
fdb.docsrc(3, "how to update a sig using a updated dict of sig's parameters", "sigd", "sigd.values()")
```

## snoop

```python
fdb.snoop()
```

```python
fdb.print()
```
