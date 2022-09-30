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

# 05_test_sig


## imports

```python
from fastdebug.utils import *
from fastdebug.core import *
```

```python
from fastcore.meta import *
import fastcore.meta as fm
```

## setups

```python
whatinside(fm, dun=True)
```

```python
g = locals()
fdb = Fastdb(test_sig, outloc=g)
```

```python
fdb.print()
```

```python
fdb.eg = """
def func_1(h,i,j): pass
test_sig(func_1, '(h, i, j)')
"""

fdb.eg = """
class T:
    def __init__(self, a, b): pass
test_sig(T, '(a, b)')
"""
fdb.eg = """
def func_2(h,i=3, j=[5,6]): pass
test_sig(func_2, '(h, i=3, j=[5, 6])')
"""
```

## documents

```python
fdb.docsrc(2, "test_sig is to test two strings with test_eq; how to turn a signature into a string;", "pprint(inspect.signature(f))", \
"inspect.signature(f)", "str(inspect.signature(f))")
```

## snoop

```python
fdb.snoop()
```

```python
fdb.docsrc(1, "test_sig(f:FunctionType or ClassType, b:str); test_sig will get f's signature as a string; \
b is a signature in string provided by the user; in fact, test_sig is to compare two strings")
```

```python
fdb.print()
```

```python

```
