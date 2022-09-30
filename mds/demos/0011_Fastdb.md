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

# 0011_Fastdb

```python
from fastdebug.utils import *
from fastdebug.core import *
```

```python
fdb = Fastdb(Fastdb.printtitle)
```

```python
fdb.docsrc(5, "how to use :=<, :=>, :=^ with format to align text to left, right, and middle")
fdb.print()
```

```python
fdb = Fastdb(Fastdb.snoop)
```

```python
fdb.print()
```

```python
fdb = Fastdb(Fastdb.create_explore_str)
```

```python
fdb.print()
```

```python
fastview(Fastdb.printtitle)
```

```python

```
