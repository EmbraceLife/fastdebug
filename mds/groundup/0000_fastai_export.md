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

# 0000_export
how to use nbdev export 

```python
#| default_exp test
```

```python
#| export
TEST = "test"
```

```python
#| export access_data
test1 = "from "
```

## nb_name

```python
from fastdebug.utils import *
```

```python
nbname = nb_name()
```

```python
nbname
```

## notebook as json

```python
import json
```

```python
if bool(nbname):
    d = json.load(open(nbname,'r'))
else: 
    d = None
```

```python
d
```

```python
all_src = []
if bool(d):
    for dct in d['cells']:
        all_src = all_src + dct['source']
all_src
```

```python
#|hide
import nbdev; nbdev.nbdev_export()
```

```python

```
