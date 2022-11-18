---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

This notebook is to explore and experiment on @radek1's notebook [simplified without need for chunking](https://www.kaggle.com/code/radek1/last-20-aids) and the [original code](https://www.kaggle.com/code/ttahara/last-aid-20) is by @ttahara


## rd: recsys - otto - last 20 aids - The purpose of this notebook - Grabbing the last 20 aid for each session of the test set, use them as prediction and submit or run local validation to see how powerful can the last 20 aids be.





## import dataset and functions

```python
import os

try: import fastkaggle
except ModuleNotFoundError:
    os.system("pip install -Uq fastkaggle")

from fastkaggle import *

# use fastdebug.utils 
if iskaggle: os.system("pip install nbdev snoop")

if iskaggle:
    path = "../input/fastdebugutils0"
    import sys
    sys.path
    sys.path.insert(1, path)
    import utils as fu
    from utils import *
else: 
    from fastdebug.utils import *
    import fastdebug.utils as fu
```

```python _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import pandas as pd

train = pd.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
test = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')

!pip install pickle5
import pickle5 as pickle

with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh:
    id2type = pickle.load(fh)
with open('../input/otto-full-optimized-memory-footprint/type2id.pkl', "rb") as fh:
    type2id = pickle.load(fh)
    
sample_sub = pd.read_csv('../input/otto-recommender-system/sample_submission.csv')
```

```python
sample_sub.head()
```

```python
test.head()
```

### rd: recsys - otto - last 20 aid - sort df by two columns - test.sort_values(['session', 'ts'])

```python
%%time

test = test.sort_values(['session', 'ts'])
test.head()
```

### rd: recsys - otto - last 20 aid - take last 20 aids from each session - test.groupby('session')['aid'].apply(lambda x: list(x)[-20:])

```python
session_aids = test.groupby('session')['aid'].apply(lambda x: list(x)[-20:])
```

```python

```

```python
session_aids.head()
```

```python
session_aids.index
```

### rd: recsys - otto - last 20 aid - loop through a Series with session as index with a list as the only column's value - for session, aids in session_aids.iteritems():


### rd: recsys - otto - last 20 aid - turn a list into a string with values connected with empty space - labels.append(' '.join([str(a) for a in aids]))

```python
%%time

session_type = []
labels = []
session_types = ['clicks', 'carts', 'orders']

for session, aids in session_aids.iteritems():
    for st in session_types:
        session_type.append(f'{session}_{st}')
        labels.append(' '.join([str(a) for a in aids]))
```

### rd: recsys - otto - last 20 aid - make a df from a dict with two lists as values - pd.DataFrame({'session_type': session_type, 'labels': labels})

```python
submission = pd.DataFrame({'session_type': session_type, 'labels': labels})
```

```python
submission.head()
```

```python
submission.to_csv('submission.csv', index=False)
```
