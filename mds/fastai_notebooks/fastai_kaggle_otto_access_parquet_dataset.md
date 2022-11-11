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

I converated the dataset for this competition from `jsonl` to `csv` and `parquet` so that it is easy to work with using our favorite set of tools! 🙂 You can find the converted dataset [here](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint).

Unfotunately, it was impossible to process this data on Kaggle due to not enough RAM. I carried out the processing on my local machine and uploaded the processed data (will share the code I used for processing as well, please see [the associate thread](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843)).

The 'type' information was represented as a string, which takes up a lot of memory. Instead, I cast it to `np.uint8`. This makes the data much smaller and easier to work with without any loss of information!

Let me walk you through how everything is set up so that you can use this data in your work.


### rd: recsys - otto - get started - copy and paste dataset path and use !ls to see what inside

```python
# Here are the files
!ls ../input/otto-full-optimized-memory-footprint/
```

### rd: recsys - otto - get started - pd.read_parquet('copy and paste the dataset path')

```python
import pandas as pd

# There is a version of the data stored as `csv` as well
# but I recommend you use `parquet` as I do here -- it is much faster

train = pd.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
test = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
```

```python
train.head()
```

The `type` column has been encoded as integers. To translate between the integer and original representations, please use the following.


### rd: recsys - otto - get started - load a function from a pickle file with pickle5, with open as fh: and pick.load(fh)

```python
!pip install pickle5

import pickle5 as pickle

with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh:
    id2type = pickle.load(fh)
with open('../input/otto-full-optimized-memory-footprint/type2id.pkl', "rb") as fh:
    type2id = pickle.load(fh)
```

Using `id2type` we can convert from integer to string representation (and we can use `type2id` to go in the other direction)

```python
id2type, type2id
```

```python
id2type
```

### rd: recsys - otto - get started - convert int back to string, train.iloc[:1000].type.map(lambda i: id2type[i])

```python
type_as_string = train.iloc[:1000].type.map(lambda i: id2type[i])
type_as_string.head()
```

And we can just as easily go back from strings to idxs.

```python
type_as_string.map(lambda i: type2id[i]).head()
```

```python
type_as_string.map(type2id).head()
```

And that's it! I hope this will speed you along in your work 🙂

If you found this useful, I would be extremely grateful if you could please upvote this notebook and [the associated dataset](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint).

Thank you so much for your support! Happy kaggling! 🥳
