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

# process_data_OTTO

[Original process_data.ipynb](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843) is given by @radek1 to make the original OTTO dataset easier to process and faster to access.

This notebook is to experiment in order to understand what the original code does.


### imports

```python
# make sure fastkaggle is install and imported
import os
```

```python
try: import fastkaggle
except ModuleNotFoundError:
    os.system("pip install -Uq fastkaggle")
```

```python
from fastkaggle import *
```

```python
# use fastdebug.utils 
if iskaggle: os.system("pip install nbdev snoop")
```

```python
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

```python
comp = "otto-recommender-system"
path = setup_comp(comp)
```

```python
list(path.ls())
```

```python
train_path = path.ls()[2]
test_path = path.ls()[1]
train_path
test_path
```

### jn: revisit process_data notebook and change get started to process_data in title. /2022-11-14


## The original code start from below


### rd: recsys - otto - process data - save a list or dict into pkl and load them - id2type = ['clicks', 'carts', 'orders'] - type2id = {a: i for i, a in enumerate(id2type)} - pd.to_pickle(id2type, 'id2type.pkl')

```python
import numpy as np
import pandas as pd
```

```python
id2type = ['clicks', 'carts', 'orders'] # I have analyzed the data
                                          # and so I know we can expect these event types
type2id = {a: i for i, a in enumerate(id2type)}

id2type, type2id
```

```python
pd.to_pickle(id2type, 'id2type.pkl')
pd.to_pickle(type2id, 'type2id.pkl')
```

```python

```

```python
# !pip install pickle5

import pickle5 as pickle

with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh:
    id2type = pickle.load(fh)
```

```python

```

### rd: recsys - otto - process data - how to process jsonl file to df (pd.read_json, chunk.iterrows) - chunks = pd.read_json(fn, lines=True, chunksize=2) - for chunk in chunks: - for row_idx, session_data in chunk.iterrows(): - sessions = [] - num_events = len(session_data.events) - sessions += ([session_data.session] * num_events)

```python
[1,2] * 2 + [3,4]
```

```python
def jsonl_to_df(fn):
    sessions = []
    aids = []
    tss = []
    types = []

    # lines: True => Read the file as a json object per line
    # chunksize=100_000 => Return JsonReader object for iteration; If this is None, the file will be read into memory all at once.
    # 100_000 == 100000, I guess _ is for easy view; each chunk will have 100,000 lines/objects
    chunks = pd.read_json(fn, lines=True, chunksize=2) # you can change to chunksize=2 to experiment
    
    for chunk in chunks: # each chunk will have 2 items (if chunksize=2)
        info = "each item is a session, each session has two items 'session' and 'events'"
        info1 = "each session has variable amount of events"
        pp(info, info1)
        for row_idx, session_data in chunk.iterrows():
            # each session_data is a pd.Series object and contain two columns: 'session' (int) and 'events' (list of dicts)
            # each dict of 'events' has keys: 'aid', 'ts', 'type', 'clicks'
            num_events = len(session_data.events) # total num of events of each session_data or each item in a chunk
            sessions += ([session_data.session] * num_events) # sessions is a list contains the same session value for every event

            pp(session_data.session, len(session_data.events), session_data.events[0], session_data.events[1])
            for event in session_data.events: # each session_data.events actually have different num of events
                aids.append(event['aid']) # aids is a list containing value of `aid` of every event
                tss.append(event['ts']) # tss is a list containing value of `ts` of every event
                types.append(type2id[event['type']]) # types is a list containing value of `type`|`id` of every event 
                # (`id` is created by Radek from above)
        return
    # now we can combine all the data info `session`, `aids`, `tss` and `types` into a DataFrame for each session_data
    return pd.DataFrame(data={'session': sessions, 'aid': aids, 'ts': tss, 'type': types})
```

```python
train_df = jsonl_to_df(train_path)
```

```python
# %%time

# # train_df = jsonl_to_df('data/train.jsonl')
# train_df = jsonl_to_df(train_path)
# train_df.type = train_df.type.astype(np.uint8) # a tiny bit of further memory footprint optimization (original type is just int class)
# # train_df.to_parquet('train.parquet', index=False)
# train_df.to_csv('train.csv', index=False)

# del train_df
```

### rd: src - recsys - otto - process data - jsonl_to_df

```python
def jsonl_to_df(fn):
    sessions = []
    aids = []
    tss = []
    types = []

    # lines: True => Read the file as a json object per line
    # chunksize=100_000 => Return JsonReader object for iteration; If this is None, the file will be read into memory all at once.
    # 100_000 == 100000, I guess _ is for easy view; each chunk will have 100,000 lines/objects
    chunks = pd.read_json(fn, lines=True, chunksize=2) # you can change to chunksize=2 to experiment
    
    for chunk in chunks: # each chunk will have 2 items (if chunksize=2)
        info = "each item is a session, each session has two items 'session' and 'events'"
        info1 = "each session has variable amount of events"
        pp(info, info1)
        for row_idx, session_data in chunk.iterrows():
            # each session_data is a pd.Series object and contain two columns: 'session' (int) and 'events' (list of dicts)
            # each dict of 'events' has keys: 'aid', 'ts', 'type', 'clicks'
            num_events = len(session_data.events) # total num of events of each session_data or each item in a chunk
            sessions += ([session_data.session] * num_events) # sessions is a list contains the same session value for every event

            pp(session_data.session, len(session_data.events), session_data.events[0], session_data.events[1])
            for event in session_data.events: # each session_data.events actually have different num of events
                aids.append(event['aid']) # aids is a list containing value of `aid` of every event
                tss.append(event['ts']) # tss is a list containing value of `ts` of every event
                types.append(type2id[event['type']]) # types is a list containing value of `type`|`id` of every event 
                # (`id` is created by Radek from above)
        return
    # now we can combine all the data info `session`, `aids`, `tss` and `types` into a DataFrame for each session_data
    return pd.DataFrame(data={'session': sessions, 'aid': aids, 'ts': tss, 'type': types})
```

### rd: recsys - otto - process data - 400MB parquet file takes up nearly 4GB ram on Kaggle


see more detailed findings of mine in the discussion [here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843#2024279)


### rd: recsys - otto - process data - use parquet over csv, why and how - test_df.type = test_df.type.astype(np.uint8) - test_df.to_parquet('test.parquet', index=False) - test_df.to_csv('test.csv', index=False)


Summary of technical features of parquet files
- Apache Parquet is column-oriented and designed to provide efficient columnar storage compared to row-based file types such as CSV.
- Parquet files were designed with complex nested data structures in mind.
- Apache Parquet is designed to support very efficient compression and encoding schemes.
- Apache Parquet generates lower storage costs for data files and maximizes the effectiveness of data queries with current cloud technologies such as Amazon Athena, Redshift Spectrum, BigQuery and Azure Data Lakes.
- Licensed under the Apache license and available for any project.

```python
%%time

# test_df = jsonl_to_df('data/test.jsonl')
test_df = jsonl_to_df(test_path)
test_df.type = test_df.type.astype(np.uint8)
test_df.to_parquet('test.parquet', index=False)
test_df.to_csv('test.csv', index=False)
```

```python
del test_df
```

### rd: recsys - otto - process data - use parquet to instead of jsonl or csv to save space on disk - os.path.getsize(path)


for details see discussion [here](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files/comments#2025116)

```python
def jsonl_to_df_type_str(fn):
    sessions = []
    aids = []
    tss = []
    types = []

    chunks = pd.read_json(fn, lines=True, chunksize=100_000) 
    
    for chunk in chunks:
        for row_idx, session_data in chunk.iterrows():
            num_events = len(session_data.events) 
            sessions += ([session_data.session] * num_events) 
            for event in session_data.events:
                aids.append(event['aid']) 
                tss.append(event['ts']) 
                types.append(event['type'])

    return pd.DataFrame(data={'session': sessions, 'aid': aids, 'ts': tss, 'type': types})
```

```python
%%time
test_df_str = jsonl_to_df_type_str(test_path)
test_df_str.to_parquet('test_keep_str.parquet', index=False)
test_df_str.to_csv('test_keep_str.csv', index=False)

test_df = jsonl_to_df(test_path)
test_df.to_parquet('test_str2int.parquet', index=False)
test_df.to_csv('test0_str2int.csv', index=False)

test_df.type = test_df.type.astype(np.uint8)
test_df.to_parquet('test_str2uint8.parquet', index=False)
test_df.to_csv('test_str2uint8.csv', index=False)
```

```python
def filesize(path): # path: str or path
    import os
    return os.path.getsize(path)
```

```python
[("jsonl=>{}".format(path), filesize(path)) for path in ['../input/otto-recommender-system/test.jsonl','test_keep_str.parquet', 'test_keep_str.csv', 'test_str2int.parquet', 'test0_str2int.csv', 'test_str2uint8.parquet', 'test_str2uint8.csv']]
```

It seems that by converting jsonl to csv can half the size, to parquet can quarter the size;

Converting type from string to int using type2id can save 34MB which is about 10% size of original test dataset during conversion from jsonl to csv; but it has no observable effect during convertion to parquet file.

Although converting type further from int to uint8 in a parquet file can shrink the size, but the amount is barely observable, about 200byte.

So, it seems that conversion from jsonl to parquet alone does the most heavy lifting in reducing size, converting from string to int and then uint8 is helpful but in this case has no significant effect in reducing the size.


### rd: recsys - otto - process data - use `uint8` instead of `int` or `str` to reduce RAM usage by 9 times - test_df.memory_usage()


read the discussion [here](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files/comments#2025129)

```python
test_df.memory_usage()
test_df_str.memory_usage()
```

### jn: process_data revisited and get the name straight for search (done) /2022-11-14
