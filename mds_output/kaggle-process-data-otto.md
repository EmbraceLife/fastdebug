# process_data_OTTO

Original code is given by @radek1 to make the original OTTO dataset easier to process and faster to access.

This notebook is to experiment in order to understand what the original code does.

## rd: recsys - otto - process data - How to convert dataset from jsonl file into parquet file to save disk tremendously, and convert type column from string to uint8 and ts column from int64 to int32 by dividing 1000 first to reduce RAM usage significantly


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

### rd: recsys - otto - process data - create vocab or map between id and type using dict and list - id2type = ['clicks', 'carts', 'orders'] - type2id = {a: i for i, a in enumerate(id2type)}


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
#                 types.append(type2id[event['type']])
                types.append(event['type']) 
                
    return pd.DataFrame(data={'session': sessions, 'aid': aids, 'ts': tss, 'type': types})
```


```python
pd.DataFrame.memory_usage??
```

### rd: recsys - otto - process data - chunks = pd.read_json(fn, lines=True, chunksize=100_000) - for chunk in chunks: - for row_idx, session_data in chunk.iterrows(): - session_data.session - for event in session_data.events: - aids.append(event['aid']) - tss.append(event['ts'])


```python
def jsonl_to_df(fn):
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
                types.append(type2id[event['type']])

    return pd.DataFrame(data={'session': sessions, 'aid': aids, 'ts': tss, 'type': types})
```

### rd: recsys - otto - process data - and check RAM of a df and save df into parquet or csv file - test_df_str.memory_usage() - test_df_str.to_parquet('test_keep_str.parquet', index=False) - test_df_str.to_csv('test_keep_str.csv', index=False)


```python
%%time
test_df_str = jsonl_to_df_type_str(test_path)
test_df_str.memory_usage()
test_df_str.to_parquet('test_keep_str.parquet', index=False)
test_df_str.to_csv('test_keep_str.csv', index=False)
```

### rd: recsys - otto - process data - How much RAM does convert string to uint8 save? - test_df.type = test_df.type.astype(np.uint8)


```python
%%time
test_df = jsonl_to_df(test_path)
```


```python
test_df.memory_usage()
```


```python
test_df.type = test_df.type.astype(np.uint8)
test_df.memory_usage()
```


```python
test_df.memory_usage()
test_df_str.memory_usage()
```

### rd: recsys - otto - process data - convert `ts` from int64 to int32 without `/1000` will lose a lot of info - (test_df_updated.ts/1000).astype(np.int32)


```python
test_df_updated = jsonl_to_df(test_path)
```

[the difference between int64, int32](http://www.ece.northwestern.edu/local-apps/matlabhelp/techdoc/ref/int8.html)


```python
test_df_updated.ts.dtype
```


```python
test_df_updated.ts[0], test_df_updated.ts.astype(np.int32)[0] 
# without dividing 1000, ts is too large to be contained by int 32, directly convert to int32 can result in info loss 
```


```python
# 1_661_724_000_278 is beyond int32 range, but 1_661_724_000 (divided by 1000) is within int32
test_df_updated.ts[0],test_df_updated.ts[0]/1000, (test_df_updated.ts/1000).astype(np.int32)[0] 
```

### rd: recsys - otto - process data - dividing ts by 1000 only affect on milisecond accuracy not second accuracy - datetime.datetime.fromtimestamp((test_df_updated.ts/1000).astype(np.int32)[100])


```python
import datetime
```


```python
datetime.date.fromtimestamp(test_df_updated.ts[100]/1000)
datetime.datetime.fromtimestamp(test_df_updated.ts[100]/1000) # the milisecond is the last item 278000
datetime.datetime.fromtimestamp((test_df_updated.ts/1000).astype(np.int32)[100]) # the milisecond is removed
```

### rd: recsys - otto - process data - How much RAM can be saved by dividing `ts` by 1000 - test_df_updated.ts = (test_df_updated.ts / 1000).astype(np.int32) 


```python
test_df_updated = jsonl_to_df(test_path)
test_df_updated.type = test_df_updated.type.astype(np.uint8)
test_df_updated.ts = (test_df_updated.ts / 1000).astype(np.int32) 
test_df_updated.memory_usage() # the RAM used by `ts` is halved
```

### rd: recsys - otto - process data - how much disk can be saved by saving jsonl to parquet - os.path.getsize(path)


```python
def filesize(path):
    import os
    return os.path.getsize(path)
```


```python
# [("jsonl=>{}".format(path), filesize(path)) for path in ['../input/otto-recommender-system/test.jsonl','test_keep_str.parquet', 'test_keep_str.csv', 'test_str2int.parquet', 'test0_str2int.csv', 'test_str2uint8.parquet', 'test_str2uint8.csv']]
```

It seems that by converting jsonl to csv can half the size, to parquet can quarter the size; 

Converting `type` from string to int using `type2id` can save 34MB which is about 10% size of original test dataset during conversion from jsonl to csv; but it has no observable effect during convertion to parquet file.

Although converting `type` further from `int` to `uint8` in a parquet file can shrink the size, but the amount is barely observable, about 200byte.

So, it seems that conversion from jsonl to parquet alone does the most heavy lifting in reducing size, converting from string to int and then uint8 is helpful but in this case has no significant effect in reducing the size. 


```python

```
