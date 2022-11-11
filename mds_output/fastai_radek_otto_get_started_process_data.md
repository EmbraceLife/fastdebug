# process_data_OTTO

[Original process_data.ipynb](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843) is given by @radek1 to make the original OTTO dataset easier to process and faster to access.

This notebook is to experiment in order to understand what the original code does.


```
# make sure fastkaggle is install and imported
import os
```


```
try: import fastkaggle
except ModuleNotFoundError:
    os.system("pip install -Uq fastkaggle")
```

    WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv



```
from fastkaggle import *
```


```
# use fastdebug.utils 
if iskaggle: os.system("pip install nbdev snoop")
```

    Collecting nbdev
      Downloading nbdev-2.3.9-py3-none-any.whl (64 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.1/64.1 kB 492.2 kB/s eta 0:00:00
    Collecting snoop
      Downloading snoop-0.4.2-py2.py3-none-any.whl (27 kB)
    Requirement already satisfied: fastcore>=1.5.27 in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.5.27)
    Collecting asttokens
      Downloading asttokens-2.1.0-py2.py3-none-any.whl (26 kB)
    Requirement already satisfied: PyYAML in /opt/conda/lib/python3.7/site-packages (from nbdev) (6.0)
    Requirement already satisfied: astunparse in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.6.3)
    Collecting ghapi>=1.0.3
      Downloading ghapi-1.0.3-py3-none-any.whl (58 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.1/58.1 kB 1.7 MB/s eta 0:00:00
    Collecting execnb>=0.1.4
      Downloading execnb-0.1.4-py3-none-any.whl (13 kB)
    Collecting watchdog
      Downloading watchdog-2.1.9-py3-none-manylinux2014_x86_64.whl (78 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.4/78.4 kB 2.1 MB/s eta 0:00:00
    Collecting cheap-repr>=0.4.0
      Downloading cheap_repr-0.5.1-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from snoop) (1.15.0)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.7/site-packages (from snoop) (2.12.0)
    Collecting executing
      Downloading executing-1.2.0-py2.py3-none-any.whl (24 kB)
    Requirement already satisfied: ipython in /opt/conda/lib/python3.7/site-packages (from execnb>=0.1.4->nbdev) (7.33.0)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.27->nbdev) (21.3)
    Requirement already satisfied: pip in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.27->nbdev) (22.1.2)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.7/site-packages (from astunparse->nbdev) (0.37.1)
    Requirement already satisfied: traitlets>=4.2 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (5.3.0)
    Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (4.8.0)
    Requirement already satisfied: pickleshare in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (3.0.30)
    Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.18.1)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.2.0)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (5.1.1)
    Requirement already satisfied: setuptools>=18.5 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (59.8.0)
    Requirement already satisfied: matplotlib-inline in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.1.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.7/site-packages (from packaging->fastcore>=1.5.27->nbdev) (3.0.9)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/conda/lib/python3.7/site-packages (from jedi>=0.16->ipython->execnb>=0.1.4->nbdev) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.7/site-packages (from pexpect>4.3->ipython->execnb>=0.1.4->nbdev) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython->execnb>=0.1.4->nbdev) (0.2.5)
    Installing collected packages: executing, cheap-repr, watchdog, asttokens, snoop, ghapi, execnb, nbdev
    Successfully installed asttokens-2.1.0 cheap-repr-0.5.1 execnb-0.1.4 executing-1.2.0 ghapi-1.0.3 nbdev-2.3.9 snoop-0.4.2 watchdog-2.1.9


    WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv



```
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


<style>.container { width:100% !important; }</style>



```
comp = "otto-recommender-system"
path = setup_comp(comp)
```


```
list(path.ls())
```




    [Path('../input/otto-recommender-system/sample_submission.csv'),
     Path('../input/otto-recommender-system/test.jsonl'),
     Path('../input/otto-recommender-system/train.jsonl')]




```
train_path = path.ls()[2]
test_path = path.ls()[1]
train_path
test_path
```




    Path('../input/otto-recommender-system/train.jsonl')






    Path('../input/otto-recommender-system/test.jsonl')



## The original code start from below

### rd: recsys - otto - get started - save a list or dict into pkl and load them


```
import numpy as np
import pandas as pd
```


```
id2type = ['clicks', 'carts', 'orders'] # I have analyzed the data
                                          # and so I know we can expect these event types
type2id = {a: i for i, a in enumerate(id2type)}

id2type, type2id
```




    (['clicks', 'carts', 'orders'], {'clicks': 0, 'carts': 1, 'orders': 2})




```
pd.to_pickle(id2type, 'id2type.pkl')
pd.to_pickle(type2id, 'type2id.pkl')
```


```

```


```
# !pip install pickle5

import pickle5 as pickle

with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh:
    id2type = pickle.load(fh)
```


```

```

### rd: recsys - otto - get started - process jsonl to df


```
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


```
train_df = jsonl_to_df(train_path)
```

    11:13:26.20 LOG:
    11:13:26.21 .... info = "each item is a session, each session has two items 'session' and 'events'"
    11:13:26.21 .... info1 = 'each session has variable amount of events'
    11:13:26.21 LOG:
    11:13:26.22 .... session_data.session = 0
    11:13:26.22 .... len(session_data.events) = 276
    11:13:26.23 .... session_data.events[0] = {'aid': 1517085, 'ts': 1659304800025, 'type': 'clicks'}
    11:13:26.23 .... session_data.events[1] = {'aid': 1563459, 'ts': 1659304904511, 'type': 'clicks'}
    11:13:26.23 LOG:
    11:13:26.23 .... session_data.session = 1
    11:13:26.23 .... len(session_data.events) = 32
    11:13:26.23 .... session_data.events[0] = {'aid': 424964, 'ts': 1659304800025, 'type': 'carts'}
    11:13:26.23 .... session_data.events[1] = {'aid': 1492293, 'ts': 1659304852871, 'type': 'clicks'}



```
# %%time

# # train_df = jsonl_to_df('data/train.jsonl')
# train_df = jsonl_to_df(train_path)
# train_df.type = train_df.type.astype(np.uint8) # a tiny bit of further memory footprint optimization (original type is just int class)
# # train_df.to_parquet('train.parquet', index=False)
# train_df.to_csv('train.csv', index=False)

# del train_df
```

### rd: recsys - otto - get started - RAM needed to process data on Kaggle, 400MB jsonl takes up nearly 4GB ram

see more detailed findings of mine in the discussion [here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843#2024279)

### rd: recsys - otto - get started - use parquet over csv, why and how

Summary of technical features of parquet files
- Apache Parquet is column-oriented and designed to provide efficient columnar storage compared to row-based file types such as CSV.
- Parquet files were designed with complex nested data structures in mind.
- Apache Parquet is designed to support very efficient compression and encoding schemes.
- Apache Parquet generates lower storage costs for data files and maximizes the effectiveness of data queries with current cloud technologies such as Amazon Athena, Redshift Spectrum, BigQuery and Azure Data Lakes.
- Licensed under the Apache license and available for any project.


```
%%time

# test_df = jsonl_to_df('data/test.jsonl')
test_df = jsonl_to_df(test_path)
test_df.type = test_df.type.astype(np.uint8)
test_df.to_parquet('test.parquet', index=False)
test_df.to_csv('test.csv', index=False)
```

    CPU times: user 2min 49s, sys: 1.5 s, total: 2min 51s
    Wall time: 2min 55s



```
del test_df
```

### rd: recsys - otto - get started - use parquet to instead of jsonl or csv to save space on disk

for details see discussion [here](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files/comments#2025116)


```
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


```
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


```
def filesize(path): # path: str or path
    import os
    return os.path.getsize(path)
```


```
[("jsonl=>{}".format(path), filesize(path)) for path in ['../input/otto-recommender-system/test.jsonl','test_keep_str.parquet', 'test_keep_str.csv', 'test_str2int.parquet', 'test0_str2int.csv', 'test_str2uint8.parquet', 'test_str2uint8.csv']]
```

It seems that by converting jsonl to csv can half the size, to parquet can quarter the size;

Converting type from string to int using type2id can save 34MB which is about 10% size of original test dataset during conversion from jsonl to csv; but it has no observable effect during convertion to parquet file.

Although converting type further from int to uint8 in a parquet file can shrink the size, but the amount is barely observable, about 200byte.

So, it seems that conversion from jsonl to parquet alone does the most heavy lifting in reducing size, converting from string to int and then uint8 is helpful but in this case has no significant effect in reducing the size.

### rd: recsys - otto - get started - use `uint8` instead of `int` or `str` to reduce RAM usage by 9 times

read the discussion [here](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files/comments#2025129)


```
test_df.memory_usage()
test_df_str.memory_usage()
```
