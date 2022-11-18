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

    WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv


    Collecting nbdev
      Downloading nbdev-2.3.9-py3-none-any.whl (64 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.1/64.1 kB 2.1 MB/s eta 0:00:00
    Collecting snoop
      Downloading snoop-0.4.2-py2.py3-none-any.whl (27 kB)
    Requirement already satisfied: fastcore>=1.5.27 in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.5.27)
    Collecting execnb>=0.1.4
      Downloading execnb-0.1.4-py3-none-any.whl (13 kB)
    Collecting ghapi>=1.0.3
      Downloading ghapi-1.0.3-py3-none-any.whl (58 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.1/58.1 kB 3.8 MB/s eta 0:00:00
    Collecting watchdog
      Downloading watchdog-2.1.9-py3-none-manylinux2014_x86_64.whl (78 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.4/78.4 kB 6.1 MB/s eta 0:00:00
    Requirement already satisfied: PyYAML in /opt/conda/lib/python3.7/site-packages (from nbdev) (6.0)
    Requirement already satisfied: astunparse in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.6.3)
    Collecting asttokens
      Downloading asttokens-2.1.0-py2.py3-none-any.whl (26 kB)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.7/site-packages (from snoop) (2.12.0)
    Collecting executing
      Downloading executing-1.2.0-py2.py3-none-any.whl (24 kB)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from snoop) (1.15.0)
    Collecting cheap-repr>=0.4.0
      Downloading cheap_repr-0.5.1-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: ipython in /opt/conda/lib/python3.7/site-packages (from execnb>=0.1.4->nbdev) (7.33.0)
    Requirement already satisfied: pip in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.27->nbdev) (22.1.2)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.27->nbdev) (21.3)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.7/site-packages (from astunparse->nbdev) (0.37.1)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.2.0)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (3.0.30)
    Requirement already satisfied: traitlets>=4.2 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (5.3.0)
    Requirement already satisfied: setuptools>=18.5 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (59.8.0)
    Requirement already satisfied: matplotlib-inline in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.1.3)
    Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.18.1)
    Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (4.8.0)
    Requirement already satisfied: pickleshare in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.7.5)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (5.1.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.7/site-packages (from packaging->fastcore>=1.5.27->nbdev) (3.0.9)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/conda/lib/python3.7/site-packages (from jedi>=0.16->ipython->execnb>=0.1.4->nbdev) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.7/site-packages (from pexpect>4.3->ipython->execnb>=0.1.4->nbdev) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython->execnb>=0.1.4->nbdev) (0.2.5)
    Installing collected packages: executing, cheap-repr, watchdog, asttokens, snoop, ghapi, execnb, nbdev
    Successfully installed asttokens-2.1.0 cheap-repr-0.5.1 execnb-0.1.4 executing-1.2.0 ghapi-1.0.3 nbdev-2.3.9 snoop-0.4.2 watchdog-2.1.9


    WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv



    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    /tmp/ipykernel_27/1165569295.py in <module>
         15     sys.path
         16     sys.path.insert(1, path)
    ---> 17     import utils as fu
         18     from utils import *
         19 else:


    ModuleNotFoundError: No module named 'utils'



```python
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

    Collecting pickle5
      Downloading pickle5-0.0.12-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (256 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m256.4/256.4 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hInstalling collected packages: pickle5
    Successfully installed pickle5-0.0.12
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
sample_sub.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_type</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12899779_clicks</td>
      <td>129004 126836 118524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12899779_carts</td>
      <td>129004 126836 118524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12899779_orders</td>
      <td>129004 126836 118524</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12899780_clicks</td>
      <td>129004 126836 118524</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12899780_carts</td>
      <td>129004 126836 118524</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session</th>
      <th>aid</th>
      <th>ts</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12899779</td>
      <td>59625</td>
      <td>1661724000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12899780</td>
      <td>1142000</td>
      <td>1661724000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12899780</td>
      <td>582732</td>
      <td>1661724058</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12899780</td>
      <td>973453</td>
      <td>1661724109</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12899780</td>
      <td>736515</td>
      <td>1661724136</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### rd: recsys - otto - last 20 aid - sort df by two columns - test.sort_values(['session', 'ts'])


```python
%%time

test = test.sort_values(['session', 'ts'])
test.head()
```

    CPU times: user 3.16 s, sys: 704 ms, total: 3.86 s
    Wall time: 3.87 s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session</th>
      <th>aid</th>
      <th>ts</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12899779</td>
      <td>59625</td>
      <td>1661724000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12899780</td>
      <td>1142000</td>
      <td>1661724000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12899780</td>
      <td>582732</td>
      <td>1661724058</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12899780</td>
      <td>973453</td>
      <td>1661724109</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12899780</td>
      <td>736515</td>
      <td>1661724136</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### rd: recsys - otto - last 20 aid - take last 20 aids from each session - test.groupby('session')['aid'].apply(lambda x: list(x)[-20:])


```python
session_aids = test.groupby('session')['aid'].apply(lambda x: list(x)[-20:])
```


```python

```


```python
session_aids.head()
```




    session
    12899779                                              [59625]
    12899780           [1142000, 582732, 973453, 736515, 1142000]
    12899781    [141736, 199008, 57315, 194067, 199008, 199008...
    12899782    [889671, 1099390, 987399, 987399, 638410, 1072...
    12899783    [255297, 1114789, 255297, 300127, 198385, 3001...
    Name: aid, dtype: object




```python
session_aids.index
```




    Int64Index([12899779, 12899780, 12899781, 12899782, 12899783, 12899784,
                12899785, 12899786, 12899787, 12899788,
                ...
                14571572, 14571573, 14571574, 14571575, 14571576, 14571577,
                14571578, 14571579, 14571580, 14571581],
               dtype='int64', name='session', length=1671803)



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

    CPU times: user 9.57 s, sys: 540 ms, total: 10.1 s
    Wall time: 10.1 s


### rd: recsys - otto - last 20 aid - make a df from a dict with two lists as values - pd.DataFrame({'session_type': session_type, 'labels': labels})


```python
submission = pd.DataFrame({'session_type': session_type, 'labels': labels})
```


```python
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>session_type</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12899779_clicks</td>
      <td>59625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12899779_carts</td>
      <td>59625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12899779_orders</td>
      <td>59625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12899780_clicks</td>
      <td>1142000 582732 973453 736515 1142000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12899780_carts</td>
      <td>1142000 582732 973453 736515 1142000</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv('submission.csv', index=False)
```
