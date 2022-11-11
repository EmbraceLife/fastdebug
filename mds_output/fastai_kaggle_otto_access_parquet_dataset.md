I converated the dataset for this competition from `jsonl` to `csv` and `parquet` so that it is easy to work with using our favorite set of tools! 🙂 You can find the converted dataset [here](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint).

Unfotunately, it was impossible to process this data on Kaggle due to not enough RAM. I carried out the processing on my local machine and uploaded the processed data (will share the code I used for processing as well, please see [the associate thread](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843)).

The 'type' information was represented as a string, which takes up a lot of memory. Instead, I cast it to `np.uint8`. This makes the data much smaller and easier to work with without any loss of information!

Let me walk you through how everything is set up so that you can use this data in your work.

### rd: recsys - otto - get started - copy and paste dataset path and use !ls to see what inside


```
# Here are the files
!ls ../input/otto-full-optimized-memory-footprint/
```

    id2type.pkl  test.parquet  train.parquet  type2id.pkl


### rd: recsys - otto - get started - pd.read_parquet('copy and paste the dataset path')


```
import pandas as pd

# There is a version of the data stored as `csv` as well
# but I recommend you use `parquet` as I do here -- it is much faster

train = pd.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
test = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
```


```
train.head()
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
      <td>0</td>
      <td>1517085</td>
      <td>1659304800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1563459</td>
      <td>1659304904</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1309446</td>
      <td>1659367439</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>16246</td>
      <td>1659367719</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1781822</td>
      <td>1659367871</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The `type` column has been encoded as integers. To translate between the integer and original representations, please use the following.

### rd: recsys - otto - get started - load a function from a pickle file with pickle5, with open as fh: and pick.load(fh)


```
!pip install pickle5

import pickle5 as pickle

with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh:
    id2type = pickle.load(fh)
with open('../input/otto-full-optimized-memory-footprint/type2id.pkl', "rb") as fh:
    type2id = pickle.load(fh)
```

    Collecting pickle5
      Downloading pickle5-0.0.12-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (256 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m256.4/256.4 kB[0m [31m921.5 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hInstalling collected packages: pickle5
    Successfully installed pickle5-0.0.12
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

Using `id2type` we can convert from integer to string representation (and we can use `type2id` to go in the other direction)


```
id2type, type2id
```




    (['clicks', 'carts', 'orders'], {'clicks': 0, 'carts': 1, 'orders': 2})




```
id2type
```




    ['clicks', 'carts', 'orders']



### rd: recsys - otto - get started - convert int back to string, train.iloc[:1000].type.map(lambda i: id2type[i])


```
type_as_string = train.iloc[:1000].type.map(lambda i: id2type[i])
type_as_string.head()
```




    0    clicks
    1    clicks
    2    clicks
    3    clicks
    4    clicks
    Name: type, dtype: object



And we can just as easily go back from strings to idxs.


```
type_as_string.map(lambda i: type2id[i]).head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: type, dtype: int64




```
type_as_string.map(type2id).head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: type, dtype: int64



And that's it! I hope this will speed you along in your work 🙂

If you found this useful, I would be extremely grateful if you could please upvote this notebook and [the associated dataset](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint).

Thank you so much for your support! Happy kaggling! 🥳
