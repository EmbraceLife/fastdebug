The original [notebook](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset) is done by Radek. I experimented to learn the techniques.

### import utils and otto dataset


```
# make sure fastkaggle is install and imported
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



# download (if necessary and return the path of the dataset)
home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
comp = 'otto-recommender-system' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
path = download_kaggle_dataset(comp, local_folder=home)#, install='fastai "timm>=0.6.2.dev0"')
# path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
```

    Requirement already satisfied: nbdev in /opt/conda/lib/python3.7/site-packages (2.3.9)
    Requirement already satisfied: snoop in /opt/conda/lib/python3.7/site-packages (0.4.2)
    Requirement already satisfied: ghapi>=1.0.3 in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.0.3)
    Requirement already satisfied: execnb>=0.1.4 in /opt/conda/lib/python3.7/site-packages (from nbdev) (0.1.4)
    Requirement already satisfied: PyYAML in /opt/conda/lib/python3.7/site-packages (from nbdev) (6.0)
    Requirement already satisfied: fastcore>=1.5.27 in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.5.27)
    Requirement already satisfied: astunparse in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.6.3)
    Requirement already satisfied: watchdog in /opt/conda/lib/python3.7/site-packages (from nbdev) (2.1.9)
    Requirement already satisfied: asttokens in /opt/conda/lib/python3.7/site-packages (from nbdev) (2.1.0)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from snoop) (1.15.0)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.7/site-packages (from snoop) (2.12.0)
    Requirement already satisfied: cheap-repr>=0.4.0 in /opt/conda/lib/python3.7/site-packages (from snoop) (0.5.1)
    Requirement already satisfied: executing in /opt/conda/lib/python3.7/site-packages (from snoop) (1.2.0)
    Requirement already satisfied: ipython in /opt/conda/lib/python3.7/site-packages (from execnb>=0.1.4->nbdev) (7.33.0)
    Requirement already satisfied: pip in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.27->nbdev) (22.1.2)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.27->nbdev) (21.3)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.7/site-packages (from astunparse->nbdev) (0.37.1)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.2.0)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (5.1.1)
    Requirement already satisfied: setuptools>=18.5 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (59.8.0)
    Requirement already satisfied: matplotlib-inline in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.1.3)
    Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.18.1)
    Requirement already satisfied: pickleshare in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (3.0.30)
    Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (4.8.0)
    Requirement already satisfied: traitlets>=4.2 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb>=0.1.4->nbdev) (5.3.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.7/site-packages (from packaging->fastcore>=1.5.27->nbdev) (3.0.9)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/conda/lib/python3.7/site-packages (from jedi>=0.16->ipython->execnb>=0.1.4->nbdev) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.7/site-packages (from pexpect>4.3->ipython->execnb>=0.1.4->nbdev) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython->execnb>=0.1.4->nbdev) (0.2.5)


    WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv



<style>.container { width:100% !important; }</style>



```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

Let us read in the `train` and `test` datasets.

### rd: recsys - otto - eda - Read parquet file with `pd.read_parquet`


```
train = pd.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
test = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
```

### rd: recsys - otto - eda - load object from a pikle file - import pickle5 as pickle - with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh: - id2type = pickle.load(fh)

Let us also read in the pickle file that will allow us to decipher the `type` information that has been encoded as integers to conserve memory.


```
!pip install pickle5

import pickle5 as pickle

with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh:
    id2type = pickle.load(fh)
```

    Requirement already satisfied: pickle5 in /opt/conda/lib/python3.7/site-packages (0.0.12)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

### rd: recsys - otto - eda - explore how lightweighted is inference vs training - test.shape[0]/train.shape[0]


```
train.shape, test.shape
```




    ((216716096, 4), (6928123, 4))



The `train` dataset contains 216716096 datapoints with `test` containing only 6928123.

Proportion of `test` to `train`:


```
test.shape[0]/train.shape[0]
```




    0.03196865912534711



The size of the test set is ~3.1% of the train set. This can give us an idea of how lightweight the inference is likely to be compared to training.


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



### rd: recsys - otto - eda - how much more of the unique sessions in train vs in test sets - test.session.unique().shape[0]/train.session.unique().shape[0]

How many session are there in `train` and `test`?


```
train.session.unique()
train.session.unique().shape
```




    array([       0,        1,        2, ..., 12899776, 12899777, 12899778],
          dtype=int32)






    (12899779,)




```
train.session.unique().shape[0], test.session.unique().shape[0]
```




    (12899779, 1671803)




```
train.session.max() + 1
```




    12899779




```
test.session.unique().shape[0]/train.session.unique().shape[0]
```




    0.12959935205091497



### rd: recsys - otto - eda - are sessions in test shorter than sessions in train?

#### rd: recsys - otto - eda - compare the histograms of train vs test on the natural log of the amount of aids in each session. The comparison of histogram can give us a sense of distribution difference - test.groupby('session')['aid'].count().apply(np.log1p).hist()

Seems the sessions in the test set are much shorter! Let's confirm this.


```
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




```
test.groupby('session')['aid'].count()
train.groupby('session')['aid'].count()
```




    session
    12899779     1
    12899780     5
    12899781    11
    12899782    70
    12899783    11
                ..
    14571577     1
    14571578     1
    14571579     1
    14571580     1
    14571581     1
    Name: aid, Length: 1671803, dtype: int64






    session
    0           276
    1            32
    2            33
    3           226
    4            19
               ... 
    12899774      2
    12899775      2
    12899776      2
    12899777      2
    12899778      2
    Name: aid, Length: 12899779, dtype: int64




```
test.groupby('session')['aid'].count().apply(np.log1p)
```




    session
    12899779    0.693147
    12899780    1.791759
    12899781    2.484907
    12899782    4.262680
    12899783    2.484907
                  ...   
    14571577    0.693147
    14571578    0.693147
    14571579    0.693147
    14571580    0.693147
    14571581    0.693147
    Name: aid, Length: 1671803, dtype: float64




```
test.groupby('session')['aid'].count().apply(np.log1p).max()
test.groupby('session')['aid'].count().apply(np.log1p).min()
```


```
test.groupby('session')['aid'].count().apply(np.log1p).hist()
```




    <AxesSubplot:>




    
![png](fastai_kaggle_radek_otto_eda_files/fastai_kaggle_radek_otto_eda_29_1.png)
    



```
train.groupby('session')['aid'].count().apply(np.log1p).hist()
```




    <AxesSubplot:>




    
![png](fastai_kaggle_radek_otto_eda_files/fastai_kaggle_radek_otto_eda_30_1.png)
    


#### rd: recsys - otto - eda - why np.log1p over np.log? return natural log and also be super accurate in floating point, train.groupby('session')['aid'].count().apply(np.log1p).hist()

##### jn: use help instead of doc_sig and chk from my utils will have less error throwing. Maybe remove doc_sig and chk from fastdebug.utils /2022-11-14


```
np.info(np.log1p) # see the notes for its actual usage here
```

    log1p(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Return the natural logarithm of one plus the input array, element-wise.
    
    Calculates ``log(1 + x)``.
    
    Parameters
    ----------
    x : array_like
        Input values.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    y : ndarray
        Natural logarithm of `1 + x`, element-wise.
        This is a scalar if `x` is a scalar.
    
    See Also
    --------
    expm1 : ``exp(x) - 1``, the inverse of `log1p`.
    
    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.
    
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = 1 + x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi, pi]`.
    
    For real-valued input data types, `log1p` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    
    For complex-valued input, `log1p` is a complex analytical function that
    has a branch cut `[-inf, -1]` and is continuous from above on it.
    `log1p` handles the floating-point negative zero as an infinitesimal
    negative number, conforming to the C99 standard.
    
    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 67. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Logarithm". https://en.wikipedia.org/wiki/Logarithm
    
    Examples
    --------
    >>> np.log1p(1e-99)
    1e-99
    >>> np.log(1 + 1e-99)
    0.0


### rd: recsys - otto - eda - whether train and test sessions have time intersection - datetime.datetime.fromtimestamp(test.ts.min()/1000) - 

#### jn: even for questions not answered, but writing up and revisit them, potential answers may flow to me more likely: use `/1000` could make the comparison more obvious? and without `/1000` can reveal the holiday season. /2022-11-14

#### jn: (qusetion answered) Radek just informed me that dividing by 1000 on the timestamps can save RAM on Kaggle. /2022-11-14

I have raised a [question](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset/comments#2024482) on why `/1000` in converting timestamp

There might be something at play here. Could the organizers have thrown us a curve ball and the train and test data are not from the same distribution?

Let's quickly look at timestamps.


```
train.ts
```




    0            1659304800
    1            1659304904
    2            1659367439
    3            1659367719
    4            1659367871
                    ...    
    216716091    1661723987
    216716092    1661723976
    216716093    1661723986
    216716094    1661723983
    216716095    1661723994
    Name: ts, Length: 216716096, dtype: int32




```
import datetime

datetime.datetime.fromtimestamp(train.ts.min()/1000), datetime.datetime.fromtimestamp(train.ts.max()/1000)
datetime.datetime.fromtimestamp(train.ts.min()), datetime.datetime.fromtimestamp(train.ts.max())
```




    (datetime.datetime(1970, 1, 20, 4, 55, 4, 800000),
     datetime.datetime(1970, 1, 20, 5, 35, 23, 999000))






    (datetime.datetime(2022, 7, 31, 22, 0),
     datetime.datetime(2022, 8, 28, 21, 59, 59))




```
import datetime

datetime.datetime.fromtimestamp(test.ts.min()/1000), datetime.datetime.fromtimestamp(test.ts.max()/1000)
datetime.datetime.fromtimestamp(test.ts.min()), datetime.datetime.fromtimestamp(test.ts.max())
```




    (datetime.datetime(1970, 1, 20, 5, 35, 24),
     datetime.datetime(1970, 1, 20, 5, 45, 28, 791000))






    (datetime.datetime(2022, 8, 28, 22, 0),
     datetime.datetime(2022, 9, 4, 21, 59, 51))



Looks like we have temporally split data. The problem is that the data doesn't come from the same period. In most geographies the beginning of September is the start of the school year!

That is the period where people are coming back from vacation, commerce resumes after a slowdown during the vacation season.

The organizers are not making this easy for us 🙂

### rd: recsys - otto - eda - are there new items in test not in train? len(set(test.aid.tolist()) - set(train.aid.tolist()))

Let's see if there are any new items in the test set that were not see in train.


```
len(set(test.aid.tolist()) - set(train.aid.tolist()))
```




    0



### rd: recsys - otto - eda - describe (check distribution of ) total number of aids of sessions between train and test, train.groupby('session')['aid'].count().describe()

So at least we have that going for us, no new items in the test set! 😊

I just scanned the forums really quickly and seems we have an answer as to why the session length differs between `train` and `test`!

First, let's look at the data more closely.


```
train.groupby('session')['aid'].count().describe()
```




    count    1.289978e+07
    mean     1.679999e+01
    std      3.357738e+01
    min      2.000000e+00
    25%      3.000000e+00
    50%      6.000000e+00
    75%      1.500000e+01
    max      5.000000e+02
    Name: aid, dtype: float64




```
test.groupby('session')['aid'].count().describe()
```




    count    1.671803e+06
    mean     4.144103e+00
    std      8.215717e+00
    min      1.000000e+00
    25%      1.000000e+00
    50%      2.000000e+00
    75%      4.000000e+00
    max      4.580000e+02
    Name: aid, dtype: float64



### rd: recsys - otto - eda - define a session (a tracking period)

And here is the [key piece of information on the forums](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363554#2015486).

Apparently, a session are all actions by a user in the tracking period. So naturally, if the tracking period is shorter, the sessions will also be shorter.

Maybe there is nothing amiss happening here.

### rd: recsys - otto - eda - whether train and test have no common sessions, - train.session.max(), test.session.min()


```
train.session.max(), test.session.min()
```




    (12899778, 12899779)



An we see that the `session_ids` are not overlapping between `train` and `test` so it will be impossible to map the users (even if we have seen them before in train). We have to assume each session is from a different user.

Now that I know a bit more about the dataset, I can't wait to start playing around with it. This is shaping up to be a very interesting problem! 😊

### jn: eda revisit is done /2022-11-14

### jn: just got Radek very detailed and helpful replies, and I did the same with more experiments and questions on the effect of convert to int32 and divde by 1000 [here](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset/comments#2028533) /2022-11-14
