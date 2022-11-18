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

The original [notebook](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset) is done by Radek. I experimented to learn the techniques.


### import utils and otto dataset

```python
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

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

Let us read in the `train` and `test` datasets.


### rd: recsys - otto - eda - Read parquet file with `pd.read_parquet`

```python
train = pd.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
test = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
```

### rd: recsys - otto - eda - load object from a pikle file - import pickle5 as pickle - with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh: - id2type = pickle.load(fh)


Let us also read in the pickle file that will allow us to decipher the `type` information that has been encoded as integers to conserve memory.

```python
!pip install pickle5

import pickle5 as pickle

with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh:
    id2type = pickle.load(fh)
```

### rd: recsys - otto - eda - explore how lightweighted is inference vs training - test.shape[0]/train.shape[0]

```python
train.shape, test.shape
```

The `train` dataset contains 216716096 datapoints with `test` containing only 6928123.

Proportion of `test` to `train`:

```python
test.shape[0]/train.shape[0]
```

The size of the test set is ~3.1% of the train set. This can give us an idea of how lightweight the inference is likely to be compared to training.

```python
train.head()
```

### rd: recsys - otto - eda - how much more of the unique sessions in train vs in test sets - test.session.unique().shape[0]/train.session.unique().shape[0]


How many session are there in `train` and `test`?

```python
train.session.unique()
train.session.unique().shape
```

```python
train.session.unique().shape[0], test.session.unique().shape[0]
```

```python
train.session.max() + 1
```

```python
test.session.unique().shape[0]/train.session.unique().shape[0]
```

### rd: recsys - otto - eda - are sessions in test shorter than sessions in train?


#### rd: recsys - otto - eda - compare the histograms of train vs test on the natural log of the amount of aids in each session. The comparison of histogram can give us a sense of distribution difference - test.groupby('session')['aid'].count().apply(np.log1p).hist()


Seems the sessions in the test set are much shorter! Let's confirm this.

```python
test.head()
```

```python
test.groupby('session')['aid'].count()
train.groupby('session')['aid'].count()
```

```python
test.groupby('session')['aid'].count().apply(np.log1p)
```

```python
test.groupby('session')['aid'].count().apply(np.log1p).max()
test.groupby('session')['aid'].count().apply(np.log1p).min()
```

```python
test.groupby('session')['aid'].count().apply(np.log1p).hist()
```

```python
train.groupby('session')['aid'].count().apply(np.log1p).hist()
```

#### rd: recsys - otto - eda - why np.log1p over np.log? return natural log and also be super accurate in floating point, train.groupby('session')['aid'].count().apply(np.log1p).hist()


##### jn: use help instead of doc_sig and chk from my utils will have less error throwing. Maybe remove doc_sig and chk from fastdebug.utils /2022-11-14

```python
np.info(np.log1p) # see the notes for its actual usage here
```

### rd: recsys - otto - eda - whether train and test sessions have time intersection - datetime.datetime.fromtimestamp(test.ts.min()/1000) - 


#### jn: even for questions not answered, but writing up and revisit them, potential answers may flow to me more likely: use `/1000` could make the comparison more obvious? and without `/1000` can reveal the holiday season. /2022-11-14


#### jn: (qusetion answered) Radek just informed me that dividing by 1000 on the timestamps can save RAM on Kaggle. /2022-11-14


I have raised a [question](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset/comments#2024482) on why `/1000` in converting timestamp


There might be something at play here. Could the organizers have thrown us a curve ball and the train and test data are not from the same distribution?

Let's quickly look at timestamps.

```python
train.ts
```

```python
import datetime

datetime.datetime.fromtimestamp(train.ts.min()/1000), datetime.datetime.fromtimestamp(train.ts.max()/1000)
datetime.datetime.fromtimestamp(train.ts.min()), datetime.datetime.fromtimestamp(train.ts.max())
```

```python
import datetime

datetime.datetime.fromtimestamp(test.ts.min()/1000), datetime.datetime.fromtimestamp(test.ts.max()/1000)
datetime.datetime.fromtimestamp(test.ts.min()), datetime.datetime.fromtimestamp(test.ts.max())
```

Looks like we have temporally split data. The problem is that the data doesn't come from the same period. In most geographies the beginning of September is the start of the school year!

That is the period where people are coming back from vacation, commerce resumes after a slowdown during the vacation season.

The organizers are not making this easy for us 🙂


### rd: recsys - otto - eda - are there new items in test not in train? len(set(test.aid.tolist()) - set(train.aid.tolist()))


Let's see if there are any new items in the test set that were not see in train.

```python
len(set(test.aid.tolist()) - set(train.aid.tolist()))
```

### rd: recsys - otto - eda - describe (check distribution of ) total number of aids of sessions between train and test, train.groupby('session')['aid'].count().describe()


So at least we have that going for us, no new items in the test set! 😊

I just scanned the forums really quickly and seems we have an answer as to why the session length differs between `train` and `test`!

First, let's look at the data more closely.

```python
train.groupby('session')['aid'].count().describe()
```

```python
test.groupby('session')['aid'].count().describe()
```

### rd: recsys - otto - eda - define a session (a tracking period)

And here is the [key piece of information on the forums](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363554#2015486).

Apparently, a session are all actions by a user in the tracking period. So naturally, if the tracking period is shorter, the sessions will also be shorter.

Maybe there is nothing amiss happening here.


### rd: recsys - otto - eda - whether train and test have no common sessions, - train.session.max(), test.session.min()

```python
train.session.max(), test.session.min()
```

An we see that the `session_ids` are not overlapping between `train` and `test` so it will be impossible to map the users (even if we have seen them before in train). We have to assume each session is from a different user.

Now that I know a bit more about the dataset, I can't wait to start playing around with it. This is shaping up to be a very interesting problem! 😊


### jn: eda revisit is done /2022-11-14


### jn: just got Radek very detailed and helpful replies, and I did the same with more experiments and questions on the effect of convert to int32 and divde by 1000 [here](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset/comments#2028533) /2022-11-14
