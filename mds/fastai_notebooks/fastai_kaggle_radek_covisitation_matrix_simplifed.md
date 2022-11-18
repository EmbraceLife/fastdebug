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

**This notebook is the annotated version of the original [notebook](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic) by @radek1**

<!-- #region -->
This notebook builds on [the work](https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix) by [@vslaykovsky](https://www.kaggle.com/vslaykovsky). As suggested by [@cdeotte](https://www.kaggle.com/cdeotte) [here](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost) I am including test information in clculating the co-visitation matrix.

Here I provide a simplified implementation that is easier to follow and that can also be more easily extended!

Please find some additional discussion on this notebook in this [thread](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364210).

**Would appreciate [your upvote on the thread](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364210) to increase visibility.** 🙏

Because the code is simpler, it was straightforward to modify the logic and make a couple of different decisions to achieve a better result.

Additionally, I am also using a version of the dataset that I shared [here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843). This further simplifies matters as we no longer have to read the data from `jasonl` files!

**If you like this notebook, please upvote! Thank you! 😊**

## You might also find useful:

* [💡 Do not disregard longer sessions -- they contribute disproportionately to the competition metric!](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364375)
* [🐘 the elephant in the room -- high cardinality of targets and what to do about this](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364722)
* [📅 Dataset for local validation created using organizer's repository (parquet files)](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534)


### Update:

* added sorting of `AIDs` in test based on number of clicks from [Multiple clicks vs latest items](https://www.kaggle.com/code/pietromaldini1/multiple-clicks-vs-latest-items) by [pietromaldini1](https://www.kaggle.com/pietromaldini1)
* added speed improvements from [Fast Co-Visitation Matrix](https://www.kaggle.com/code/pietromaldini1/multiple-clicks-vs-latest-items) by [dpalbrecht](https://www.kaggle.com/dpalbrecht)
* added weigting by type of `AIDs` in test from [Item type vs multiple clicks vs latest items](https://www.kaggle.com/code/ingvarasgalinskas/item-type-vs-multiple-clicks-vs-latest-items) by [ingvarasgalinskas](https://www.kaggle.com/ingvarasgalinskas)
<!-- #endregion -->

## imports

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
```

```python
import pandas as pd
import numpy as np
```

## rd: recsys - otto - covisitation_simplified - load the dataset files and have a peek at them


### rd: recsys - otto - covisitation_simplified - read parquet and csv from dataset, pd.read_parquet('copy & paste the file path'), pd.read_csv('copy & paste path')

```python
train = pd.read_parquet('../input/otto-full-optimized-memory-footprint//train.parquet')
test = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')

sample_sub = pd.read_csv('../input/otto-recommender-system//sample_submission.csv')
```

```python
train.head()
```

```python
test.head()
```

```python
sample_sub.head()
```

## rd: recsys - otto - covisitation_simplified - taking subset for fast experiment


### rd: recsys - otto - covisitation_simplified - subset based on entire sessions, train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session'], train[train.session.isin(lucky_sessions_train)]

```python
fraction_of_sessions_to_use = 0.000001 # 0.001 is recommended, but 0.000001 can finish in less than 4 minutes
```

```python
train.shape # how many rows
```

```python
train.drop_duplicates(['session']).shape # how many unique sessions (drop rows with the same session)
```

```python
subset_of_train_no_duplicate = train.sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
subset_of_train_no_duplicate.shape # take 0.000001 from entire train
```

```python
lucky_sessions_train = train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
lucky_sessions_train.shape # take 0.000001 from a dataframe in which each row is an unique session
```

```python
lucky_sessions_train.head()
lucky_sessions_train.reset_index(drop=True).head() # make index easier to see
```

```python
train.session.isin(lucky_sessions_train).sum() # how many rows under the 13 sessions
```

```python
if fraction_of_sessions_to_use != 1:
    lucky_sessions_train = train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
    subset_of_train = train[train.session.isin(lucky_sessions_train)]
    
    lucky_sessions_test = test.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
    subset_of_test = test[test.session.isin(lucky_sessions_test)]
else:
    subset_of_train = train
    subset_of_test = test
```

```python
subset_of_train.shape
```

```python
subset_of_train
```

## rd: recsys - otto - covisitation_simplified - some setups before training


### rd: recsys - otto - covisitation_simplified - use session column as index, subset_of_train.index = pd.MultiIndex.from_frame(subset_of_train[['session']]), [['session']] as Series not DataFrame

```python
# see the difference between 
subset_of_train[['session']], subset_of_train['session']
```

```python
subset_of_train.index = pd.MultiIndex.from_frame(subset_of_train[['session']])
subset_of_train
```

```python
subset_of_test.index = pd.MultiIndex.from_frame(subset_of_test[['session']])
subset_of_test
```

### rd: recsys - otto - covisitation_simplified - get starting and end timestamp for all sessions from start of training set to end of test set, train.ts.min() test.ts.max()

```python
min_ts = train.ts.min()
max_ts = test.ts.max()

min_ts, max_ts
```

### rd: recsys - otto - covisitation_simplified - use defaultdict + Counter to count occurences, next_AIDs = defaultdict(Counter)

```python
from collections import defaultdict, Counter
next_AIDs = defaultdict(Counter)
next_AIDs
```

### rd: recsys - otto - covisitation_simplified - use test for training, get subsets, sessions, sessions_train, sessions_test - subsets = pd.concat([subset_of_train, subset_of_test]) - sessions = subsets.session.unique()

```python
subsets = pd.concat([subset_of_train, subset_of_test]) # use test set for training too
sessions = subsets.session.unique() # unique sessions in subsets
sessions_train = subset_of_train.session.unique() # unique sessions in train
sessions_test = subset_of_test.session.unique() # unique sessions in test
len(sessions), len(sessions_train), len(sessions_test)
```

```python
subsets
```

```python
sessions[:5], sessions[-5:], sessions_train[:5], sessions_test[-5:]
```

## rd: recsys - otto - covisitation_simplified - Training: when one aid occurred, keep track of what other aid occurred, how often they occurred. Do this for every aid across both train and test sessions.


### rd: recsys - otto - covisitation_simplified - loop every chunk_size number unique sessions - for i in range(0, sessions.shape[0], chunk_size): 
### rd: recsys - otto - covisitation_simplified - take a chunk_size number of sessions (each session in its entirety, ie, probably with many rows) from subsets as current_chunk - current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) 

```python
chunk_size = 30_000

for i in range(0, sessions.shape[0], chunk_size): # loop every 30_000 unique sessions
    
    #### take a chunk_size number of sessions (each session in its entirety, ie., many rows) from subsets as current_chunk
    
    pp(i, sessions[i], sessions.shape[0]-1, i+chunk_size-1, min(sessions.shape[0]-1, i+chunk_size-1))
    current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) # to remove index column
    pp(current_chunk.head(), current_chunk.tail())
    break

    #### In current_chunk, from each session (in its entirety) only takes the last/latest 30 events and combine them to update current_chunk
#     pp(range(-30,0), list(range(-30,0)))
    current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
#     pp(current_chunk[:31], current_chunk[30:61])
#     pp(doc_sig(pd.DataFrame.reset_index))
#     pp(doc_sig(current_chunk.groupby('session', as_index=False).nth))

    #### merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session)
    consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
#     pp(consecutive_AIDs[:30])
#     pp(consecutive_AIDs[29:61])

    #### remove all the rows which aid_x == aid_y (remove the row when the two articles are the same)
    consecutive_AIDs = consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
    
    #### add a column named 'days_elapsed' which shows how many days passed between the two aids in a session
    consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
#     pp(consecutive_AIDs[:61])

    #### keep the rows if the two aids of a session are occurred within the same day on the right order
    consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
    
    #### among all sessions or all users, for each article occurred, count how often the other articles are occurred
    for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
        next_AIDs[aid_x][aid_y] += 1
    
```

### rd: recsys - otto - covisitation_simplified - In current_chunk, from each session (in its entirety) only takes the last/latest 30 events/rows and combine them to update current_chunk (focus on the latest 30 events to save computations) - current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)

```python
chunk_size = 30_000

for i in range(0, sessions.shape[0], chunk_size): # loop every 30_000 unique sessions
    
    #### take a chunk_size number of sessions (each session in its entirety, ie., many rows) from subsets as current_chunk
    
#     pp(i, sessions[i], sessions.shape[0]-1, i+chunk_size-1, min(sessions.shape[0]-1, i+chunk_size-1))
    current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) # to remove index column
#     pp(current_chunk.head(), current_chunk.tail())
#     break

    #### In current_chunk, from each session (in its entirety) only takes the last/latest 30 events and combine them to update current_chunk
    pp(range(-30,0), list(range(-30,0)))
    current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
    pp(current_chunk[:31], current_chunk[29:61])
    pp(doc_sig(pd.DataFrame.reset_index))
    pp(doc_sig(current_chunk.groupby('session', as_index=False).nth))
    break

    #### merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session)
    consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
#     pp(consecutive_AIDs[:30])
#     pp(consecutive_AIDs[29:61])

    #### remove all the rows which aid_x == aid_y (remove the row when the two articles are the same)
    consecutive_AIDs = consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
    
    #### add a column named 'days_elapsed' which shows how many days passed between the two aids in a session
    consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
#     pp(consecutive_AIDs[:61])

    #### keep the rows if the two aids of a session are occurred within the same day on the right order
    consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
    
    #### among all sessions or all users, for each article occurred, count how often the other articles are occurred
    for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
        next_AIDs[aid_x][aid_y] += 1
    
```

### rd: recsys - otto - covisitation_simplified - merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session) - consecutive_AIDs = current_chunk.merge(current_chunk, on='session')

```python
chunk_size = 30_000

for i in range(0, sessions.shape[0], chunk_size): # loop every 30_000 unique sessions
    
    #### take a chunk_size number of sessions (each session in its entirety, ie., many rows) from subsets as current_chunk
    
#     pp(i, sessions[i], sessions.shape[0]-1, i+chunk_size-1, min(sessions.shape[0]-1, i+chunk_size-1))
    current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) # to remove index column
#     pp(current_chunk.head(), current_chunk.tail())
#     break

    #### In current_chunk, from each session (in its entirety) only takes the last/latest 30 events and combine them to update current_chunk
#     pp(range(-30,0), list(range(-30,0)))
    current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
#     pp(current_chunk[:31], current_chunk[29:61])
#     pp(doc_sig(pd.DataFrame.reset_index))
#     pp(doc_sig(current_chunk.groupby('session', as_index=False).nth))
#     break

    #### merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session)
    consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
    pp(consecutive_AIDs[:30])
    pp(consecutive_AIDs[29:61])
    break

    #### remove all the rows which aid_x == aid_y (remove the row when the two articles are the same)
    consecutive_AIDs = consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
    
    #### add a column named 'days_elapsed' which shows how many days passed between the two aids in a session
    consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
#     pp(consecutive_AIDs[:61])

    #### keep the rows if the two aids of a session are occurred within the same day on the right order
    consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
    
    #### among all sessions or all users, for each article occurred, count how often the other articles are occurred
    for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
        next_AIDs[aid_x][aid_y] += 1
    
```

### rd: recsys - otto - covisitation_simplified - remove all the rows which aid_x == aid_y (remove the row when the two articles are the same) as they are meaningless - consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]

### rd: recsys - otto - covisitation_simplified - add a column named 'days_elapsed' which shows how many days passed between the two aids in a session - consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)

### rd: recsys - otto - covisitation_simplified - keep the rows if the two aids of a session are occurred within the same day on the right order (one after the other) -     consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]

```python
chunk_size = 30_000

for i in range(0, sessions.shape[0], chunk_size): # loop every 30_000 unique sessions
    
    #### take a chunk_size number of sessions (each session in its entirety, ie., many rows) from subsets as current_chunk
    
#     pp(i, sessions[i], sessions.shape[0]-1, i+chunk_size-1, min(sessions.shape[0]-1, i+chunk_size-1))
    current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) # to remove index column
#     pp(current_chunk.head(), current_chunk.tail())
#     break

    #### In current_chunk, from each session (in its entirety) only takes the last/latest 30 events and combine them to update current_chunk
#     pp(range(-30,0), list(range(-30,0)))
    current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
#     pp(current_chunk[:31], current_chunk[29:61])
#     pp(doc_sig(pd.DataFrame.reset_index))
#     pp(doc_sig(current_chunk.groupby('session', as_index=False).nth))
#     break

    #### merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session)
    consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
#     pp(consecutive_AIDs[:30])
#     pp(consecutive_AIDs[29:61])
#     break

    #### remove all the rows which aid_x == aid_y (remove the row when the two articles are the same)
    consecutive_AIDs = consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
    
    #### add a column named 'days_elapsed' which shows how many days passed between the two aids in a session
    consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
    pp(consecutive_AIDs[:20])

    #### keep the rows if the two aids of a session are occurred within the same day on the right order
    consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
    pp(consecutive_AIDs[:20])
    break
    
    #### among all sessions or all users, for each article occurred, count how often the other articles are occurred
    for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
        next_AIDs[aid_x][aid_y] += 1
    
```

### rd: recsys - otto - covisitation_simplified - among all sessions/rows selected (regardless which session we are looking at), for each aid occurred, count how often the other aids are occurred -     for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']): next_AIDs[aid_x][aid_y] += 1

```python
chunk_size = 30_000

for i in range(0, sessions.shape[0], chunk_size): # loop every 30_000 unique sessions
    
    #### take a chunk_size number of sessions (each session in its entirety, ie., many rows) from subsets as current_chunk
    
#     pp(i, sessions[i], sessions.shape[0]-1, i+chunk_size-1, min(sessions.shape[0]-1, i+chunk_size-1))
    current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) # to remove index column
#     pp(current_chunk.head(), current_chunk.tail())
#     break

    #### In current_chunk, from each session (in its entirety) only takes the last/latest 30 events and combine them to update current_chunk
#     pp(range(-30,0), list(range(-30,0)))
    current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
#     pp(current_chunk[:31], current_chunk[29:61])
#     pp(doc_sig(pd.DataFrame.reset_index))
#     pp(doc_sig(current_chunk.groupby('session', as_index=False).nth))
#     break

    #### merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session)
    consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
#     pp(consecutive_AIDs[:30])
#     pp(consecutive_AIDs[29:61])
#     break

    #### remove all the rows which aid_x == aid_y (remove the row when the two articles are the same)
    consecutive_AIDs = consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
    
    #### add a column named 'days_elapsed' which shows how many days passed between the two aids in a session
    consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
#     pp(consecutive_AIDs[:20])

    #### keep the rows if the two aids of a session are occurred within the same day on the right order
    consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
#     pp(consecutive_AIDs[:20])
#     break
    
    #### among all sessions or all users, for each article occurred, count how often the other articles are occurred
    for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
        next_AIDs[aid_x][aid_y] += 1
    pp(next_AIDs)
    break
```

### rd: src - recsys - otto - covisitation_simplified

```python
chunk_size = 30_000

for i in range(0, sessions.shape[0], chunk_size): # loop every 30_000 unique sessions
    
    #### take a chunk_size number of sessions (each session in its entirety, ie., many rows) from subsets as current_chunk
    
#     pp(i, sessions[i], sessions.shape[0]-1, i+chunk_size-1, min(sessions.shape[0]-1, i+chunk_size-1))
    current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) # to remove index column
#     pp(current_chunk.head(), current_chunk.tail())
#     break

    #### In current_chunk, from each session (in its entirety) only takes the last/latest 30 events and combine them to update current_chunk
#     pp(range(-30,0), list(range(-30,0)))
    current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
#     pp(current_chunk[:31], current_chunk[29:61])
#     pp(doc_sig(pd.DataFrame.reset_index))
#     pp(doc_sig(current_chunk.groupby('session', as_index=False).nth))
#     break

    #### merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session)
    consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
#     pp(consecutive_AIDs[:30])
#     pp(consecutive_AIDs[29:61])
#     break

    #### remove all the rows which aid_x == aid_y (remove the row when the two articles are the same)
    consecutive_AIDs = consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
    
    #### add a column named 'days_elapsed' which shows how many days passed between the two aids in a session
    consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
#     pp(consecutive_AIDs[:20])

    #### keep the rows if the two aids of a session are occurred within the same day on the right order
    consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
#     pp(consecutive_AIDs[:20])
#     break
    
    #### among all sessions or all users, for each article occurred, count how often the other articles are occurred
    for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
        next_AIDs[aid_x][aid_y] += 1
#     pp(next_AIDs)
#     break
```

### rd: recsys - otto - covisitation_simplified - remove some data objects to save RAM

```python
del train, subset_of_train, subsets # to save RAM
```

## rd: recsys - otto - covisitation_simplified - make predictions


### rd: recsys - otto - covisitation_simplified - group the test set by session, under each session, put all aids into a list, and put all action types into another list - test.reset_index(drop=True).groupby('session')['aid'].apply(list)

```python
test, test.reset_index(drop=True)
```

```python
test_session_AIDs = test.reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types = test.reset_index(drop=True).groupby('session')['type'].apply(list)
```

```python
test_session_AIDs
test_session_types
```

### rd: recsys - otto - covisitation_simplified - setup, create some containers, such as labels, no_data, no_data_all_aids - type_weight_multipliers = {0: 1, 1: 6, 2: 3} - session_types = ['clicks', 'carts', 'orders']

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']
```

### rd: recsys - otto - covisitation_simplified - loop every session, access all of its aids and types - for AIDs, types in zip(test_session_AIDs, test_session_types):
### rd: recsys - otto - covisitation_simplified - when there are >= 20 aids in a session: 
### rd: recsys - otto - covisitation_simplified - assign logspaced weight to each aid under each session, as the latter aids should have higher weight/probability to occur than the earlier aids. - if len(AIDs) >= 20: - weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']
idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
        pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
        pp(weights)
        pp(doc_sig(np.logspace))
        break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
#         pp(AIDs) # there are repeated aids
        for aid,w,t in zip(AIDs,weights,types): 
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
            
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
#         pp(aids_temp.items())
#         pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
#         pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         pp(aids_temp, sorted_aids)
#         break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
#         if len(AIDs) > 10:
#             pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])), doc_sig(dict.fromkeys))
# https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
#             pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
#             break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []
        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: candidates += [aid for aid, count in next_AIDs[AID].most_common(20)]
#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many times a test session have no aid in next_AIDs from training
        #### count how many new other aids offerred by next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
#         pp(AIDs, labels, no_data, no_data_all_aids)
#         idx += 1
#         if idx > 2: break
```

### rd: recsys - otto - covisitation_simplified - create a defaultdict (if no value to the key, set value to 0) - aids_temp=defaultdict(lambda: 0)

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']

idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
#         pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         pp(weights)
#         pp(doc_sig(np.logspace))
#         break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
        pp(aids_temp['a'], help(defaultdict))
        break
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
#         pp(AIDs) # there are repeated aids
        for aid,w,t in zip(AIDs,weights,types): 
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
            
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
#         pp(aids_temp.items())
#         pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
#         pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         pp(aids_temp, sorted_aids)
#         break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
#         if len(AIDs) > 10:
#             pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])), doc_sig(dict.fromkeys))
# https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
#             pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
#             break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []
        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: candidates += [aid for aid, count in next_AIDs[AID].most_common(20)]
#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many times a test session have no aid in next_AIDs from training
        #### count how many new other aids offerred by next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
#         pp(AIDs, labels, no_data, no_data_all_aids)
#         idx += 1
#         if idx > 2: break
```

### rd: recsys - otto - covisitation_simplified - loop each aid, weight, event_type of a session: - for aid,w,t in zip(AIDs,weights,types):
### rd: recsys - otto - covisitation_simplified - Within each session, accumulate the weight for each aid based on its occurences, event_type and logspaced weight; save the accumulated weight as value and each aid as key into a defaultdict (aids_temp), no duplicated aid here in this dict, and every session has its own aid_temp - aids_temp[aid] += w * type_weight_multipliers[t]

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']

idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
#         pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         pp(weights)
#         pp(doc_sig(np.logspace))
#         break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
#         pp(aids_temp['a'], help(defaultdict))
#         break
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
        for aid,w,t in zip(AIDs,weights,types):
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
        pp(AIDs, aids_temp) 
        break
        
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
#         pp(aids_temp.items())
#         pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
#         pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         pp(aids_temp, sorted_aids)
#         break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
#         if len(AIDs) > 10:
#             pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])), doc_sig(dict.fromkeys))
# https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
#             pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
#             break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []
        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: candidates += [aid for aid, count in next_AIDs[AID].most_common(20)]
#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many times a test session have no aid in next_AIDs from training
        #### count how many new other aids offerred by next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
#         pp(AIDs, labels, no_data, no_data_all_aids)
#         idx += 1
#         if idx > 2: break
```

### rd: recsys - otto - covisitation_simplified - sort a defaultdict from largest weight to smallest weight of all aids in each session???, and then put its keys into a list named sorted_aids - sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]

### rd: recsys - otto - covisitation_simplified - store the first 20 aids (the most weighted or most likely aids to be acted upon within a session) into the list 'labels' -         labels.append(sorted_aids[:20])

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']

idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
#         pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         pp(weights)
#         pp(doc_sig(np.logspace))
#         break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
#         pp(aids_temp['a'], help(defaultdict))
#         break
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
        for aid,w,t in zip(AIDs,weights,types): 
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
#         pp(AIDs, aids_temp) 
#         break
        
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
        pp(aids_temp.items())
        pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
        pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        pp(aids_temp, sorted_aids)
        pp(help(sorted))
        break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
#         if len(AIDs) > 10:
#             pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])), doc_sig(dict.fromkeys))
# https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
#             pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
#             break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []
        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: candidates += [aid for aid, count in next_AIDs[AID].most_common(20)]
#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many times a test session have no aid in next_AIDs from training
        #### count how many new other aids offerred by next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
#         pp(AIDs, labels, no_data, no_data_all_aids)
#         idx += 1
#         if idx > 2: break
```




### rd: recsys - otto - covisitation_simplified - when there are < 20 aids in a session: - if len(AIDs) > 10:
### rd: recsys - otto - covisitation_simplified - within each test session, reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs - AIDs = list(dict.fromkeys(AIDs[::-1]))

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']

idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
#         pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         pp(weights)
#         pp(doc_sig(np.logspace))
#         break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
#         pp(aids_temp['a'], help(defaultdict))
#         break
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
        for aid,w,t in zip(AIDs,weights,types): 
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
#         pp(AIDs, aids_temp) 
#         break
        
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
#         pp(aids_temp.items())
#         pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
#         pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         pp(aids_temp, sorted_aids)
#         pp(help(sorted))
#         break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
        if len(AIDs) > 10:
            pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])))
# https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
            pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
            break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []
        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: candidates += [aid for aid, count in next_AIDs[AID].most_common(20)]
#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many times a test session have no aid in next_AIDs from training
        #### count how many new other aids offerred by next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
#         pp(AIDs, labels, no_data, no_data_all_aids)
#         idx += 1
#         if idx > 2: break
```

### rd: recsys - otto - covisitation_simplified - keep track the length of AIDs and create an empty list named candidates - AIDs_len_start = len(AIDs)
### rd: recsys - otto - covisitation_simplified - (within a session) for each AID inside AIDs: if AID is in the keys of next_AIDs (from training), then take the 20 most common other aids occurred (from next_AIDs) when AID occurred, into a list and add this list into the list named candidate (not a list of list, just a merged list). Each candidate in its full size has len(AIDs) * 20 number of other aids, which can have duplicated ids. -         for AID in AIDs: - if AID in next_AIDs: - candidates = candidates + [aid for aid, count in next_AIDs[AID].most_common(20)]

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']

idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
#         pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         pp(weights)
#         pp(doc_sig(np.logspace))
#         break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
#         pp(aids_temp['a'], help(defaultdict))
#         break
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
        for aid,w,t in zip(AIDs,weights,types): 
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
#         pp(AIDs, aids_temp) 
#         break
        
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
#         pp(aids_temp.items())
#         pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
#         pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         pp(aids_temp, sorted_aids)
#         pp(help(sorted))
#         break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
#         if len(AIDs) > 10:
#             pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])))
# # https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
#             pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
#             break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []

        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: 
                candidates = candidates + [aid for aid, count in next_AIDs[AID].most_common(20)]

#         if not bool(candidates): pp(AIDs)
#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many times a test session have no aid in next_AIDs from training
        #### count how many new other aids offerred by next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
#         pp(AIDs, labels, no_data, no_data_all_aids)
#         idx += 1
#         if idx > 2: break
```

### rd: recsys - otto - covisitation_simplified - find the first 40 most common aids in a candidate (for a session); and if they (these aids) are not found in AIDs then merge them into AIDs list, so that a session has a updated AIDs list (which most likely to occur) - AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
### rd: recsys - otto - covisitation_simplified - give the first 20 aids from AIDs to labels (a list); count how many test sessions whose aids are not seen in next_AIDs from training; count how many test sessions don't receive additional aids from next_AIDs  labels.append(AIDs[:20]) - if candidates == []: no_data += 1 - if AIDs_len_start == len(AIDs): no_data_all_aids += 1

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']

idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
#         pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         pp(weights)
#         pp(doc_sig(np.logspace))
#         break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
#         pp(aids_temp['a'], help(defaultdict))
#         break
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
        for aid,w,t in zip(AIDs,weights,types): 
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
#         pp(AIDs, aids_temp) 
#         break
        
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
#         pp(aids_temp.items())
#         pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
#         pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         pp(aids_temp, sorted_aids)
#         pp(help(sorted))
#         break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
#         if len(AIDs) > 10:
#             pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])))
# # https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
#             pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
#             break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []
#         with snoop:
#             idx = 0
        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: 
                candidates = candidates + [aid for aid, count in next_AIDs[AID].most_common(20)]
#                 idx+=1
#                 if idx > 2: break


#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many test sessions whose aids are not seen in next_AIDs from training
        #### count how many test sessions don't receive additional aids from next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
        pp(AIDs, labels, no_data, no_data_all_aids)
        idx += 1
        if idx > 2: break
```

### rd: src - recsys - otto - covisitation_simplified

```python
labels = []

no_data = 0
no_data_all_aids = 0
type_weight_multipliers = {0: 1, 1: 6, 2: 3}
session_types = ['clicks', 'carts', 'orders']

idx = 0
for AIDs, types in zip(test_session_AIDs, test_session_types): #### for each session, get all of its aids and types
    if len(AIDs) >= 20:
        #### assign logspaced weight to each arcticle id under each session
#         pp(AIDs, len(AIDs), np.logspace(0.1,1,len(AIDs),base=2, endpoint=True), np.logspace(0.1,1,len(AIDs),base=2, endpoint=False))
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         pp(weights)
#         pp(doc_sig(np.logspace))
#         break

        #### create a defaultdict to store weight for each id article under a session
        aids_temp=defaultdict(lambda: 0)
#         pp(aids_temp['a'], help(defaultdict))
#         break
    
        #### loop each article id, its weight and its type from all aids, their weights, and types
        for aid,w,t in zip(AIDs,weights,types): 
        
            #### Within each session, accumulate the weight for each aid based on its occurences, 
            #### event_type and logspaced weight; save the accumulated weight for each aid into a defaultdict, 
            #### no duplicated aid here in this dict
            aids_temp[aid] += w * type_weight_multipliers[t]
#         pp(AIDs, aids_temp) 
#         break
        
        #### sort a defaultdict from largest value to smallest, and then put its keys into a list
#         pp(aids_temp.items())
#         pp(sorted(aids_temp.items(), key=lambda item: -item[1]))
#         pp(sorted(aids_temp.items(), key=lambda item: item[1])) 
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         pp(aids_temp, sorted_aids)
#         pp(help(sorted))
#         break

        #### store the first 20 aids into the list 'labels'
        labels.append(sorted_aids[:20])
        
    else:
        
        #### reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs
#         if len(AIDs) > 10:
#             pp(AIDs, AIDs[::-1], dict.fromkeys(AIDs[::-1]), list(dict.fromkeys(AIDs[::-1])))
# # https://stackoverflow.com/questions/31633635/what-is-the-meaning-of-inta-1-in-python [::-1] means reverse order
#             pp(AIDs, list(dict.fromkeys(AIDs[::-1])), list(dict.fromkeys(AIDs[::1])), list(dict.fromkeys(AIDs)))
#             break
        AIDs = list(dict.fromkeys(AIDs[::-1]))
    
        #### keep track the length of AIDs
        AIDs_len_start = len(AIDs)
        
        candidates = []
#         with snoop:
#             idx = 0
        for AID in AIDs:
            #### if AID is in the keys of next_AIDs, then take the 20 most common other aids occurred when AID occurred into a list
            #### and add this list into the list candidate (not a list of list, just a merged list)
            if AID in next_AIDs: 
                candidates = candidates + [aid for aid, count in next_AIDs[AID].most_common(20)]
#                 idx+=1
#                 if idx > 2: break


#         if len(candidates) > 0: 
#             pp(candidates, Counter(candidates), Counter(candidates).most_common(40))
#             break

        #### use Counter to find the first 40 most common aid in candidate, and if they are not in AIDs then merge them into AIDs list
        AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
        #### append the first 20 aids to labels; 
        #### also count how many test sessions whose aids are not seen in next_AIDs from training
        #### count how many test sessions don't receive additional aids from next_AIDs
        labels.append(AIDs[:20])
        if candidates == []: no_data += 1
        if AIDs_len_start == len(AIDs): no_data_all_aids += 1
            
#         pp(AIDs, labels, no_data, no_data_all_aids)
#         idx += 1
#         if idx > 2: break
```

## rd: recsys - otto - covisitation_simplified - prepare results to CSV


### rd: recsys - otto - covisitation_simplified - make a list of lists (labels) into a list of strings (labels_as_strings) - labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

```python
labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]
pp(labels[:2], labels_as_strings[:2])
```

### rd: recsys - otto - covisitation_simplified - give each list of label strings a session number - predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

```python
pp(len(test_session_AIDs.index), len(labels_as_strings), test_session_AIDs.head())
predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})
predictions.head()
```

### rd: recsys - otto - covisitation_simplified - multi-objective means 'clicks', 'carts', and 'orders'; and we make the same predictions on them - session_types = ['clicks', 'carts', 'orders'] - for st in session_types: - modified_predictions = predictions.copy() - modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}' - prediction_dfs.append(modified_predictions)

```python
prediction_dfs = []
session_types = ['clicks', 'carts', 'orders']
for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)
```

```python
pp(prediction_dfs[0][:5], prediction_dfs[1][:5], prediction_dfs[2][:5])
```

### rd: recsys - otto - covisitation_simplified - get the csv file ready, stack on each other. - submission = pd.concat(prediction_dfs).reset_index(drop=True) - submission.to_csv('submission.csv', index=False) - submission.head()


the order of the rows does not matter, according to the dataset official [site](https://github.com/otto-de/recsys-dataset#submission-format)

```python
submission = pd.concat(prediction_dfs).reset_index(drop=True)
submission.to_csv('submission.csv', index=False)
submission.head()
print(f'Test sessions that we did not manage to extend based on the co-visitation matrix: {no_data_all_aids}')
```

```python
sample_sub.head()
```

The following plot (combined with the information printed above) is quite significant -- it show us how much data we are still missing, how many predictions are not at their maximum allowable length.

And there is never a point in not outputting all 20 AIDs for any given prediction!

```python
from matplotlib import pyplot as plt

plt.hist([len(l) for l in labels]);
plt.suptitle('Distribution of predicted sequence lengths');
```

```python
# fraction_of_sessions_to_use = 0.000001 # takes about 3 mins to run

# import pandas as pd
# import numpy as np

# train = pd.read_parquet('../input/otto-full-optimized-memory-footprint//train.parquet')
# test = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')

# sample_sub = pd.read_csv('../input/otto-recommender-system//sample_submission.csv')

# if fraction_of_sessions_to_use != 1:
#     lucky_sessions_train = train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
#     subset_of_train = train[train.session.isin(lucky_sessions_train)]
    
#     lucky_sessions_test = test.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
#     subset_of_test = test[test.session.isin(lucky_sessions_test)]
# else:
#     subset_of_train = train
#     subset_of_test = test

# subset_of_train.index = pd.MultiIndex.from_frame(subset_of_train[['session']])
# subset_of_test.index = pd.MultiIndex.from_frame(subset_of_test[['session']])

# chunk_size = 30_000
# min_ts = train.ts.min()
# max_ts = test.ts.max()

# from collections import defaultdict, Counter
# next_AIDs = defaultdict(Counter)

# subsets = pd.concat([subset_of_train, subset_of_test])
# sessions = subsets.session.unique()
# for i in range(0, sessions.shape[0], chunk_size):
#     current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True)
#     current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
#     consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
#     consecutive_AIDs = consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
#     consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
#     consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
    
#     for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']):
#         next_AIDs[aid_x][aid_y] += 1
    
# del train, subset_of_train, subsets

# session_types = ['clicks', 'carts', 'orders']
# test_session_AIDs = test.reset_index(drop=True).groupby('session')['aid'].apply(list)
# test_session_types = test.reset_index(drop=True).groupby('session')['type'].apply(list)

# labels = []

# no_data = 0
# no_data_all_aids = 0
# type_weight_multipliers = {0: 1, 1: 6, 2: 3}
# for AIDs, types in zip(test_session_AIDs, test_session_types):
#     if len(AIDs) >= 20:
#         weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
#         aids_temp=defaultdict(lambda: 0)
#         for aid,w,t in zip(AIDs,weights,types): 
#             aids_temp[aid]+= w * type_weight_multipliers[t]
            
#         sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
#         labels.append(sorted_aids[:20])
#     else:
#         AIDs = list(dict.fromkeys(AIDs[::-1]))
#         AIDs_len_start = len(AIDs)
        
#         candidates = []
#         for AID in AIDs:
#             if AID in next_AIDs: candidates += [aid for aid, count in next_AIDs[AID].most_common(20)]
#         AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
        
#         labels.append(AIDs[:20])
#         if candidates == []: no_data += 1
#         if AIDs_len_start == len(AIDs): no_data_all_aids += 1

# # >>> outputting results to CSV

# labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

# predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})
```

```python
# pp(len(test_session_AIDs.index), len(labels_as_strings))
```

```python
# predictions
# prediction_dfs = []

# for st in session_types:
#     modified_predictions = predictions.copy()
#     modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
#     prediction_dfs.append(modified_predictions)

# submission = pd.concat(prediction_dfs).reset_index(drop=True)
# submission.to_csv('submission.csv', index=False)

# print(f'Test sessions that we did not manage to extend based on the co-visitation matrix: {no_data_all_aids}')
```

```python

```
