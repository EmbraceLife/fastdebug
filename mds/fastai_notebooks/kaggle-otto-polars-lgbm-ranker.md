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

In this notebook we will train an LGBM Ranker.

In his very informative post, [Recommendation Systems for Large Datasets](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364721) [@ravishah1](https://www.kaggle.com/ravishah1) explains how re-ranking models are the industry standard for dealing with datasets like we are presented with in this competition, that is ones with high cardinality categories!

Earlier in this competition I shared a notebook [co-visitation matrix - simplified, imprvd logic 🔥](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic) which introduces the co-visitation matrix that can be used for candidate generation and scoring. (to read more about co-visitation matrices and how they work, please see [💡 What is the co-visiation matrix, really?](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365358))

Here, we will only look at ranking. I don't expect this notebook to achieve a particularly good score, but it will provide all the low level plumbing needed for training ranking models. One will be able to build on it and improve the result (via for instance adding new candidates generated using co-visitation matrices!).

For data processing we will use [polars](https://www.pola.rs/). Polars is a very interesting library that I wanted to try for a very long time now. It is written in Rust and embraces running on multiple cores. And I must say it delivers! I liked the API quite a bit and its speed (though in that department `cudf` would still be my first choice!). I am however not touching my GPU quata on Kaggle just yet as I have a couple of things lined up that I would like to share with you that definitely will require the GPU! 🙂

**Would appreciate [your upvote on the accompanying thread](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366194) to increase visibility.** 🙏

To simplify the code, I am using a version of the dataset that I shared [here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843). No need for dealing with `jsonl` files any longer as it's all `parquet` files now! (Specifically, I am using a version of this dataset that I preprared for local validation [in this notebook](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework).)

**If you like this notebook, please upvote! Thank you! 😊**

You might also find useful:

* [💡 What is the co-visiation matrix, really?](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365358)
* [🐘 the elephant in the room -- high cardinality of targets and what to do about this](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364722)
* [💡 Best hyperparams for the co-visitation matrix based on HPO study with 30 runs](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365153)
* [💡A robust local validation framework 🚀🚀🚀](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework)
* [📅 Dataset for local validation created using organizer's repository (parquet files)](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534)


## rd: recsys - otto - LGBM Ranker - my utils

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

## rd: recsys - otto - LGBM Ranker - use polars to load Radek's local validation dataset

```python
!pip install polars
```

### rd: recsys - otto - LGBM Ranker - import polars as pl - train = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test.parquet') - train_labels = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test_labels.parquet')

```python
import polars as pl
```

```python
train = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test.parquet')
# test_labels.parquet is well prepared for local validation calculation
train_labels = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test_labels.parquet')
```

### rd: recsys - otto - LGBM Ranker - use local validation dataset save lots of data processing time (I should not have tried to do it for every run of a notebook); in this dataset there are train, test, and valid sets which are all splitted from the original training set; valid set is turned into test_label in the format good for local validation score calculation. In this notebook, LGBM Ranker model is trained on train set (actually test set) and test_labels (actually processed from valid sets), so the model is trained with a smaller amount of data for experiment and time saving.


### rd: recsys - otto - LGBM Ranker - todo: train LGBM Ranker on the entire training set


### rd: recsys - otto - LGBM Ranker - We can check their datetime to confirm the dataset length - datetime.datetime.fromtimestamp(real_train['ts'].min()), datetime.datetime.fromtimestamp(real_train['ts'].max())

```python
real_train = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/train.parquet')
```

```python
real_train['ts'].min(), real_train['ts'].max()

import datetime
datetime.datetime.fromtimestamp(real_train['ts'].min()), datetime.datetime.fromtimestamp(real_train['ts'].max()) # 3 weeks

datetime.datetime.fromtimestamp(train['ts'].min()), datetime.datetime.fromtimestamp(train['ts'].max()) # 1 weeks
```

### rd: recsys - otto - LGBM Ranker - check session intersection

```python
train['session'].unique().shape[0], train_labels['session'].unique().shape[0]
```

```python
len(set(train['session']).intersection(set(train_labels['session'])))
```

> We are calculating the scores that we used for creating co-vistation matrices! We know they carry signal, so let's provde this information to our `LGBM Ranker`!

### rd: recsys - otto - LGBM Ranker - question: where/which notebook did Radek calc the scores "we used for creating co-vistation matrices!" in the first place? What do we know aobut their sygnal?


## rd: recsys - otto - LGBM Ranker - calc and add the features to train for LGBM Ranker model


### rd: recsys - otto - LGBM Ranker - use pp and return to debug every line of the functions (see src below)


### rd: recsys - otto - LGBM Ranker - (polars) select all existing columns, select "session" col to apply cumcount, reverse, over('session'), and rename it action_num_reverse_chrono - df.select([pl.col('*'),pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono')])


### rd: recsys - otto - LGBM Ranker - (polars) select all existing columns, select "session" col to apply count, over('session'), rename to 'session_length'


### rd: recsys - otto - LGBM Ranker - (polars) add or overwrite more column (a Series with an exprssion to calc) to df, named log_recency_score which calc log_recency_score - df.with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)


### rd: recsys - otto - LGBM Ranker - (polars) create a pl Series by apply a lambda to a column - pl.Series(df['type'].apply(lambda x: type_weights[x]) * df['log_recency_score'])


### rd: recsys - otto - LGBM Ranker - (polars) add or replace a column to df - df.with_column(type_weighted_log_recency_score.alias('type_weighted_log_recency_score'))

```python
def add_action_num_reverse_chrono(df):
    "add a column named action_num_reverse_chrono, which is to count the num of rows/events in each session and give each row an index, \
    then reverse the index while the rows remain unchanged."
    res = df.select([
        pl.col('*'),
        pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono')
    ])
    return res

def add_session_length(df):
    "add a column named session_length, which count num of rows/events in each session and put the count on each row of the session"
    res = df.select([
        pl.col('*'),
        pl.col('session').count().over('session').alias('session_length')
    ])
#     pp(res.head(10))
#     return
    return res

def add_log_recency_score(df):
    "add a column named log_recency_score, which calc log_recency_score. But no idea what does it mean!!!"
    linear_interpolation = 0.1 + ((1-0.1) / (df['session_length']-1)) * (df['session_length']-df['action_num_reverse_chrono']-1)
    res = df.with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)
#     pp(res.head(10))
#     return 
    return res

def add_type_weighted_log_recency_score(df):
    type_weights = {0:1, 1:6, 2:3} # this is difference from the weight given by organizer clicks=0.1, carts=0.3, orders=0.6
    type_weighted_log_recency_score = pl.Series(df['type'].apply(lambda x: type_weights[x]) * df['log_recency_score'])
    return df.with_column(type_weighted_log_recency_score.alias('type_weighted_log_recency_score'))

def apply(df, pipeline):
    for f in pipeline:
        df = f(df)
    return df
```

```python
pipeline = [add_action_num_reverse_chrono, add_session_length, add_log_recency_score, add_type_weighted_log_recency_score]
```

```python
train = apply(train, pipeline)
```

All done!


## rd: recsys - otto - LGBM Ranker - questions on the feature engineering 


### rd: recsys - otto - LGBM Ranker - Are there two ways of doing co-visitation features? [answered](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker/comments#2037807) by Radek


### rd: recsys - otto - LGBM Ranker - question - why weights are different - The 0,1,2 refers to `clicks`, `carts` and `orders`, but why the values are so different? (0.1, 0.3, 0.6 vs 1,6,3) [answered](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker/comments#2037868) by @radek1


In the local validation [notebook](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework?scriptVersionId=110252868&cellId=22), you used the weights of clicks, carts and orders provided by the organizer to calculate score: `local_validation_score = (recall_per_type * pd.Series({'clicks': 0.10, 'carts': 0.30, 'orders': 0.60})).sum()`

[Here](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker/comments#2037807) in this LGBM Ranker notebook and the co-visitation matrix notebook you used `type_weight_multipliers = {0: 1, 1: 6, 2: 3}`.

The 0,1,2 refers to `clicks`, `carts` and `orders`, but why the values are so different? (0.1, 0.3, 0.6 vs 1,6,3)

I must have missed something big here, could you help me understand the usages of weights in these two places? Thank you so much @radek1

```python
train.head()
```

## rd: recsys - otto - LGBM Ranker - process our labels to merge them onto our train set.

```python
train_labels = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test_labels.parquet')
train_labels.head()
```

### rd: recsys - otto - LGBM Ranker - (polars) 'explode' a list of aids as a single value in a column into multiple rows of a column - train_labels.explode('ground_truth')


### rd: recsys - otto - LGBM Ranker - (polars) add two columns to df by rename one and overwrite another column - train_labels.explode('ground_truth').with_columns([pl.col('ground_truth').alias('aid'),pl.col('type').apply(lambda x: type2id[x])])


### rd: recsys - otto - LGBM Ranker - (polars) select 3 columns of a df - train_labels[['session', 'type', 'aid']]

```python
type2id = {"clicks": 0, "carts": 1, "orders": 2}

train_labels = train_labels.explode('ground_truth').with_columns([
    pl.col('ground_truth').alias('aid'),
    pl.col('type').apply(lambda x: type2id[x])
])[['session', 'type', 'aid']]
```

### rd: recsys - otto - LGBM Ranker - (polars) overwrite 3 columns by casting each to a different type - train_labels.with_columns([pl.col('session').cast(pl.datatypes.Int32),pl.col('type').cast(pl.datatypes.UInt8),pl.col('aid').cast(pl.datatypes.Int32)])

```python
train_labels = train_labels.with_columns([
    pl.col('session').cast(pl.datatypes.Int32),
    pl.col('type').cast(pl.datatypes.UInt8),
    pl.col('aid').cast(pl.datatypes.Int32)
])
```

### rd: recsys - otto - LGBM Ranker - (polars) add a column by filling in a literal value - train_labels.with_column(pl.lit(1).alias('gt'))

```python
train_labels = train_labels.with_column(pl.lit(1).alias('gt'))
```

### rd: recsys - otto - LGBM Ranker - (polars) merge or join two dfs on 3 columns - train.join(train_labels, how='left', on=['session', 'type', 'aid'])


### rd: recsys - otto - LGBM Ranker - (polars) overwrite a column by filling null with 0 - train.join(train_labels, how='left', on=['session', 'type', 'aid']).with_column(pl.col('gt').fill_null(0))

```python
train = train.join(train_labels, how='left', on=['session', 'type', 'aid']).with_column(pl.col('gt').fill_null(0))
```

```python
train.head()
```

## rd: recsys - otto - LGBM Ranker - how to group and compress all rows of a session into a single row/value for the session

```python
def get_session_lenghts(df):
    return df.groupby('session').agg([
        pl.col('session').count().alias('session_length')
    ])['session_length'].to_numpy()
```

### rd: recsys - otto - LGBM Ranker - (polars) group all rows of a session, agg or compress into a single row with the value of count of the rows in the session - train.groupby('session').agg([pl.col('session').count().alias('session_length')])
### rd: recsys - otto - LGBM Ranker - (polars) select a single column from a df and convert it to a numpy array - df['session_length'].to_numpy()


```python
train.groupby('session').agg([
        pl.col('session').count().alias('session_length')
    ]).head()
```

```python
session_lengths_train = get_session_lenghts(train)
```

## rd: recsys - otto - LGBM Rander - Build and Train a LGBM Ranker


### rd: recsys - otto - LGBM Rander - import and build a LGBM Ranker - from lightgbm.sklearn import LGBMRanker - ranker = LGBMRanker(objective="lambdarank",metric="ndcg",boosting_type="dart",n_estimators=20,importance_type='gain',)

```python
from lightgbm.sklearn import LGBMRanker
```

```python
ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    n_estimators=20,
    importance_type='gain',
)
```

### rd: recsys - otto - LGBM Rander - find features and target columns for LGMB Ranker model - train.columns - feature_cols = ['aid', 'type', 'action_num_reverse_chrono', 'session_length', 'log_recency_score', 'type_weighted_log_recency_score']- target = 'gt'

```python
train.columns
```

```python
feature_cols = ['aid', 'type', 'action_num_reverse_chrono', 'session_length', 'log_recency_score', 'type_weighted_log_recency_score']
target = 'gt'
```

### rd: recsys - otto - LGBM Rander - get features column names and target and group for training the LGMB Ranker model with 


### rd: recsys - otto - LGBM Rander - train the model with feature columns, target column and group the rows using session_length_train - ranker = ranker.fit(train[feature_cols].to_pandas(),train[target].to_pandas(),group=session_lengths_train,)

```python
ranker = ranker.fit(
    train[feature_cols].to_pandas(),
    train[target].to_pandas(),
    group=session_lengths_train,
)
```

### rd: recsys - otto - LGBM Rander - question: can we do local validation on LGBM Ranker model? But if we want to train the model with train.parquet and use test.parquet and test_labels.parquet to do local validation to see how good the model is. However, we can't just use train.parquet, we will have to do random split on train.parquet to have a new train.parquet and train_labels.parquet. So, I wonder whether @radek1 could update his dataset to include them.


## rd: recsys - otto - LGBM Ranker - load test set, process it to get features, and make predictions


### rd: recsys - otto - LGBM Ranker - load and process test set - test = pl.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet') - test = apply(test, pipeline)


### rd: recsys - otto - LGBM Ranker - use model to predict with the feature columns of the test set - scores = ranker.predict(test[feature_cols].to_pandas())

```python
test = pl.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
test = apply(test, pipeline)
```

```python
scores = ranker.predict(test[feature_cols].to_pandas())
```

## rd: recsys - otto - LGBM Ranker - from predictions to submission df


### rd: recsys - otto - LGBM Ranker - add a column of score to test dataframe - test = test.with_columns(pl.Series(name='score', values=scores))


### rd: recsys - otto - LGBM Ranker - sort test dataframe by 2 columns 'session' and 'score' and reverse the order (session from high to low, and then score from high to low within a session) - test.sort(['session', 'score'], reverse=True)


### rd: recsys - otto - LGBM Ranker - take every row of a session and compress them into a single row/value which is a list of the first 20 aids of the session (return a 2-column df) - test.groupby('session').agg([pl.col('aid').limit(20).list()])

```python
test = test.with_columns(pl.Series(name='score', values=scores))
test_predictions = test.sort(['session', 'score'], reverse=True).groupby('session').agg([
    pl.col('aid').limit(20).list()
])
```

```python
test_predictions.columns
```

## rd: recsys - otto - LGBM Ranker - make the submission


### rd: recsys - otto - LGBM Ranker - loop every session number and aid list - for session, preds in zip(test_predictions['session'].to_numpy(), test_predictions['aid'].to_numpy()):


### rd: recsys - otto - LGBM Ranker - turn a list into a string of aids separated by " " - l = ' '.join(str(p) for p in preds)


### rd: recsys - otto - LGBM Ranker - create a list to contain the string of aids and a list to contain session + type - labels.append(l) - session_types.append(f'{session}_{session_type}')


### rd: recsys - otto - LGBM Ranker - create a dataframe with a dict of two lists - submission = pl.DataFrame({'session_type': session_types, 'labels': labels})


### rd: recsys - otto - LGBM Ranker - (polars) write dataframe into csv file - submission.write_csv('submission.csv')

```python
session_types = []
labels = []

for session, preds in zip(test_predictions['session'].to_numpy(), test_predictions['aid'].to_numpy()):
    l = ' '.join(str(p) for p in preds)
    for session_type in ['clicks', 'carts', 'orders']:
        labels.append(l)
        session_types.append(f'{session}_{session_type}')
```

```python
submission = pl.DataFrame({'session_type': session_types, 'labels': labels})
submission.write_csv('submission.csv')
```

## Journey


### jn: last two days 2022-11-19-21, I was consumed by how to refactor local validation notebook and the last 20 aid notebook. The problem is that processing data takes a lot of time even on Kaggle and many places can go wrong when running the whole thing with version control. After some reflections, I retrain myself to the following: 1. try not to change the original notebook codes as much as I can; 2. add detailed and searchable comments for each line of code necessarily; 3. go through all the good notebooks and revisit commented notebooks daily to improve the search experience /2022-11-21


### jn: what an amazing discussion and thought process by @radek1 and @cdeotte https://www.kaggle.com/competitions/otto-recommender-system/discussion/366474 /2022-11-21


## rd: recsys - otto - LGBM Ranker - todo: 1. read this post and discussion to improve on this ranker model https://www.kaggle.com/competitions/otto-recommender-system/discussion/366474; 2. read this post to have a general understanding of XGB or LGBM Ranker and more https://www.kaggle.com/competitions/otto-recommender-system/discussion/366477

```python

```
