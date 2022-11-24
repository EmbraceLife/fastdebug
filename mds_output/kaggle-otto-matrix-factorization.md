## rd: recsys - otto - matrix factorization - 
### rd: recsys - otto - matrix factorization - todos: learn more of ideas of co-visitation, matrix factorization, embeddings, and how do they relate to each other
### rd: recsys - otto - matrix factorization - todos: before moving onto exploring the ideas and gain proper understanding of those ideas, I should get myself very very familiar with the codes and what they do

A co-visitation matrix is essentially an "analog" approximation to matrix factorization! I talk a bit more about this idea here: [💡 What is the co-visitation matrix, really?](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365358).

But matrix factorization has a lot of advantages as compared to co-visitation matrices. First of all, it can make better use of data -- it operates on the notion of similarity between categories. We can construct a more powerful representation if our model understands that aid `1` is similar to aid `142` as opposed to it treating each aid as an atomic entity (this is the jump from unigram/bigram/trigram models to word2vec in NLP).

Let us thus train a matrix factorization model and replace the co-visitation matrices with it!

Now, I don't expect that the first version of the model will be particularly well tuned. There has already been a lot of work put into co-visitation matrices and in the later versions we work off 3 different matrices, one for each category of actions! A similar progression can and will happen with matrix factorization 🙂 This notebook hopefully will enable us to jumpstart this type of exploration 🙂

To streamline the work, we will use data in `parquet` format. (Here is the notebook [💡 [Howto] Full dataset as parquet/csv files](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files) and here is [the most up-to-date version of the dataset](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint), no need for dealing with `jasonl` files and the associated mess any longer! Please upvote if you find this useful!)

For data processing we will use [polars](https://www.pola.rs/). `Polars` has a much smaller memory footprint than `pandas` and is quite fast. Plus it has really clean, intuitive API.

Let's get to work! 🙂

**If you like this notebook, please smash the upvote button! Thank you! 🙏**


## Other resources you might find useful:

* [💡 [Howto] Full dataset as parquet/csv files](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files) -- basing your work off parquet files without having to iterate over `jsonl` can save you a lot of time!
* [🐘 the elephant in the room -- high cardinality of targets and what to do about this](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364722)
* [💡 Best hyperparams for the co-visitation matrix based on HPO study with 30 runs](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365153)
* [💡A robust local validation framework 🚀🚀🚀](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework)
* [📅 Dataset for local validation created using organizer's repository (parquet files)](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364534)

## import my utils


```
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


<style>.container { width:100% !important; }</style>


## rd: recsys - otto - matrix factorization - takes 50+ mins to run, based on version 6 of @radek1's notebook

## rd: recsys - otto - matrix factorization - @radek1 provides us ways to improve on this notebook

## rd: recsys - otto - matrix factorization - use polars to read train.parquet and test.parquet - !pip install polars - import polars as pl - train = pl.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')


```
!pip install polars

import polars as pl

train = pl.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
test = pl.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
```

We need to create `aid-aid` pairs to train our matrix factorization model!

Let's us grab the pairs both from the train and test set.

## rd: recsys - otto - matrix factorization - build aid-aid pairs of train + test - this is the entire training set for this notebook - no other columns are used

### rd: recsys - otto - matrix factorization - stack train on top of test - pl.concat([train, test])

### rd: recsys - otto - matrix factorization - group by session, aggregate/compress rows of column 'aid' in the session into a single row and aggregate column 'aid' and shift forward by 1 and name it 'aid_next' - df.groupby('session').agg([pl.col('aid'),pl.col('aid').shift(-1).alias('aid_next')])

### rd: recsys - otto - matrix factorization - explode the values of each row in columns 'aid' and 'aid_next' - df.explode(['aid', 'aid_next'])

### rd: recsys - otto - matrix factorization - remove all nulls in all rows - df.drop_nulls()

### rd: recsys - otto - matrix factorization - select only specified columns - df[['aid', 'aid_next']]


```
%%time

train_pairs = (pl.concat([train, test])
    .groupby('session').agg([
        pl.col('aid'),
        pl.col('aid').shift(-1).alias('aid_next')
    ])
    .explode(['aid', 'aid_next'])
    .drop_nulls()
)[['aid', 'aid_next']]
```

### rd: recsys - otto - matrix factorization - how many rows and memory useage of train_pairs - ppn(train_pairs.shape) - train_pairs.to_pandas().memory_usage()


```
train_pairs.shape[0] / 1_000_000, ppn(train_pairs.shape[0])
```

That is 209 million pairs created in 40 seconds without running out of RAM! 🙂 Not too bad


```
train_pairs.head()
train_pairs.to_pandas().memory_usage()
```

Let's see what is the cardinality of our aids -- we will need this to create the embedding layer.

## rd: recsys - otto - matrix factorization - help(polars_df) to find out more of the usages

### rd: recsys - otto - matrix factorization - sort a single or multiple columns in reverse order or not - train_pairs.sort([pl.col("aid"), pl.col("aid_next")],reverse=[True, False],)

### rd: recsys - otto - matrix factorization - (polars) find unique rows - len(train_pairs.to_pandas().aid.unique()), len(train_pairs.to_pandas().aid_next.unique()), 

### rd: recsys - otto - matrix factorization - question: what does cardinality really mean? - cardinality_aids = max(train_pairs['aid'].max(), train_pairs['aid_next'].max()) - It seems to be the largest aid number


```
cardinality_aids = max(train_pairs['aid'].max(), train_pairs['aid_next'].max())
```


```
ppn(cardinality_aids)
```

We will have up to `1855602` -- that is a lot! But our matrix factorization model will be able to handle this.

Let's construct a `PyTorch` dataset and `dataloader`.

## rd: recsys - otto - matrix factorization - build a ClicksDataset

### rd: recsys - otto - matrix factorization - import what needed from torch - import torch, from torch import nn, from torch.utils.data import Dataset, DataLoader

### rd: recsys - otto - matrix factorization - how to build a ClickDataset class - build __init__, __getitem__, __len__ - see src below


```
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class ClicksDataset(Dataset):
    def __init__(self, pairs):
        self.aid1 = pairs['aid'].to_numpy()
        self.aid2 = pairs['aid_next'].to_numpy()
    def __getitem__(self, idx):
        aid1 = self.aid1[idx]
        aid2 = self.aid2[idx]
        return [aid1, aid2]
    def __len__(self):
        return len(self.aid1)

train_ds = ClicksDataset(train_pairs[:-10_000_000])
valid_ds = ClicksDataset(train_pairs[10_000_000:])
```

Let us see how quickly we can iterate over a single epoch with a batch size of `65536`.

### rd: recsys - otto - matrix factorization - instantiate a ClicksDataset and create a DataLoader - train_ds = ClicksDataset(train_pairs) - train_dl_pytorch = DataLoader(train_ds, 65536, True, num_workers=2)


```
train_ds = ClicksDataset(train_pairs)
train_dl_pytorch = DataLoader(train_ds, 65536, True, num_workers=2)
```

### rd: recsys - otto - matrix factorization - loop every batch of 65536 samples and access data from each sample and time the process - %%time - for batch in train_dl_pytorch: aid1, aid2 = batch[0], batch[1]


```
%%time
for batch in train_dl_pytorch:
    aid1, aid2 = batch[0], batch[1]
```

## rd: recsys - otto - matrix factorization - why torch Dataset and DataLoader take so long to access data? indexing into the the arrays and collating results into batches is very computationally expensive.

Oh dear, that took forever! Mind you, were are not doing anything here, apart from iterating over the dataset for a single epoch (and that is without validation!).

The reason this is taking so long is that indexing into the the arrays and collating results into batches is very computationally expensive.

There are ways to work around this but they require writing a lot of code (you could use the iterable-style dataset). And still our solution wouldn't be particularly well optimized.

## rd: recsys - otto - matrix factorization - Merlin DataLoader can rescue torch DataLoader with great speed, but kaggle's GPU RAM is too small so we have to use Merlin DataLoader with CPU RAM.

Let us do something else instead!

We will use a brand new [Merlin Dataloader](https://github.com/NVIDIA-Merlin/dataloader). It is a library that my team launched just a couple of days ago 🙂

Now this library shines when you have a GPU, which is what you generally want when training DL models. But, alas, Kaggle gives you only 13 GB of RAM on a kernel with a GPU, and that wouldn't allow us to process our dataset!

Let's see how far we can get with CPU only.

## rd: recsys - otto - matrix factorization - how to install Merlin DataLoader - !pip install merlin-dataloader


```
!pip install merlin-dataloader
```

### rd: recsys - otto - matrix factorization - save dataframe into parquet files - train_pairs[:-10_000_000].to_pandas().to_parquet('train_pairs.parquet') - train_pairs[-10_000_000:].to_pandas().to_parquet('valid_pairs.parquet')

### rd: recsys - otto - matrix factorization - what to import from merlin - from merlin.loader.torch import Loader - from merlin.io import Dataset

### rd: recsys - otto - matrix factorization - merlin dataloader can access dataset directly from disk with parquet files and make into Datase and Loader - train_ds = Dataset('train_pairs.parquet') - train_dl_merlin = Loader(train_ds, 65536, True)

### rd: recsys - otto - matrix factorization - access  - %%time - for batch in train_dl_merlin: - aid1, aid2 = batch[0], batch[1]

### rd: recsys - otto - matrix factorization - help(df) to learn more of how to use merlin-dataloader

We can read data directly from the disk -- even better!

Let's write our datasets to disk.


```
train_pairs[:-10_000_000].to_pandas().to_parquet('train_pairs.parquet')
train_pairs[-10_000_000:].to_pandas().to_parquet('valid_pairs.parquet')
```


```
from merlin.loader.torch import Loader 
from merlin.io import Dataset

train_ds = Dataset('train_pairs.parquet')
train_dl_merlin = Loader(train_ds, 65536, True)
```


```
%%time

for batch in train_dl_merlin:
    aid1, aid2 = batch[0], batch[1]
```

That is much better 🙂. Let's train our matrix factorization model!

## rd: recsys - otto - matrix factorization - how to build a layer/model of MatrixFactorization

### rd: recsys - otto - matrix factorization - how to initialize to create an embedding function -     def __init__(self, n_aids, n_factors): - super().__init__() - self.aid_factors = nn.Embedding(n_aids, n_factors, sparse=True)

### rd: recsys - otto - matrix factorization - how to write the forward function -     def forward(self, aid1, aid2): - aid1 = self.aid_factors(aid1) - aid2 = self.aid_factors(aid2) - return (aid1 * aid2).sum(dim=1)

## rd: recsys - otto - matrix factorization - how to write a AverageMeter class

### rd: recsys - otto - matrix factorization - how to write __init__, reset, update, __str__

## rd: recsys - otto - matrix factorization - create a Dataset and DataLoader with merlin-dataloader for valid set

### rd: recsys - otto - matrix factorization - from valid_pairs.parquet - valid_ds = Dataset('valid_pairs.parquet') - valid_dl_merlin = Loader(valid_ds, 65536, True)


```
class MatrixFactorization(nn.Module):
    def __init__(self, n_aids, n_factors):
        super().__init__()
        self.aid_factors = nn.Embedding(n_aids, n_factors, sparse=True)
        
    def forward(self, aid1, aid2):
        aid1 = self.aid_factors(aid1)
        aid2 = self.aid_factors(aid2)
        return (aid1 * aid2).sum(dim=1)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

valid_ds = Dataset('valid_pairs.parquet')
valid_dl_merlin = Loader(valid_ds, 65536, True)
```

## rd: recsys - otto - matrix factorization - Instantiate a MatrixFactorization model and create an optimizer and loss function

### rd: recsys - otto - matrix factorization - Instantiate a model with MatrixFactorization - model = MatrixFactorization(cardinality_aids+1, 32)

### rd: recsys - otto - matrix factorization - Create an optimizer - from torch.optim import SparseAdam - num_epochs=1- lr=0.1 - optimizer = SparseAdam(model.parameters(), lr=lr)

### rd: recsys - otto - matrix factorization - Create a loss function - criterion = nn.BCEWithLogitsLoss()


```
from torch.optim import SparseAdam

num_epochs=1
lr=0.1

model = MatrixFactorization(cardinality_aids+1, 32)
optimizer = SparseAdam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
```

## rd: recsys - otto - matrix factorization - train matrix factorization model: forward, backward, trainloss and accuracy (see src below)

### rd: recsys - otto - matrix factorization - what are negative output and what they are  for? see radek's answer [here](https://www.kaggle.com/code/radek1/matrix-factorization-pytorch-merlin-dataloader/comments#2039714)


```
%%time

for epoch in range(num_epochs):
    for batch, _ in train_dl_merlin: # question: why this line of code is different from the demo above? 
        model.train()
        losses = AverageMeter('Loss', ':.4e')
            
        aid1, aid2 = batch['aid'], batch['aid_next']
        output_pos = model(aid1, aid2)
        output_neg = model(aid1, aid2[torch.randperm(aid2.shape[0])]) # explained nicely by Radek in the link above
        
        output = torch.cat([output_pos, output_neg])
        targets = torch.cat([torch.ones_like(output_pos), torch.zeros_like(output_pos)]) # explained nicely by Radek in the link above
        loss = criterion(output, targets)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
    
    # running local validation below ? (I think so)
    with torch.no_grad():
        accuracy = AverageMeter('accuracy')
        for batch, _ in valid_dl_merlin: # what is the other thing here in _?
            aid1, aid2 = batch['aid'], batch['aid_next']
            output_pos = model(aid1, aid2)
            output_neg = model(aid1, aid2[torch.randperm(aid2.shape[0])])
            accuracy_batch = torch.cat([output_pos.sigmoid() > 0.5, output_neg.sigmoid() < 0.5]).float().mean()
            accuracy.update(accuracy_batch, aid1.shape[0])
            
    print(f'{epoch+1:02d}: * TrainLoss {losses.avg:.3f}  * Accuracy {accuracy.avg:.3f}')
```

Let's grab the embeddings!

## rd: recsys - otto - matrix factorization - extract embeddings from the model for use later - embeddings = model.aid_factors.weight.detach().numpy()


```
embeddings = model.aid_factors.weight.detach().numpy()
```


```
embeddings.shape
```

## rd: recsys - otto - matrix factorization - create an instance of AnnoyIndex for approximate nearest neighbor search

### rd: recsys - otto - matrix factorization - import and create an instance of AnnoyIndex - from annoy import AnnoyIndex - index = AnnoyIndex(32, 'euclidean')

### rd: recsys - otto - matrix factorization - load embeddings to annoyindex - for i, v in enumerate(embeddings): - index.add_item(i, v)

### rd: recsys - otto - matrix factorization - build a forest of n trees - index.build(10) - question: do we must run this line?


```
%%time

from annoy import AnnoyIndex

index = AnnoyIndex(32, 'euclidean')
for i, v in enumerate(embeddings):
    index.add_item(i, v)
    
index.build(10)
```

###  recsys - otto - matrix factorization - Now for any `aid`, we can find its nearest neighbor - index.get_nns_by_item(123, 10)


```
index.get_nns_by_item(123, 10)
```

Let's create a submission! 🙂

## rd: recsys - otto - matrix factorization - make predictions on every session of test set with co-visitation matrix heuristic when a session has more than 20 aids and use nearest neights to generate candidates when a session has less than 20 aids

### rd: recsys - otto - matrix factorization - group aids/types of each session into a list - test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list) - test_session_types = test.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

### rd: recsys - otto - matrix factorization - where to get defaultdict - from collections import defaultdict

### rd: recsys - otto - matrix factorization - how to read data from csv file - sample_sub = pd.read_csv('../input/otto-recommender-system//sample_submission.csv')

### rd: recsys - otto - matrix factorization - how to give different weights to clicks, carts and orders - type_weight_multipliers = {0: 1, 1: 6, 2: 3}

### rd: recsys - otto - matrix factorization - how to loop each session to access the list of aids and the list of types - for AIDs, types in zip(test_session_AIDs, test_session_types):

### rd: recsys - otto - matrix factorization - create a list of time_weights which log2-increase itself as time goes - weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1 # time weighted aid 

### rd: recsys - otto - matrix factorization - create a defaultdict to store data if no value is given to the key then use 0 as value -         aids_temp=defaultdict(lambda: 0)

###  rd: recsys - otto - matrix factorization - loop every aid, weight and type from the lists of aids, weights and types of a session  - for aid,w,t in zip(AIDs,weights,types):  - then accumulate the weight of an aid based on its type_weight, time_weight and occurrences, and then save aid and its accumulated weight into the defaultdict - aids_temp[aid]+= w * type_weight_multipliers[t]

### rd: recsys - otto - matrix factorization - aid_temp is a special defaultdict which make the weight zero/0 if no weight of the aid is given or initialize every aid's weight value to be 0 - aids_temp=defaultdict(lambda: 0) 


```
# code experiment
from collections import defaultdict
aids_temp=defaultdict(lambda: 0)
aids_temp[1]
aids_temp[1] += 3
aids_temp[1]
```




    0






    3



###  rd: recsys - otto - matrix factorization -  sort a dict by its values and reverse the order (from highest value to the lowest value), and then take the keys into a list -    sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]


```
[k for k, v in sorted({1:3, 2:4, 3:1}.items(), key=lambda item: -item[1])]
```




    [2, 1, 3]



###  rd: recsys - otto - matrix factorization  - take the first 20 aids from the list sorted_aids and append to the list labels -  labels.append(sorted_aids[:20])

### rd: recsys - otto - matrix factorization - if there are less than 20 aids in a session, then select the last aid and use approximate nearest neighbor search and our embeddings to generate candidates, and then combine the existing aids with candidates, and select the first 20 as labels

###  rd: recsys - otto - matrix factorization - reverse the order of a list - AIDs[::-1] - remove the duplicates  - dict.fromkeys(AIDs[::-1]) - and put the rest into a list  - list(...)  - AIDs = list(dict.fromkeys(AIDs[::-1]))

### rd: recsys - otto - matrix factorization - get the most recent aid - most_recent_aid = AIDs[0]

### rd: recsys - otto - matrix factorization - get 21 nearest neights of the most recent aid - nns = index.get_nns_by_item(most_recent_aid, 21)[1:]

### rd: recsys - otto - matrix factorization - combine two lists and take the first 20 aids from it and append to the list labels - labels.append((AIDs+nns)[:20])


```
import pandas as pd
import numpy as np

from collections import defaultdict

sample_sub = pd.read_csv('../input/otto-recommender-system//sample_submission.csv')

session_types = ['clicks', 'carts', 'orders']
test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types = test.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

labels = []

type_weight_multipliers = {0: 1, 1: 6, 2: 3}
for AIDs, types in zip(test_session_AIDs, test_session_types):
    if len(AIDs) >= 20:
        # if we have enough aids (over equals 20) we don't need to look for candidates! we just use the old logic
        weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1 # time weighted aid 
        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(AIDs,weights,types): 
            aids_temp[aid]+= w * type_weight_multipliers[t]
            
        sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
        labels.append(sorted_aids[:20])
    else:
        # here we don't have 20 aids to output -- we will use approximate nearest neighbor search and our embeddings
        # to generate candidates!
        AIDs = list(dict.fromkeys(AIDs[::-1]))
        
        # let's grab the most recent aid
        most_recent_aid = AIDs[0]
        
        # and look for some neighbors!
        nns = index.get_nns_by_item(most_recent_aid, 21)[1:]
                        
        labels.append((AIDs+nns)[:20])
```

Let's now pull it all together and write to a file,

## rd: recsys - otto - matrix factorization - make the format right for submission: predictions/labels of a session needs to be in a long string separated by '_'; the session id needs to attach with its type in a string too.

### rd: recsys - otto - matrix factorization - turn a list of lists of values into a list of a long string with values connected with  '_' - labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

### rd: recsys - otto - matrix factorization - create a pd.DataFrame with a dict of (a Series and a list) - predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

### rd: recsys - otto - matrix factorization - make a copy of a dataframe -  modified_predictions = predictions.copy()

### rd: recsys - otto - matrix factorization - change a column's values' type from int to str and then their string values -     modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'

### rd: recsys - otto - matrix factorization - append a dataframe into a list -     prediction_dfs.append(modified_predictions)

### rd: recsys - otto - matrix factorization - stack a list of dataframe vertically and drop the original index - submission = pd.concat(prediction_dfs).reset_index(drop=True)

### rd: recsys - otto - matrix factorization - write a dataframe into a csv file without the index carried from the dataframe - submission.to_csv('submission.csv', index=False)


```
labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})

prediction_dfs = []

for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)

submission = pd.concat(prediction_dfs).reset_index(drop=True)
submission.to_csv('submission.csv', index=False)
```

## rd: recsys - otto - matrix factorization - what to do next on this notebook

And we are done!


**If you like this notebook, please smash the upvote button! Thank you! 😊**

There are many ways in which this can be expanded:
* we can train on the GPU
* we can train for longer
* maybe we would get better results if we were to filter our train data by type?
* should we train only on adjacent aids? maybe we should expand the neighborhood we train on

We can keep asking ourselves many questions like this 🙂 Now we have a framework to start answering them!

Thank you for reading! Happy Kaggling! 🙌

## Journey

### jn: well, I kind of finished all the notebooks shared by Radek on otto at the moment, my goal is to be able to experiment further on these notebooks freely on my own. The next step is to familiarize those notebooks, read more on the models even come back to the videos by Xavier /2022-11-23

### jn: speaking of things I could do to benefit myself and others and contribute to Radek's work, I could do the random split on the training set for his local validation dataset: 1) how to do random split like organizer's script, 2) apply it to the training set of the local validation set /2022-11-23


```

```
