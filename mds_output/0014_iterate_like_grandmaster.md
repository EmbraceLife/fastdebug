# 0014_iterate_like_grandmaster
---
skip_exec: true
---
## Iterate like a grandmaster

**Note**: If you're fairly new to Kaggle, NLP, or Transformers, I strongly recommend you read my [Getting Started](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) notebook first, and then come back to this one.

---

### what most of kaggle notebooks are like

There's a lot of impressive notebooks around on Kaggle, but they often  fall into one of two categories:

- Exploratory Data Analysis (EDA) notebooks with lots of pretty charts, but not much focus on understanding the key issues that will make a difference in the competition
- Training/inference notebooks with little detail about *why* each step was chosen.

### What does it take to become a grandmaster of Kaggle competition

In this notebook I'll try to give a taste of how a competitions grandmaster might tackle the [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/) competition. The focus generally should be two things:

1. Creating an effective validation set
1. Iterating rapidly to find changes which improve results on the validation set.

If you can do these two things, then you can try out lots of experiments and find what works, and what doesn't. Without these two things, it will be nearly impossible to do well in a Kaggle competition (and, indeed, to create highly accurate models in real life!)

### the goals of this notebook: creating appropriate validation set and keep code concise and simple

I will show a couple of different ways to create an appropriate validation set, and will explain how to expand them into an appropriate cross-validation system. I'll use just plain HuggingFace Transformers for everything, and will keep the code concise and simple. The more code you have, the more you have to maintain, and the more chances there are to make mistakes. So keep it simple!

OK, let's get started...

### how to download kaggle dataset to work locally

It's nice to be able to run things locally too, to save your Kaggle GPU hours, so set a variable to make it easy to see where we are, and download what we need:


```
from pathlib import Path
import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    !pip install -Uqq fastai
else:
    import zipfile,kaggle
    path = Path('us-patent-phrase-to-phrase-matching')
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)
```

    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow-io 0.21.0 requires tensorflow-io-gcs-filesystem==0.21.0, which is not installed.
    explainable-ai-sdk 1.3.2 requires xai-image-widget, which is not installed.
    tensorflow 2.6.2 requires numpy~=1.19.2, but you have numpy 1.20.3 which is incompatible.
    tensorflow 2.6.2 requires six~=1.15.0, but you have six 1.16.0 which is incompatible.
    tensorflow 2.6.2 requires typing-extensions~=3.7.4, but you have typing-extensions 3.10.0.2 which is incompatible.
    tensorflow 2.6.2 requires wrapt~=1.12.1, but you have wrapt 1.13.3 which is incompatible.
    tensorflow-transform 1.5.0 requires absl-py<0.13,>=0.9, but you have absl-py 0.15.0 which is incompatible.
    tensorflow-transform 1.5.0 requires numpy<1.20,>=1.16, but you have numpy 1.20.3 which is incompatible.
    tensorflow-transform 1.5.0 requires pyarrow<6,>=1, but you have pyarrow 6.0.1 which is incompatible.
    tensorflow-transform 1.5.0 requires tensorflow!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<2.8,>=1.15.2, but you have tensorflow 2.6.2 which is incompatible.
    tensorflow-serving-api 2.7.0 requires tensorflow<3,>=2.7.0, but you have tensorflow 2.6.2 which is incompatible.
    flake8 4.0.1 requires importlib-metadata<4.3; python_version < "3.8", but you have importlib-metadata 4.11.3 which is incompatible.
    apache-beam 2.34.0 requires dill<0.3.2,>=0.3.1.1, but you have dill 0.3.4 which is incompatible.
    apache-beam 2.34.0 requires httplib2<0.20.0,>=0.8, but you have httplib2 0.20.2 which is incompatible.
    apache-beam 2.34.0 requires pyarrow<6.0.0,>=0.15.1, but you have pyarrow 6.0.1 which is incompatible.
    aioitertools 0.10.0 requires typing_extensions>=4.0; python_version < "3.10", but you have typing-extensions 3.10.0.2 which is incompatible.
    aiobotocore 2.1.2 requires botocore<1.23.25,>=1.23.24, but you have botocore 1.24.20 which is incompatible.[0m


### how fastai gives us a lot of basic imports like np, pd, plt etc; what does fastai.imports provide

A lot of the basic imports you'll want (`np`, `pd`, `plt`, etc) are provided by fastai, so let's grab them in one line:


```
from fastai.imports import *
```

## Import and EDA

### how to check what inside the dataset folder

Set a path to our data. Use `pathlib.Path` because it makes everything so much easier, and make it work automatically regardless if you're working on your own PC or on Kaggle!


```
if iskaggle: path = Path('../input/us-patent-phrase-to-phrase-matching')
path.ls()
```




    (#3) [Path('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv'),Path('../input/us-patent-phrase-to-phrase-matching/train.csv'),Path('../input/us-patent-phrase-to-phrase-matching/test.csv')]



### how to check what inside the train.csv file in the dataset folder

Let's look at the training set:


```
df = pd.read_csv(path/'train.csv')
df
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
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37d61fd2272659b1</td>
      <td>abatement</td>
      <td>abatement of pollution</td>
      <td>A47</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7b9652b17b68b7a4</td>
      <td>abatement</td>
      <td>act of abating</td>
      <td>A47</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36d72442aefd8232</td>
      <td>abatement</td>
      <td>active catalyst</td>
      <td>A47</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5296b0c19e1ce60e</td>
      <td>abatement</td>
      <td>eliminating process</td>
      <td>A47</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54c1e3b9184cb5b6</td>
      <td>abatement</td>
      <td>forest region</td>
      <td>A47</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36468</th>
      <td>8e1386cbefd7f245</td>
      <td>wood article</td>
      <td>wooden article</td>
      <td>B44</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>36469</th>
      <td>42d9e032d1cd3242</td>
      <td>wood article</td>
      <td>wooden box</td>
      <td>B44</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>36470</th>
      <td>208654ccb9e14fa3</td>
      <td>wood article</td>
      <td>wooden handle</td>
      <td>B44</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>36471</th>
      <td>756ec035e694722b</td>
      <td>wood article</td>
      <td>wooden material</td>
      <td>B44</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>36472</th>
      <td>8d135da0b55b8c88</td>
      <td>wood article</td>
      <td>wooden substrate</td>
      <td>B44</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
<p>36473 rows × 5 columns</p>
</div>



### how to check what inside the test.csv file in the dataset folder

...and the test set:


```
eval_df = pd.read_csv(path/'test.csv')
len(eval_df)
```




    36




```
eval_df.head()
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
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4112d61851461f60</td>
      <td>opc drum</td>
      <td>inorganic photoconductor drum</td>
      <td>G02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09e418c93a776564</td>
      <td>adjust gas flow</td>
      <td>altering gas flow</td>
      <td>F23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36baf228038e314b</td>
      <td>lower trunnion</td>
      <td>lower locating</td>
      <td>B60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1f37ead645e7f0c8</td>
      <td>cap component</td>
      <td>upper portion</td>
      <td>D06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>71a5b6ad068d531f</td>
      <td>neural stimulation</td>
      <td>artificial neural network</td>
      <td>H04</td>
    </tr>
  </tbody>
</table>
</div>



### how to check the distribution of the column `target` of the training set dataframe

Let's look at the distribution of values of `target`:


```
df.target.value_counts()
```




    composition                    24
    data                           22
    metal                          22
    motor                          22
    assembly                       21
                                   ..
    switching switch over valve     1
    switching switch off valve      1
    switching over valve            1
    switching off valve             1
    wooden substrate                1
    Name: target, Length: 29340, dtype: int64



### what info do we get from reading the distribution of different values of the target column of the training set

We see that there's nearly as many unique targets as items in the training set, so they're nearly but not quite unique. Most importantly, we can see that these generally contain very few words (1-4 words in the above sample).

Let's check `anchor`:


```
df.anchor.value_counts()
```




    component composite coating              152
    sheet supply roller                      150
    source voltage                           140
    perfluoroalkyl group                     136
    el display                               135
                                            ... 
    plug nozzle                                2
    shannon                                    2
    dry coating composition1                   2
    peripheral nervous system stimulation      1
    conduct conducting material                1
    Name: anchor, Length: 733, dtype: int64



### how to check the distribution of different values of the context column of training set

We can see here that there's far fewer unique values (just 733) and that again they're very short (2-4 words in this sample).

Now we'll do `context`


```
df.context.value_counts()
```




    H01    2186
    H04    2177
    G01    1812
    A61    1477
    F16    1091
           ... 
    B03      47
    F17      33
    B31      24
    A62      23
    F26      18
    Name: context, Length: 106, dtype: int64



### how to get the distribution of the different section names embedded inside the context column, and create a column named section based on the data

These are just short codes. Some of them have very few examples (18 in the smallest case) The first character is the section the patent was filed under -- let's create a column for that and look at the distribution:


```
df['section'] = df.context.str[0]
df.section.value_counts()
```




    B    8019
    H    6195
    G    6013
    C    5288
    A    4094
    F    4054
    E    1531
    D    1279
    Name: section, dtype: int64



### how to view the distribution of continuous data or column like the column score of the training set

It seems likely that these sections might be useful, since they've got quite a bit more data in each.

Finally, we'll take a look at a histogram of the scores:


```
df.score.hist();
```


    
![png](0014_iterate_like_grandmaster_files/0014_iterate_like_grandmaster_41_0.png)
    


There's a small number that are scored `1.0` - here's a sample:


```
df[df.score==1]
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
      <th>id</th>
      <th>anchor</th>
      <th>target</th>
      <th>context</th>
      <th>score</th>
      <th>section</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>473137168ebf7484</td>
      <td>abatement</td>
      <td>abating</td>
      <td>F24</td>
      <td>1.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>158</th>
      <td>621b048d70aa8867</td>
      <td>absorbent properties</td>
      <td>absorbent characteristics</td>
      <td>D01</td>
      <td>1.0</td>
      <td>D</td>
    </tr>
    <tr>
      <th>161</th>
      <td>bc20a1c961cb073a</td>
      <td>absorbent properties</td>
      <td>absorption properties</td>
      <td>D01</td>
      <td>1.0</td>
      <td>D</td>
    </tr>
    <tr>
      <th>311</th>
      <td>e955700dffd68624</td>
      <td>acid absorption</td>
      <td>absorption of acid</td>
      <td>B08</td>
      <td>1.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>315</th>
      <td>3a09aba546aac675</td>
      <td>acid absorption</td>
      <td>acid absorption</td>
      <td>B08</td>
      <td>1.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36398</th>
      <td>913141526432f1d6</td>
      <td>wiring trough</td>
      <td>wiring troughs</td>
      <td>F16</td>
      <td>1.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>36435</th>
      <td>ee0746f2a8ecef97</td>
      <td>wood article</td>
      <td>wood articles</td>
      <td>B05</td>
      <td>1.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>36440</th>
      <td>ecaf479135cf0dfd</td>
      <td>wood article</td>
      <td>wooden article</td>
      <td>B05</td>
      <td>1.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>36464</th>
      <td>8ceaa2b5c2d56250</td>
      <td>wood article</td>
      <td>wood article</td>
      <td>B44</td>
      <td>1.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>36468</th>
      <td>8e1386cbefd7f245</td>
      <td>wood article</td>
      <td>wooden article</td>
      <td>B44</td>
      <td>1.0</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
<p>1154 rows × 6 columns</p>
</div>



We can see from this that these are just minor rewordings of the same concept, and isn't likely to be specific to `context`. Any pretrained model should be pretty good at finding these already.

## Training

### libraries needed for creating and training models here

Time to import the stuff we'll need for training:


```
from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer
```


```
if iskaggle:
    !pip install -q datasets
import datasets
from datasets import load_dataset, Dataset, DatasetDict
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m


### how to quite down warning messages

HuggingFace Transformers tends to be rather enthusiastic about spitting out lots of warnings, so let's quieten it down for our sanity:


```
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)
```

I tried to find a model that I could train reasonably at home in under two minutes, but got reasonable accuracy from. I found that deberta-v3-small fits the bill, so let's use it:


```
model_nm = 'microsoft/deberta-v3-small'
```

We can now create a tokenizer for this model. Note that pretrained models assume that text is tokenized in a particular way. In order to ensure that your tokenizer matches your model, use the `AutoTokenizer`, passing in your model name.


```
tokz = AutoTokenizer.from_pretrained(model_nm)
```


    Downloading:   0%|          | 0.00/52.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/578 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/2.35M [00:00<?, ?B/s]


We'll need to combine the context, anchor, and target together somehow. There's not much research as to the best way to do this, so we may need to iterate a bit. To start with, we'll just combine them all into a single string. The model will need to know where each section starts, so we can use the special separator token to tell it:


```
sep = tokz.sep_token
sep
```




    '[SEP]'



Let's now created our combined column:


```
df['inputs'] = df.context + sep + df.anchor + sep + df.target
```

Generally we'll get best performance if we convert pandas DataFrames into HuggingFace Datasets, so we'll convert them over, and also rename the score column to what Transformers expects for the dependent variable, which is `label`:


```
ds = Dataset.from_pandas(df).rename_column('score', 'label')
eval_ds = Dataset.from_pandas(eval_df)
```

To tokenize the data, we'll create a function (since that's what `Dataset.map` will need):


```
def tok_func(x): return tokz(x["inputs"])
```

Let's try tokenizing one input and see how it looks


```
tok_func(ds[0])
```




    {'input_ids': [1, 336, 5753, 2, 47284, 2, 47284, 265, 6435, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}



The only bit we care about at the moment is `input_ids`. We can see in the tokens that it starts with a special token `1` (which represents the start of text), and then has our three fields separated by the separator token `2`. We can check the indices of the special token IDs like so:


```
tokz.all_special_tokens
```




    ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']



We can now tokenize the input. We'll use batching to speed it up, and remove the columns we no longer need:


```
inps = "anchor","target","context"
tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))
```


      0%|          | 0/37 [00:00<?, ?ba/s]


Looking at the first item of the dataset we should see the same information as when we checked `tok_func` above:


```
tok_ds[0]
```




    {'label': 0.5,
     'input_ids': [1, 336, 5753, 2, 47284, 2, 47284, 265, 6435, 2],
     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}



## Creating a validation set

According to [this post](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/315220), the private test anchors do not overlap with the training set. So let's do the same thing for our validation set.

First, create a randomly shuffled list of anchors:


```
anchors = df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)
anchors[:5]
```




    array(['time digital signal', 'antiatherosclerotic', 'filled interior',
           'dispersed powder', 'locking formation'], dtype=object)



Now we can pick some proportion (e.g 25%) of these anchors to go in the validation set:


```
val_prop = 0.25
val_sz = int(len(anchors)*val_prop)
val_anchors = anchors[:val_sz]
```

Now we can get a list of which rows match `val_anchors`, and get their indices:


```
is_val = np.isin(df.anchor, val_anchors)
idxs = np.arange(len(df))
val_idxs = idxs[ is_val]
trn_idxs = idxs[~is_val]
len(val_idxs),len(trn_idxs)
```




    (9116, 27357)



Our training and validation `Dataset`s can now be selected, and put into a `DatasetDict` ready for training:


```
dds = DatasetDict({"train":tok_ds.select(trn_idxs),
             "test": tok_ds.select(val_idxs)})
```

BTW, a lot of people do more complex stuff for creating their validation set, but with a dataset this large there's not much point. As you can see, the mean scores in the two groups are very similar despite just doing a random shuffle:


```
df.iloc[trn_idxs].score.mean(),df.iloc[val_idxs].score.mean()
```




    (0.3623021530138539, 0.3613426941641071)



## Initial model

Let's now train our model! We'll need to specify a metric, which is the correlation coefficient provided by numpy (we need to return a dictionary since that's how Transformers knows what label to use):


```
def corr(eval_pred): return {'pearson': np.corrcoef(*eval_pred)[0][1]}
```

We pick a learning rate and batch size that fits our GPU, and pick a reasonable weight decay and small number of epochs:


```
lr,bs = 8e-5,128
wd,epochs = 0.01,4
```

Three epochs might not sound like much, but you'll see once we train that most of the progress can be made in that time, so this is good for experimentation.

Transformers uses the `TrainingArguments` class to set up arguments. We'll use a cosine scheduler with warmup, since at fast.ai we've found that's pretty reliable. We'll use fp16 since it's much faster on modern GPUs, and saves some memory. We evaluate using double-sized batches, since no gradients are stored so we can do twice as many rows at a time.


```
def get_trainer(dds):
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=corr)
```


```
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=wd, report_to='none')
```

We can now create our model, and `Trainer`, which is a class which combines the data and model together (just like `Learner` in fastai):


```
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
               tokenizer=tokz, compute_metrics=corr)
```


    Downloading:   0%|          | 0.00/273M [00:00<?, ?B/s]


Let's train our model!


```
trainer.train();
```



    <div>

      <progress value='856' max='856' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [856/856 04:19, Epoch 4/4]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Pearson</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.025167</td>
      <td>0.798359</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.025149</td>
      <td>0.803286</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.035300</td>
      <td>0.024344</td>
      <td>0.815202</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.035300</td>
      <td>0.024549</td>
      <td>0.815378</td>
    </tr>
  </tbody>
</table><p>


## Improving the model

We now want to start iterating to improve this. To do that, we need to know whether the model gives stable results. I tried training it 3 times from scratch, and got a range of outcomes from 0.808-0.810. This is stable enough to make a start - if we're not finding improvements that are visible within this range, then they're not very significant! Later on, if and when we feel confident that we've got the basics right, we can use cross validation and more epochs of training.

Iteration speed is critical, so we need to quickly be able to try different data processing and trainer parameters. So let's create a function to quickly apply tokenization and create our `DatasetDict`:


```
def get_dds(df):
    ds = Dataset.from_pandas(df).rename_column('score', 'label')
    tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))
    return DatasetDict({"train":tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)})
```

...and also a function to create a `Trainer`:


```
def get_model(): return AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

def get_trainer(dds, model=None):
    if model is None: model = get_model()
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=corr)
```

Let's now try out some ideas...

Perhaps using the special separator character isn't a good idea, and we should use something we create instead. Let's see if that makes things better. First we'll change the separator and create the `DatasetDict`:


```
sep = " [s] "
df['inputs'] = df.context + sep + df.anchor + sep + df.target
dds = get_dds(df)
```


      0%|          | 0/37 [00:00<?, ?ba/s]


...and create and train a model. 


```
get_trainer(dds).train()
```



    <div>

      <progress value='856' max='856' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [856/856 04:29, Epoch 4/4]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Pearson</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.024909</td>
      <td>0.797869</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.024800</td>
      <td>0.812953</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.031600</td>
      <td>0.024418</td>
      <td>0.818292</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.031600</td>
      <td>0.024037</td>
      <td>0.819089</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=856, training_loss=0.023822079752093165, metrics={'train_runtime': 269.6383, 'train_samples_per_second': 405.833, 'train_steps_per_second': 3.175, 'total_flos': 582160588599300.0, 'train_loss': 0.023822079752093165, 'epoch': 4.0})



That's looking quite a bit better, so we'll keep that change.

Often changing to lowercase is helpful. Let's see if that helps too:


```
df['inputs'] = df.inputs.str.lower()
dds = get_dds(df)
get_trainer(dds).train()
```


      0%|          | 0/37 [00:00<?, ?ba/s]




    <div>

      <progress value='856' max='856' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [856/856 04:34, Epoch 4/4]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Pearson</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.025170</td>
      <td>0.798002</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.024433</td>
      <td>0.815301</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.031500</td>
      <td>0.024575</td>
      <td>0.818443</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.031500</td>
      <td>0.024150</td>
      <td>0.818868</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=856, training_loss=0.023811190484840178, metrics={'train_runtime': 274.7918, 'train_samples_per_second': 398.222, 'train_steps_per_second': 3.115, 'total_flos': 582160588599300.0, 'train_loss': 0.023811190484840178, 'epoch': 4.0})



That one is less clear. We'll keep that change too since most times I run it, it's a little better.

## Special tokens

What if we made the patent section a special token? Then potentially the model might learn to recognize that different sections need to be handled in different ways. To do that, we'll use, e.g. `[A]` for section A. We'll then add those as special tokens:


```
df['sectok'] = '[' + df.section + ']'
sectoks = list(df.sectok.unique())
tokz.add_special_tokens({'additional_special_tokens': sectoks})
```




    8



We concatenate the section token to the start of our inputs:


```
df['inputs'] = df.sectok + sep + df.context + sep + df.anchor.str.lower() + sep + df.target
dds = get_dds(df)
```


      0%|          | 0/37 [00:00<?, ?ba/s]


Since we've added more tokens, we need to resize the embedding matrix in the model:


```
model = get_model()
model.resize_token_embeddings(len(tokz))
```




    Embedding(128009, 768)



Now we're ready to train:


```
trainer = get_trainer(dds, model=model)
trainer.train()
```



    <div>

      <progress value='856' max='856' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [856/856 04:55, Epoch 4/4]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Pearson</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>0.024835</td>
      <td>0.797996</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>0.024412</td>
      <td>0.812386</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.031800</td>
      <td>0.024019</td>
      <td>0.820914</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.031800</td>
      <td>0.024220</td>
      <td>0.820187</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=856, training_loss=0.023861238889605084, metrics={'train_runtime': 296.1519, 'train_samples_per_second': 369.5, 'train_steps_per_second': 2.89, 'total_flos': 695409809982180.0, 'train_loss': 0.023861238889605084, 'epoch': 4.0})



It looks like we've made another bit of an improvement!

There's plenty more things you could try. Here's some thoughts:

- Try a model pretrained on legal vocabulary. E.g. how about [BERT for patents](https://huggingface.co/anferico/bert-for-patents)?
- You'd likely get better results by using a sentence similarity model. Did you know that there's a [patent similarity model](https://huggingface.co/AI-Growth-Lab/PatentSBERTa) you could try?
- You could also fine-tune any HuggingFace model using the full patent database (which is provided in BigQuery), before applying it to this dataset
- Replace the patent context field with the description of that context provided by the patent office
- ...and try out your own ideas too!

Before submitting a model, retrain it on the full dataset, rather than just the 75% training subset we've used here. Create a function like the ones above to make that easy for you!"

## Cross-validation


```
n_folds = 4
```

Once you've gotten the low hanging fruit, you might want to use cross-validation to see the impact of minor changes. This time we'll use [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html#sklearn.model_selection.StratifiedGroupKFold), partly just to show a different approach to before, and partly because it will give us slightly better balanced datasets.


```
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=n_folds)
```

Here's how to split the data frame into `n_folds` groups, with non-overlapping anchors and matched scores, after randomly shuffling the rows:


```
df = df.sample(frac=1, random_state=42)
scores = (df.score*100).astype(int)
folds = list(cv.split(idxs, scores, df.anchor))
folds
```




    [(array([    0,     1,     2, ..., 36469, 36471, 36472]),
      array([    8,    13,    14, ..., 36453, 36464, 36470])),
     (array([    0,     1,     5, ..., 36470, 36471, 36472]),
      array([    2,     3,     4, ..., 36459, 36461, 36462])),
     (array([    1,     2,     3, ..., 36467, 36470, 36472]),
      array([    0,     7,    11, ..., 36468, 36469, 36471])),
     (array([    0,     2,     3, ..., 36469, 36470, 36471]),
      array([    1,     5,     9, ..., 36465, 36467, 36472]))]



We can now create a little function to split into training and validation sets based on a fold:


```
def get_fold(folds, fold_num):
    trn,val = folds[fold_num]
    return DatasetDict({"train":tok_ds.select(trn), "test": tok_ds.select(val)})
```

Let's try it out:


```
dds = get_fold(folds, 0)
dds
```




    DatasetDict({
        train: Dataset({
            features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 27346
        })
        test: Dataset({
            features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
            num_rows: 9127
        })
    })



We can now pass this into `get_trainer` as we did before. If we have, say, 4 folds, then doing that for each fold will give us 4 models, and 4 sets of predictions and metrics. You could ensemble the 4 models to get a stronger model, and can also average the 4 metrics to get a more accurate assessment of your model. Here's how to get the final epoch metrics from a trainer:


```
metrics = [o['eval_pearson'] for o in trainer.state.log_history if 'eval_pearson' in o]
metrics[-1]
```




    0.8201874392079798



I hope you've found this a helpful guide to improving your results in this competition - and on Kaggle more generally! If you like it, please remember to give it an upvote, and don't hesitate to add a comment if you have any questions or thoughts to add. And if the ideas here are helpful to you in creating your models, I'd really appreciate a link back to this notebook or a comment below to let me know what helped.


```

```
