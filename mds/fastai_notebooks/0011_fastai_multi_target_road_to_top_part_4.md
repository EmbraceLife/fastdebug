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

# 0011_fastai_multi_target_road_to_top_part_4

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

```python
# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *
```

### what is multi-target for; why would multi-target be useful


This is part 4 of the [Road to the Top](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1) series, in which I show the process I used to tackle the [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) competition. The first three parts show the process I took that lead to four 1st place submissions. This part onwards discusses various extensions and advanced tricks you might want to consider -- but which I don't promise will necessarily help your score (you'll need to do your own experiments to figure this out!) If you haven't read the rest of this series yet, I'd suggest starting at [part 1](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1).

In this notebook we're going to build a model that doesn't just predict what disease the rice paddy has, but also predicts what kind of rice is shown.

This might sound like a bad idea. After all, doesn't that mean that the model has *more* to do? Mightn't it get rather distracted from its main task, which is to identify paddy disease?

Perhaps... But in previous projects I've often found the opposite to be true, especially when training for quite a few epochs. By giving the model more signal about what is present in a picture, it may be able to use this information to find more interesting features that predict our target of interest. For instance, perhaps some of the features of disease change between varieties.


## Multi-output `DataLoader`


### how to get dataset and libraries ready for Kaggle's Paddy Doctor competition


First we'll repeat the steps we used last time to access the data and ensure all the latest libraries are installed:

```python
comp = 'paddy-disease-classification'
path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
from fastai.vision.all import *
set_seed(42)

from fastcore.parallel import *
trn_path = path/'train_images'
```

### how to read a csv into a pandas dataframe and select a column to be its index column


Here's the CSV that Kaggle provides, showing the variety of rice contained in each image -- we'll make `image_id` the index of our data frame so that we can look up images directly to grab their variety:

```python
df = pd.read_csv(path/'train.csv', index_col='image_id')
df.head()
```

### how to extract a particular info from a row using an index from the index column and the column of interest with `df.loc`


Pandas uses the `loc` attribute to look up rows by index. Here's how we can get the variety of image `100330.jpg`, for instance:

```python
df.loc['100330.jpg', 'variety']
```

### how to get the rice variety of each image with `df.loc` and the path from `get_image_files`


Our DataBlock will be using `get_image_files` to get the list of training images, which returns `Path` objects. Therefore, to look up an item to get its variety, we'll need to pass its `name`. Here's a function which does just that:

```python
def get_variety(p): return df.loc[p.name, 'variety']
```

### how to use `DataBlock` to create dataloaders; what is DataBlock


We're now ready to create our `DataLoaders`. To do this, we'll use the `DataBlock` API, which is a flexible and convenient way to plug pieces of a data processing pipeline together:

```python
dls = DataBlock(
    blocks=(ImageBlock,CategoryBlock,CategoryBlock),
    n_inp=1,
    get_items=get_image_files,
    get_y = [parent_label,get_variety],
    splitter=RandomSplitter(0.2, seed=42),
    item_tfms=Resize(192, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75)
).dataloaders(trn_path)
```

<!-- #region -->
Here's an explanation of each line:

```python
blocks=(ImageBlock,CategoryBlock,CategoryBlock),
```

The `DataBlock` will create 3 things from each file: an image (the contents of the file), and 2 categorical variables (the disease and the variety).

```python
n_inp=1,
```

There is `1` input (the image) -- and therefore the other two variables (the two categories) are outputs.

```python
get_items=get_image_files,
```

Use `get_image_files` to get a list of inputs.

```python
get_y = [parent_label,get_variety],
```

To create the two outputs for each file, call two functions: `parent_label` (from fastai) and `get_variety` (defined above).

```python
splitter=RandomSplitter(0.2, seed=42),
```

Randomly split the input into 80% train and 20% validation sets.

```python
item_tfms=Resize(192, method='squish'),
batch_tfms=aug_transforms(size=128, min_scale=0.75)
```

These are the same item and batch transforms we've used in previous notebooks.
<!-- #endregion -->

### how to display 6 images from a batch


Let's take a look at part of a batch of this data:

```python
dls.show_batch(max_n=6)
```

We can see that fastai has created both the image input and two categorical outputs that we requested!


## Replicating the disease model


### how to create disease error-rate and disease entropy loss functions to receive 3 instead of 2 inputs


Now we'll replicate the same disease model we've made before, but have it work with this new data.

The key difference is that our metrics and loss will now receive three things instead of two: the model outputs (i.e. the metric and loss function inputs), and the two targets (disease and variety). Therefore, we need to define slight variations of our metric (`error_rate`) and loss function (`cross_entropy`) to pass on just the `disease` target:

```python
def disease_err(inp,disease,variety): return error_rate(inp,disease)
def disease_loss(inp,disease,variety): return F.cross_entropy(inp,disease)
```

### how to create a vision learner using `vision_learner` which specifies `loss_func`, `metrics`, `n_out` (num of outputs)


We're now ready to create our learner.

There's just one wrinkle to be aware of. Now that our `DataLoaders` is returning multiple targets, fastai doesn't know how many outputs our model will need. Therefore we have to pass `n_out` when we create our `Learner` -- we need `10` outputs, one for each possible disease:

```python
arch = 'convnext_small_in22k'
learn = vision_learner(dls, arch, loss_func=disease_loss, metrics=disease_err, n_out=10).to_fp16()
lr = 0.01
```

When we train this model we should get similar results to what we've seen with similar models before:

```python
learn.fine_tune(5, lr)
```

## Multi-target model


### how to make the vision_learner to return 20 outputs, 10 for diseases, 10 for varieties; how to use to_fp16


In order to predict both the probability of each disease, and of each variety, we'll now need the model to output a tensor of length 20, since there are 10 possible diseases, and 10 possible varieties. We can do this by setting `n_out=20`:

```python
learn = vision_learner(dls, arch, n_out=20).to_fp16()
```

### how to split the inputs for creating entropy loss for disease and variety


We can define `disease_loss` just like we did previously, but with one important change: the input tensor is now length 20, not 10, so it doesn't match the number of possible diseases. We can pick whatever part of the input we want to be used to predict disease. Let's use the first 10 values:

```python
def disease_loss(inp,disease,variety): return F.cross_entropy(inp[:,:10],disease)
```

That means we can do the same thing for predicting variety, but use the last 10 values of the input, and set the target to `variety` instead of `disease`:

```python
def variety_loss(inp,disease,variety): return F.cross_entropy(inp[:,10:],variety)
```

### how to combine both disease loss and variety loss together to be the model loss


Our overall loss will then be the sum of these two losses:

```python
def combine_loss(inp,disease,variety): return disease_loss(inp,disease,variety)+variety_loss(inp,disease,variety)
```

### how to split the input data to calc both disease error rate and variety error rate


It would be useful to view the error rate for each of the outputs too, so let's do the same thing for out metrics:

```python
def disease_err(inp,disease,variety): return error_rate(inp[:,:10],disease)
def variety_err(inp,disease,variety): return error_rate(inp[:,10:],variety)

err_metrics = (disease_err,variety_err)
```

### how to wrap all error rates and losses into a tuple `all_metrics` and give it to `metrics` of `vision_learner`


It's useful to see the loss for each of the outputs too, so we'll add those as metrics:

```python
all_metrics = err_metrics+(disease_loss,variety_loss)
```

We're now ready to create and train our `Learner`:

```python
learn = vision_learner(dls, arch, loss_func=combine_loss, metrics=all_metrics, n_out=20).to_fp16()
```

```python
learn.fine_tune(5, lr)
```

## Conclusion


### how can multi-target model be useful to us


So, is this useful?

Well... if you actually want a model that predicts multiple things, then yes, definitely! But as to whether it's going to help us better predict rice disease, I honestly don't know. I haven't come across any research that tackles this important question: when can a multi-target model improve the accuracy of the individual targets compared to a single target model? (That doesn't mean it doesn't exist of course -- perhaps it does and I haven't found it yet...)

I've certainly found in previous projects that there are cases where improvements to single targets can be made by using a multi-target model. I'd guess that it'll be most useful when you're having problems with overfitting. So try doing this with more epochs, and let me know how you go!

If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. And if you have any questions or comments, please pop them below -- I read every comment I receive.

```python
# This is what I use to push my notebook from my home PC to Kaggle
if not iskaggle:
    push_notebook('jhoward', 'multi-target-road-to-the-top-part-4',
                  title='Multi-target: Road to the Top, Part 4',
                  file='11-multitask.ipynb',
                  competition=comp, private=False, gpu=True)
```
