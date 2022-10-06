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

# 0010_fastai_scaling_up_road_to_top_part_3

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

This is part 3 of the [Road to the Top](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1) series, in which I show the process I used to tackle the [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) competition, leading to four 1st place submissions. The previous notebook is available here: [part 2](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1).


## Memory and gradient accumulation


### how to get the train_val dataset folder/path ready; how to get the test set images files ready


First we'll repeat the steps we used last time to access the data and ensure all the latest libraries are installed, and we'll also grab the files we'll need for the test set:

```python
comp = 'paddy-disease-classification'
path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
from fastai.vision.all import *
set_seed(42)

tst_files = get_image_files(path/'test_images').sorted()
```

### how to quickly train an ensemble of larger models with larger inputs on Kaggle


In this analysis our goal will be to train an ensemble of larger models with larger inputs. The challenge when training such models is generally GPU memory. Kaggle GPUs have 16280MiB of memory available, as at the time of writing. I like to try out my notebooks on my home PC, then upload them -- but I still need them to run OK on Kaggle (especially if it's a code competition, where this is required). My home PC has 24GiB cards, so just because it runs OK at home doesn't mean it'll run OK on Kaggle.
 
It's really helpful to be able to quickly try a few models and image sizes and find out what will run successfully. To make this quick, we can just grab a small subset of the data for running short epochs -- the memory use will still be the same, but it'll be much faster.

One easy way to do this is to simply pick a category with few files in it. Here's our options:


### how to find out the num of files in each disease class using `pandas.value_counts`

```python
df = pd.read_csv(path/'train.csv')
df.label.value_counts()
```

### how to choose a data folder which has the least num of image files for training


Let's use *bacterial_panicle_blight* since it's the smallest:

```python
trn_path = path/'train_images'/'bacterial_panicle_blight'
```

### how `fine_tune` differ from `fit_one_cycle`


Now we'll set up a `train` function which is very similar to the steps we used for training in the last notebook. But there's a few significant differences...

The first is that I'm using a `finetune` argument to pick whether we are going to run the `fine_tune()` method, or the `fit_one_cycle()` method -- the latter is faster since it doesn't do an initial fine-tuning of the head. 


### how to create a `train` function to do either fine_tune + tta or fit_one_cycle for all layers without freezing; how to add `gradient accumulation` to `train`


When we fine tune in this function I also have it calculate and return the TTA predictions on the test set, since later on we'll be ensembling the TTA results of a number of models. Note also that we no longer have `seed=42` in the `ImageDataLoaders` line -- that means we'll have different training and validation sets each time we call this. That's what we'll want for ensembling, since it means that each model will use slightly different data.

The more important change is that I've added an `accum` argument to implement *gradient accumulation*. As you'll see in the code below, this does two things:

1. Divide the batch size by `accum`
1. Add the `GradientAccumulation` callback, passing in `accum`.

```python
def train(arch, size, item=Resize(480, method='squish'), accum=1, finetune=True, epochs=12):
    dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, item_tfms=item,
        batch_tfms=aug_transforms(size=size, min_scale=0.75), bs=64//accum)
    cbs = GradientAccumulation(64) if accum else []
    learn = vision_learner(dls, arch, metrics=error_rate, cbs=cbs).to_fp16()
    if finetune:
        learn.fine_tune(epochs, 0.01)
        return learn.tta(dl=dls.test_dl(tst_files))
    else:
        learn.unfreeze()
        learn.fit_one_cycle(epochs, 0.01)
```

### what does gradient accumulation do?


*Gradient accumulation* refers to a very simple trick: rather than updating the model weights after every batch based on that batch's gradients, instead keep *accumulating* (adding up) the gradients for a few batches, and them update the model weights with those accumulated gradients. In fastai, the parameter you pass to `GradientAccumulation` defines how many batches of gradients are accumulated. Since we're adding up the gradients over `accum` batches, we therefore need to divide the batch size by that same number. 


### What benefits does gradient accumulation bring


The resulting training loop is nearly mathematically identical to using the original batch size, but the amount of memory used is the same as using a batch size `accum` times smaller!


### how does gradient accumulation work under the hood

<!-- #region -->
For instance, here's a basic example of a single epoch of a training loop without gradient accumulation:

```python
for x,y in dl:
    calc_loss(coeffs, x, y).backward()
    coeffs.data.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()
```

Here's the same thing, but with gradient accumulation added (assuming a target effective batch size of 64):

```python
count = 0            # track count of items seen since last weight update
for x,y in dl:
    count += len(x)  # update count based on this minibatch size
    calc_loss(coeffs, x, y).backward()
    if count>64:     # count is greater than accumulation target, so do weight update
        coeffs.data.sub_(coeffs.grad * lr)
        coeffs.grad.zero_()
        count=0      # reset count
```

The full implementation in fastai is only a few lines of code -- here's the [source code](https://github.com/fastai/fastai/blob/master/fastai/callback/training.py#L26).

To see the impact of gradient accumulation, consider this small model:
<!-- #endregion -->

```python
train('convnext_small_in22k', 128, epochs=1, accum=1, finetune=False)
```

### how to find out how much gpu memory is used; and how to free up the gpu memory


Let's create a function to find out how much memory it used, and also to then clear out the memory for the next run:

```python
import gc
def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()
```

```python
report_gpu()
```

So with `accum=1` the GPU used around 5GB RAM. Let's try `accum=2`:

```python
train('convnext_small_in22k', 128, epochs=1, accum=2, finetune=False)
report_gpu()
```

As you see, the RAM usage has now gone down to 4GB. It's not halved since there's other overhead involved (for larger models this overhead is likely to be relatively lower).

Let's try `4`:

```python
train('convnext_small_in22k', 128, epochs=1, accum=4, finetune=False)
report_gpu()
```

The memory use is even lower!


## Checking memory use


### how to check the gpu memory usage of large models with large image inputs


We'll now check the memory use for each of the architectures and sizes we'll be training later, to ensure they all fit in 16GB RAM. For each of these, I tried `accum=1` first, and then doubled it any time the resulting memory use was over 16GB. As it turns out, `accum=2` was what I needed for every case.

First, `convnext_large`:

```python
train('convnext_large_in22k', 224, epochs=1, accum=2, finetune=False)
report_gpu()
```

```python
train('convnext_large_in22k', (320,240), epochs=1, accum=2, finetune=False)
report_gpu()
```

Here's `vit_large`. This one is very close to going over the 16280MiB we've got on Kaggle!

```python
train('vit_large_patch16_224', 224, epochs=1, accum=2, finetune=False)
report_gpu()
```

Then finally our `swinv2` and `swin` models:

```python
train('swinv2_large_window12_192_22k', 192, epochs=1, accum=2, finetune=False)
report_gpu()
```

```python
train('swin_large_patch4_window7_224', 224, epochs=1, accum=2, finetune=False)
report_gpu()
```

## Running the models


### how to use a dictionary to organize all models and their item and batch transformation setups


Using the previous notebook, I tried a bunch of different architectures and preprocessing approaches on small models, and picked a few which looked good. We'll using a `dict` to list our the preprocessing approaches we'll use for each architecture of interest based on that analysis:

```python
res = 640,480
```

```python
models = {
    'convnext_large_in22k': {
        (Resize(res), 224),
        (Resize(res), (320,224)),
    }, 'vit_large_patch16_224': {
        (Resize(480, method='squish'), 224),
        (Resize(res), 224),
    }, 'swinv2_large_window12_192_22k': {
        (Resize(480, method='squish'), 192),
        (Resize(res), 192),
    }, 'swin_large_patch4_window7_224': {
        (Resize(480, method='squish'), 224),
        (Resize(res), 224),
    }
}
```

### how to train all the selected models with transformation setups and save the tta results into a list


We'll need to switch to using the full training set of course!

```python
trn_path = path/'train_images'
```

Now we're ready to train all these models. Remember that each is using a different training and validation set, so the results aren't directly comparable.

We'll append each set of TTA predictions on the test set into a list called `tta_res`.

```python
tta_res = []

for arch,details in models.items():
    for item,size in details:
        print('---',arch)
        print(size)
        print(item.name)
        tta_res.append(train(arch, size, item=item, accum=2)) #, epochs=1))
        gc.collect()
        torch.cuda.empty_cache()
```

## Ensembling


### how to save all the tta results (a list) into a pickle file


Since this has taken quite a while to run, let's save the results, just in case something goes wrong!

```python
save_pickle('tta_res.pkl', tta_res)
```

### how to get all the predictions from a list of results in which each result contains a prediction and a target for each row of test set


`Learner.tta` returns predictions and targets for each rows. We just want the predictions:

```python
tta_prs = first(zip(*tta_res))
```

### why and how to double the weights for vit models in the ensembles


Originally I just used the above predictions, but later I realised in my experiments on smaller models that `vit` was a bit better than everything else, so I decided to give those double the weight in my ensemble. I did that by simply adding the to the list a second time (we could also do this by using a weighted average):

```python
tta_prs += tta_prs[2:4]
```

### what is the simplest way of doing ensembling


An *ensemble* simply refers to a model which is itself the result of combining a number of other models. The simplest way to do ensembling is to take the average of the predictions of each model:

```python
avg_pr = torch.stack(tta_prs).mean(0)
avg_pr.shape
```

### how to all the classes or vocab of the dataset using dataloaders; how to prepare the csv for kaggle submission


That's all that's needed to create an ensemble! Finally, we copy the steps we used in the last notebook to create a submission file:

```python
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=224, min_scale=0.75))
```

```python
idxs = avg_pr.argmax(dim=1)
vocab = np.array(dls.vocab)
ss = pd.read_csv(path/'sample_submission.csv')
ss['label'] = vocab[idxs]
ss.to_csv('subm.csv', index=False)
```

### how to submit to kaggle using fastkaggle api


Now we can submit:

```python
if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'part 3 v2', comp)
```

That's it -- at the time of creating this analysis, that got easily to the top of the leaderboard! Here are the four submissions I entered, each of which was better than the last, and each of which was ranked #1:

<img src="https://user-images.githubusercontent.com/346999/174503966-65005151-8f28-4f8b-b3c3-212cf74014f1.png" width="400">

*Edit: Actually the one that got to the top of the leaderboard timed out when I ran it on Kaggle Notebooks, so I had to remove two of the runs from the ensemble. There's only a very small difference in accuracy however.*


Going from bottom to top, here's what each one was:

1. `convnext_small` trained for 12 epochs, with TTA
1. `convnext_large` trained the same way
1. The ensemble in this notebook, with `vit` models not over-weighted
1. The ensemble in this notebook, with `vit` models over-weighted.


## Conclusion


### how fastai can superbly simply the codes and standardize processes


The key takeaway I hope to get across from this series so far is that you can get great results in image recognition using very little code and a very standardised approach, and that with a rigorous process you can improve in significant steps. Our training function, including data processing and TTA, is just half a dozen lines of code, plus another 7 lines of code to ensemble the models and create a submission file!

If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. If you have any questions or comments, please pop them below -- I read every comment I receive!

```python
# This is what I use to push my notebook from my home PC to Kaggle

if not iskaggle:
    push_notebook('jhoward', 'scaling-up-road-to-the-top-part-3',
                  title='Scaling Up: Road to the Top, Part 3',
                  file='10-scaling-up-road-to-the-top-part-3.ipynb',
                  competition=comp, private=False, gpu=True)
```

```python

```
