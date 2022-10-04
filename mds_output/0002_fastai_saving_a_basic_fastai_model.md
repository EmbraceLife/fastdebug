# 0002_fastai_saving_a_basic_fastai_model

This is a minimal example showing how to train a fastai model on Kaggle, and save it so you can use it in your app.


```
# Make sure we've got the latest version of fastai:
# !pip install -Uqq fastai
```

## what to import to handle vision problems in fastai

First, import all the stuff we need from fastai:


```
from fastai.vision.all import *
```

## how to download and decompress datasets prepared by fastai

This is a dataset of cats and dogs


```
#|eval: false
path = untar_data(URLs.PETS)/'images'
```

## how to tell it is a cat by reading filename

We need a way to label our images as dogs or cats. In this dataset, pictures of cats are given a filename that starts with a capital letter:


```
def is_cat(x): return x[0].isupper() 
```

## how to create dataloaders with `from_name_func`

Now we can create our `DataLoaders`:


```
#|eval: false
dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))
```

## how to create a pretrained model with resnet18 and error_rate; how to fine tune it 3 epochs

... and train our model, a resnet18 (to keep it small and fast):


```
#|eval: false
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
      warnings.warn(
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





    <div>
      <progress value='0' class='' max='1' style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00&lt;?]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
      <progress value='19' class='' max='92' style='width:300px; height:20px; vertical-align: middle;'></progress>
      20.65% [19/92 00:12&lt;00:48 0.7980]
    </div>




```

```

## how to export model to a pickle file and download it from Kaggle

Now we can export our trained `Learner`. This contains all the information needed to run the model:


```
#|eval: false
learn.export('model.pkl')
```

Finally, open the Kaggle sidebar on the right if it's not already, and find the section marked "Output". Open the `/kaggle/working` folder, and you'll see `model.pkl`. Click on it, then click on the menu on the right that appears, and choose "Download". After a few seconds, your model will be downloaded to your computer, where you can then create your app that uses the model.

## how to convert ipynb to md


```
from fastdebug.utils import *
import fastdebug.utils as fu
```


<style>.container { width:100% !important; }</style>



```
ipy2md()
```

    [jupytext] Reading /Users/Natsume/Documents/fastdebug/nbs/2022part1/0002_fastai_Saving_Model_fastai.ipynb in format ipynb
    [jupytext] Writing /Users/Natsume/Documents/fastdebug/nbs/2022part1/0002_fastai_Saving_Model_fastai.md
    cp to : /Users/Natsume/Documents/divefastai/Debuggable/jupytext
    move to : /Users/Natsume/Documents/fastdebug/mds/2022part1/


    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/2022part1/0002_fastai_Saving_Model_fastai.ipynb to markdown
    [NbConvertApp] Writing 4849 bytes to /Users/Natsume/Documents/fastdebug/nbs/2022part1/0002_fastai_Saving_Model_fastai.md


    move to : /Users/Natsume/Documents/fastdebug/mds_output



```

```


```

```
