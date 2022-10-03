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

# 0002_fastai_Saving_Model_fastai


This is a minimal example showing how to train a fastai model on Kaggle, and save it so you can use it in your app.

```python
# Make sure we've got the latest version of fastai:
# !pip install -Uqq fastai
```

## what to import to handle vision problems in fastai


First, import all the stuff we need from fastai:

```python
from fastai.vision.all import *
```

## how to download and decompress datasets prepared by fastai


This is a dataset of cats and dogs

```python
#|eval: false
path = untar_data(URLs.PETS)/'images'
```

## how to tell it is a cat by reading filename


We need a way to label our images as dogs or cats. In this dataset, pictures of cats are given a filename that starts with a capital letter:

```python
def is_cat(x): return x[0].isupper() 
```

## how to create dataloaders with `from_name_func`


Now we can create our `DataLoaders`:

```python
#|eval: false
dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))
```

## how to create a pretrained model with resnet18 and error_rate; how to fine tune it 3 epochs


... and train our model, a resnet18 (to keep it small and fast):

```python
#|eval: false
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

```python

```

## how to export model to a pickle file and download it from Kaggle


Now we can export our trained `Learner`. This contains all the information needed to run the model:

```python
#|eval: false
learn.export('model.pkl')
```

Finally, open the Kaggle sidebar on the right if it's not already, and find the section marked "Output". Open the `/kaggle/working` folder, and you'll see `model.pkl`. Click on it, then click on the menu on the right that appears, and choose "Download". After a few seconds, your model will be downloaded to your computer, where you can then create your app that uses the model.


## how to convert ipynb to md

```python
from fastdebug.utils import *
import fastdebug.utils as fu
```

```python
ipy2md()
```

```python

```

```python

```
