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

# 0001_Is it a bird? Creating a model from your own data


## Useful Course sites


**Official course site**:  for lesson [1](https://course.fast.ai/Lessons/lesson1.html)    

**Official notebooks** [repo](https://github.com/fastai/course22), on [nbviewer](https://nbviewer.org/github/fastai/course22/tree/master/)

Official **Is it a bird** [notebook](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data) on kaggle     


```python
%load_ext autoreload
%autoreload 2
```

## How to use autoreload


This [documentation](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html) has a helpful example.


put these two lines in the top of this notebook
```
%load_ext autoreload
%autoreload 2
```


so that, when I updated `fastdebug` library, I don't need to rerun `import fastdebug.utils ....` and it should reload the library for me automatically.


## How to install and update libraries

```python
# !mamba update -q -y fastai
```

```python
# !pip install -Uqq duckduckgo_search
```

## Know a little about the libraries

```python
from fastdebug.utils import *
from fastdebug.core import *
```

### what is fastai

```python
import fastai
```

```python
whichversion("fastai")
```

```python
whatinside(fastai, lib=True)
```

```python
import fastai.losses as fl
```

```python
whatinside(fl, dun=True)
```

### what is duckduckgo

```python
import duckduckgo_search
```

```python
whichversion("duckduckgo_search")
```

```python
whatinside(duckduckgo_search)
```

```python
whatinside(duckduckgo_search, func=True)
```

```python

```

```python

```

```python

```

## How to use fastdebug with fastai notebooks


### what is fastdebug

```python
from fastdebug.utils import *
from fastdebug.core import *
import fastdebug.utils as fu
import fastdebug.core as core
```

```python
whatinside(fu,dun=True)
```

```python
whatinside(core, dun=True)
```

```python
inspect_class(Fastdb)
```

```python

```

### Did I document it in a notebook before?


run `push-code-new` in teminal to convert all current notebooks into mds     

so that the followign search will get me the latest result if I did document similar things

```python
fastnbs("what is fastdebug")
```

I can also extract all the notebook subheadings with the function below    
and to check whether I have documented something similar by `cmd + f` and search keywords there

```python
fastlistnbs()
```

### Did I document it in a src before?

```python
fastcodes("how to access parameters")
```

I can check all the commented src files.

```python
fastsrcs()
```

I can print out all the learning points as comments inside each src file    

However, I need to figure out a way to extract them nicely from the files    

Todos: how to comment src for list extraction

```python
fastlistsrcs()
```

## how to download images

```python
from duckduckgo_search import ddg_images
from fastcore.all import *
```

```python
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
```

```python
#|eval: false
#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('bird photos', max_images=1)
urls[0]
```

```python
#|eval: false
from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```

```python
#|eval: false
download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```

### how to create folders using path; how to search and download images in folders; how to resize images 


Our searches seem to be giving reasonable results, so let's grab 200 examples of each of "bird" and "forest" photos, and save each group of photos to a different folder:

```python
#|eval: false
searches = 'forest','bird'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)
```

## Train my model


### How to find and unlink images not properly downloaded


Some photos might not download correctly which could cause our model training to fail, so we'll remove them:

```python
#|eval: false
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```

```python

```

### How to create a DataLoaders with DataBlock; how to view data with it


To train a model, we'll need DataLoaders:     

1) a training set (the images used to create a model) and 

2) a validation set (the images used to check the accuracy of a model -- not used during training). 

We can view sample images from it:

```python
#|eval: false
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)
```

### How to build my model with dataloaders and pretrained model; how to train my model


Now we're ready to train our model. The fastest widely used computer vision model is resnet18. You can train this in a few minutes, even on a CPU! (On a GPU, it generally takes under 10 seconds...)

fastai comes with a helpful `fine_tune()` method which automatically uses best practices for fine tuning a pre-trained model, so we'll use that.

```python
#|eval: false
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

```python

```

### How to predict with my model; how to avoid running cells in nbdev_prepare

```python
#|eval: false
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
```

## Check fastnbs and fastlistnbs

```python
fastlistnbs()
```

```python
fastnbs("how unlink images properly")
```

## search and display section input and output in notebooks

```python
#| export utils
def add_py_block(file_fullname):
#     with open(file_fullname, 'r') as file:
#         for count, l in enumerate(file):    
    with open(file_fullname, 'r') as file:
        entire = file.read()   
        parts = entire.split("```")
        new_entire = ""
        for idx, p in zip(range(len(parts)), parts):
            if idx % 2 == 0:
                new_entire = new_entire + p + "```python"
            else: 
                new_entire = new_entire + p + "```"
        return new_entire
```

```python
#| export utils
def fastnbsrc(question:str, accu:float=0.8, nb=True, db=False):
    "using keywords to search learning points (a section title and a section itself) from my documented fastai notebooks"
    questlst = question.split(' ')
    all_nbs, folder, ipynbs, ipyfolder, mds_out, mds_out_fd = get_all_nbs()
    for file_fullname in mds_out:
        file_name = file_fullname.split('/')[-1]
        with open(file_fullname, 'r') as file:
            for count, l in enumerate(file):
                truelst = [q.lower() in l.lower() for q in questlst]
                pct = sum(truelst)/len(truelst)
                if pct >= accu and (l.startswith("##") or l.startswith("###") or l.startswith("####")):
                    if db: 
                        head1 = f"keyword match is {pct}, Found a section: in {file_name}"
                        head1 = highlight(str(pct), head1)
                        head1 = highlight(file_name, head1)
                        display_md(head1)
                        highlighted_line = highlight(question, l, db=db)                        
                        print()
                    new_entire = add_py_block(file_fullname)
                    display_block(l, new_entire, keywords=question)
                    if nb: openNB(file_name, db=db)
```

```python

```

```python
fastnbs("what is fastai")
```

```python
fastnbsrc("what is fastai")
```

```python
all_nbs, folder, ipynbs, ipyfolder, mds_out, mds_out_fd = get_all_nbs()
mds_out
```

```python
fastlistnbs()
```

```python

```
