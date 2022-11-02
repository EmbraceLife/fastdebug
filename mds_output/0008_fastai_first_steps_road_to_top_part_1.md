# 0008_fastai_first_steps_road_to_top_part_1
---
skip_exec: true
---

```
#| default_exp delete_road_top1
```


```
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>



```
# fastlistnbs("howto")
```

### ht: install fastkaggle if not available


```
# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *
```

### ht: iterate like a grandmaster

In [Iterate Like a Grandmaster](https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster) I explained that when working on a Kaggle project:

> ...the focus generally should be two things:
> 
> 1. Creating an effective validation set
> 2. Iterating rapidly to find changes which improve results on the validation set.

Here I'm going to go further, showing the process I used to tackle the [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) competition, leading to four submissions in a row which all were (at the time of submission) in 1st place, each one more accurate than the last. You might be surprised to discover that the process of doing this was nearly entirely mechanistic and didn't involve any consideration of the actual data or evaluation details at all.

This notebook is the first in a series showing every step of the process. At the end of this notebook we'll have a basic submission; by the end of the series you'll see how I got to the top of the table!:

<img src="https://user-images.githubusercontent.com/346999/174389920-60d67ead-0f36-41d0-9649-e23b08720c8a.png" width="600"/>

### what are the related walkthrus on paddy doctor competition

As a special extra, I'm also opening up early a selection of "walkthru" videos that we've been preparing for the new upcoming fast.ai course. Each day I do a walkthru with fast.ai fellows and registered students, and we record those sessions. They'll all be released at the same time as the next course (probably August 2022), but I'm releasing the ones covering this competition right now! Here they are:

- [Walkthru 8](https://www.youtube.com/watch?v=-Scs4gbwWXg)
- [Walkthru 9](https://www.youtube.com/watch?v=EK5wJRzffas)
- [Walkthru 10](https://youtu.be/zhBRynq9Yvo)
- [Walkthru 11](https://youtu.be/j-zMF2VirA8)
- [Walkthru 12](https://youtu.be/GuCkpjXHdTc)
- [Walkthru 13](https://youtu.be/INrkhUGCXHg)

## ht: download and access kaggle competition dataset

### ht: set up before downloading
go to kaggle.com, account, api, and click create a new api token

then `cp kaggle.json ~/.kaggle/`

go to the competition site and join the competition, and get the fullname of the competition for downloading the dataset

First, we'll get the data. I've just created a new library called [fastkaggle](https://fastai.github.io/fastkaggle/) which has a few handy features, including getting the data for a competition correctly regardless of whether we're running on Kaggle or elsewhere. Note you'll need to first accept the competition rules and join the competition, and you'll need your kaggle API key file `kaggle.json` downloaded if you're running this somewhere other than on Kaggle. `setup_comp` is the function we use in `fastkaggle` to grab the data, and install or upgrade our needed python modules when we're running on Kaggle:

### src: setup_comp(comp, install='fastai "timm>=0.6.2.dev0")
Get a path to data for `competition`, downloading it if needed


```
@snoop
def setup_comp(competition, install=''):
    "Get a path to data for `competition`, downloading it if needed"
    if iskaggle:
        if install:
            os.system(f'pip install -Uqq {install}')
        return Path('../input')/competition
    else:
        path = Path(competition)
        api = import_kaggle()
        if not path.exists():
            import zipfile
            api.competition_download_cli(str(competition))
            zipfile.ZipFile(f'{competition}.zip').extractall(str(competition))
        return path
# File:      ~/mambaforge/lib/python3.9/site-packages/fastkaggle/core.py
# Type:      function
```


```
"on kaggle" if iskaggle else "not on kaggle"
```




    'not on kaggle'




```
# api = import_kaggle()
# lst_cmp=api.competitions_list()
# lst_cmp[0].__dict__
# lst_cmp
```


```
comp = 'paddy-disease-classification'
path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
```

    19:08:29.63 >>> Call to setup_comp in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_82945/1008324063.py", line 2
    19:08:29.63 ...... competition = 'paddy-disease-classification'
    19:08:29.63 ...... install = 'fastai "timm>=0.6.2.dev0"'
    19:08:29.63    2 | def setup_comp(competition, install=''):
    19:08:29.63    4 |     if iskaggle:
    19:08:29.63    9 |         path = Path(competition)
    19:08:29.63 .............. path = Path('paddy-disease-classification')
    19:08:29.63   10 |         api = import_kaggle()
    19:08:29.66 .............. api = <kaggle.api.kaggle_api_extended.KaggleApi object>
    19:08:29.66   11 |         if not path.exists():
    19:08:29.66   15 |         return path
    19:08:29.66 <<< Return value from setup_comp: Path('paddy-disease-classification')



```
path.ls()
```




    (#4) [Path('paddy-disease-classification/test_images'),Path('paddy-disease-classification/train.csv'),Path('paddy-disease-classification/train_images'),Path('paddy-disease-classification/sample_submission.csv')]




```
def setup_comp(competition, install=''):
    "Get a path to data for `competition`, downloading it if needed"
    if iskaggle:
        if install:
            os.system(f'pip install -Uqq {install}')
        return Path('../input')/competition
    else:
        path = Path(competition)
        api = import_kaggle()
        if not path.exists():
            import zipfile
            api.competition_download_cli(str(competition))
            zipfile.ZipFile(f'{competition}.zip').extractall(str(competition))
        return path
# File:      ~/mambaforge/lib/python3.9/site-packages/fastkaggle/core.py
# Type:      function
```

### ht: reproducibility in training

Now we can import the stuff we'll need from fastai, set a seed (for reproducibility -- just for the purposes of making this notebook easier to write; I don't recommend doing that in your own analysis however) and check what's in the data:


```
from fastai.vision.all import *
set_seed(42)
```

## ht: data - access dataset

### ht: data - map subfolders content
use `path.ls()` and `check_subfolders_img(path)` to see what inside each subfolders


```
path.ls()
```




    (#4) [Path('paddy-disease-classification/test_images'),Path('paddy-disease-classification/train.csv'),Path('paddy-disease-classification/train_images'),Path('paddy-disease-classification/sample_submission.csv')]



### src: check_subfolders_img(path, db=False)


```
# @snoop
def check_subfolders_img(path, db=False):
    from pathlib import Path
    for entry in path.iterdir():
        if entry.is_file():
            print(f'{str(entry.absolute())}')
    for entry in path.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and len(entry.ls(file_exts=image_extensions)) > 5:
            print(f'{str(entry.parent.absolute())}: {len(entry.ls(file_exts=image_extensions))}  {entry.name}')
#             print(entry.name, f': {len(entry.ls(file_exts=[".jpg", ".png", ".jpeg", ".JPG", ".jpg!d"]))}') # how to include both png and jpg
            if db:
                for e in entry.ls(): # check any image file which has a different suffix from those above
                    if e.is_file() and not e.name.startswith(".") and e.suffix not in image_extensions and e.suffix not in [".ipynb", ".py"]:
    #                 if e.suffix not in [".jpg", ".png", ".jpeg", ".JPG", ".jpg!d"]:
                        pp(e.suffix, e)
                        try:
                            pp(Image.open(e).width)
                        except:
                            print(f"{e} can't be opened")
    #                     pp(Image.open(e).width if e.suffix in image_extensions)
        elif entry.is_dir() and not entry.name.startswith("."): 
#             with snoop:
            count_files_in_subfolders(entry)
```


```
check_subfolders_img(path)
```

    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/train.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification/sample_submission.csv
    /Users/Natsume/Documents/fastdebug/nbs/fastai_notebooks/paddy-disease-classification: 3469  test_images
    dead_heart : nbs/fastai_notebooks/paddy-disease-classification/train_images
    dead_heart : 1442
    bacterial_panicle_blight : nbs/fastai_notebooks/paddy-disease-classification/train_images
    bacterial_panicle_blight : 337
    bacterial_leaf_blight : nbs/fastai_notebooks/paddy-disease-classification/train_images
    bacterial_leaf_blight : 479
    brown_spot : nbs/fastai_notebooks/paddy-disease-classification/train_images
    brown_spot : 965
    hispa : nbs/fastai_notebooks/paddy-disease-classification/train_images
    hispa : 1594
    downy_mildew : nbs/fastai_notebooks/paddy-disease-classification/train_images
    downy_mildew : 620
    blast : nbs/fastai_notebooks/paddy-disease-classification/train_images
    blast : 1738
    normal : nbs/fastai_notebooks/paddy-disease-classification/train_images
    normal : 1764
    bacterial_leaf_streak : nbs/fastai_notebooks/paddy-disease-classification/train_images
    bacterial_leaf_streak : 380
    tungro : nbs/fastai_notebooks/paddy-disease-classification/train_images
    tungro : 1088


### ht: data - extract all images for test and train 
using `get_image_files(path)` for both test folder and train folder


```
test_files = get_image_files(path/"test_images")
train_files = get_image_files(path/"train_images")
```


```
test_files
train_files
```




    (#3469) [Path('paddy-disease-classification/test_images/202919.jpg'),Path('paddy-disease-classification/test_images/200868.jpg'),Path('paddy-disease-classification/test_images/200698.jpg'),Path('paddy-disease-classification/test_images/200840.jpg'),Path('paddy-disease-classification/test_images/201586.jpg'),Path('paddy-disease-classification/test_images/203391.jpg'),Path('paddy-disease-classification/test_images/202931.jpg'),Path('paddy-disease-classification/test_images/202925.jpg'),Path('paddy-disease-classification/test_images/203385.jpg'),Path('paddy-disease-classification/test_images/200854.jpg')...]






    (#10407) [Path('paddy-disease-classification/train_images/dead_heart/110369.jpg'),Path('paddy-disease-classification/train_images/dead_heart/105002.jpg'),Path('paddy-disease-classification/train_images/dead_heart/106279.jpg'),Path('paddy-disease-classification/train_images/dead_heart/108254.jpg'),Path('paddy-disease-classification/train_images/dead_heart/104308.jpg'),Path('paddy-disease-classification/train_images/dead_heart/107629.jpg'),Path('paddy-disease-classification/train_images/dead_heart/110355.jpg'),Path('paddy-disease-classification/train_images/dead_heart/100146.jpg'),Path('paddy-disease-classification/train_images/dead_heart/103329.jpg'),Path('paddy-disease-classification/train_images/dead_heart/105980.jpg')...]




```
fastnbs("get_image_files")
```


### <mark style="background-color: #FFFF00">get_image_files</mark> 




The current section is heading 3.


```python

```

```python
check(get_image_files)
```

```python
files_train = get_image_files(path/"train")
files_valid = get_image_files(path/"valid")
files_test = get_image_files(path/"test")
len(files_train), len(files_valid), len(files_test)
```

```python
files_train[:5]
```

```python
#| export
def get_img_paths(query, train, valid, test):
    from fastai.data.external import untar_data
    from fastai.data.transforms import get_image_files
    path = untar_data(search_data_url(query))
    files_train = get_image_files(path/train)
    files_valid = get_image_files(path/valid)
    files_test = get_image_files(path/test)
    print(f'train: {len(files_train)}, valid: {len(files_valid)}, test: {len(files_test)}')
    return files_train, files_valid, files_test
```

```python
files_train, files_valid, files_test = get_img_paths("mnist var", "train", "valid", "test")
```

```python

```

start of another heading 3
### get_labels



[Open `groundup_002_get_data_ready` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/groundup_002_get_data_ready.ipynb#get_image_files
)



### <mark style="background-color: #FFFF00">get_image_files</mark> , get_files, image_extensions




The current section is heading 3.

to extract all image files recursively from all subfolders of a parent path

```python
from fastai.data.transforms import _get_files
```

```python
not None
```

```python
def _get_files(p, # path
               fs, # list of filenames
               extensions=None):
    "get the fullnames for the list of filenames of a path"
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```

```python
def get_files(path, extensions=None, recurse=True, folders=None, followlinks=True):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."
    path = Path(path)
    folders=L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)): # returns (dirpath, dirnames, filenames)
            if len(folders) !=0 and i==0: d[:] = [o for o in d if o in folders]
            else:                         d[:] = [o for o in d if not o.startswith('.')]
            if len(folders) !=0 and i==0 and '.' not in folders: continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    return L(res)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```

```python
len(image_extensions)
".aspx" in image_extensions
```

```python
def get_image_files(path, recurse=True, folders=None):
    "Get image files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=image_extensions, recurse=recurse, folders=folders)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/data/transforms.py
```

```python
count_files_in_subfolders(dino)
get_image_files(dino)
count_files_in_subfolders(bird)
get_image_files(bird)
```

```python
Image.open(Path('forest_or_bird/bird/3b7a4112-9d77-4d8f-8b4c-a01e2ca1ecea.aspx'))
```

start of another heading 3
### parent_label(o)



[Open `0001_fastai_is_it_a_bird` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0001_fastai_is_it_a_bird.ipynb#get_image_files,-get_files,-image_extensions
)



[Open `0001_fastai_is_it_a_bird` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data)



### how to get the rice variety of each image with `df.loc` and the path from `get_image_files`




The current section is heading 3.



Our DataBlock will be using `get_image_files` to get the list of training images, which returns `Path` objects. Therefore, to look up an item to get its variety, we'll need to pass its `name`. Here's a function which does just that:

```python
def get_variety(p): return df.loc[p.name, 'variety']
```

start of another heading 3
### how to use `DataBlock` to create dataloaders; what is DataBlock



[Open `0011_fastai_multi_target_road_to_top_part_4` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0011_fastai_multi_target_road_to_top_part_4.ipynb#how-to-get-the-rice-variety-of-each-image-with-`df.loc`-and-the-path-from-`get_image_files`
)



[Open `0011_fastai_multi_target_road_to_top_part_4` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/multi-target-road-to-the-top-part-4)


...and take a look at one:

### ht: data - display images from test_files or train_files
use `randomdisplay(path, size, db=False)` to display images from a folder or a L list of images such as `test_files` or `train_files`

### src: randomdisplay(path, size, db=False)
display a random images from a L list (eg., test_files, train_files) of image files or from a path/folder of images.\
    the image filename is printed as well


```
import pathlib
type(path) == pathlib.PosixPath
type(train_files) == L
```




    True






    True




```
#| export utils
# @snoop
def randomdisplay(path, size, db=False):
    "display a random images from a L list (eg., test_files, train_files) of image files or from a path/folder of images.\
    the image filename is printed as well"
# https://www.geeksforgeeks.org/python-random-module/
    import random
    import pathlib
    from fastai.vision.all import PILImage
    if type(path) == pathlib.PosixPath:
        rand = random.randint(0,len(path.ls())-1) 
        file = path.ls()[rand]
    elif type(path) == L:
        rand = random.randint(0,len(path)-1) 
        file = path[rand]
    im = PILImage.create(file)
    if db: pp(im.width, im.height, file)
    pp(file)
    return im.to_thumb(size)
```


```
randomdisplay(test_files, 128)
randomdisplay(train_files, 200)
randomdisplay(path/"train_images/dead_heart", 128)
```

    19:13:02.91 LOG:
    19:13:02.91 .... file = Path('paddy-disease-classification/test_images/201589.jpg')





    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_39_1.png)
    



    19:13:02.93 LOG:
    19:13:02.93 .... file = Path('paddy-disease-classification/train_images/hispa/104841.jpg')





    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_39_3.png)
    



    19:13:02.95 LOG:
    19:13:02.95 .... file = Path('paddy-disease-classification/train_images/dead_heart/108036.jpg')





    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_39_5.png)
    



### how to use `fastcore.parallel` to quickly access size of all images; how to count the occurance of each unique value in a pandas 

Looks like the images might be 480x640 -- let's check all their sizes. This is faster if we do it in parallel, so we'll use fastcore's `parallel` for this:


```
from fastcore.parallel import *

def f(o): return PILImage.create(o).size
sizes = parallel(f, files, n_workers=8)
pd.Series(sizes).value_counts()
```




    (480, 640)    10403
    (640, 480)        4
    dtype: int64



### how to create an image dataloaders; how to setup `item_tfms` and `batch_tfms` on image sizes; why to start with the smallest sizes first; how to display images in batch

They're nearly all the same size, except for a few. Because of those few, however, we'll need to make sure we always resize each image to common dimensions first, otherwise fastai won't be able to create batches. For now, we'll just squish them to 480x480 images, and then once they're in batches we do a random resized crop down to a smaller size, along with the other default fastai augmentations provided by `aug_transforms`. We'll start out with small resized images, since we want to be able to iterate quickly:


```
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)
```


    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_45_0.png)
    


## Our first model

### how to pick the first pretrained model for our model; how to build our model based on the selected pretrained model

Let's create a model. To pick an architecture, we should look at the options in [The best vision models for fine-tuning](https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning). I like the looks of `resnet26d`, which is the fastest resolution-independent model which gets into the top-15 lists there.


```
learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()
```

### how to find the learning rate for our model

Let's see what the learning rate finder shows:


```
learn.lr_find(suggest_funcs=(valley, slide))
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    SuggestedLRs(valley=0.0008317637839354575, slide=0.0030199517495930195)




    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_52_3.png)
    


`lr_find` generally recommends rather conservative learning rates, to ensure that your model will train successfully. I generally like to push it a bit higher if I can. Let's train a few epochs and see how it looks:


```
learn.fine_tune(3, 0.01)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




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
    <tr>
      <td>0</td>
      <td>1.805964</td>
      <td>1.233453</td>
      <td>0.403652</td>
      <td>00:14</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




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
    <tr>
      <td>0</td>
      <td>1.129876</td>
      <td>0.785891</td>
      <td>0.266218</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.777808</td>
      <td>0.456637</td>
      <td>0.143681</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.557498</td>
      <td>0.407197</td>
      <td>0.136473</td>
      <td>00:15</td>
    </tr>
  </tbody>
</table>


We're now ready to build our first submission. Let's take a look at the sample Kaggle provided to see what it needs to look like:

## Submitting to Kaggle

### how to check the kaggle submission sample csv file


```
ss = pd.read_csv(path/'sample_submission.csv')
ss
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
      <th>image_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>200001.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200002.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200003.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200004.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200005.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3464</th>
      <td>203465.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3465</th>
      <td>203466.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3466</th>
      <td>203467.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3467</th>
      <td>203468.jpg</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3468</th>
      <td>203469.jpg</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3469 rows × 2 columns</p>
</div>



### how to sort the files in the test set in the alphabetical order; how to create dataloaders for the test set based on the dataloaders of the training set

OK so we need a CSV containing all the test images, in alphabetical order, and the predicted label for each one. We can create the needed test set using fastai like so:


```
tst_files = get_image_files(path/'test_images').sorted()
tst_dl = dls.test_dl(tst_files)
```

### how to make predictions for all test set; and what does `learn.get_preds` return

We can now get the probabilities of each class, and the index of the most likely class, from this test set (the 2nd thing returned by `get_preds` are the targets, which are blank for a test set, so we discard them):


```
probs,_,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
idxs
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>










    TensorBase([7, 8, 7,  ..., 8, 7, 5])



### how to access all the classes of labels with dataloaders

These need to be mapped to the names of each of these diseases, these names are stored by fastai automatically in the `vocab`:


```
dls.vocab
```




    ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']



### how to map classes to each idx from the predictions

We can create an apply this mapping using pandas:


```
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
results
```




    0            hispa
    1           normal
    2            hispa
    3            blast
    4            blast
               ...    
    3464    dead_heart
    3465         hispa
    3466        normal
    3467         hispa
    3468    dead_heart
    Name: idxs, Length: 3469, dtype: object



### how to save result into csv file

Kaggle expects the submission as a CSV file, so let's save it, and check the first few lines:


```
ss['label'] = results
ss.to_csv('subm.csv', index=False)
!head subm.csv
```

    image_id,label
    200001.jpg,hispa
    200002.jpg,normal
    200003.jpg,hispa
    200004.jpg,blast
    200005.jpg,blast
    200006.jpg,brown_spot
    200007.jpg,dead_heart
    200008.jpg,brown_spot
    200009.jpg,hispa


### how to submit to kaggle with fastkaggle api

Let's submit this to kaggle. We can do it from the notebook if we're running on Kaggle, otherwise we can use the API:


```
if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'initial rn26d 128px', comp)
```

    100%|██████████████████████████████████████████| 70.0k/70.0k [00:05<00:00, 13.8kB/s]





    Successfully submitted to Paddy Doctor: Paddy Disease Classification



Success! We successfully created a submission.

## Conclusion

### what is the most important thing for your first model

Our initial submission is not very good (top 80% of teams) but it only took a minute to train. The important thing is that we have a good starting point to iterate from, and we can do rapid iterations. Every step from loading the data to creating the model to submitting to Kaggle is all automated and runs quickly.

Therefore, we can now try lots of things quickly and easily and use those experiments to improve our results. In the next notebook, we'll do exactly that!

If you found this notebook useful, please remember to click the little up-arrow at the top to upvote it, since I like to know when people have found my work useful, and it helps others find it too. And if you have any questions or comments, please pop them below -- I read every comment I receive!

## Addendum

### how to quickly push your local notebook to become kaggle notebook online

`fastkaggle` also provides a function that pushes a notebook to Kaggle Notebooks. I wrote this notebook on my own machine, and pushed it to Kaggle from there -- here's the command I used:


```
if not iskaggle:
    push_notebook('jhoward', 'first-steps-road-to-the-top-part-1',
                  title='First Steps: Road to the Top, Part 1',
                  file='first-steps-road-to-the-top-part-1.ipynb',
                  competition=comp, private=False, gpu=True)
```

    Kernel version 10 successfully pushed.  Please check progress at https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1



```

```


```
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>



```
nb_name()
```




    '0008_fastai_first_steps_road_to_top_part_1.ipynb'




```
ipy2md()
```

    [jupytext] Reading /Users/Natsume/Documents/fastdebug/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1.ipynb in format ipynb
    [jupytext] Writing /Users/Natsume/Documents/fastdebug/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1.md
    cp to : /Users/Natsume/Documents/divefastai/Debuggable/jupytext
    move to : /Users/Natsume/Documents/fastdebug/mds/2022part1/


    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1.ipynb to markdown


    copy to : /Users/Natsume/Documents/fastdebug/mds_output
    move to : /Users/Natsume/Documents/divefastai/Debuggable/nbconvert


    [NbConvertApp] Support files will be in 0008_fastai_first_steps_road_to_top_part_1_files/
    [NbConvertApp] Making directory /Users/Natsume/Documents/fastdebug/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1_files
    [NbConvertApp] Making directory /Users/Natsume/Documents/fastdebug/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1_files
    [NbConvertApp] Making directory /Users/Natsume/Documents/fastdebug/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1_files
    [NbConvertApp] Writing 20191 bytes to /Users/Natsume/Documents/fastdebug/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1.md



```
fastnbs("push kaggle")
```


### how to quickly <mark style="background-color: #ffff00">push</mark>  your local notebook to become <mark style="background-color: #FFFF00">kaggle</mark>  notebook online






`fastkaggle` also provides a function that pushes a notebook to Kaggle Notebooks. I wrote this notebook on my own machine, and pushed it to Kaggle from there -- here's the command I used:

```python
if not iskaggle:
    push_notebook('jhoward', 'first-steps-road-to-the-top-part-1',
                  title='First Steps: Road to the Top, Part 1',
                  file='first-steps-road-to-the-top-part-1.ipynb',
                  competition=comp, private=False, gpu=True)
```

```python

```

```python
from fastdebug.utils import *
```

```python
nb_name()
```

```python
ipy2md()
```

```python
fastnbs("push kaggle")
```

```python

```




[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/2022part1/0008_fastai_first_steps_road_to_top_part_1.ipynb)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)

