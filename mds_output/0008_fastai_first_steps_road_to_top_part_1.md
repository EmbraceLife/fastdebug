# 0008_fastai_first_steps_road_to_top_part_1
---
skip_exec: true
---

```
#| default_exp delete_road_top1
```

### jn: help other is the best way forward

**Reflection on Radek's 1st newsletter**

One way to summarize Radek's secret to success is the following: 

> No matter which stage of journey in the deep learning or any subject, when you are doing your best to help others to learn what you learnt and what you are dying to find out, and if you persist, you will be happy and successful. 

I have dreamed of such hypothesis many times when I motivated myself to share online, and Radek proved it to be solid and true! No time to waste now!

Another extremely simple but shocking secret to Radek's success is, in his words (now I can recite):

> I would suspend my disbelief and do exactly what Jeremy Howard told us to do in the lectures

What Jeremy told us to do is loud and clear, the 4 steps (watch, experiment, reproduce, apply elsewhere). More importantly, they are true and working if one holds onto it like Radek did. 

Why I am always trying to do something different? Why couldn't I just follow this great advice right from the start? I walked [a long way around it](https://twitter.com/shendusuipian/status/1587429658621988871?s=20&t=zjz1OlYRt7yJJ8HVBdsqoA) and luckily I get my sense back and move onto the second step now. 

## ht: imports - vision


```
from fastdebug.utils import *
from fastai.vision.all import *
```


<style>.container { width:100% !important; }</style>


### ht: fu - whatinside, show_doc, fastlistnbs, fastnbs


```
# whatinside(fu, dun=True) # see all functions defined in fastdebug.utils
```


```
# show_doc(fastlistnbs)
# show_doc(fastnbs)
```


```
# fastlistnbs("srcode")
# fastlistnbs("howto")
# fastlistnbs("doc")

```


```
# fastnbs("ht: data - check")
# fastnbs("src: check_siz")
```

### ht: imports - fastkaggle 


```
# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *
```

### ht: imports - use mylib in kaggle
- upload my fastdebug.utils module as a dataset to kaggle, to create a dataset

- and in one kaggle notebook, go to the top right, under Data/Input/ find the dataset and copy file path, and run the cell below to import it

- when updating the library, go to the dataset page, click top right `...` and click `update version` to upload the latest version of your script


```
# !pip install nbdev snoop

# path = "../input/fastdebugutils0"
# import sys
# sys.path
# sys.path.insert(1, path)
# import utils as fu
```

### ht: imports - fastkaggle - push libs to kaggle


```
# lib_path = Path('/root/kaggle_datasets') # not working
# lib_path = Path('../input/kaggle_datasets') # it's working, but I can't find it in kaggle
# username = 'danielliao'
```


```
# libs = ['fastcore','timm']
# create_libs_datasets(libs,lib_path,username)
```

### ht: fu - git - when a commit takes too long

1. cp repo (removed the large dataset) elsewhere in your file system
2. git clone your-repo --depth 1
3. cp everything in repo except .git folder to the latest repo just downloaded
4. git push to update

### jn: how to iterate or make one step forward at at time

This is Jeremy showing us how to iterate, ie., increment learning one tiny step every time every day

In [Iterate Like a Grandmaster](https://www.kaggle.com/code/jhoward/iterate-like-a-grandmaster) I explained that when working on a Kaggle project:

> ...the focus generally should be two things:
> 
> 1. Creating an effective validation set
> 2. Iterating rapidly to find changes which improve results on the validation set.

Here I'm going to go further, showing the process I used to tackle the [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) competition, leading to four submissions in a row which all were (at the time of submission) in 1st place, each one more accurate than the last. You might be surprised to discover that the process of doing this was nearly entirely mechanistic and didn't involve any consideration of the actual data or evaluation details at all.

This notebook is the first in a series showing every step of the process. At the end of this notebook we'll have a basic submission; by the end of the series you'll see how I got to the top of the table!:

<img src="https://user-images.githubusercontent.com/346999/174389920-60d67ead-0f36-41d0-9649-e23b08720c8a.png" width="600"/>

As a special extra, I'm also opening up early a selection of "walkthru" videos that we've been preparing for the new upcoming fast.ai course. Each day I do a walkthru with fast.ai fellows and registered students, and we record those sessions. They'll all be released at the same time as the next course (probably August 2022), but I'm releasing the ones covering this competition right now! Here they are:

- [Walkthru 8](https://www.youtube.com/watch?v=-Scs4gbwWXg)
- [Walkthru 9](https://www.youtube.com/watch?v=EK5wJRzffas)
- [Walkthru 10](https://youtu.be/zhBRynq9Yvo)
- [Walkthru 11](https://youtu.be/j-zMF2VirA8)
- [Walkthru 12](https://youtu.be/GuCkpjXHdTc)
- [Walkthru 13](https://youtu.be/INrkhUGCXHg)

## ht: data_download - kaggle competition dataset

### ht: data_download - join, `kaggle.json`, `setup_comp`

go to kaggle.com, go to 'account', 'api', and click 'create a new api token'

then `cp kaggle.json ~/.kaggle/`

go to the competition site and join the competition, and get the fullname of the competition for downloading the dataset

running `setup_comp(comp, install='fastai "timm>=0.6.2.dev0")` to download the dataset

First, we'll get the data. I've just created a new library called [fastkaggle](https://fastai.github.io/fastkaggle/) which has a few handy features, including getting the data for a competition correctly regardless of whether we're running on Kaggle or elsewhere. Note you'll need to first accept the competition rules and join the competition, and you'll need your kaggle API key file `kaggle.json` downloaded if you're running this somewhere other than on Kaggle. `setup_comp` is the function we use in `fastkaggle` to grab the data, and install or upgrade our needed python modules when we're running on Kaggle:

### doc: setup_comp(comp, local_folder='', install='fastai "timm>=0.6.2.dev0")

override `fastkaggle.core.setup_comp` for my use

If on kaggle, download and install required libraries to work with, and return a path linking to the dataset

If on local machine, download the dataset to the path based on local_folder if the path is not available and return the path

```python
@snoop
def setup_comp(competition, local_folder='', install=''):
    "Get a path to data for `competition`, downloading it if needed"
    if iskaggle:
        if install:
            os.system(f'pip install -Uqq {install}')
        return Path('../input')/competition
    else:
        path = Path(local_folder + competition)
        api = import_kaggle()
        if not path.exists():
            import zipfile
#             pp(doc_sig(api.competition_download_cli))
#             return
            api.competition_download_cli(str(competition), path=path)
            zipfile.ZipFile(f'{local_folder + competition}.zip').extractall(str(local_folder + competition))
        return path
# File:      ~/mambaforge/lib/python3.9/site-packages/fastkaggle/core.py
# Type:      function
```


```
show_doc(fastkaggle.core.setup_comp)
```




---

### setup_comp

>      setup_comp (competition, install='')

Get a path to data for `competition`, downloading it if needed




```
src(fastkaggle.core.setup_comp)
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
    


### ht: fu - debug every srcline without breaking

withe magic of `return` we can use `@snoop`, `pp`, `doc_sig`, `chk`, `src` to do debug anything we want


```
src(chk)
```

    def chk(obj):
        "return obj's type, length and type if available"
        tp = type(obj)
        length = obj.__len__() if hasattr(obj, '__len__') else "no length"
        shape = obj.shape if hasattr(obj, 'shape') else "no shape"
        return tp, length, shape
    


### ht: fu - (de)activate snoop without commenting out using `snoopon()` and `snoopoff()`  

### src: setup_compsetup_comp(comp, local_folder='', install='fastai "timm>=0.6.2.dev0")


```
snoopon()
```


```
@snoop
def setup_comp(competition, local_folder='', install=''):
    "Get a path to data for `competition`, downloading it if needed"
    if iskaggle:
        if install:
            os.system(f'pip install -Uqq {install}')
        return Path('../input')/competition
    else:
        path = Path(local_folder + competition)
        api = import_kaggle()
        if not path.exists():
            import zipfile
#             pp(doc_sig(api.competition_download_cli))
#             return
            api.competition_download_cli(str(competition), path=path)
            zipfile.ZipFile(f'{local_folder + competition}.zip').extractall(str(local_folder + competition))
        return path
# File:      ~/mambaforge/lib/python3.9/site-packages/fastkaggle/core.py
# Type:      function
```


```
comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
local = "/Users/Natsume/Documents/"
path = setup_comp(comp, local_folder=local, install='fastai "timm>=0.6.2.dev0"')
```

    14:44:39.03 >>> Call to setup_comp in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_39922/1633869195.py", line 2
    14:44:39.03 ...... competition = 'paddy-disease-classification'
    14:44:39.03 ...... local_folder = '/Users/Natsume/Documents/'
    14:44:39.03 ...... install = 'fastai "timm>=0.6.2.dev0"'
    14:44:39.03    2 | def setup_comp(competition, local_folder='', install=''):
    14:44:39.03    4 |     if iskaggle:
    14:44:39.03    9 |         path = Path(local_folder + competition)
    14:44:39.03 .............. path = Path('/Users/Natsume/Documents/paddy-disease-classification')
    14:44:39.03   10 |         api = import_kaggle()
    14:44:39.05 .............. api = <kaggle.api.kaggle_api_extended.KaggleApi object>
    14:44:39.05   11 |         if not path.exists():
    14:44:39.05   17 |         return path
    14:44:39.05 <<< Return value from setup_comp: Path('/Users/Natsume/Documents/paddy-disease-classification')



```
snoopoff()
```


```
path.ls()
```




    (#4) [Path('/Users/Natsume/Documents/paddy-disease-classification/test_images'),Path('/Users/Natsume/Documents/paddy-disease-classification/train.csv'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images'),Path('/Users/Natsume/Documents/paddy-disease-classification/sample_submission.csv')]



### ht: data_prep reproducibility in training a model

Now we can import the stuff we'll need from fastai, set a seed (for reproducibility -- just for the purposes of making this notebook easier to write; I don't recommend doing that in your own analysis however) and check what's in the data:


```
# hts
```


```
set_seed(42)
```

## ht: data - access dataset

### ht: data_access - map subfolders content with `check_subfolders_img`

use `path.ls()` and `check_subfolders_img(path)` to see what inside each subfolders


```
path.ls()
```




    (#4) [Path('/Users/Natsume/Documents/paddy-disease-classification/test_images'),Path('/Users/Natsume/Documents/paddy-disease-classification/train.csv'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images'),Path('/Users/Natsume/Documents/paddy-disease-classification/sample_submission.csv')]



### src: check_subfolders_img(path, db=False)


```
#| export utils 
from fastai.data.transforms import image_extensions
```


```
#| export utils
# @snoop
def check_subfolders_img(path, db=False):
    from pathlib import Path
    for entry in path.iterdir():
        if entry.is_file():
            print(f'{str(entry.absolute())}')
    addup = 0
    for entry in path.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and len(entry.ls(file_exts=image_extensions)) > 5:
            addup += len(entry.ls(file_exts=image_extensions))
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
            check_subfolders_img(entry)
    print(f"addup num: {addup}")
```


```
check_subfolders_img(path)
```

    /Users/Natsume/Documents/paddy-disease-classification/train.csv
    /Users/Natsume/Documents/paddy-disease-classification/sample_submission.csv
    /Users/Natsume/Documents/paddy-disease-classification: 3469  test_images
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1442  dead_heart
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 337  bacterial_panicle_blight
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 479  bacterial_leaf_blight
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 965  brown_spot
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1594  hispa
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 620  downy_mildew
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1738  blast
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1764  normal
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 380  bacterial_leaf_streak
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1088  tungro
    addup num: 10407
    addup num: 3469


### ht: data_access - extract all images for test and train with `get_image_files`


```
test_files = get_image_files(path/"test_images")
train_files = get_image_files(path/"train_images")
```


```
test_files
train_files
```




    (#3469) [Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/202919.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/200868.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/200698.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/200840.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/201586.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/203391.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/202931.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/202925.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/203385.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/200854.jpg')...]






    (#10407) [Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/110369.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/105002.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/106279.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/108254.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/104308.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/107629.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/110355.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/100146.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/103329.jpg'),Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/105980.jpg')...]




```
# fastnbs("src: get_image_files")
```

...and take a look at one:

### ht: data_access - display an image from test_files or train_files with `randomdisplay`

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
snoopon()
```


```
#| export utils
# @snoop
def randomdisplay(path, size=128, db=False):
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




    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_68_0.png)
    






    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_68_1.png)
    






    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_68_2.png)
    




```
snoopoff()
```

### ht: data_prep - remove images that fail to open with `remove_failed(path)`

#### why must remove failed images

images failed to open must be removed, otherwise it will cause errors during training.


```
# fastnbs("remove_failed")
# verify_images??
```


```
remove_failed(path)
```

    before running remove_failed:
    /Users/Natsume/Documents/paddy-disease-classification/train.csv
    /Users/Natsume/Documents/paddy-disease-classification/sample_submission.csv
    /Users/Natsume/Documents/paddy-disease-classification: 3469  test_images
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1442  dead_heart
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 337  bacterial_panicle_blight
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 479  bacterial_leaf_blight
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 965  brown_spot
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1594  hispa
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 620  downy_mildew
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1738  blast
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1764  normal
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 380  bacterial_leaf_streak
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1088  tungro
    addup num: 10407
    addup num: 3469
    total num: 13876
    num offailed: 0
    
    after running remove_failed:
    /Users/Natsume/Documents/paddy-disease-classification/train.csv
    /Users/Natsume/Documents/paddy-disease-classification/sample_submission.csv
    /Users/Natsume/Documents/paddy-disease-classification: 3469  test_images
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1442  dead_heart
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 337  bacterial_panicle_blight
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 479  bacterial_leaf_blight
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 965  brown_spot
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1594  hispa
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 620  downy_mildew
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1738  blast
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1764  normal
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 380  bacterial_leaf_streak
    /Users/Natsume/Documents/paddy-disease-classification/train_images: 1088  tungro
    addup num: 10407
    addup num: 3469


### ht: data_prep - describe sizes of all images with `check_sizes_img`

Looks like the images might be 480x640 -- let's check all their sizes. This is faster if we do it in parallel, so we'll use fastcore's `parallel` for this:


```
PILImage
```




    fastai.vision.core.PILImage



### src: check_sizes_img(files)



```
#| export utils
def f(o, sz=None): 
    im = None
    if sz and PILImage.create(o).size == sz:
        im = PILImage.create(o).to_thumb(500)
    return PILImage.create(o).size, im
```


```
# from fastcore.parallel import parallel
# doc(parallel)
```


```
#| export utils 
from fastcore.meta import delegates
```


```
#| export utils
# @snoop
@delegates(f)
def check_sizes_img(files, **kwargs):
    "use fastcore.parallel to quickly find out the different sizes of all images and their occurences. \
    output images with specific sizes if specified in `sz`"
    from fastcore.parallel import parallel
    res = parallel(f, files, n_workers=8, **kwargs) # add sz as a keyword to parallel
    sizes = [size for size, im in res]
    imgs = [im for size, im in res if im != None]
    pp(pd.Series(sizes).value_counts())  
    pp(imgs)
    if len(imgs):
        for im in imgs:
            im.to_thumb(125).show()
    return imgs
```


```
check_sizes_img(train_files)
```


```
imgs = check_sizes_img(test_files, sz = (640, 480))
```

### qt: how to display a list of images?


```
imgs[0] 
imgs[1]

```

### ht: data_loaders - create a dataloader from a folder with `ImageDataLoaders.from_folder`

#### qt: why must all images have the same dimensions?

They're nearly all the same size, except for a few. Because of those few, however, we'll need to make sure we always resize each image to common dimensions first, otherwise fastai won't be able to create batches. 

For now, we'll just squish them to 480x480 images, and then once they're in batches we do a random resized crop down to a smaller size, along with the other default fastai augmentations provided by `aug_transforms`. 

We'll start out with small resized images, since we want to be able to iterate quickly:


```
# fastnbs("src: ImageDataLoaders")
# ImageDataLoaders??
```

### doc: ImageDataLoaders.from_folder

To create a DataLoader obj from a folder. 

eg., give it `trn_path` (folder has subfolders like train, test even valid), `valid_pct` (split a portion for validation set), `seed` (set a seed for reproducibility), `item_tfms` (do transforms to each item), and `batch_tfms` (do transformations on batches)

```python
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)
```

```python
@classmethod
@delegates(DataLoaders.from_dblock)
def from_folder(cls:ImageDataLoaders, path, train='train', valid='valid', valid_pct=None, seed=None, vocab=None, item_tfms=None,
                batch_tfms=None, img_cls=PILImage, **kwargs):
    "Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)"
    # get the splitter function to split training and validation sets
    splitter = GrandparentSplitter(train_name=train, valid_name=valid) if valid_pct is None else RandomSplitter(valid_pct, seed=seed)
    # get the function to extract image files from using get_image_files in different spices
    get_items = get_image_files if valid_pct else partial(get_image_files, folders=[train, valid])
    # create a DataBlock object to organise all the data processing functions or callbacks
    dblock = DataBlock(blocks=(ImageBlock(img_cls), CategoryBlock(vocab=vocab)),
                       get_items=get_items,
                       splitter=splitter,
                       get_y=parent_label,
                       item_tfms=item_tfms,
                       batch_tfms=batch_tfms)
    # return a dataloaders created from the given DataBlock object above calling DataBlock.dataloaders
    return cls.from_dblock(dblock, path, path=path, **kwargs)
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/data.py
# Type:      method
```


```
show_doc(ImageDataLoaders.from_folder)
```

### src: ImageDataLoaders.from_folder


```
fu.snoopon()
```


```
from __future__ import annotations # to ensure path:str|Path='.' can work
```


```
class ImageDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for computer vision problems"
    @classmethod
    @snoop
    @delegates(DataLoaders.from_dblock)
    def from_folder(cls, path, train='train', valid='valid', valid_pct=None, seed=None, vocab=None, item_tfms=None,
                    batch_tfms=None, **kwargs):
        "Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)"
        splitter = GrandparentSplitter(train_name=train, valid_name=valid) if valid_pct is None else RandomSplitter(valid_pct, seed=seed)
        get_items = get_image_files if valid_pct else partial(get_image_files, folders=[train, valid])
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
                           get_items=get_items,
                           splitter=splitter,
                           get_y=parent_label,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        pp(doc_sig(cls.from_dblock))
        pp(inspect.getsource(cls.from_dblock))
        return cls.from_dblock(dblock, path, path=path, **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_path_func(cls, path, fnames, label_func, valid_pct=0.2, seed=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from list of `fnames` in `path`s with `label_func`"
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           splitter=RandomSplitter(valid_pct, seed=seed),
                           get_y=label_func,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, fnames, path=path, **kwargs)

    @classmethod
    def from_name_func(cls,
        path:str|Path, # Set the default path to a directory that a `Learner` can use to save files like models
        fnames:list, # A list of `os.Pathlike`'s to individual image files
        label_func:callable, # A function that receives a string (the file name) and outputs a label
        **kwargs
    ) -> DataLoaders:
        "Create from the name attrs of `fnames` in `path`s with `label_func`"
        if sys.platform == 'win32' and isinstance(label_func, types.LambdaType) and label_func.__name__ == '<lambda>':
            # https://medium.com/@jwnx/multiprocessing-serialization-in-python-with-pickle-9844f6fa1812
            raise ValueError("label_func couldn't be lambda function on Windows")
        f = using_attr(label_func, 'name')
        return cls.from_path_func(path, fnames, f, **kwargs)

    @classmethod
    def from_path_re(cls, path, fnames, pat, **kwargs):
        "Create from list of `fnames` in `path`s with re expression `pat`"
        return cls.from_path_func(path, fnames, RegexLabeller(pat), **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_name_re(cls, path, fnames, pat, **kwargs):
        "Create from the name attrs of `fnames` in `path`s with re expression `pat`"
        return cls.from_name_func(path, fnames, RegexLabeller(pat), **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None,
                y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from `df` using `fn_col` and `label_col`"
        pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
        if y_block is None:
            is_multi = (is_listy(label_col) and len(label_col) > 1) or label_delim is not None
            y_block = MultiCategoryBlock if is_multi else CategoryBlock
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)
        dblock = DataBlock(blocks=(ImageBlock, y_block),
                           get_x=ColReader(fn_col, pref=pref, suff=suff),
                           get_y=ColReader(label_col, label_delim=label_delim),
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, df, path=path, **kwargs)

    @classmethod
    def from_csv(cls, path, csv_fname='labels.csv', header='infer', delimiter=None, **kwargs):
        "Create from `path/csv_fname` using `fn_col` and `label_col`"
        df = pd.read_csv(Path(path)/csv_fname, header=header, delimiter=delimiter)
        return cls.from_df(df, path=path, **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_lists(cls, path, fnames, labels, valid_pct=0.2, seed:int=None, y_block=None, item_tfms=None, batch_tfms=None,
                   **kwargs):
        "Create from list of `fnames` and `labels` in `path`"
        if y_block is None:
            y_block = MultiCategoryBlock if is_listy(labels[0]) and len(labels[0]) > 1 else (
                RegressionBlock if isinstance(labels[0], float) else CategoryBlock)
        dblock = DataBlock.from_columns(blocks=(ImageBlock, y_block),
                           splitter=RandomSplitter(valid_pct, seed=seed),
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, (fnames, labels), path=path, **kwargs)
# File:           ~/mambaforge/lib/python3.9/site-packages/fastai/vision/data.py
# Type:           type
# Subclasses:     
```


```
trn_path = path/"train_images"
# path.ls()
```


```
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)
```


```
fu.snoopoff()
```

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

`lr_find` generally recommends rather conservative learning rates, to ensure that your model will train successfully. I generally like to push it a bit higher if I can. Let's train a few epochs and see how it looks:


```
learn.fine_tune(3, 0.01)
```

We're now ready to build our first submission. Let's take a look at the sample Kaggle provided to see what it needs to look like:

## Submitting to Kaggle

### how to check the kaggle submission sample csv file


```
ss = pd.read_csv(path/'sample_submission.csv')
ss
```

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

### how to access all the classes of labels with dataloaders

These need to be mapped to the names of each of these diseases, these names are stored by fastai automatically in the `vocab`:


```
dls.vocab
```

### how to map classes to each idx from the predictions

We can create an apply this mapping using pandas:


```
mapping = dict(enumerate(dls.vocab))
results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
results
```

### how to save result into csv file

Kaggle expects the submission as a CSV file, so let's save it, and check the first few lines:


```
ss['label'] = results
ss.to_csv('subm.csv', index=False)
!head subm.csv
```

### how to submit to kaggle with fastkaggle api

Let's submit this to kaggle. We can do it from the notebook if we're running on Kaggle, otherwise we can use the API:


```
if not iskaggle:
    from kaggle import api
    api.competition_submit_cli('subm.csv', 'initial rn26d 128px', comp)
```

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


```

```


```
from fastdebug.utils import *
```


```
nb_name()
```


```
ipy2md()
```


```
fastnbs("push kaggle")
```
