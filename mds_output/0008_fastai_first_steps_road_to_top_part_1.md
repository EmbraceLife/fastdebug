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

    22:17:06.96 >>> Call to setup_comp in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_47000/1633869195.py", line 2
    22:17:06.96 ...... competition = 'paddy-disease-classification'
    22:17:06.96 ...... local_folder = '/Users/Natsume/Documents/'
    22:17:06.96 ...... install = 'fastai "timm>=0.6.2.dev0"'
    22:17:06.96    2 | def setup_comp(competition, local_folder='', install=''):
    22:17:06.96    4 |     if iskaggle:
    22:17:06.96    9 |         path = Path(local_folder + competition)
    22:17:06.96 .............. path = Path('/Users/Natsume/Documents/paddy-disease-classification')
    22:17:06.96   10 |         api = import_kaggle()
    22:17:06.98 .............. api = <kaggle.api.kaggle_api_extended.KaggleApi object>
    22:17:06.98   11 |         if not path.exists():
    22:17:06.98   17 |         return path
    22:17:06.98 <<< Return value from setup_comp: Path('/Users/Natsume/Documents/paddy-disease-classification')



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

    22:17:07.52 LOG:
    22:17:07.67 .... file = Path('/Users/Natsume/Documents/paddy-disease-classification/test_images/201216.jpg')





    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_68_1.png)
    



    22:17:07.67 LOG:
    22:17:07.68 .... file = Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/bacterial_leaf_blight/104518.jpg')





    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_68_3.png)
    



    22:17:07.69 LOG:
    22:17:07.69 .... file = Path('/Users/Natsume/Documents/paddy-disease-classification/train_images/dead_heart/100966.jpg')





    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_68_5.png)
    




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




    []




```
imgs = check_sizes_img(test_files, sz = (640, 480))
```

### qt: how to display a list of images?


```
imgs[0] 
imgs[1]

```




    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_86_0.png)
    






    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_86_1.png)
    



### ht: data_loaders - create a dataloader from a folder with `ImageDataLoaders.from_folder`

A dataloader prepares training and validation sets, transformations to each image, and batch transformations to each batch of images

#### qt: why must all images have the same dimensions? how to resolve this problem?

They're nearly all the same size, except for a few. Because of those few, however, we'll need to make sure we always resize each image to common dimensions first, otherwise fastai won't be able to create batches. 

For now, we'll just squish them to 480x480 images, and then once they're in batches we do a random resized crop down to a smaller size, along with the other default fastai augmentations provided by `aug_transforms`. 

#### qt: why should we start with small resized images
We'll start out with small resized images, since we want to be able to iterate quickly:


```
# fastnbs("src: ImageDataLoaders")
# ImageDataLoaders??
```

### doc: ImageDataLoaders.from_folder

To create a DataLoader obj from a folder, and a dataloader prepares functions for splitting training and validation sets, extracting images and labels, each item transformations, and batch transformations.

eg., give it `trn_path` (folder has subfolders like train, test or even valid), `valid_pct` (split a portion from train to create validation set), `seed` (set a seed for reproducibility), `item_tfms` (do transforms to each item), and `batch_tfms` (do transformations on batches)

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




---

[source](https://github.com/fastai/fastai/blob/master/fastai/vision/data.py#LNone){target="_blank" style="float:right; font-size:smaller"}

### ImageDataLoaders.from_folder

>      ImageDataLoaders.from_folder (path, train='train', valid='valid',
>                                    valid_pct=None, seed=None, vocab=None,
>                                    item_tfms=None, batch_tfms=None,
>                                    img_cls=<class
>                                    'fastai.vision.core.PILImage'>, bs:int=64,
>                                    val_bs:int=None, shuffle:bool=True,
>                                    device=None)

Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)

|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| path | str \| Path | . | Path to put in `DataLoaders` |
| train | str | train |  |
| valid | str | valid |  |
| valid_pct | NoneType | None |  |
| seed | NoneType | None |  |
| vocab | NoneType | None |  |
| item_tfms | NoneType | None |  |
| batch_tfms | NoneType | None |  |
| img_cls | BypassNewMeta | PILImage |  |
| bs | int | 64 | Size of batch |
| val_bs | int | None | Size of batch for validation `DataLoader` |
| shuffle | bool | True | Whether to shuffle data |
| device | NoneType | None | Device to put `DataLoaders` |



### src: ImageDataLoaders.from_folder


```
snoopon()
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

    22:17:27.98 >>> Call to ImageDataLoaders.from_folder in File "/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/ipykernel_47000/3243071519.py", line 6
    22:17:27.98 .......... cls = <class '__main__.ImageDataLoaders'>
    22:17:27.98 .......... path = Path('/Users/Natsume/Documents/paddy-disease-classification/train_images')
    22:17:27.98 .......... train = 'train'
    22:17:27.98 .......... valid = 'valid'
    22:17:27.98 .......... valid_pct = 0.2
    22:17:27.98 .......... seed = 42
    22:17:27.98 .......... vocab = None
    22:17:27.98 .......... item_tfms = Resize -- {'size': (480, 480), 'method': 'squish...encodes
    22:17:27.98                        (TensorPoint,object) -> encodes
    22:17:27.98                        decodes: 
    22:17:27.98 .......... batch_tfms = [Flip -- {'size': None, 'mode': 'bilinear', 'pad_...encodes
    22:17:27.98                         (TensorPoint,object) -> encodes
    22:17:27.98                         decodes: , Brightness -- {'max_lighting': 0.2, 'p': 1.0, 'd...ncodes: (TensorImage,object) -> encodes
    22:17:27.98                         decodes: , RandomResizedCropGPU -- {'size': (128, 128), 'mi... encodes
    22:17:27.98                         (TensorMask,object) -> encodes
    22:17:27.98                         decodes: ]
    22:17:27.98 .......... len(batch_tfms) = 3
    22:17:27.98 .......... kwargs = {}
    22:17:27.98    6 |     def from_folder(cls, path, train='train', valid='valid', valid_pct=None, seed=None, vocab=None, item_tfms=None,
    22:17:27.98    9 |         splitter = GrandparentSplitter(train_name=train, valid_name=valid) if valid_pct is None else RandomSplitter(valid_pct, seed=seed)
    22:17:27.98 .............. splitter = <function RandomSplitter.<locals>._inner>
    22:17:27.98   10 |         get_items = get_image_files if valid_pct else partial(get_image_files, folders=[train, valid])
    22:17:27.98 .............. get_items = <function get_image_files>
    22:17:27.98   11 |         dblock = DataBlock(blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
    22:17:27.98   12 |                            get_items=get_items,
    22:17:27.98   13 |                            splitter=splitter,
    22:17:27.98   14 |                            get_y=parent_label,
    22:17:27.98   15 |                            item_tfms=item_tfms,
    22:17:27.98   16 |                            batch_tfms=batch_tfms)
    22:17:27.98   11 |         dblock = DataBlock(blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
    22:17:27.98 .............. dblock = <fastai.data.block.DataBlock object>
    22:17:27.98   17 |         pp(doc_sig(cls.from_dblock))
    22:17:27.98 LOG:
    22:17:28.03 .... doc_sig(cls.from_dblock) = ('no mro',
    22:17:28.03                                  'Create a dataloaders from a given `dblock`',
    22:17:28.03                                  <Signature (dblock, source, path: 'str | Path' = '.', bs: 'int' = 64, val_bs: 'int' = None, shuffle: 'bool' = True, device=None, **kwargs)>)
    22:17:28.03   18 |         pp(inspect.getsource(cls.from_dblock))
    22:17:28.03 LOG:
    22:17:28.03 .... inspect.getsource(cls.from_dblock) = ('    @classmethod\n'
    22:17:28.03                                            '    def from_dblock(cls, \n'
    22:17:28.03                                            '        dblock, # `DataBlock` object\n'
    22:17:28.03                                            '        source, # Source of data. Can be `Path` to files\n'
    22:17:28.03                                            "        path:str|Path='.', # Path to put in `DataLoaders`\n"
    22:17:28.03                                            '        bs:int=64, # Size of batch\n'
    22:17:28.03                                            '        val_bs:int=None, # Size of batch for validation `DataLoader`\n'
    22:17:28.03                                            '        shuffle:bool=True, # Whether to shuffle data\n'
    22:17:28.03                                            '        device=None, # Device to put `DataLoaders`\n'
    22:17:28.03                                            '        **kwargs\n'
    22:17:28.03                                            '    ):\n'
    22:17:28.03                                            '        return dblock.dataloaders(source, path=path, bs=bs, val_bs=val_bs, '
    22:17:28.03                                            'shuffle=shuffle, device=device, **kwargs)\n')
    22:17:28.03   19 |         return cls.from_dblock(dblock, path, path=path, **kwargs)
    22:17:29.06 <<< Return value from ImageDataLoaders.from_folder: <fastai.data.core.DataLoaders object>



    
![png](0008_fastai_first_steps_road_to_top_part_1_files/0008_fastai_first_steps_road_to_top_part_1_102_1.png)
    



```
snoopoff()
```

### ht: data_loaders - apply transformations to each image with `item_tfms = Resize(480, method='squish')`

Besides gathering all the images, dataloaders need to prepare transformations eg., resize to each image. This is done with `item_tfms = Resize(480, method='squish')`

### ht: data_loaders - apply image augmentations to each batch of data with `batch_tfms = aug_transforms`

After transformations done to each image, we can apply image augmentations like flip, rotate, zoom, wrap, lighting to each batch of images

### doc: aug_transforms(size=128, min_scale=0.75)

to return a list of image augmentations (transformations) for doing custom flip, rotate, zoom, warp, lighting


```
# The official doc is easy to understand and the explanation to each arg is very helpful.
show_doc(aug_transforms)
```




---

[source](https://github.com/fastai/fastai/blob/master/fastai/vision/augment.py#LNone){target="_blank" style="float:right; font-size:smaller"}

### aug_transforms

>      aug_transforms (mult:float=1.0, do_flip:bool=True, flip_vert:bool=False,
>                      max_rotate:float=10.0, min_zoom:float=1.0,
>                      max_zoom:float=1.1, max_lighting:float=0.2,
>                      max_warp:float=0.2, p_affine:float=0.75,
>                      p_lighting:float=0.75, xtra_tfms:list=None,
>                      size:Union[int,tuple]=None, mode:str='bilinear',
>                      pad_mode='reflection', align_corners=True, batch=False,
>                      min_scale=1.0)

Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms.

|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| mult | float | 1.0 | Multiplication applying to `max_rotate`,`max_lighting`,`max_warp` |
| do_flip | bool | True | Random flipping |
| flip_vert | bool | False | Flip vertically |
| max_rotate | float | 10.0 | Maximum degree of rotation |
| min_zoom | float | 1.0 | Minimum zoom |
| max_zoom | float | 1.1 | Maximum zoom |
| max_lighting | float | 0.2 | Maximum scale of changing brightness |
| max_warp | float | 0.2 | Maximum value of changing warp per |
| p_affine | float | 0.75 | Probability of applying affine transformation |
| p_lighting | float | 0.75 | Probability of changing brightnest and contrast |
| xtra_tfms | list | None | Custom Transformations |
| size | int \| tuple | None | Output size, duplicated if one value is specified |
| mode | str | bilinear | PyTorch `F.grid_sample` interpolation |
| pad_mode | str | reflection | A `PadMode` |
| align_corners | bool | True | PyTorch `F.grid_sample` align_corners |
| batch | bool | False | Apply identical transformation to entire batch |
| min_scale | float | 1.0 | Minimum scale of the crop, in relation to image area |




```
src(aug_transforms)
```

    def aug_transforms(
        mult:float=1.0, # Multiplication applying to `max_rotate`,`max_lighting`,`max_warp`
        do_flip:bool=True, # Random flipping
        flip_vert:bool=False, # Flip vertically
        max_rotate:float=10., # Maximum degree of rotation
        min_zoom:float=1., # Minimum zoom 
        max_zoom:float=1.1, # Maximum zoom 
        max_lighting:float=0.2, # Maximum scale of changing brightness 
        max_warp:float=0.2, # Maximum value of changing warp per
        p_affine:float=0.75, # Probability of applying affine transformation
        p_lighting:float=0.75, # Probability of changing brightnest and contrast 
        xtra_tfms:list=None, # Custom Transformations
        size:int|tuple=None, # Output size, duplicated if one value is specified
        mode:str='bilinear', # PyTorch `F.grid_sample` interpolation
        pad_mode=PadMode.Reflection, # A `PadMode`
        align_corners=True, # PyTorch `F.grid_sample` align_corners
        batch=False, # Apply identical transformation to entire batch
        min_scale=1. # Minimum scale of the crop, in relation to image area
    ):
        "Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms."
        res,tkw = [],dict(size=size if min_scale==1. else None, mode=mode, pad_mode=pad_mode, batch=batch, align_corners=align_corners)
        max_rotate,max_lighting,max_warp = array([max_rotate,max_lighting,max_warp])*mult
        if do_flip: res.append(Dihedral(p=0.5, **tkw) if flip_vert else Flip(p=0.5, **tkw))
        if max_warp:   res.append(Warp(magnitude=max_warp, p=p_affine, **tkw))
        if max_rotate: res.append(Rotate(max_deg=max_rotate, p=p_affine, **tkw))
        if min_zoom<1 or max_zoom>1: res.append(Zoom(min_zoom=min_zoom, max_zoom=max_zoom, p=p_affine, **tkw))
        if max_lighting:
            res.append(Brightness(max_lighting=max_lighting, p=p_lighting, batch=batch))
            res.append(Contrast(max_lighting=max_lighting, p=p_lighting, batch=batch))
        if min_scale!=1.: xtra_tfms = RandomResizedCropGPU(size, min_scale=min_scale, ratio=(1,1)) + L(xtra_tfms)
        return setup_aug_tfms(res + L(xtra_tfms))
    


### src: aug_transforms(size=128, min_scale=0.75)


```
snoopon()
```


```
# @snoop
def aug_transforms(
    mult:float=1.0, # Multiplication applying to `max_rotate`,`max_lighting`,`max_warp`
    do_flip:bool=True, # Random flipping
    flip_vert:bool=False, # Flip vertically
    max_rotate:float=10., # Maximum degree of rotation
    min_zoom:float=1., # Minimum zoom 
    max_zoom:float=1.1, # Maximum zoom 
    max_lighting:float=0.2, # Maximum scale of changing brightness 
    max_warp:float=0.2, # Maximum value of changing warp per
    p_affine:float=0.75, # Probability of applying affine transformation
    p_lighting:float=0.75, # Probability of changing brightnest and contrast 
    xtra_tfms:list=None, # Custom Transformations
    size:int|tuple=None, # Output size, duplicated if one value is specified
    mode:str='bilinear', # PyTorch `F.grid_sample` interpolation
    pad_mode=PadMode.Reflection, # A `PadMode`
    align_corners=True, # PyTorch `F.grid_sample` align_corners
    batch=False, # Apply identical transformation to entire batch
    min_scale=1. # Minimum scale of the crop, in relation to image area
):
    "Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms."
    res,tkw = [],dict(size=size if min_scale==1. else None, mode=mode, pad_mode=pad_mode, batch=batch, align_corners=align_corners)
    max_rotate,max_lighting,max_warp = array([max_rotate,max_lighting,max_warp])*mult
    if do_flip: res.append(Dihedral(p=0.5, **tkw) if flip_vert else Flip(p=0.5, **tkw))
    if max_warp:   res.append(Warp(magnitude=max_warp, p=p_affine, **tkw))
    if max_rotate: res.append(Rotate(max_deg=max_rotate, p=p_affine, **tkw))
    if min_zoom<1 or max_zoom>1: res.append(Zoom(min_zoom=min_zoom, max_zoom=max_zoom, p=p_affine, **tkw))
    if max_lighting:
        res.append(Brightness(max_lighting=max_lighting, p=p_lighting, batch=batch))
        res.append(Contrast(max_lighting=max_lighting, p=p_lighting, batch=batch))
    if min_scale!=1.: xtra_tfms = RandomResizedCropGPU(size, min_scale=min_scale, ratio=(1,1)) + L(xtra_tfms)
    pp(res, L(xtra_tfms), doc_sig(setup_aug_tfms))
    return setup_aug_tfms(res + L(xtra_tfms))
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/augment.py
# Type:      function
```


```
aug_transforms(size=128, min_scale=0.75)
```

    22:17:30.03 LOG:
    22:17:30.04 .... res = [Flip -- {'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True, 'p': 0.5}:
    22:17:30.04            encodes: (TensorImage,object) -> encodes
    22:17:30.04            (TensorMask,object) -> encodes
    22:17:30.04            (TensorBBox,object) -> encodes
    22:17:30.04            (TensorPoint,object) -> encodes
    22:17:30.04            decodes: ,
    22:17:30.04             Warp -- {'magnitude': 0.2, 'p': 1.0, 'draw_x': None, 'draw_y': None, 'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'batch': False, 'align_corners': True, 'mode_mask': 'nearest'}:
    22:17:30.04            encodes: (TensorImage,object) -> encodes
    22:17:30.04            (TensorMask,object) -> encodes
    22:17:30.04            (TensorBBox,object) -> encodes
    22:17:30.04            (TensorPoint,object) -> encodes
    22:17:30.04            decodes: ,
    22:17:30.04             Rotate -- {'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True, 'p': 1.0}:
    22:17:30.04            encodes: (TensorImage,object) -> encodes
    22:17:30.04            (TensorMask,object) -> encodes
    22:17:30.04            (TensorBBox,object) -> encodes
    22:17:30.04            (TensorPoint,object) -> encodes
    22:17:30.04            decodes: ,
    22:17:30.04             Zoom -- {'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True, 'p': 1.0}:
    22:17:30.04            encodes: (TensorImage,object) -> encodes
    22:17:30.04            (TensorMask,object) -> encodes
    22:17:30.04            (TensorBBox,object) -> encodes
    22:17:30.04            (TensorPoint,object) -> encodes
    22:17:30.04            decodes: ,
    22:17:30.04             Brightness -- {'max_lighting': 0.2, 'p': 1.0, 'draw': None, 'batch': False}:
    22:17:30.04            encodes: (TensorImage,object) -> encodes
    22:17:30.04            decodes: ,
    22:17:30.04             Contrast -- {'max_lighting': 0.2, 'p': 1.0, 'draw': None, 'batch': False}:
    22:17:30.04            encodes: (TensorImage,object) -> encodes
    22:17:30.04            decodes: ]
    22:17:30.04 .... L(xtra_tfms) = [RandomResizedCropGPU -- {'size': (128, 128), 'min_scale': 0.75, 'ratio': (1, 1), 'mode': 'bilinear', 'valid_scale': 1.0, 'max_scale': 1.0, 'mode_mask': 'nearest', 'p': 1.0}:
    22:17:30.04                     encodes: (TensorImage,object) -> encodes
    22:17:30.04                     (TensorBBox,object) -> encodes
    22:17:30.04                     (TensorPoint,object) -> encodes
    22:17:30.04                     (TensorMask,object) -> encodes
    22:17:30.04                     decodes: ]
    22:17:30.04 .... doc_sig(setup_aug_tfms) = ('no mro',
    22:17:30.04                                 'Go through `tfms` and combines together affine/coord or lighting transforms',
    22:17:30.04                                 <Signature (tfms)>)





    [Flip -- {'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True, 'p': 0.5}:
     encodes: (TensorImage,object) -> encodes
     (TensorMask,object) -> encodes
     (TensorBBox,object) -> encodes
     (TensorPoint,object) -> encodes
     decodes: ,
     Brightness -- {'max_lighting': 0.2, 'p': 1.0, 'draw': None, 'batch': False}:
     encodes: (TensorImage,object) -> encodes
     decodes: ,
     RandomResizedCropGPU -- {'size': (128, 128), 'min_scale': 0.75, 'ratio': (1, 1), 'mode': 'bilinear', 'valid_scale': 1.0, 'max_scale': 1.0, 'mode_mask': 'nearest', 'p': 1.0}:
     encodes: (TensorImage,object) -> encodes
     (TensorBBox,object) -> encodes
     (TensorPoint,object) -> encodes
     (TensorMask,object) -> encodes
     decodes: ]




```
snoopoff()
```

## Our first model

### ht: learner - model arch - how to pick the first to try

Let's create a model. To pick an architecture, we should look at the options in [The best vision models for fine-tuning](https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning). I like the looks of `resnet26d`, which is the fastest resolution-independent model which gets into the top-15 lists there.


```
# !pip install "timm>=0.6.2.dev0"
```

### ht: learner - vision_learner - build a learner for vision

### doc: vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()

Give `vision_learner` a dataloader and a model architecture string, it returns a learner which create the specified model object and put model, dls together for handling training for vision problems

Besides dataloader and model arch, a learners prepares a lot of things like loss func, opt func, lr, splitter, cbs, metrics, weight_decay, batch_norm, etc.

How to fill in the details description of the args of `vision_learner`? By exploring the source?


```
show_doc(vision_learner)
```




---

[source](https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py#LNone){target="_blank" style="float:right; font-size:smaller"}

### vision_learner

>      vision_learner (dls, arch, normalize=True, n_out=None, pretrained=True,
>                      loss_func=None, opt_func=<function Adam>, lr=0.001,
>                      splitter=None, cbs=None, metrics=None, path=None,
>                      model_dir='models', wd=None, wd_bn_bias=False,
>                      train_bn=True, moms=(0.95, 0.85, 0.95), cut=None,
>                      init=<function kaiming_normal_>, custom_head=None,
>                      concat_pool=True, pool=True, lin_ftrs=None, ps=0.5,
>                      first_bn=True, bn_final=False, lin_first=False,
>                      y_range=None, n_in=3)

Build a vision learner from `dls` and `arch`

|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| dls |  |  |  |
| arch |  |  |  |
| normalize | bool | True |  |
| n_out | NoneType | None |  |
| pretrained | bool | True |  |
| loss_func | NoneType | None |  |
| opt_func | function | Adam |  |
| lr | float | 0.001 |  |
| splitter | NoneType | None |  |
| cbs | NoneType | None |  |
| metrics | NoneType | None |  |
| path | NoneType | None | learner args |
| model_dir | str | models |  |
| wd | NoneType | None |  |
| wd_bn_bias | bool | False |  |
| train_bn | bool | True |  |
| moms | tuple | (0.95, 0.85, 0.95) |  |
| cut | NoneType | None |  |
| init | function | kaiming_normal_ |  |
| custom_head | NoneType | None |  |
| concat_pool | bool | True |  |
| pool | bool | True | model & head args |
| lin_ftrs | NoneType | None |  |
| ps | float | 0.5 |  |
| first_bn | bool | True |  |
| bn_final | bool | False |  |
| lin_first | bool | False |  |
| y_range | NoneType | None |  |
| n_in | int | 3 |  |



```python
@snoop
@delegates(create_vision_model)
def vision_learner(dls, arch, normalize=True, n_out=None, pretrained=True, 
        # learner args
        loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
        model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95), # wd = weight_decay, bn = batch_norm, momentum
        # model & head args
        cut=None, init=nn.init.kaiming_normal_, custom_head=None, concat_pool=True, pool=True, # not sure about cut, lin_ftrs, ps
        lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None, **kwargs):
    "Build a vision learner from `dls` and `arch`"
    # get the num of output from dls if not given
    if n_out is None: n_out = get_c(dls)
    # n_out should be extracted without error from dls.c or the dataset
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    # get arch's _default_meta info such as "cut" value and "split" function
    meta = model_meta.get(arch, _default_meta)
    # customize arg values for the model's instantiation and put them into a dict 
    model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
                      first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    # set n_in to be 3 if not given by kwargs
    n_in = kwargs['n_in'] if 'n_in' in kwargs else 3
    # if arch is a string, then create the model from timm, and normalize the dataset if specified
    if isinstance(arch, str):
        # use timm to customize a model, and return a model object and its config info in a dict
        model,cfg = create_timm_model(arch, n_out, default_split, pretrained, **model_args)
        # use model's cfg mean and std to normalize data as a tfm during after_batch
        if normalize: _timm_norm(dls, cfg, pretrained, n_in)
    else:
        if normalize: _add_norm(dls, meta, pretrained, n_in)
        model = create_vision_model(arch, n_out, pretrained=pretrained, **model_args)
    # pick the splitter func to split train and validation set
    splitter = ifnone(splitter, meta['split'])
    # create a learner object which group dls, model, loss_func, opt_func, lr, splitter, cbs, metrics... to handle training
    learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)
    # if the model is pretrained, freeze the model to the last parameter group
    if pretrained: learn.freeze()
    # keep track of args for loggers
    store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    return learn
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/learner.py
# Type:      function
```

### src: vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()


```
snoopon()
```


```
from fastai.vision.learner import *
from fastai.vision.learner import  _default_meta, _add_norm, _timm_norm
from fastai.callback.fp16 import to_fp16
```


```
@snoop
@delegates(create_vision_model)
def vision_learner(dls, arch, normalize=True, n_out=None, pretrained=True, 
        # learner args
        loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=None, cbs=None, metrics=None, path=None,
        model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95,0.85,0.95),
        # model & head args
        cut=None, init=nn.init.kaiming_normal_, custom_head=None, concat_pool=True, pool=True,
        lin_ftrs=None, ps=0.5, first_bn=True, bn_final=False, lin_first=False, y_range=None, **kwargs):
    "Build a vision learner from `dls` and `arch`"
#     pp(src(get_c))
    # get the num of output from dls if not given
    if n_out is None: n_out = get_c(dls)
    # n_out should be extracted without error from dls.c or the dataset
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    pp(arch, _default_meta, doc_sig(model_meta.get))
    # get arch's _default_meta info such as "cut" value and "split" function
    meta = model_meta.get(arch, _default_meta)
    # customize arg values for the model's instantiation and put them into a dict 
    model_args = dict(init=init, custom_head=custom_head, concat_pool=concat_pool, pool=pool, lin_ftrs=lin_ftrs, ps=ps,
                      first_bn=first_bn, bn_final=bn_final, lin_first=lin_first, y_range=y_range, **kwargs)
    # set n_in to be 3 if not given by kwargs
    n_in = kwargs['n_in'] if 'n_in' in kwargs else 3
    # if arch is a string, then create the model from timm, and normalize the dataset if specified
    if isinstance(arch, str):
#         pp(doc_sig(create_timm_model))
        # use timm to customize a model, and return a model object and its config info in a dict
        model,cfg = create_timm_model(arch, n_out, default_split, pretrained, **model_args)
#         pp(doc_sig(_timm_norm))
#         pp(src(_timm_norm))
        # use model's cfg mean and std to normalize data as a tfm during after_batch
        if normalize: _timm_norm(dls, cfg, pretrained, n_in)
    else:
        if normalize: _add_norm(dls, meta, pretrained, n_in)
        model = create_vision_model(arch, n_out, pretrained=pretrained, **model_args)
    # pick the splitter func to split train and validation set
    splitter = ifnone(splitter, meta['split'])
#     pp(doc_sig(Learner))
    # create a learner object which group dls, model, loss_func, opt_func, lr, splitter, cbs, metrics... to handle training
    learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                   metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias, train_bn=train_bn, moms=moms)

#     pp(doc_sig(learn.freeze))
    # if the model is pretrained, freeze the model to the last parameter group
    if pretrained: learn.freeze()
    # keep track of args for loggers
    store_attr('arch,normalize,n_out,pretrained', self=learn, **kwargs)
    return learn
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/vision/learner.py
# Type:      function
```


```
learn = vision_learner(dls, 'resnet26d', metrics=error_rate, path='.')
learn = learn.to_fp16()
```


```
snoopoff()
```

### ht: learner - find learning rate with `lr_find`

Let's see what the learning rate finder shows:


```
show_doc(learn.lr_find)
```




---

[source](https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#LNone){target="_blank" style="float:right; font-size:smaller"}

### Learner.lr_find

>      Learner.lr_find (start_lr=1e-07, end_lr=10, num_it=100, stop_div=True,
>                       show_plot=True, suggest_funcs=<function valley>)

Launch a mock training to find a good learning rate and return suggestions based on `suggest_funcs` as a named tuple



```python
@patch
def lr_find(self:Learner, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, show_plot=True, suggest_funcs=(SuggestionMethod.Valley)):
    "Launch a mock training to find a good learning rate and return suggestions based on `suggest_funcs` as a named tuple"
    n_epoch = num_it//len(self.dls.train) + 1
    cb=LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
    with self.no_logging(): self.fit(n_epoch, cbs=cb)
    if suggest_funcs is not None:
        lrs, losses = tensor(self.recorder.lrs[num_it//10:-5]), tensor(self.recorder.losses[num_it//10:-5])
        nan_idxs = torch.nonzero(torch.isnan(losses.view(-1)))
        if len(nan_idxs) > 0:
            drop_idx = min(nan_idxs)
            lrs = lrs[:drop_idx]
            losses = losses[:drop_idx]
        _suggestions, nms = [], []
        for func in tuplify(suggest_funcs):
            nms.append(func.__name__ if not isinstance(func, partial) else func.func.__name__) # deal with partials
            _suggestions.append(func(lrs, losses, num_it))
        
        SuggestedLRs = collections.namedtuple('SuggestedLRs', nms)
        lrs, pnts = [], []
        for lr, pnt in _suggestions:
            lrs.append(lr)
            pnts.append(pnt)
        if show_plot: self.recorder.plot_lr_find(suggestions=pnts, nms=nms)
        return SuggestedLRs(*lrs)

    elif show_plot: self.recorder.plot_lr_find()
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/callback/schedule.py
# Type:      method
```


```
learn.lr_find(suggest_funcs=(valley, slide))
```

    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torch/amp/autocast_mode.py:198: UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
      warnings.warn('User provided device_type of \'cuda\', but CUDA is not available. Disabling')
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/torch/cuda/amp/grad_scaler.py:115: UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.
      warnings.warn("torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")




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



<div>
  <progress value='4' class='' max='130' style='width:300px; height:20px; vertical-align: middle;'></progress>
  3.08% [4/130 00:23&lt;12:32 3.8080]
</div>




    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Input In [96], in <cell line: 1>()
    ----> 1 learn.lr_find(suggest_funcs=(valley, slide))


    File ~/mambaforge/lib/python3.9/site-packages/fastai/callback/schedule.py:293, in lr_find(self, start_lr, end_lr, num_it, stop_div, show_plot, suggest_funcs)
        291 n_epoch = num_it//len(self.dls.train) + 1
        292 cb=LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
    --> 293 with self.no_logging(): self.fit(n_epoch, cbs=cb)
        294 if suggest_funcs is not None:
        295     lrs, losses = tensor(self.recorder.lrs[num_it//10:-5]), tensor(self.recorder.losses[num_it//10:-5])


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:256, in Learner.fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
        254 self.opt.set_hypers(lr=self.lr if lr is None else lr)
        255 self.n_epoch = n_epoch
    --> 256 self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:245, in Learner._do_fit(self)
        243 for epoch in range(self.n_epoch):
        244     self.epoch=epoch
    --> 245     self._with_events(self._do_epoch, 'epoch', CancelEpochException)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:239, in Learner._do_epoch(self)
        238 def _do_epoch(self):
    --> 239     self._do_epoch_train()
        240     self._do_epoch_validate()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:231, in Learner._do_epoch_train(self)
        229 def _do_epoch_train(self):
        230     self.dl = self.dls.train
    --> 231     self._with_events(self.all_batches, 'train', CancelTrainException)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:199, in Learner.all_batches(self)
        197 def all_batches(self):
        198     self.n_iter = len(self.dl)
    --> 199     for o in enumerate(self.dl): self.one_batch(*o)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:227, in Learner.one_batch(self, i, b)
        225 b = self._set_device(b)
        226 self._split(b)
    --> 227 self._with_events(self._do_one_batch, 'batch', CancelBatchException)


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:212, in Learner._do_one_batch(self)
        210 self('after_loss')
        211 if not self.training or not len(self.yb): return
    --> 212 self._with_events(self._backward, 'backward', CancelBackwardException)
        213 self._with_events(self._step, 'step', CancelStepException)
        214 self.opt.zero_grad()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:193, in Learner._with_events(self, f, event_type, ex, final)
        192 def _with_events(self, f, event_type, ex, final=noop):
    --> 193     try: self(f'before_{event_type}');  f()
        194     except ex: self(f'after_cancel_{event_type}')
        195     self(f'after_{event_type}');  final()


    File ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py:201, in Learner._backward(self)
    --> 201 def _backward(self): self.loss_grad.backward()


    File ~/mambaforge/lib/python3.9/site-packages/torch/_tensor.py:388, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        341 r"""Computes the gradient of current tensor w.r.t. graph leaves.
        342 
        343 The graph is differentiated using the chain rule. If the tensor is
       (...)
        385         used to compute the attr::tensors.
        386 """
        387 if has_torch_function_unary(self):
    --> 388     return handle_torch_function(
        389         Tensor.backward,
        390         (self,),
        391         self,
        392         gradient=gradient,
        393         retain_graph=retain_graph,
        394         create_graph=create_graph,
        395         inputs=inputs)
        396 torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)


    File ~/mambaforge/lib/python3.9/site-packages/torch/overrides.py:1498, in handle_torch_function(public_api, relevant_args, *args, **kwargs)
       1492     warnings.warn("Defining your `__torch_function__ as a plain method is deprecated and "
       1493                   "will be an error in future, please define it as a classmethod.",
       1494                   DeprecationWarning)
       1496 # Use `public_api` instead of `implementation` so __torch_function__
       1497 # implementations can do equality/identity comparisons.
    -> 1498 result = torch_func_method(public_api, types, args, kwargs)
       1500 if result is not NotImplemented:
       1501     return result


    File ~/mambaforge/lib/python3.9/site-packages/fastai/torch_core.py:378, in TensorBase.__torch_function__(cls, func, types, args, kwargs)
        376 if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
        377 if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
    --> 378 res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        379 dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        380 if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)


    File ~/mambaforge/lib/python3.9/site-packages/torch/_tensor.py:1121, in Tensor.__torch_function__(cls, func, types, args, kwargs)
       1118     return NotImplemented
       1120 with _C.DisableTorchFunction():
    -> 1121     ret = func(*args, **kwargs)
       1122     if func in get_default_nowrap_functions():
       1123         return ret


    File ~/mambaforge/lib/python3.9/site-packages/torch/_tensor.py:396, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        387 if has_torch_function_unary(self):
        388     return handle_torch_function(
        389         Tensor.backward,
        390         (self,),
       (...)
        394         create_graph=create_graph,
        395         inputs=inputs)
    --> 396 torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)


    File ~/mambaforge/lib/python3.9/site-packages/torch/autograd/__init__.py:173, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        168     retain_graph = create_graph
        170 # The reason we repeat same the comment below is that
        171 # some Python versions print out the first line of a multi-line function
        172 # calls in the traceback and some print out the last line
    --> 173 Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
        174     tensors, grad_tensors_, retain_graph, create_graph, inputs,
        175     allow_unreachable=True, accumulate_grad=True)


    KeyboardInterrupt: 


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
