# Make Learning Fastai Uncool

## Todos

- a workflow: use nbdev to split and assembly code/notebooks starting with kaggle notebook 1 
- start to work on Radek's notebooks on OTTO competition [started]
- display all important forum posts of mine and others I admire [done]
- what do I most like to work on and share (kaggle notebooks dissection)



```
#| hide
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>


## use `fastlistnbs` to check menu


```
show_doc(fastlistnbs)
```




---

[source](https://github.com/EmbraceLife/fastdebug/blob/master/fastdebug/utils.py#LNone){target="_blank" style="float:right; font-size:smaller"}

### fastlistnbs

>      fastlistnbs (query='all', flt_fd='src')

display section headings of notebooks, filter options: fastai, part2, groundup, src_fastai,src_fastcore, all

|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| query | str | all | "howto", "srcode", "journey", "question", "doc", "radek", "practice", "links", or "all" |
| flt_fd | str | src | other options: "groundup", "part2", "all" |




```
fastnbs("src: fastlistnbs")
```


### <mark style="background-color: #ffff00">src:</mark>  <mark style="background-color: #FFFF00">fastlistnbs</mark> (query, fld_fd), hts




heading 3.


```python
#| export 
import pandas as pd
```

```python
#| export
hts = pd.Series(list(map(lambda x: "ht: " + x, "imports, data_download, data_access, data_prep, data_loaders, cbs_tfms, learner, fit, pred, fu".split(", "))))
```

```python
hts
```

```python
#| export
def fastlistnbs(query="all", # "howto", "srcode", "journey", "question", "doc", "radek", "practice", "links", or "all"
                flt_fd="src"): # other options: "groundup", "part2", "all"
    "display section headings of notebooks, filter options: fastai, part2, groundup, src_fastai,\
src_fastcore, all"
    nbs, folder, _, _, _, _ = get_all_nbs()
    nb_rt = ""
    nbs_fd = []
    for nb in nbs:
        if flt_fd == "fastai" and "_fastai_" in nb.split("/")[-1] and not "_fastai_pt2" in nb.split("/")[-1]: 
            nbs_fd.append(nb)
        elif flt_fd == "part2" and "_fastai_pt2" in nb.split("/")[-1]:
            nbs_fd.append(nb)
        elif flt_fd == "groundup" and "groundup_" in nb.split("/")[-1]:            
            nbs_fd.append(nb)
        elif flt_fd == "src" and "fast" in nb.split("/")[-1]:
            nbs_fd.append(nb)
        elif flt_fd == "all": 
            nbs_fd.append(nb)
        else: 
            continue      

    if query != "howto":
        for nb_rt in nbs_fd:
            with open(nb_rt, 'r') as file:
                found = False
                for idx, l in enumerate(file):
                    if "##" in l:
                        if query == "howto" and "ht:" in l:
                            if l.count("#") == 2: print()
                            print(l, end="") # no extra new line between each line printed   
                            found = True
                        elif query == "srcode" and "src:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True
                        elif query == "doc" and "doc:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True                        
                        elif query == "journey" and "jn:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True
                        elif query == "question" and "qt:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True
                        elif query == "radek" and "rd:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True
                        elif query == "practice" and "pt:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True
                        elif query == "links" and "lk:" in l:
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True                            
                        elif query == "all": 
                            if l.count("#") == 2: print()                        
                            print(l, end="") 
                            found = True                        
                if found: print(nb_rt + "\n")
    else:
        for idx, o in enumerate(hts):
            print('{:=<157}'.format(f"step {idx}: {o}"))
            for nb_rt in nbs_fd:
                with open(nb_rt, 'r') as file:
                    found = False
                    for idx, l in enumerate(file):
                        if "##" in l:
                            if o in l:
                                if l.count("#") == 2: print()
                                print(l, end="") # no extra new line between each line printed   
                                found = True                   
                    if found: print(nb_rt + "\n")
```

```python
# for i, o in enumerate("imports, data-download, data-access, data-prep, data-loaders, cbs-tfms, learner, fit, pred".split(", ")):
#     i, o
```

```python
fastlistnbs("howto")
# fastlistnbs("doc")
# fastlistnbs("srcode")
# fastlistnbs("journey")

```

Next, heading 2
## fastlistsrcs



[Open `01_utils` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/01_utils.ipynb#src:-fastlistnbs(query,-fld_fd),-hts
)



```
hts
```




    0          ht: imports
    1    ht: data_download
    2      ht: data_access
    3        ht: data_prep
    4     ht: data_loaders
    5         ht: cbs_tfms
    6          ht: learner
    7              ht: fit
    8             ht: pred
    9               ht: fu
    dtype: object



## use fastnbs to dive in


```
show_doc(fastnbs)
```




---

[source](https://github.com/EmbraceLife/fastdebug/blob/master/fastdebug/utils.py#LNone){target="_blank" style="float:right; font-size:smaller"}

### fastnbs

>      fastnbs (question:str, filter_folder='src', strict=False, output=False,
>               accu:float=0.8, nb=True, db=False)

check with fastlistnbs() to skim through all the learning points as section titles; then use fastnotes() to find interesting lines which can be notes or codes, and finally use fastnbs() display the entire learning points section including notes and codes.

|    | **Type** | **Default** | **Details** |
| -- | -------- | ----------- | ----------- |
| question | str |  | query options, "rd: adept practitioner", "doc: ImageDataLoaders", "src: DataBlock", "ht: git", "jn: help others is the way" |
| filter_folder | str | src | options: src, all, |
| strict | bool | False | loose search keyword, not as the first query word |
| output | bool | False | True for nice print of cell output |
| accu | float | 0.8 |  |
| nb | bool | True |  |
| db | bool | False |  |



## Search my Journey 


```
fastlistnbs("journey")
```

    ### jn: help other is the best way forward
    ### jn: how to iterate or make one step forward at at time
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    



```
fastnbs("jn: help other")
```


### <mark style="background-color: #ffff00">jn:</mark>  <mark style="background-color: #ffff00">help</mark>  <mark style="background-color: #FFFF00">other</mark>  is the best way forward




heading 3.



**Reflection on Radek's 1st newsletter**

One way to summarize Radek's secret to success is the following: 

> No matter which stage of journey in the deep learning or any subject, when you are doing your best to help others to learn what you learnt and what you are dying to find out, and if you persist, you will be happy and successful. 

I have dreamed of such hypothesis many times when I motivated myself to share online, and Radek proved it to be solid and true! No time to waste now!

Another extremely simple but shocking secret to Radek's success is, in his words (now I can recite):

> I would suspend my disbelief and do exactly what Jeremy Howard told us to do in the lectures

What Jeremy told us to do is loud and clear, the 4 steps (watch, experiment, reproduce, apply elsewhere). More importantly, they are true and working if one holds onto it like Radek did. 

Why I am always trying to do something different? Why couldn't I just follow this great advice right from the start? I walked [a long way around it](https://twitter.com/shendusuipian/status/1587429658621988871?s=20&t=zjz1OlYRt7yJJ8HVBdsqoA) and luckily I get my sense back and move onto the second step now. 


Next, heading 2
## ht: imports - vision



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb#jn:-help-other-is-the-best-way-forward
)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)


## Search docs


```
fastlistnbs("doc")
```

    ### doc: DataBlock.datasets(source, verbose)
    ### doc: DataBlock.dataloaders
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0001_fastai_is_it_a_bird.md
    
    ### doc: setup_comp(comp, local_folder='', install='fastai "timm>=0.6.2.dev0")
    ### doc: ImageDataLoaders.from_folder
    ### doc: aug_transforms(size=128, min_scale=0.75)
    ### doc: vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()
    ### doc: learn.export(fname='export.pkl', pickle_module=pickle, pickle_protocol=2)
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    ### doc: ImageDataLoaders.from_name_func(path: 'str | Path', fnames: 'list', label_func: 'callable', **kwargs) -> 'DataLoaders'
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md
    



```
fastnbs("doc: ImageDataLoaders")
```


### <mark style="background-color: #ffff00">doc:</mark>  <mark style="background-color: #FFFF00">imagedataloaders</mark> .from_folder




heading 3.



To create a DataLoader obj from a folder, and a dataloader prepares functions for splitting training and validation sets, extracting images and labels, each item transformations, and batch transformations.

eg., give it `trn_path` (folder has subfolders like train, test or even valid), `valid_pct` (split a portion from train to create validation set), `seed` (set a seed for reproducibility), `item_tfms` (do transforms to each item), and `batch_tfms` (do transformations on batches)

<!-- #region -->
```python
dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=128, min_scale=0.75))

dls.show_batch(max_n=6)
```
<!-- #endregion -->

<!-- #region -->
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
<!-- #endregion -->

```python
show_doc(ImageDataLoaders.from_folder)
```

Next, heading 3
### src: ImageDataLoaders.from_folder



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb#doc:-ImageDataLoaders.from_folder
)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)



### <mark style="background-color: #ffff00">doc:</mark>  <mark style="background-color: #FFFF00">imagedataloaders</mark> .from_name_func(path: 'str | path', fnames: 'list', label_func: 'callable', **kwargs) -> 'dataloaders'




heading 3.



official: "Create from the name attrs of `fnames` in `path`s with `label_func`"

use `using_attr(label_func, 'name')` as `f`, and pass `f` to `from_path_func` to create a DataLoaders (which later passed to a learner)

from_name_func: because the label is inside the name of the image filename

label_func: is to get the targe or label from the name of the image filename

path: is the string name or path for the folder which is to store models

fnames: all the image/data filenames to be used for the model, get_image_files(path) can return a L list of image filenames/path

`f = using_attr(label_func, 'name')`: make sure `is_cat` is to work on the `name` of a image filename. (see example inside source below)

```python
# fastnbs("DataBlock.dataloaders")
# fastnbs("DataBlock.datasets")
# fastnbs("Datasets")
```

Next, heading 3
### src: ImageDataLoaders.from_name_func(path: 'str | Path', fnames: 'list', label_func: 'callable', **kwargs) -> 'DataLoaders'



[Open `0002_fastai_saving_a_basic_fastai_model` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.ipynb#doc:-ImageDataLoaders.from_name_func(path:-'str-|-Path',-fnames:-'list',-label_func:-'callable',-**kwargs)-->-'DataLoaders'
)



[Open `0002_fastai_saving_a_basic_fastai_model` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/saving-a-basic-fastai-model)


## Search source code


```
fastlistnbs("srcode")
```

    ### src: itemgot(self:L, *idxs)
    ### src: itemgetter
    ### src: DataBlock.datasets((source, verbose)
    ### src: DataBlock.dataloaders
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0001_fastai_is_it_a_bird.md
    
    ### src: setup_compsetup_comp(comp, local_folder='', install='fastai "timm>=0.6.2.dev0")
    ### src: check_subfolders_img(path, db=False)
    ### src: randomdisplay(path, size, db=False)
    ### src: check_sizes_img(files)
    ### src: ImageDataLoaders.from_folder
    ### src: aug_transforms(size=128, min_scale=0.75)
    ### src: vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    ### src: ImageDataLoaders.from_name_func(path: 'str | Path', fnames: 'list', label_func: 'callable', **kwargs) -> 'DataLoaders'
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md
    



```
fastnbs("src: ImageDataLoaders.from_name")
```


### <mark style="background-color: #ffff00">src:</mark>  <mark style="background-color: #FFFF00">imagedataloaders.from_name</mark> _func(path: 'str | path', fnames: 'list', label_func: 'callable', **kwargs) -> 'dataloaders'




heading 3.


```python
doc_sig(ImageDataLoaders.from_name_func)
doc_sig(get_image_files)
```

```python
from __future__ import annotations # to ensure path:str|Path='.' can work
```

```python
# DataLoaders.from_dblock??
```

```python
class ImageDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for computer vision problems"
    @classmethod
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
    @snoop
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
        pp(doc_sig(using_attr), f)
        pp(f(fnames[0]), fnames[0].name) # no need to worry about getting the name out of the filename
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

```python
# fastnbs("DataBlock.__")

# fastnbs("get_image_files")
# fastnbs("Resize", "src", True)
```

```python
#|eval: false
dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), 
    valid_pct=0.2, 
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))
```

```python
class ImageDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for computer vision problems"
    @classmethod
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

```python
@docs
class DataLoaders(GetAttr):
    "Basic wrapper around several `DataLoader`s."
    _default='train'
    def __init__(self, 
        *loaders, # `DataLoader` objects to wrap
        path:str|Path='.', # Path to store export objects
        device=None # Device to put `DataLoaders`
    ):
        self.loaders,self.path = list(loaders),Path(path)
        if device is not None or hasattr(loaders[0],'to'): self.device = device

    def __getitem__(self, i): return self.loaders[i]
    def __len__(self): return len(self.loaders)
    def new_empty(self):
        loaders = [dl.new(dl.dataset.new_empty()) for dl in self.loaders]
        return type(self)(*loaders, path=self.path, device=self.device)

    def _set(i, self, v): self.loaders[i] = v
    train   ,valid    = add_props(lambda i,x: x[i], _set)
    train_ds,valid_ds = add_props(lambda i,x: x[i].dataset)

    @property
    def device(self): return self._device

    @device.setter
    def device(self, 
        d # Device to put `DataLoaders`
    ):
        for dl in self.loaders: dl.to(d)
        self._device = d

    def to(self, 
        device # Device to put `DataLoaders`
    ):
        self.device = device
        return self
            
    def _add_tfms(self, tfms, event, dl_idx):
        "Adds `tfms` to `event` on `dl`"
        if(isinstance(dl_idx,str)): dl_idx = 0 if(dl_idx=='train') else 1
        dl_tfms = getattr(self[dl_idx], event)
        apply(dl_tfms.add, tfms)
        
    def add_tfms(self,
        tfms, # List of `Transform`(s) or `Pipeline` to apply
        event, # When to run `Transform`. Events mentioned in `TfmdDL`
        loaders=None # List of `DataLoader` objects to add `tfms` to
    ):
        "Adds `tfms` to `events` on `loaders`"
        if(loaders is None): loaders=range(len(self.loaders))
        if not is_listy(loaders): loaders = listify(loaders)
        for loader in loaders:
            self._add_tfms(tfms,event,loader)      

    def cuda(self): return self.to(device=default_device())
    def cpu(self):  return self.to(device=torch.device('cpu'))

    @classmethod
    def from_dsets(cls, 
        *ds, # `Datasets` object(s)
        path:str|Path='.', # Path to put in `DataLoaders`
        bs:int=64, # Size of batch
        device=None, # Device to put `DataLoaders`
        dl_type=TfmdDL, # Type of `DataLoader`
        **kwargs
    ):
        default = (True,) + (False,) * (len(ds)-1)
        defaults = {'shuffle': default, 'drop_last': default}
        tfms = {k:tuple(Pipeline(kwargs[k]) for i in range_of(ds)) for k in _batch_tfms if k in kwargs}
        kwargs = merge(defaults, {k: tuplify(v, match=ds) for k,v in kwargs.items() if k not in _batch_tfms}, tfms)
        kwargs = [{k: v[i] for k,v in kwargs.items()} for i in range_of(ds)]
        return cls(*[dl_type(d, bs=bs, **k) for d,k in zip(ds, kwargs)], path=path, device=device)

    @classmethod
    def from_dblock(cls, 
        dblock, # `DataBlock` object
        source, # Source of data. Can be `Path` to files
        path:str|Path='.', # Path to put in `DataLoaders`
        bs:int=64, # Size of batch
        val_bs:int=None, # Size of batch for validation `DataLoader`
        shuffle:bool=True, # Whether to shuffle data
        device=None, # Device to put `DataLoaders`
        **kwargs
    ):
        return dblock.dataloaders(source, path=path, bs=bs, val_bs=val_bs, shuffle=shuffle, device=device, **kwargs)

    _docs=dict(__getitem__="Retrieve `DataLoader` at `i` (`0` is training, `1` is validation)",
               train="Training `DataLoader`",
               valid="Validation `DataLoader`",
               train_ds="Training `Dataset`",
               valid_ds="Validation `Dataset`",
               to="Use `device`",
               add_tfms="Add `tfms` to `loaders` for `event",
               cuda="Use accelerator if available",
               cpu="Use the cpu",
               new_empty="Create a new empty version of `self` with the same transforms",
               from_dblock="Create a dataloaders from a given `dblock`")
# File:           ~/mambaforge/lib/python3.9/site-packages/fastai/data/core.py
# Type:           type
# Subclasses:     ImageDataLoaders, SegmentationDataLoaders
```

```python

```

Next, heading 3
### vision_learner(dls, resnet18, metrics=error_rate)



[Open `0002_fastai_saving_a_basic_fastai_model` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.ipynb#src:-ImageDataLoaders.from_name_func(path:-'str-|-Path',-fnames:-'list',-label_func:-'callable',-**kwargs)-->-'DataLoaders'
)



[Open `0002_fastai_saving_a_basic_fastai_model` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/saving-a-basic-fastai-model)



```

```

## Search howto


```
# show_doc(fastnotes)
# show_doc(fastlistsrcs)

# how to do notes from now on? make them notebooks first for using fastlistnbs and fastnbs
# make howto internal steps breakable?
fastlistnbs("howto")
```

    step 0: ht: imports==========================================================================================================================================
    
    ## ht: imports - vision
    ### ht: imports - fastkaggle 
    ### ht: imports - upload and update mylib in kaggle
    ### ht: imports - fastkaggle - push libs to kaggle with `create_libs_datasets`
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 1: ht: data_download====================================================================================================================================
    
    ## ht: data_download - kaggle competition dataset
    ### ht: data_download - join, `kaggle.json`, `setup_comp` for local use
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 2: ht: data_access======================================================================================================================================
    ### ht: data_access - map subfolders content with `check_subfolders_img`
    ### ht: data_access - extract all images for test and train with `get_image_files`
    ### ht: data_access - display an image from test_files or train_files with `randomdisplay`
    ### ht: data_access - select a subset from each subfolder with `get_image_files_subset`
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 3: ht: data_prep========================================================================================================================================
    ### ht: data_prep reproducibility in training a model
    ### ht: data_prep - remove images that fail to open with `remove_failed(path)`
    ### ht: data_prep - describe sizes of all images with `check_sizes_img`
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 4: ht: data_loaders=====================================================================================================================================
    ### ht: data_loaders - create a dataloader from a folder with `ImageDataLoaders.from_folder`
    ### ht: data_loaders - apply transformations to each image with `item_tfms = Resize(480, method='squish')`
    ### ht: data_loaders - apply image augmentations to each batch of data with `batch_tfms = aug_transforms`
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 5: ht: cbs_tfms=========================================================================================================================================
    step 6: ht: learner==========================================================================================================================================
    ### ht: learner - model arch - how to pick the first to try
    ### ht: learner - vision_learner - build a learner for vision
    ### ht: learner - find learning rate with `learn.lr_find(suggest_funcs=(valley, slide))`
    ### ht: learner - save model with `learn.export`
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 7: ht: fit==============================================================================================================================================
    step 8: ht: pred=============================================================================================================================================
    step 9: ht: fu===============================================================================================================================================
    ### ht: fu - whatinside, show_doc, fastlistnbs, fastnbs
    ### ht: fu - git - when a commit takes too long
    ### ht: fu - debug every srcline without breaking
    ### ht: fu - (de)activate snoop without commenting out using `snoopon()` and `snoopoff()`  
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    



```
fastnbs('ht: learner save model')
```


### <mark style="background-color: #ffff00">ht:</mark>  <mark style="background-color: #ffff00">learner</mark>  - <mark style="background-color: #ffff00">save</mark>  <mark style="background-color: #FFFF00">model</mark>  with `learn.export`




heading 3.



Next, heading 3
### doc: learn.export(fname='export.pkl', pickle_module=pickle, pickle_protocol=2)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb#ht:-learner---save-model-with-`learn.export`
)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)



```
fastnbs("doc: learn.export")
```


### <mark style="background-color: #ffff00">doc:</mark>  <mark style="background-color: #FFFF00">learn.export</mark> (fname='export.pkl', pickle_module=pickle, pickle_protocol=2)




heading 3.



return: nothing, but saved a model as a pkl file in `learn.path` with the name given by `fname`

we can change the folder for storing model by changing `learn.path`

we can give a detailed name to specify the model

<!-- #region -->
```python
@patch
def export(self:Learner, fname='export.pkl', pickle_module=pickle, pickle_protocol=2):
    "Export the content of `self` without the items and the optimizer state for inference"
    if rank_distrib(): return # don't export if child proc
    self._end_cleanup()
    old_dbunch = self.dls
    self.dls = self.dls.new_empty()
    state = self.opt.state_dict() if self.opt is not None else None
    self.opt = None
    with warnings.catch_warnings():
        #To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        # the folder is defined by self.path
        torch.save(self, self.path/fname, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
    self.create_opt()
    if state is not None: self.opt.load_state_dict(state)
    self.dls = old_dbunch
# File:      ~/mambaforge/lib/python3.9/site-packages/fastai/learner.py
# Type:      method
```
<!-- #endregion -->

```python
learn.path
```

```python
learn.path = Path('models')
```

```python
learn.path
```

```python
learn.export("paddy_10pct_resnet26d_10epochs.pkl")
```

```python

```

Next, heading 3
### qt: How many epochs should I train in general in this early stage with 10% dataset without gpu



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb#doc:-learn.export(fname='export.pkl',-pickle_module=pickle,-pickle_protocol=2)
)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)


## Search questions (notes and Q&A on forum) 


```
fastlistnbs("question")
```

    ### qt: how to display a list of images?
    #### qt: why must all images have the same dimensions? how to resolve this problem?
    #### qt: why should we start with small resized images
    ### qt: How many epochs should I train in general in this early stage with 10% dataset without gpu
    ### qt: how to display video and embed webpage
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    



```
hts
```




    0          ht: imports
    1    ht: data_download
    2      ht: data_access
    3        ht: data_prep
    4     ht: data_loaders
    5         ht: cbs_tfms
    6          ht: learner
    7              ht: fit
    8             ht: pred
    9               ht: fu
    dtype: object




```
# fastnbs("ht: load")
```

## Search Meta


```
fastlistnbs("radek")
```

    
    ## rd: The problem with theory
    ### rd: How Hinton approaches and discover theories
    ### rd: Why must practice before theory
    ### rd: How to make practice before theory
    
    ## rd: For best effects, use one cup of theory, one cup of practice. Rinse and repeat
    ### rd: Is your learning of theory takes you where you want to go
    ### rd: What is the darker effect of just learning theory
    ### rd: How to combine practice and theory as a learner of image classification
    ### rd: What benefits does the practice + theory approach offer
    
    ## rd: Practice wins every time
    
    ## rd: Programming is all about what you have to say
    ### rd: What makes one a good programmer? (give me the most wrong answer)
    ### rd: How is your ability to program ultimately measured
    ### rd: What is the only way to really understand a piece of syntax
    ### rd: How to learn a new language by reading and writing
    ### rd: Why knowing the programming language is only a starting point
    ### rd: Why domain knowledge comes first? 
    ### rd: What is the fastest way to learn to program?
    
    ## rd: The secret of good developers
    ### rd: Why it is costly to enter your zone of work
    ### rd: What's wrong with switching context during work
    ### rd: Tips to achieve a long uninterrupted sessions
    
    ## rd: The best way to improve as a developer
    ### rd: How to be better at something
    ### rd:  What being a developer is all about
    ### rd: How to start to read code
    ### rd: How to start to write code
    ### rd: Just do it
    
    ## rd: How to use your tools to achieve a state of flow
    ### rd: What is a state of flow and why it is great to be in
    ### rd: What developers often do for achieving a state of flow
    ### rd: What can prevent you dive into a state of flow
    ### rd: How to get into a state of flow more often
    
    ## rd: Use reality as your mirror
    ### rd: Why testing beliefs and assumptions
    ### rd: How Radek overcomed the fear of tweeting
    ### rd: How to face the unknown
    
    ## rd: Do genuine work (it compounds)
    ### rd: Why writing is better than thinking
    ### rd: Why sharing is even better than writing alone
    ### rd: What is the best/fastest way to learn
    
    ## rd: The hidden game of machine learning
    ### rd: What are the essential questions of machine learning
    ### rd: What makes you an adept practitioner? 
    ### rd: What makes you a great practitioner? 
    ### rd: What can lead to a tragic consequence of your model? 
    ### rd: How to gain a deeper understanding of the ability to generalize to unseen data?
    
    ## rd: How to structure a healthy machine learning project
    ### rd: ingredient 1 - train-validation-test split
    ### rd: ingredient 2 - what is a baseline
    ### rd: what does a simplest baseline look like
    ### rd: Why is having a baseline so crucial? 
    ### rd: What to experiment on after baseline
    ### rd: Why running entire pipeline regularly?
    ### rd: How to allow time to do the rest
    ### rd: A beautiful sum up by Radek
    
    ## rd: How to win Kaggle
    ### rd: even getting started can take a while
    ### rd: Why Kaggle competition takes time
    ### rd: Join the competition early for creativity
    ### rd: What to do with Kaggle forum
    ### rd: How to read papers for kaggle
    ### rd: What about blog posts on papers
    ### rd: What to tweak daily
    ### rd: How to build a baseline for kaggle
    ### rd: What's the objective of initial models
    ### rd: The importance of validation split
    ### rd: how do we measure whether we got it right?
    ### rd: Why ensemble is a must
    ### rd: cross-validation is a must (not clear to me)
    ### rd: Sum up
    
    ## rd: How to Pick Up What They Put Down by swyx
    ### rd: Who are they?
    ### rd: What do you mean by “put down”?
    ### rd: How do I pick “it” up? 
    ### rd: Love what you do above and tag your mentor
    ### rd: What happens when you do this?
    ### rd: Why does this work on them?
    ### rd: Why does this work on you?
    ### rd: How to get started
    
    ## rd: The Permissionless Apprenticeship
    ### rd: How to have their attention
    ### rd: Why working for them without asking
    ### rd: How to work for them
    ### rd: The worse scenario of working for them
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/00_fastai_Meta_learning_Radek.md
    



```
fastnbs("rd: experiment")
```


### <mark style="background-color: #ffff00">rd:</mark>  what to <mark style="background-color: #FFFF00">experiment</mark>  on after baseline




heading 3.


> However, it is important that we **not** run the experiments solely to tweak the **hyperparameters**, especially early in the training. We want to invest our time where it matters. That usually means **exploring a larger set of architectures and developing diagnostic code**. The more we learn about **how our models are performing**, the better. Additionally, **trying out different architectures can be a great source of insight on what is possible**.


> As we build on our baseline, the idea is to keep moving in small steps. Just as we didn’t want to go from having no pipeline to training a complex model, neither do we want to jump straight to it now. Before we train an elaborate, state-of-the- art deep learning model, we might want to **try random forest[31] or a simple model consisting only of fully connected layers**.

> [31] Random Forest is a great algorithm to reach for, given the chance! It is very quick to train and it doesn’t overfit. On many datasets, it can match the performance of more complex methods. It also lends itself extremely well to interpretation. For an example in which a Random Forest classifier matched the performance of a neural network, see this repository. Additionally, it enabled a deeper understanding of the factors at play, which would be very hard or impossible to get at using the RNN!


> Extending this way of thinking to the implementation of our most complex model, we probably **don**’t want to go about **building it in one sweep**. We might want to **begin by implementing the loss and bolting it onto a simple model consisting of only fully connected layers**. The **next** step might be to implement **some set of layers** and **pass a batch from our dataloader through it**. Ensuring we **get something other than all zeros and an output of the correct shape is a very valuable check**.[32]

> [32] This can be taken further, where we **look at the means and standard deviations of intermittent layers**, **monitor for how many dead ReLus** we have, and so on. In many instances, just a single check will suffice as it provides a great value for the time invested.

> The idea is to always move in **small increments**, using simpler models as **a stepping stone to more complex** ones.

Next, heading 3
### pt: how to explore architectures and developing diagnostic code in tiny steps (above)



[Open `00_fastai_Meta_learning_Radek` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/00_fastai_Meta_learning_Radek.ipynb#rd:-What-to-experiment-on-after-baseline
)


## Search my practice


```
fastlistnbs("practice")
```

    ### pt: what does Radek mean by benchmarks and where are they and how to use them
    ### pt: how my tool help me stay in the flow
    ### pt: what is my voice and how could I help others (myself)?
    ### pt: how to explore architectures and developing diagnostic code in tiny steps (above)
    #### pt: how to find (what) valuable insights from simple model with subset of data?
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/00_fastai_Meta_learning_Radek.md
    



```
fastnbs("pt: my tool")
```


### <mark style="background-color: #ffff00">pt:</mark>  how <mark style="background-color: #ffff00">my</mark>  <mark style="background-color: #FFFF00">tool</mark>  help me stay in the flow




heading 3.


Building index.ipynb using fastlistnbs and fastnbs on howto, doc, src, question, radek is the way to get closer to a state of flow 

Next, heading 2
## rd: Use reality as your mirror



[Open `00_fastai_Meta_learning_Radek` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/00_fastai_Meta_learning_Radek.ipynb#pt:-how-my-tool-help-me-stay-in-the-flow
)


## Search links


```
fastlistnbs("links")
```

    ### lk: Jeremy and Paddy
    ### lk: Daniel's fastai forum posts
    ### lk: Radek and OTTO
    ### lk: ilovescience on kaggle
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_links_forums_kaggle_github.md
    



```
fastnbs("lk: Radek Otto")
```


### <mark style="background-color: #ffff00">lk:</mark>  <mark style="background-color: #ffff00">radek</mark>  and <mark style="background-color: #FFFF00">otto</mark> 




heading 3.

Radek on [twitter](https://twitter.com/radekosmulski)

How to get started [post](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364062)

Baseline model [notebook](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic/notebook?scriptVersionId=110068977)

Full dataset converted to csv/parquet format [post](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843) with notebooks

Next, heading 3
### lk: ilovescience on kaggle



[Open `fastai_links_forums_kaggle_github` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/fastai_links_forums_kaggle_github.ipynb#lk:-Radek-and-OTTO
)



```

```
