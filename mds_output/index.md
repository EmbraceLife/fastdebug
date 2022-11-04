# Make Learning Fastai Uncool


```
#| hide
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>


## `fastlistnbs` to read menu, `fastnbs` to dive in


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
| query | str | all | howto, srcode, journey, question, doc, radek, or all |
| flt_fd | str | src | other options: "groundup", "part2", "all" |




```
# fastlistnbs??
```


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
fastnbs("jn: help")
```


### <mark style="background-color: #ffff00">jn:</mark>  <mark style="background-color: #FFFF00">help</mark>  other is the best way forward




The current section is heading 3.



**Reflection on Radek's 1st newsletter**

One way to summarize Radek's secret to success is the following: 

> No matter which stage of journey in the deep learning or any subject, when you are doing your best to help others to learn what you learnt and what you are dying to find out, and if you persist, you will be happy and successful. 

I have dreamed of such hypothesis many times when I motivated myself to share online, and Radek proved it to be solid and true! No time to waste now!

Another extremely simple but shocking secret to Radek's success is, in his words (now I can recite):

> I would suspend my disbelief and do exactly what Jeremy Howard told us to do in the lectures

What Jeremy told us to do is loud and clear, the 4 steps (watch, experiment, reproduce, apply elsewhere). More importantly, they are true and working if one holds onto it like Radek did. 

Why I am always trying to do something different? Why couldn't I just follow this great advice right from the start? I walked [a long way around it](https://twitter.com/shendusuipian/status/1587429658621988871?s=20&t=zjz1OlYRt7yJJ8HVBdsqoA) and luckily I get my sense back and move onto the second step now. 


start of heading 2
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
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    ### doc: ImageDataLoaders.from_name_func(path: 'str | Path', fnames: 'list', label_func: 'callable', **kwargs) -> 'DataLoaders'
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md
    



```
fastnbs("doc: DataBlock.datasets")
```


### <mark style="background-color: #ffff00">doc:</mark>  <mark style="background-color: #FFFF00">datablock.datasets</mark> (source, verbose)




The current section is heading 3.

get data items from the source, and split the items and use `fastai.data.core.Datasets` to create the datasets


start of another heading 3
### src: DataBlock.datasets((source, verbose)



[Open `0001_fastai_is_it_a_bird` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0001_fastai_is_it_a_bird.ipynb#doc:-DataBlock.datasets(source,-verbose)
)



[Open `0001_fastai_is_it_a_bird` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data)


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




The current section is heading 3.


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

start of another heading 3
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


<style>.container { width:100% !important; }</style>


    step 0: ht: imports==========================================================================================================================================
    
    ## ht: imports - vision
    ### ht: imports - fastkaggle 
    ### ht: imports - use mylib in kaggle
    ### ht: imports - fastkaggle - push libs to kaggle
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 1: ht: data_download====================================================================================================================================
    
    ## ht: data_download - kaggle competition dataset
    ### ht: data_download - join, `kaggle.json`, `setup_comp`
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    step 2: ht: data_access======================================================================================================================================
    ### ht: data_access - map subfolders content with `check_subfolders_img`
    ### ht: data_access - extract all images for test and train with `get_image_files`
    ### ht: data_access - display an image from test_files or train_files with `randomdisplay`
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
    ### ht: learner - find learning rate with `lr_find`
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

```

## Search questions (notes and Q&A on forum) 


```
fastlistnbs("question")
```

    ### qt: how to display a list of images?
    #### qt: why must all images have the same dimensions? how to resolve this problem?
    #### qt: why should we start with small resized images
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

### Search Meta


```
fastlistnbs("radek")
```

    #### rd: What is The hidden game of machine learning? 
    #### rd: What makes you an adept practitioner? 
    #### rd: What makes you a great practitioner? 
    #### rd: What can lead to a tragic consequence of your model? 
    #### rd: How to gain a deeper understanding of the ability to generalize to unseen data?
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/00_fastai_Meta_learning_Radek.md
    



```

```
