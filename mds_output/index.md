```
#| hide
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>


# Make Learning Fastai Uncool

## Todos

### 2022.11.7

- [x] a workflow: 1. create a pure verion of code for training models; 2. create code for exploring src; 3. create links to open both codes 
- [ ] redo kaggle paddy part 1 with new workflow
- [ ] share my notes of meta learning and ask Radek for permession of sharing
- [ ] start to work on Radek's notebooks on OTTO competition 
- [ ] display all important forum posts of mine and others I admire 
- [ ] what do I most like to work on and share (kaggle notebooks dissection as a beginner)

## How to use this notebook

### use `fastlistnbs` to check menu


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

```

### use `fastnbs` to dive in


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
| question | str |  | see fastlistnbs() results for what to search "doc: ", "rd: ", "src: ", "ht: ", "jn: ", "qt: ", "pt:" |
| filter_folder | str | src | options: src, all, |
| strict | bool | False | loose search keyword, not as the first query word |
| output | bool | False | True for nice print of cell output |
| accu | float | 0.8 |  |
| nb | bool | True |  |
| db | bool | False |  |




```
fastnbs("src: fastnbs")
```


### <mark style="background-color: #ffff00">src:</mark>  <mark style="background-color: #FFFF00">fastnbs</mark> (question, filter_folder="src", ...)




heading 3.


```python
#| export
# @snoop
def fastnbs(question:str, # see fastlistnbs() results for what to search "doc: ", "rd: ", "src: ", "ht: ", "jn: ", "qt: ", "pt:"
            filter_folder="src", # options: src, all,
            strict=False, # loose search keyword, not as the first query word
            output=False, # True for nice print of cell output
            accu:float=0.8, 
            nb=True, 
            db=False):
    "check with fastlistnbs() to skim through all the learning points as section titles; \
then use fastnotes() to find interesting lines which can be notes or codes, and finally \
use fastnbs() display the entire learning points section including notes and codes."
    questlst = question.split(' ')
    mds_no_output, folder, ipynbs, ipyfolder, mds_output, output_fd, pys, py_folder = get_all_nbs()
    if not output: mds = mds_no_output
    else: mds = mds_output
        
    for file_path in mds:
        if filter_folder == "fastai" and "_fastai_" in file_path and not "_fastai_pt2_" in file_path:
            file_fullname = file_path
        elif filter_folder == "part2" and "_fastai_pt2_" in file_path:
            file_fullname = file_path
        elif filter_folder == "src" and "fast" in file_path:
            file_fullname = file_path            
        elif filter_folder == "all": 
            file_fullname = file_path
        else: continue

        file_name = file_fullname.split('/')[-1]
        with open(file_fullname, 'r') as file:
            for count, l in enumerate(file):
                if l.startswith("## ") or l.startswith("### ") or l.startswith("#### "):
                    truelst = [q.lower() in l.lower() for q in questlst]
                    pct = sum(truelst)/len(truelst)
                    ctn = l.split("# ```")[1] if "# ```" in l else l.split("# ")[1] if "# " in l else l.split("# `")
                    if strict:
                        if pct >= accu and ctn.startswith(questlst[0]): # make sure the heading start with the exact quest word
                            if db: 
                                head1 = f"keyword match is {pct}, Found a section: in {file_name}"
                                head1 = highlight(str(pct), head1)
                                head1 = highlight(file_name, head1)
                                display_md(head1)
                                highlighted_line = highlight(question, l, db=db)                        
        #                         print()
                            display_block(l, file_fullname, output=output, keywords=question)
                            if nb: # to link a notebook with specific heading
                                if "# ```" in l: openNB(file_name, l.split("```")[1].replace(" ", "-"), db=db)
                                else: openNB(file_name, l.split("# ")[1].replace(" ", "-"), db=db)

                                openNBKaggle(file_name, db=db)
                    else: 
                        if pct >= accu: # make sure the heading start with the exact quest word
                            if db: 
                                head1 = f"keyword match is {pct}, Found a section: in {file_name}"
                                head1 = highlight(str(pct), head1)
                                head1 = highlight(file_name, head1)
                                display_md(head1)
                                highlighted_line = highlight(question, l, db=db)                        
        #                         print()
                            display_block(l, file_fullname, output=output, keywords=question)
                            if nb: # to link a notebook with specific heading
                                if "# ```" in l: openNB(file_name, l.split("```")[1].replace(" ", "-"), db=db)
                                else: openNB(file_name, l.split("# ")[1].replace(" ", "-"), db=db)

                                openNBKaggle(file_name, db=db)
```

```python
"# this is me".split("# ")[1].replace(" ", "-")
"# thisisme".split("# ")[1].replace(" ", "-")
```

```python
# fastnbs("Snoop them together in one go", output=True)
# fastnbs("how matrix multiplication")
# fastnbs("how matrix multiplication", folder="fastai")
# fastnbs("show_image", "src")
# fastnbs("module", "src")
# fastnbs("module", "src", False)
# fastnbs("module")
# fastnbs("apply")
# fastnbs("get all nbs")
```

```python
"### ```show_image(b, a, c)```".split("```")[1].replace(" ", "-")
```

Next, heading 2
## fastcodes



[Open `01_utils` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/01_utils.ipynb#src:-fastnbs(question,-filter_folder="src",-...)
)


## Search my Journey 


```

```


```
fastlistnbs("journey")
```

    jn by dates: ========
    ### jn: help other is the best way forward  /2022-11-2
    ### jn: combine experimenting notebooks with reading and writing codes /2022-11-5
    ### jn: Why start to try Kaggle Recommendation competition OTTO now /2022-11-7
    ### jn: The inner voice often reminds me: You won't succeed no matter how hard you try. But with all the amazing alumni like Radek, Moody etc in fastai community I can't help trying /2022-11-7
    ### jn: A get-started  [post](https://www.kaggle.com/competitions/otto-recommender-system) on recsys by Radek /2022-11-8
    #### jn: don't worry many theories or models or equations about recsys I don't understand as good libraries of recsys should have them all in code. As long as I can use the code in practice, I should be able to understand them in the end /2022-11-9
    #### jn: I should run code and experiment notebooks daily when even I don't have a whole picture of recsys /2022-11-9
    ### jn: Radek - do tell the world what you are up to /2022-11-9
    ### jn: Radek - Don't worry about how your writing is written /2022-11-9
    ### jn: Radek - let Kaggle be a life changing experience /2022-11-9
    ### jn: Radek - how to do project based learning /2022-11-9
    ### jn: I preordered walk with fastai, but I will primarily focus on Radek and OTTO /2022-11-10
    ### jn: as Radek suggested, I should focus on notebooks and discussions shared on Kaggle OTTO to grow myself bit by bit /2022-11-10
    ### jn: I have experimented `process_data.ipynb` and `eda on OTTO.ipynb` by Radek /2022-11-10
    ### jn: I started to read https://github.com/otto-de/recsys-dataset which has a lot helpful info for beginners /2022-11-10
    ### jn: record all techniques learnt within a context (using "recsys - otto - get started") is much more helpful than extract it from the context it's learnt /2022-11-11
    ### jn: starting to work on Radek's [notebook](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files?scriptVersionId=109945227) on how to use the otto dataset in parquet file /2022-11-11



```
fastnbs("jn: The nner voice often reminds me")
```


### <mark style="background-color: #ffff00">jn:</mark>  <mark style="background-color: #ffff00">the</mark>  inner <mark style="background-color: #ffff00">voice</mark>  <mark style="background-color: #ffff00">often</mark>  <mark style="background-color: #ffff00">reminds</mark>  <mark style="background-color: #FFFF00">me</mark> : you won't succeed no matter how hard you try. but with all <mark style="background-color: #ffff00">the</mark>  amazing alumni like radek, moody etc in fastai community i can't help trying /2022-11-7




heading 3.


Next, heading 3
### jn: combine experimenting notebooks with reading and writing codes /2022-11-5



[Open `00_fastai_Meta_learning_Radek` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/00_fastai_Meta_learning_Radek.ipynb#jn:-The-inner-voice-often-reminds-me:-You-won't-succeed-no-matter-how-hard-you-try.-But-with-all-the-amazing-alumni-like-Radek,-Moody-etc-in-fastai-community-I-can't-help-trying-/2022-11-7
)


## Search docs


```
fastlistnbs("doc")
```

    ### doc: DataBlock.datasets(source, verbose)
    ### doc: DataBlock.dataloaders
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0001_fastai_is_it_a_bird.md
    
    ### doc: download_kaggle_dataset(comp, local_folder='', install='fastai "timm>=0.6.2.dev0")
    ### doc: check_subfolders_img(path, db=False)
    ### doc: randomdisplay(path, size, db=False)
    ### doc: remove_failed
    ### doc: ImageDataLoaders.from_folder
    ### doc: aug_transforms(size=128, min_scale=0.75)
    ### doc: vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()
    ### doc: learn.export(fname='export.pkl', pickle_module=pickle, pickle_protocol=2)
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    ### doc: ImageDataLoaders.from_name_func(path: 'str | Path', fnames: 'list', label_func: 'callable', **kwargs) -> 'DataLoaders'
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md
    
    jn by dates: ========



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

To run actual source code with snoop, but first search with `fastlistnbs` and `export_open_py`


```
fastlistnbs("srcode")
```

    ### src: itemgot(self:L, *idxs)
    ### src: itemgetter
    ### src: DataBlock.datasets((source, verbose)
    ### src: DataBlock.dataloaders
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0001_fastai_is_it_a_bird.md
    
    ### src: download_kaggle_dataset
    ### src: check_subfolders_img(path, db=False)
    ### src: randomdisplay(path, size, db=False)
    ### src: remove_failed
    ### src: check_sizes_img(files)
    ### src: ImageDataLoaders.from_folder
    ### src: aug_transforms(size=128, min_scale=0.75)
    ### src: vision_learner(dls, 'resnet26d', metrics=error_rate, path='.').to_fp16()
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.md
    
    ### src: ImageDataLoaders.from_name_func(path: 'str | Path', fnames: 'list', label_func: 'callable', **kwargs) -> 'DataLoaders'
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/0002_fastai_saving_a_basic_fastai_model.md
    
    jn by dates: ========



```
fastnbs("src: download_kaggle")
```


### <mark style="background-color: #ffff00">src:</mark>  <mark style="background-color: #FFFF00">download_kaggle</mark> _dataset




heading 3.


```python
export_open_py()
```

```python
exec(export_open_py().replace('pyfile', 'src_download'))
```

Next, heading 3
### ht: fu - debug every srcline without breaking



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb#src:-download_kaggle_dataset
)



[Open `0008_fastai_first_steps_road_to_top_part_1` in Jupyter Notebook on Kaggle](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)



```
fastnbs("src: export")
```


## <mark style="background-color: #ffff00">src:</mark>  <mark style="background-color: #FFFF00">export</mark> _nbdev




heading 2.


```python
#| export
# calling from a different notebook, nbdev_export() will cause error, this is why use exec() to call in a different notebook
export_nbdev = "import nbdev; nbdev.nbdev_export()"
```

```python
exec(export_nbdev)
```

```python

```

Next, heading 2
## download_kaggle_dataset



[Open `01_utils` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/lib/01_utils.ipynb#src:-export_nbdev
)


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
    ### ht: imports - imports fastdebug.utils on Kaggle
    ### ht: imports - import for vision
    ### ht: imports - upload and update mylib in kaggle
    ### ht: imports - create libs as datasets for kaggle with `create_libs_datasets`
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
    
    jn by dates: ========



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
    
    jn by dates: ========



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

## Search Mentors, like "radek"


```
# fastnbs("rd: process jsonl to df")
```


```
fastlistnbs("radek")
```


<style>.container { width:100% !important; }</style>


    ### rd: recsys - otto - get started - Andrew Ng recsys old videos 2022-11-7
    ### rd:  recsys - otto - get started - Andrew Ng on new recsys videos 2022-11-8
    #### rd: recsys - otto - get started - The best recsys intro video recommended by Radek
    #### rd: recsys - otto - get started - What is session based recommendations and current development of recsys
    #### rd: recsys - otto - get started - transformers - post recommended by Radek
    ### rd: recsys - otto - get started - Intro of recsys (video) by Xavier Amatriain
    ### rd: recsys - otto - get started - advices from Radek - how to get started on recsys with OTTO
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_Radek_OTTO.md
    
    ### rd: recsys - otto - get started - subset the training set based on entire sessions, train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session'], train[train.session.isin(lucky_sessions_train)]
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/kaggle_otto_Radek_covisiation_matrix.md
    
    ### rd: recsys - otto - get started - save a list or dict into pkl and load them
    ### rd: recsys - otto - get started - process jsonl to df
    ### rd: recsys - otto - get started - RAM needed to process data on Kaggle, 400MB jsonl takes up nearly 4GB ram
    ### rd: recsys - otto - get started - use parquet over csv, why and how
    ### rd: recsys - otto - get started - use parquet to instead of jsonl or csv to save space on disk
    ### rd: recsys - otto - get started - use `uint8` instead of `int` or `str` to reduce RAM usage by 9 times
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_radek_otto_get_started_process_data.md
    
    ### rd: recsys - otto - get started - copy and paste dataset path and use !ls to see what inside
    ### rd: recsys - otto - get started - pd.read_parquet('copy and paste the dataset path')
    ### rd: recsys - otto - get started - load a function from a pickle file with pickle5, with open as fh: and pick.load(fh)
    ### rd: recsys - otto - get started - convert int back to string, train.iloc[:1000].type.map(lambda i: id2type[i])
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_otto_access_parquet_dataset.md
    
    ### rd: recsys - otto - get started - Read parquet file with `pd.read_parquet`
    ### rd: recsys - otto - get started - find all the unique sessions, train.session.unique()
    ### rd: recsys - otto - get started - group 'aid' by 'session' and count, test.groupby('session')['aid'].count()
    ### rd: recsys - otto - get started - return natural log and also be super accurate in floating point, train.groupby('session')['aid'].count().apply(np.log1p).hist()
    ### rd: recsys - otto - get started - train and test sessions have no time intersection, datetime.datetime.fromtimestamp, question
    ### rd: recsys - otto - get started - no new items in test, len(set(test.aid.tolist()) - set(train.aid.tolist()))
    ### rd: recsys - otto - get started - train and test have different session length distributions, train.groupby('session')['aid'].count().describe()
    ### rd: recsys - otto - get started - define a session (a tracking period)
    ### rd: recsys - otto - get started - train and test have no common sessions, train.session.max(), test.session.min()
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_radek_otto_eda.md
    
    
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
    ### rd: cross-validation (???)
    ### rd: Sum up
    
    ## rd: How to Pick Up What They Put Down by swyx
    ### rd: Who are they?
    ### rd: What do you mean by “put down”?
    ### rd: How do I pick “it” up? 
    ### rd: Love what you do above and tag them
    ### rd: What happens when you do this?
    ### rd: Why does this work on them?
    ### rd: Why does this work on you?
    ### rd: How to get started
    
    ## rd: The Permissionless Apprenticeship by jackbutcher
    ### rd: How to have their attention
    ### rd: Why working for them without asking
    ### rd: How to work for them
    ### rd: The worse scenario of working for them
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/00_fastai_Meta_learning_Radek.md
    
    jn by dates: ========



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
    
    jn by dates: ========



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
    ### lk: Radek on animal language models
    ### lk: Radek and OTTO
    ### lk: ilovescience on kaggle
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_links_forums_kaggle_github.md
    
    jn by dates: ========



```
fastnbs("lk: Daniel's fastai forum pos")
```


### <mark style="background-color: #ffff00">lk:</mark>  <mark style="background-color: #ffff00">daniel's</mark>  <mark style="background-color: #ffff00">fastai</mark>  <mark style="background-color: #ffff00">forum</mark>  <mark style="background-color: #FFFF00">pos</mark> ts




heading 3.

  
[Exploring fastai with Excel and Python](https://forums.fast.ai/t/exploring-fastai-with-excel-and-python/97426)
  
[Hidden Docs of Fastcore](https://forums.fast.ai/t/hidden-docs-of-fastcore/98455)
  
[Help: reading fastcore and fastai docs](https://forums.fast.ai/t/help-reading-fastcore-and-fastai-docs/100168) 


Next, heading 3
### lk: Radek on animal language models



[Open `fastai_links_forums_kaggle_github` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/fastai_links_forums_kaggle_github.ipynb#lk:-Daniel's-fastai-forum-posts
)



```

```

## Search pyfiles


```
openpy()
```

    ['/Users/Natsume/Documents/fastdebug/fastdebug/fastaiTD.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_utils_remove_failed.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/kaggle_paddy_pt1.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_multiprocess_counter.ipynb',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fast_src_download_kaggle_dataset.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/kaggle_utils_dataset.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_utils_randomdisplay.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_check_subfolders_img.py']



```
openpy('kaggle_utils')
```


[Open `kaggle_utils_dataset` in Jupyter Notebook locally](http://localhost:8888/tree/fastdebug/kaggle_utils_dataset.py)



```

```
