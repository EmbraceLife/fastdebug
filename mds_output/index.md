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
# fastnbs("jn: even for questions not answered")
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
    ### jn: newsletter - Radek - do tell the world what you are up to /2022-11-9
    ### jn: newsletter - Radek - Don't worry about how your writing is written /2022-11-9
    ### jn: newsletter - Radek - let Kaggle be a life changing experience /2022-11-9
    ### jn: newsletter - Radek - how to do project based learning /2022-11-9
    ### jn: I preordered walk with fastai, but I will primarily focus on Radek and OTTO /2022-11-10
    ### jn: as Radek suggested, I should focus on notebooks and discussions shared on Kaggle OTTO to grow myself bit by bit /2022-11-10
    ### jn: I have experimented `process_data.ipynb` and `eda on OTTO.ipynb` by Radek /2022-11-10
    ### jn: I started to read https://github.com/otto-de/recsys-dataset which has a lot helpful info for beginners /2022-11-10
    ### jn: record all techniques learnt within a context (using "recsys - otto - get started") is much more helpful than extract it from the context it's learnt /2022-11-11
    ### jn: starting to work on Radek's [notebook](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files?scriptVersionId=109945227) on how to use the otto dataset in parquet file /2022-11-11
    ### jn: started the notebook [co-visitation matrix - simplified, imprvd logic 🔥](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic) and half way through. At first the notebook was intimidating even though Radek made the codes easy to follow, and after experiment line by line I can taste the refresh and interesting flavor from this notebook /2022-11-11
    ### jn: in the discussion word2vec is frequently mentioned, I will at some point [search embedding](https://forums.fast.ai/t/exploring-fastai-with-excel-and-python/97426/4?u=daniel) in fastai lectures (videos) for related content (search [colab](https://colab.research.google.com/drive/102_vWdSfRxw8SI61CED1B9uVE2cJxpCC?usp=sharing)) /2022-11-11
    ### jn: read radek explaining what is co-visitation matrix on [twitter](https://twitter.com/radekosmulski/status/1590909701797007360) for a second time (first time on Kaggle), it certainly makes more sense to me after last night's work on the notebook. The next notebook to explore is train-validation split notebook recommended [here](https://twitter.com/radekosmulski/status/1590909730469294080?s=20&t=hTs07NKjbCWpz5sAXxJLwg) /2022-11-11
    ### jn: another useful notebook study group [repo](https://github.com/jcatanza/Fastai-A-Code-First-Introduction-To-Natural-Language-Processing-TWiML-Study-Group) on NLP course by Rachel /2022-11-12
    ### jn: radek's summary on Xavier's lecture 1-2 on [twitter](https://twitter.com/radekosmulski/status/1565716248083566592) /2022-11-12
    ### jn: I don't believe I will succeed as a DL practitioner no matter how great examples or paths Jeremy, Radek etc have set for me. But because it's such as a great pleasure to witness these amazing human beings' stories, I will try my best to follow their paths in particular Radek's anyway despite how much I disbelieve I could make it /2022-11-12
    ### jn: Radek and Jeremy said it numerously to share your work publicly. I think one way I feel good to share is to tweet a list of what I learnt from each of Radek's kaggle notebooks. /2022-11-12
    ### jn: when I saw this big picture, I can feel there are a lot of new stuff to learn here. It is tempting and can lead to many hours go without result. However, I will follow Radek's advice to stay with the Kaggle discussion and kernels, make sure I make the most out of them first before I explore new waters /2022-11-13
    ### jn: I have finished the annotation of Radek's co-visitation matrix simplified notebook, and have a good feel of what does co-visitation matrix do here. But still there is more work on it can be learnt, e.g., all the previous notebooks [1st](https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix) [2nd](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost), [3nd](https://www.kaggle.com/code/ingvarasgalinskas/item-type-vs-multiple-clicks-vs-latest-items) which Radek based to built the simplifed notebook /2022-11-13
    ### jn: I need to finish up the annotation on Radek's EDA notebook properly /2022-11-13
    ### jn: following Radek's advice, I realize that there are also other great kagglers to learn from like [Chris Deotte](https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost) /2022-11-13
    ### jn: process_data revisited and get the name straight for search (done) /2022-11-14
    ### jn: there are too many notebooks to try and hard to figure out which one to try first, but I should work on local validation tracks by Radek next /2022-11-14
    ### jn: to gradually conquer my fear, I should run both subset and full dataset to submit as did Radek with co-visitation matrix notebook. check my [leaderboard](https://www.kaggle.com/competitions/otto-recommender-system/leaderboard#) /2022-11-14
    ### jn: a helpful thing to all is to organize all otto kernels from easy to advanced /2022-11-14
    ### jn: well, it is a defeated night that I can't run the local validation notebook as RAM runs out. Todos: I need to find a more efficient way to search for codes learnt from Radek's notebooks; maybe rd: recsys - otto - codes - actual funcs example /2022-11-14
    ### jn: what does `density` suppose to tell us and how it is calcuated? I have raised an [issue](https://github.com/otto-de/recsys-dataset/issues/2) in the otto dataset github /2022-11-14
    ### jn: I should find time to work on the [TBA](https://github.com/otto-de/recsys-dataset#dataset-statistics) of the dataset which helps me know better of the dataset and also it is the lower hanging fruits to contribute to the repo /2022-11-14
    ### jn: actually as the test set is not fully available due to competition, and after the competition test set statistics will be made available. I have done the basic statistics for the competition test set, can be seen [here](https://www.kaggle.com/code/danielliao/eda-an-overview-of-the-full-dataset?scriptVersionId=110913371) /2022-11-14
    ### jn: revisit process_data notebook and change get started to process_data in title. /2022-11-14
    ### jn: process_data revisited and get the name straight for search (done) /2022-11-14
    ##### jn: use help instead of doc_sig and chk from my utils will have less error throwing. Maybe remove doc_sig and chk from fastdebug.utils /2022-11-14
    #### jn: even for questions not answered, but writing up and revisit them, potential answers may flow to me more likely: use `/1000` could make the comparison more obvious? and without `/1000` can reveal the holiday season. /2022-11-14
    #### jn: (qusetion answered) Radek just informed me that dividing by 1000 on the timestamps can save RAM on Kaggle. /2022-11-14
    ### jn: eda revisit is done /2022-11-14
    ### jn: just got Radek very detailed and helpful replies, and I did the same with more experiments and questions on the effect of convert to int32 and divde by 1000 [here](https://www.kaggle.com/code/radek1/eda-an-overview-of-the-full-dataset/comments#2028533) /2022-11-14
    ### jn: still can't failed to run local validation notebook of Radek because of RAM running out even though no one else reported the issue. kind of stuck here, but other notebooks I run seem fine including the one using GPU. I should keep trying other notebooks. Also I have started to writing up a map of otto for myself and other beginners like myself. /2022-11-15
    ### jn: radek - newsletter - why popularity on twitter is not what you really need (>90% knows little of you) /2022-11-15
    ### jn: radek - newsletter - what kind of followers do you need (insider of industry) /2022-11-15
    ### jn: radek - newsletter - how did Radek get his third job? (sharing an interesting project) /2022-11-15
    ### jn: radek - newsletter - how to ask for help by sharing (an example) /2022-11-15
    ### jn: out of desperation (keep running out of RAM by copying and running local validation notebook from Radek, all versions are tried) I started to experiment every line of Radek's local validation notebook, and to my surprise I get the notebook to run without error this time by removing `/1000` from one line of code. Then I figured out what problem does `/1000` cause which seems to be a bug. However, I have no idea how such a 'bug' cause me to run out of RAM and no problem for Radek and seemingly everybody else.  /2022-11-16
    ### jn: I figured out a way to debug every line of a block of code contains loop of loop with easy: make the block of code into a function and use return and print and run the function. /2022-11-16
    ### jn: (todos) After reading Radek's 3rd newsletter, I had an idea to create a beginner's map to recsys through otto competition. However, I am very slow on writing it up as great kaggler like Radek and Chris are faster in making new notebooks than I can read and experiment them. What shall I do? /2022-11-16
    ### jn: (todos) I have finished debug every line of code for local validation notebook, but I still need to revisit it to form a big picture as a whole, also I need to read more on recall metrics of otto dataset  /2022-11-16
    ### jn: I want to finish the beginners' map first at least the first version of it to have a sense of the scale of work to cover /2022-11-17



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
# fastnbs("rd: src")
```


```
fastlistnbs("radek")
```

    
    ## rd: recsys - otto - process data - How to convert dataset from jsonl file into parquet file to save disk tremendously, and convert type column from string to uint8 and ts column from int64 to int32 by dividing 1000 first to reduce RAM usage significantly
    ### rd: recsys - otto - process data - create vocab or map between id and type using dict and list - id2type = ['clicks', 'carts', 'orders'] - type2id = {a: i for i, a in enumerate(id2type)}
    ### rd: recsys - otto - process data - detail annotated source code of json_to_df
    ### rd: recsys - otto - process data - chunks = pd.read_json(fn, lines=True, chunksize=100_000) - for chunk in chunks: - for row_idx, session_data in chunk.iterrows(): - session_data.session - for event in session_data.events: - aids.append(event['aid']) - tss.append(event['ts'])
    ### rd: recsys - otto - process data - and check RAM of a df and save df into parquet or csv file - test_df_str.memory_usage() - test_df_str.to_parquet('test_keep_str.parquet', index=False) - test_df_str.to_csv('test_keep_str.csv', index=False)
    ### rd: recsys - otto - process data - How much RAM does convert string to uint8 save? - test_df.type = test_df.type.astype(np.uint8)
    ### rd: recsys - otto - process data - convert `ts` from int64 to int32 without `/1000` will lose a lot of info - (test_df_updated.ts/1000).astype(np.int32)
    ### rd: recsys - otto - process data - dividing ts by 1000 only affect on milisecond accuracy not second accuracy - datetime.datetime.fromtimestamp((test_df_updated.ts/1000).astype(np.int32)[100])
    ### rd: recsys - otto - process data - How much RAM can be saved by dividing `ts` by 1000 - test_df_updated.ts = (test_df_updated.ts / 1000).astype(np.int32) 
    ### rd: recsys - otto - process data - use parquet to instead of jsonl or csv to save space on disk - os.path.getsize(path)
    ### rd: recsys - otto - process data - 400MB parquet file takes up nearly 4GB ram on Kaggle
    ### rd: recsys - otto - process data - use parquet over csv, why and how - test_df.type = test_df.type.astype(np.uint8) - test_df.to_parquet('test.parquet', index=False) - test_df.to_csv('test.csv', index=False)
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/kaggle-otto-process-data.md
    
    ### rd: recsys - otto - get started - Andrew Ng recsys old videos 2022-11-7
    ### rd:  recsys - otto - get started - Andrew Ng on new recsys videos 2022-11-8
    #### rd: recsys - otto - get started - The best recsys intro video recommended by Radek
    #### rd: recsys - otto - get started - What is session based recommendations and current development of recsys
    #### rd: recsys - otto - get started - transformers - post recommended by Radek
    ### rd: recsys - otto - get started - Intro of recsys (video) by Xavier Amatriain
    ### rd: recsys - otto - get started - advices from Radek - how to get started on recsys with OTTO
    ### jn: well, it is a defeated night that I can't run the local validation notebook as RAM runs out. Todos: I need to find a more efficient way to search for codes learnt from Radek's notebooks; maybe rd: recsys - otto - codes - actual funcs example /2022-11-14
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_Radek_OTTO.md
    
    ### rd: recsys - otto - big pic - what is candidate generation
    ### rd: recsys - otto - big pic - what is ranking
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_otto_radek_recsys_overview.md
    
    
    ## rd: recsys - otto - process data - How to convert dataset from jsonl file into parquet file to save disk tremendously, and convert type column from string to uint8 and ts column from int64 to int32 by dividing 1000 first to reduce RAM usage significantly
    ### rd: recsys - otto - process data - create vocab or map between id and type using dict and list - id2type = ['clicks', 'carts', 'orders'] - type2id = {a: i for i, a in enumerate(id2type)}
    ### rd: recsys - otto - process data - chunks = pd.read_json(fn, lines=True, chunksize=100_000) - for chunk in chunks: - for row_idx, session_data in chunk.iterrows(): - session_data.session - for event in session_data.events: - aids.append(event['aid']) - tss.append(event['ts'])
    ### rd: recsys - otto - process data - and check RAM of a df and save df into parquet or csv file - test_df_str.memory_usage() - test_df_str.to_parquet('test_keep_str.parquet', index=False) - test_df_str.to_csv('test_keep_str.csv', index=False)
    ### rd: recsys - otto - process data - How much RAM does convert string to uint8 save? - test_df.type = test_df.type.astype(np.uint8)
    ### rd: recsys - otto - process data - convert `ts` from int64 to int32 without `/1000` will lose a lot of info - (test_df_updated.ts/1000).astype(np.int32)
    ### rd: recsys - otto - process data - dividing ts by 1000 only affect on milisecond accuracy not second accuracy - datetime.datetime.fromtimestamp((test_df_updated.ts/1000).astype(np.int32)[100])
    ### rd: recsys - otto - process data - How much RAM can be saved by dividing `ts` by 1000 - test_df_updated.ts = (test_df_updated.ts / 1000).astype(np.int32) 
    ### rd: recsys - otto - process data - how much disk can be saved by saving jsonl to parquet - os.path.getsize(path)
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/kaggle-process-data-otto.md
    
    ### rd: recsys - otto - process data - save a list or dict into pkl and load them - id2type = ['clicks', 'carts', 'orders'] - type2id = {a: i for i, a in enumerate(id2type)} - pd.to_pickle(id2type, 'id2type.pkl')
    ### rd: recsys - otto - process data - how to process jsonl file to df (pd.read_json, chunk.iterrows) - chunks = pd.read_json(fn, lines=True, chunksize=2) - for chunk in chunks: - for row_idx, session_data in chunk.iterrows(): - sessions = [] - num_events = len(session_data.events) - sessions += ([session_data.session] * num_events)
    ### rd: src - recsys - otto - process data - jsonl_to_df
    ### rd: recsys - otto - process data - 400MB parquet file takes up nearly 4GB ram on Kaggle
    ### rd: recsys - otto - process data - use parquet over csv, why and how - test_df.type = test_df.type.astype(np.uint8) - test_df.to_parquet('test.parquet', index=False) - test_df.to_csv('test.csv', index=False)
    ### rd: recsys - otto - process data - use parquet to instead of jsonl or csv to save space on disk - os.path.getsize(path)
    ### rd: recsys - otto - process data - use `uint8` instead of `int` or `str` to reduce RAM usage by 9 times - test_df.memory_usage()
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_radek_otto_get_started_process_data.md
    
    ### rd: recsys - otto - access parquet - copy and paste dataset path - !ls ../input/otto-full-optimized-memory-footprint/
    ### rd: recsys - otto - access parquet - pd.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
    ### rd: recsys - otto - access parquet - load a function from a pickle file - import pickle5 as pickle - with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh: - id2type = pickle.load(fh)
    ### rd: recsys - otto - access parquet - access the first 1000 rows and convert int back to string, train.iloc[:1000].type.map(lambda i: id2type[i])
    ### rd: recsys - otto - access parquet - how to use Series, map, lambda, dict together - type_as_string.map(lambda i: type2id[i])
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/kaggle-access-parquet-otto.md
    
    ### rd: recsys - otto - get started - what are unigram, bigram, trigram models
    ### rd: recsys - otto - get started - How RNN improve on uni-bi-trigram models
    ### rd: recsys - otto - get started - How co-visitation matrix relate to embeddings?
    ### rd: recsys - otto - get started - limitations and resembles of co-visitation matrix
    ### rd: recsys - otto - get started - build word2vec as a resemblence of co-visitation matrix
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_otto_Radek_covisiation_matrix_discussion.md
    
    
    ## rd: recsys - otto - covisitation_simplified - load the dataset files and have a peek at them
    ### rd: recsys - otto - covisitation_simplified - read parquet and csv from dataset, pd.read_parquet('copy & paste the file path'), pd.read_csv('copy & paste path')
    
    ## rd: recsys - otto - covisitation_simplified - taking subset for fast experiment
    ### rd: recsys - otto - covisitation_simplified - subset based on entire sessions, train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session'], train[train.session.isin(lucky_sessions_train)]
    
    ## rd: recsys - otto - covisitation_simplified - some setups before training
    ### rd: recsys - otto - covisitation_simplified - use session column as index, subset_of_train.index = pd.MultiIndex.from_frame(subset_of_train[['session']]), [['session']] as Series not DataFrame
    ### rd: recsys - otto - covisitation_simplified - get starting and end timestamp for all sessions from start of training set to end of test set, train.ts.min() test.ts.max()
    ### rd: recsys - otto - covisitation_simplified - use defaultdict + Counter to count occurences, next_AIDs = defaultdict(Counter)
    ### rd: recsys - otto - covisitation_simplified - use test for training, get subsets, sessions, sessions_train, sessions_test - subsets = pd.concat([subset_of_train, subset_of_test]) - sessions = subsets.session.unique()
    
    ## rd: recsys - otto - covisitation_simplified - Training: when one aid occurred, keep track of what other aid occurred, how often they occurred. Do this for every aid across both train and test sessions.
    ### rd: recsys - otto - covisitation_simplified - loop every chunk_size number unique sessions - for i in range(0, sessions.shape[0], chunk_size): 
    ### rd: recsys - otto - covisitation_simplified - take a chunk_size number of sessions (each session in its entirety, ie, probably with many rows) from subsets as current_chunk - current_chunk = subsets.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index(drop=True) 
    ### rd: recsys - otto - covisitation_simplified - In current_chunk, from each session (in its entirety) only takes the last/latest 30 events/rows and combine them to update current_chunk (focus on the latest 30 events to save computations) - current_chunk = current_chunk.groupby('session', as_index=False).nth(list(range(-30,0))).reset_index(drop=True)
    ### rd: recsys - otto - covisitation_simplified - merge an session of its entirety onto itself (help to see the relation between one aid and every other aid within each session) - consecutive_AIDs = current_chunk.merge(current_chunk, on='session')
    ### rd: recsys - otto - covisitation_simplified - remove all the rows which aid_x == aid_y (remove the row when the two articles are the same) as they are meaningless - consecutive_AIDs[consecutive_AIDs.aid_x != consecutive_AIDs.aid_y]
    ### rd: recsys - otto - covisitation_simplified - add a column named 'days_elapsed' which shows how many days passed between the two aids in a session - consecutive_AIDs['days_elapsed'] = (consecutive_AIDs.ts_y - consecutive_AIDs.ts_x) / (24 * 60 * 60)
    ### rd: recsys - otto - covisitation_simplified - keep the rows if the two aids of a session are occurred within the same day on the right order (one after the other) -     consecutive_AIDs = consecutive_AIDs[(consecutive_AIDs.days_elapsed >= 0) & (consecutive_AIDs.days_elapsed <= 1)]
    ### rd: recsys - otto - covisitation_simplified - among all sessions/rows selected (regardless which session we are looking at), for each aid occurred, count how often the other aids are occurred -     for aid_x, aid_y in zip(consecutive_AIDs['aid_x'], consecutive_AIDs['aid_y']): next_AIDs[aid_x][aid_y] += 1
    ### rd: src - recsys - otto - covisitation_simplified
    ### rd: recsys - otto - covisitation_simplified - remove some data objects to save RAM
    
    ## rd: recsys - otto - covisitation_simplified - make predictions
    ### rd: recsys - otto - covisitation_simplified - group the test set by session, under each session, put all aids into a list, and put all action types into another list - test.reset_index(drop=True).groupby('session')['aid'].apply(list)
    ### rd: recsys - otto - covisitation_simplified - setup, create some containers, such as labels, no_data, no_data_all_aids - type_weight_multipliers = {0: 1, 1: 6, 2: 3} - session_types = ['clicks', 'carts', 'orders']
    ### rd: recsys - otto - covisitation_simplified - loop every session, access all of its aids and types - for AIDs, types in zip(test_session_AIDs, test_session_types):
    ### rd: recsys - otto - covisitation_simplified - when there are >= 20 aids in a session: 
    ### rd: recsys - otto - covisitation_simplified - assign logspaced weight to each aid under each session, as the latter aids should have higher weight/probability to occur than the earlier aids. - if len(AIDs) >= 20: - weights=np.logspace(0.1,1,len(AIDs),base=2, endpoint=True)-1
    ### rd: recsys - otto - covisitation_simplified - create a defaultdict (if no value to the key, set value to 0) - aids_temp=defaultdict(lambda: 0)
    ### rd: recsys - otto - covisitation_simplified - loop each aid, weight, event_type of a session: - for aid,w,t in zip(AIDs,weights,types):
    ### rd: recsys - otto - covisitation_simplified - Within each session, accumulate the weight for each aid based on its occurences, event_type and logspaced weight; save the accumulated weight as value and each aid as key into a defaultdict (aids_temp), no duplicated aid here in this dict, and every session has its own aid_temp - aids_temp[aid] += w * type_weight_multipliers[t]
    ### rd: recsys - otto - covisitation_simplified - sort a defaultdict from largest weight to smallest weight of all aids in each session???, and then put its keys into a list named sorted_aids - sorted_aids=[k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
    ### rd: recsys - otto - covisitation_simplified - store the first 20 aids (the most weighted or most likely aids to be acted upon within a session) into the list 'labels' -         labels.append(sorted_aids[:20])
    ### rd: recsys - otto - covisitation_simplified - when there are < 20 aids in a session: - if len(AIDs) > 10:
    ### rd: recsys - otto - covisitation_simplified - within each test session, reverse the order of AIDs, remove the duplicated, put into a list, reassign it to AIDs - AIDs = list(dict.fromkeys(AIDs[::-1]))
    ### rd: recsys - otto - covisitation_simplified - keep track the length of AIDs and create an empty list named candidates - AIDs_len_start = len(AIDs)
    ### rd: recsys - otto - covisitation_simplified - (within a session) for each AID inside AIDs: if AID is in the keys of next_AIDs (from training), then take the 20 most common other aids occurred (from next_AIDs) when AID occurred, into a list and add this list into the list named candidate (not a list of list, just a merged list). Each candidate in its full size has len(AIDs) * 20 number of other aids, which can have duplicated ids. -         for AID in AIDs: - if AID in next_AIDs: - candidates = candidates + [aid for aid, count in next_AIDs[AID].most_common(20)]
    ### rd: recsys - otto - covisitation_simplified - find the first 40 most common aids in a candidate (for a session); and if they (these aids) are not found in AIDs then merge them into AIDs list, so that a session has a updated AIDs list (which most likely to occur) - AIDs += [AID for AID, cnt in Counter(candidates).most_common(40) if AID not in AIDs]
    ### rd: recsys - otto - covisitation_simplified - give the first 20 aids from AIDs to labels (a list); count how many test sessions whose aids are not seen in next_AIDs from training; count how many test sessions don't receive additional aids from next_AIDs  labels.append(AIDs[:20]) - if candidates == []: no_data += 1 - if AIDs_len_start == len(AIDs): no_data_all_aids += 1
    ### rd: src - recsys - otto - covisitation_simplified
    
    ## rd: recsys - otto - covisitation_simplified - prepare results to CSV
    ### rd: recsys - otto - covisitation_simplified - make a list of lists (labels) into a list of strings (labels_as_strings) - labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]
    ### rd: recsys - otto - covisitation_simplified - give each list of label strings a session number - predictions = pd.DataFrame(data={'session_type': test_session_AIDs.index, 'labels': labels_as_strings})
    ### rd: recsys - otto - covisitation_simplified - multi-objective means 'clicks', 'carts', and 'orders'; and we make the same predictions on them - session_types = ['clicks', 'carts', 'orders'] - for st in session_types: - modified_predictions = predictions.copy() - modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}' - prediction_dfs.append(modified_predictions)
    ### rd: recsys - otto - covisitation_simplified - get the csv file ready, stack on each other. - submission = pd.concat(prediction_dfs).reset_index(drop=True) - submission.to_csv('submission.csv', index=False) - submission.head()
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_radek_covisitation_matrix_simplifed.md
    
    ### rd: recsys - otto - eda - Read parquet file with `pd.read_parquet`
    ### rd: recsys - otto - eda - load object from a pikle file - import pickle5 as pickle - with open('../input/otto-full-optimized-memory-footprint/id2type.pkl', "rb") as fh: - id2type = pickle.load(fh)
    ### rd: recsys - otto - eda - explore how lightweighted is inference vs training - test.shape[0]/train.shape[0]
    ### rd: recsys - otto - eda - how much more of the unique sessions in train vs in test sets - test.session.unique().shape[0]/train.session.unique().shape[0]
    ### rd: recsys - otto - eda - are sessions in test shorter than sessions in train?
    #### rd: recsys - otto - eda - compare the histograms of train vs test on the natural log of the amount of aids in each session. The comparison of histogram can give us a sense of distribution difference - test.groupby('session')['aid'].count().apply(np.log1p).hist()
    #### rd: recsys - otto - eda - why np.log1p over np.log? return natural log and also be super accurate in floating point, train.groupby('session')['aid'].count().apply(np.log1p).hist()
    ### rd: recsys - otto - eda - whether train and test sessions have time intersection - datetime.datetime.fromtimestamp(test.ts.min()/1000) - 
    ### rd: recsys - otto - eda - are there new items in test not in train? len(set(test.aid.tolist()) - set(train.aid.tolist()))
    ### rd: recsys - otto - eda - describe (check distribution of ) total number of aids of sessions between train and test, train.groupby('session')['aid'].count().describe()
    ### rd: recsys - otto - eda - define a session (a tracking period)
    ### rd: recsys - otto - eda - whether train and test have no common sessions, - train.session.max(), test.session.min()
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/fastai_kaggle_radek_otto_eda.md
    
    
    ## rd: recsys - otto - last 20 aids - The purpose of this notebook - Grabbing the last 20 aid for each session of the test set, use them as prediction and submit or run local validation to see how powerful can the last 20 aids be.
    ### rd: recsys - otto - last 20 aid - sort df by two columns - test.sort_values(['session', 'ts'])
    ### rd: recsys - otto - last 20 aid - take last 20 aids from each session - test.groupby('session')['aid'].apply(lambda x: list(x)[-20:])
    ### rd: recsys - otto - last 20 aid - loop through a Series with session as index with a list as the only column's value - for session, aids in session_aids.iteritems():
    ### rd: recsys - otto - last 20 aid - turn a list into a string with values connected with empty space - labels.append(' '.join([str(a) for a in aids]))
    ### rd: recsys - otto - last 20 aid - make a df from a dict with two lists as values - pd.DataFrame({'session_type': session_type, 'labels': labels})
    /Users/Natsume/Documents/fastdebug/mds/fastai_notebooks/kaggle-otto-rd-last-20-aids.md
    
    
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


## <mark style="background-color: #ffff00">rd:</mark>  recsys - otto - covisitation_simplified - taking subset for fast <mark style="background-color: #FFFF00">experiment</mark> 




heading 2.



### rd: recsys - otto - covisitation_simplified - subset based on entire sessions, train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session'], train[train.session.isin(lucky_sessions_train)]

```python
fraction_of_sessions_to_use = 0.000001 # 0.001 is recommended, but 0.000001 can finish in less than 4 minutes
```

```python
train.shape # how many rows
```

```python
train.drop_duplicates(['session']).shape # how many unique sessions (drop rows with the same session)
```

```python
subset_of_train_no_duplicate = train.sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
subset_of_train_no_duplicate.shape # take 0.000001 from entire train
```

```python
lucky_sessions_train = train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
lucky_sessions_train.shape # take 0.000001 from a dataframe in which each row is an unique session
```

```python
lucky_sessions_train.head()
lucky_sessions_train.reset_index(drop=True).head() # make index easier to see
```

```python
train.session.isin(lucky_sessions_train).sum() # how many rows under the 13 sessions
```

```python
if fraction_of_sessions_to_use != 1:
    lucky_sessions_train = train.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
    subset_of_train = train[train.session.isin(lucky_sessions_train)]
    
    lucky_sessions_test = test.drop_duplicates(['session']).sample(frac=fraction_of_sessions_to_use, random_state=42)['session']
    subset_of_test = test[test.session.isin(lucky_sessions_test)]
else:
    subset_of_train = train
    subset_of_test = test
```

```python
subset_of_train.shape
```

```python
subset_of_train
```

Next, heading 2
## rd: recsys - otto - covisitation_simplified - some setups before training



[Open `fastai_kaggle_radek_covisitation_matrix_simplifed` in Jupyter Notebook locally](http://localhost:8888/tree/nbs/fastai_notebooks/fastai_kaggle_radek_covisitation_matrix_simplifed.ipynb#rd:-recsys---otto---covisitation_simplified---taking-subset-for-fast-experiment
)



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

    ['/Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_utils_remove_failed.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/kaggle_paddy_pt1.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fast_src_download_kaggle_dataset.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/kaggle_utils_dataset.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_utils_randomdisplay.py',
     '/Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_check_subfolders_img.py']



```
openpy("utils dataset")
```

    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 6
    # make sure fastkaggle is install and imported
    import os
    
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    /Users/Natsume/Documents/fastdebug/fastdebug/kaggle_utils_dataset.py



[Open `kaggle_utils_dataset` in Jupyter Notebook locally](http://localhost:8888/tree/fastdebug/kaggle_utils_dataset.py)



```
openpy("all")
```

    # %%
    # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb.
    
    # %% auto 0
    __all__ = ['home', 'comp', 'path', 'test_files', 'train_files']
    
    # %% ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb 2
    # this is a notebook for receiving code snippet from other notebooks
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 8
    # make sure fastkaggle is install and imported
    import os
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 9
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 11
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 12
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 14
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 49
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 56
    # set seed for reproducibility
    set_seed(42)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 65
    # map the content of all subfolders of images
    check_subfolders_img(path)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 68
    # to extract all images from a folder recursively (for subfolders)
    test_files = get_image_files(path/"test_images")
    train_files = get_image_files(path/"train_images")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 78
    # to display a random image from a path 
    randomdisplay(train_files, 200)
    randomdisplay(path/"train_images/dead_heart", 128)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 88
    # remove all images which fail to open
    remove_failed(path)
    
    
    # %%
    @snoop
    def remove_failed(path):
    #     from fastai.vision.all import get_image_files, parallel
        print("before running remove_failed:")
        check_subfolders_img(path)
        failed = verify_images(get_image_files(path))
        print(f"total num: {len(get_image_files(path))}")
        print(f"num offailed: {len(failed)}")
        failed.map(Path.unlink)
        print()
        print("after running remove_failed:")
        check_subfolders_img(path)
    # File:      ~/Documents/fastdebug/fastdebug/utils.py
    # Type:      function
    
    
    # %%
    remove_failed(path)
    
    # %%
    /Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_utils_remove_failed.py
    # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb.
    
    # %% auto 0
    __all__ = ['home', 'comp', 'path', 'test_files', 'train_files']
    
    # %% ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb 2
    # this is a notebook for receiving code snippet from other notebooks
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 6
    # make sure fastkaggle is install and imported
    import os
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 7
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 9
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 10
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 12
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 47
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 54
    # set seed for reproducibility
    set_seed(42)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 63
    # map the content of all subfolders of images
    check_subfolders_img(path)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 66
    # to extract all images from a folder recursively (for subfolders)
    test_files = get_image_files(path/"test_images")
    train_files = get_image_files(path/"train_images")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 76
    # to display a random image from a path 
    randomdisplay(train_files, 200)
    randomdisplay(path/"train_images/dead_heart", 128)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 86
    # remove all images which fail to open
    remove_failed(path)/Users/Natsume/Documents/fastdebug/fastdebug/kaggle_paddy_pt1.py
    # %%
    # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb.
    
    # %% auto 0
    __all__ = ['home', 'comp', 'path']
    
    # %% ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb 2
    # this is a notebook for receiving code snippet from other notebooks
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 7
    # make sure fastkaggle is install and imported
    import os
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 8
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 10
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 11
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 13
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 49
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 57
    @snoop
    def download_kaggle_dataset(competition, local_folder='', install=''):
        "override from fastkaggle.core.setup_comp. \
    Return a path of the `local_folder` where `competition` dataset stored, \
    downloading it if needed"
        if iskaggle:
            if install:
                os.system(f'pip install -Uqq {install}')
            return Path('../input')/competition
        else:
            path = Path(local_folder + competition)
            api = import_kaggle()
            if not path.exists():
                import zipfile
                api.competition_download_cli(str(competition), path=path)
                zipfile.ZipFile(f'{local_folder + competition}.zip').extractall(str(local_folder + competition))
            return path
    # File:      ~/Documents/fastdebug/fastdebug/utils.py
    # Type:      function
    
    
    # %%
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    
    # %%
    /Users/Natsume/Documents/fastdebug/fastdebug/fast_src_download_kaggle_dataset.py
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 6
    # make sure fastkaggle is install and imported
    import os
    
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    /Users/Natsume/Documents/fastdebug/fastdebug/kaggle_utils_dataset.py
    # %%
    # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb.
    
    # %% auto 0
    __all__ = ['home', 'comp', 'path', 'test_files', 'train_files']
    
    # %% ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb 2
    # this is a notebook for receiving code snippet from other notebooks
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 8
    # make sure fastkaggle is install and imported
    import os
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 9
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 11
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 12
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 14
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 49
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 56
    # set seed for reproducibility
    set_seed(42)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 65
    # map the content of all subfolders of images
    check_subfolders_img(path)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 68
    # to extract all images from a folder recursively (for subfolders)
    test_files = get_image_files(path/"test_images")
    train_files = get_image_files(path/"train_images")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 78
    # to display a random image from a path 
    randomdisplay(train_files, 200)
    randomdisplay(path/"train_images/dead_heart", 128)
    
    
    # %%
    @snoop
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
    # File:      ~/Documents/fastdebug/fastdebug/utils.py
    # Type:      function
    
    
    # %%
    randomdisplay(train_files, 200)
    
    # %%
    /Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_utils_randomdisplay.py
    # %%
    # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb.
    
    # %% auto 0
    __all__ = ['home', 'comp', 'path']
    
    # %% ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb 2
    # this is a notebook for receiving code snippet from other notebooks
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 8
    # make sure fastkaggle is install and imported
    import os
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 9
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 11
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 12
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 14
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 49
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 56
    # set seed for reproducibility
    set_seed(42)
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 64
    check_subfolders_img(path)
    
    
    # %%
    @snoop
    def check_subfolders_img(path:Path, # a Path object
                             db=False):
        "map the image contents of all subfolders of the path"
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
    # File:      ~/Documents/fastdebug/fastdebug/utils.py
    # Type:      function
    
    
    # %%
    check_subfolders_img(path)
    
    # %%
    /Users/Natsume/Documents/fastdebug/fastdebug/fastai_src_check_subfolders_img.py



```
openpy('fast_src_download_kaggle_dataset.py')
```

    # %%
    # AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb.
    
    # %% auto 0
    __all__ = ['home', 'comp', 'path']
    
    # %% ../nbs/fastai_notebooks/0000_fastai_kaggle_paddy_001.ipynb 2
    # this is a notebook for receiving code snippet from other notebooks
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 7
    # make sure fastkaggle is install and imported
    import os
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 8
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 10
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 11
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 13
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 49
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    
    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 57
    @snoop
    def download_kaggle_dataset(competition, local_folder='', install=''):
        "override from fastkaggle.core.setup_comp. \
    Return a path of the `local_folder` where `competition` dataset stored, \
    downloading it if needed"
        if iskaggle:
            if install:
                os.system(f'pip install -Uqq {install}')
            return Path('../input')/competition
        else:
            path = Path(local_folder + competition)
            api = import_kaggle()
            if not path.exists():
                import zipfile
                api.competition_download_cli(str(competition), path=path)
                zipfile.ZipFile(f'{local_folder + competition}.zip').extractall(str(local_folder + competition))
            return path
    # File:      ~/Documents/fastdebug/fastdebug/utils.py
    # Type:      function
    
    
    # %%
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    
    # %%
    /Users/Natsume/Documents/fastdebug/fastdebug/fast_src_download_kaggle_dataset.py



[Open `fast_src_download_kaggle_dataset` in Jupyter Notebook locally](http://localhost:8888/tree/fastdebug/fast_src_download_kaggle_dataset.py)



```
openpy("utils dataset")
```

    # %% ../nbs/fastai_notebooks/0008_fastai_first_steps_road_to_top_part_1.ipynb 6
    # make sure fastkaggle is install and imported
    import os
    
    try: import fastkaggle
    except ModuleNotFoundError:
        os.system("pip install -Uq fastkaggle")
    
    from fastkaggle import *
    
    # use fastdebug.utils 
    if iskaggle: os.system("pip install nbdev snoop")
    
    if iskaggle:
        path = "../input/fastdebugutils0"
        import sys
        sys.path
        sys.path.insert(1, path)
        import utils as fu
        from utils import *
    else: 
        from fastdebug.utils import *
        import fastdebug.utils as fu
    
    # import for dealing with vision problem
    from fastai.vision.all import *
    
    # download (if necessary and return the path of the dataset)
    home = "/Users/Natsume/Documents/fastdebug/kaggle_datasets/"
    comp = 'paddy-disease-classification' # https://www.kaggle.com/competitions/paddy-disease-classification/submissions
    path = download_kaggle_dataset(comp, local_folder=home, install='fastai "timm>=0.6.2.dev0"')
    # path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
    /Users/Natsume/Documents/fastdebug/fastdebug/kaggle_utils_dataset.py



[Open `kaggle_utils_dataset` in Jupyter Notebook locally](http://localhost:8888/tree/fastdebug/kaggle_utils_dataset.py)



```

```
