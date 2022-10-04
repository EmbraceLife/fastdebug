# 0003_fastai_which_image_model_best

*The data, concept, and initial implementation of this notebook was done in Colab by Ross Wightman, the creator of timm. I (Jeremy Howard) did some refactoring, curating, and expanding of the analysis, and added prose.*

## timm

[PyTorch Image Models](https://timm.fast.ai/) (timm) is a wonderful library by Ross Wightman which provides state-of-the-art pre-trained computer vision models. It's like Huggingface Transformers, but for computer vision instead of NLP (and it's not restricted to transformers-based models)!

Ross has been kind enough to help me understand how to best take advantage of this library by identifying the top models. I'm going to share here so of what I've learned from him, plus some additional ideas.

## how to git clone TIMM analysis data; how to enter a directory with %cd

Ross regularly benchmarks new models as they are added to timm, and puts the results in a CSV in the project's GitHub repo. To analyse the data, we'll first clone the repo:


```
#| eval: false
! git clone --depth 1 https://github.com/rwightman/pytorch-image-models.git
%cd pytorch-image-models/results
```

## how to read a csv file with pandas

Using Pandas, we can read the two CSV files we need, and merge them together.


```
import pandas as pd
df_results = pd.read_csv('results-imagenet.csv')
```

## how to merge data with pandas; how to create new column with pandas; how to string extract with regex expression; how to select columns up to a particular column with pandas; how to do loc in pandas; how to select a group of columns using str.contains and regex

We'll also add a "family" column that will allow us to group architectures into categories with similar characteristics:

Ross has told me which models he's found the most usable in practice, so I'll limit the charts to just look at these. (I also include VGG, not because it's good, but as a comparison to show how far things have come in the last few years.)


```
def get_data(part, col):
    df = pd.read_csv(f'benchmark-{part}-amp-nhwc-pt111-cu113-rtx3090.csv').merge(df_results, on='model')
    df['secs'] = 1. / df[col]
    df['family'] = df.model.str.extract('^([a-z]+?(?:v2)?)(?:\d|_|$)')
    df = df[~df.model.str.endswith('gn')]
    df.loc[df.model.str.contains('in22'),'family'] = df.loc[df.model.str.contains('in22'),'family'] + '_in22'
    df.loc[df.model.str.contains('resnet.*d'),'family'] = df.loc[df.model.str.contains('resnet.*d'),'family'] + 'd'
    return df[df.family.str.contains('^re[sg]netd?|beit|convnext|levit|efficient|vit|vgg')]
```


```
df = get_data('infer', 'infer_samples_per_sec')
```

## Inference results

### how to scatterplot with plotly.express; how to set the plot's width, height, size, title, x, y, log_x, color, hover_name, hover_data; 

Here's the results for inference performance (see the last section for training performance). In this chart:

- the x axis shows how many seconds it takes to process one image (**note**: it's a log scale)
- the y axis is the accuracy on Imagenet
- the size of each bubble is proportional to the size of images used in testing
- the color shows what "family" the architecture is from.

Hover your mouse over a marker to see details about the model. Double-click in the legend to display just one family. Single-click in the legend to show or hide a family.

**Note**: on my screen, Kaggle cuts off the family selector and some plotly functionality -- to see the whole thing, collapse the table of contents on the right by clicking the little arrow to the right of "*Contents*".


```
import plotly.express as px
w,h = 1000,800

def show_all(df, title, size):
    return px.scatter(df, width=w, height=h, size=df[size]**2, title=title,
        x='secs',  y='top1', log_x=True, color='family', hover_name='model', hover_data=[size])
```


```
show_all(df, 'Inference', 'infer_img_size')
```

That number of families can be a bit overwhelming, so I'll just pick a subset which represents a single key model from each of the families that are looking best in our plot. I've also separated convnext models into those which have been pretrained on the larger 22,000 category imagenet sample (`convnext_in22`) vs those that haven't (`convnext`). (Note that many of the best performing models were trained on the larger sample -- see the papers for details before coming to conclusions about the effectiveness of these architectures more generally.)

### how to scatterplot on a subgroup of data using regex and plotly


```
subs = 'levit|resnetd?|regnetx|vgg|convnext.*|efficientnetv2|beit'
```

In this chart, I'll add lines through the points of each family, to help see how they compare -- but note that we can see that a linear fit isn't actually ideal here! It's just there to help visually see the groups.


```
def show_subs(df, title, size):
    df_subs = df[df.family.str.fullmatch(subs)]
    return px.scatter(df_subs, width=w, height=h, size=df_subs[size]**2, title=title,
        trendline="ols", trendline_options={'log_x':True},
        x='secs',  y='top1', log_x=True, color='family', hover_name='model', hover_data=[size])
```


```
show_subs(df, 'Inference', 'infer_img_size')
```

From this, we can see that the *levit* family models are extremely fast for image recognition, and clearly the most accurate amongst the faster models. That's not surprising, since these models are a hybrid of the best ideas from CNNs and transformers, so get the benefit of each. In fact, we see a similar thing even in the middle category of speeds -- the best is the ConvNeXt, which is a pure CNN, but which takes advantage of ideas from the transformers literature.

For the slowest models, *beit* is the most accurate -- although we need to be a bit careful of interpreting this, since it's trained on a larger dataset (ImageNet-21k, which is also used for *vit* models).

I'll add one other plot here, which is of speed vs parameter count. Often, parameter count is used in papers as a proxy for speed. However, as we see, there is a wide variation in speeds at each level of parameter count, so it's really not a useful proxy.

(Parameter count may be be useful for identifying how much memory a model needs, but even for that it's not always a great proxy.)


```
px.scatter(df, width=w, height=h,
    x='param_count_x',  y='secs', log_x=True, log_y=True, color='infer_img_size',
    hover_name='model', hover_data=['infer_samples_per_sec', 'family']
)
```

## Training results

We'll now replicate the above analysis for training performance. First we grab the data:


```
tdf = get_data('train', 'train_samples_per_sec')
```

Now we can repeat the same *family* plot we did above:


```
show_all(tdf, 'Training', 'train_img_size')
```

...and we'll also look at our chosen subset of models:


```
show_subs(tdf, 'Training', 'train_img_size')
```

Finally, we should remember that speed depends on hardware. If you're using something other than a modern NVIDIA GPU, your results may be different. In particular, I suspect that transformers-based models might have worse performance in general on CPUs (although I need to study this more to be sure).

### convert ipynb to md


```
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>



```
ipy2md()
```

    [jupytext] Reading /Users/Natsume/Documents/fastdebug/nbs/2022part1/0003_fastai_which_image_model_best.ipynb in format ipynb
    [jupytext] Writing /Users/Natsume/Documents/fastdebug/nbs/2022part1/0003_fastai_which_image_model_best.md
    cp to : /Users/Natsume/Documents/divefastai/Debuggable/jupytext
    move to : /Users/Natsume/Documents/fastdebug/mds/2022part1/


    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/2022part1/0003_fastai_which_image_model_best.ipynb to markdown
    [NbConvertApp] Writing 7289 bytes to /Users/Natsume/Documents/fastdebug/nbs/2022part1/0003_fastai_which_image_model_best.md


    move to : /Users/Natsume/Documents/fastdebug/mds_output



```
fastnbs("export model")
```


## how to  <mark style="background-color: #ffff00">export</mark>   <mark style="background-color: #FFFF00">model</mark>  to a pickle file and download it from kaggle






Now we can export our trained `Learner`. This contains all the information needed to run the model:

```python
#|eval: false
learn.export('model.pkl')
```

Finally, open the Kaggle sidebar on the right if it's not already, and find the section marked "Output". Open the `/kaggle/working` folder, and you'll see `model.pkl`. Click on it, then click on the menu on the right that appears, and choose "Download". After a few seconds, your model will be downloaded to your computer, where you can then create your app that uses the model.






[Open `0002_fastai_Saving_Model_fastai` in Jupyter Notebook](http://localhost:8888/tree/nbs/2022part1/0002_fastai_Saving_Model_fastai.ipynb)



```
fastlistnbs()
```

    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0001_fastai_is_it_a_bird.md
    ## Useful Course sites
    ## How to use autoreload
    ## How to install and update libraries
    ## Know a little about the libraries
    ### what is fastai
    ### what is duckduckgo
    ## How to use fastdebug with fastai notebooks
    ### how to use fastdebug
    ### Did I document it in a notebook before?
    ### Did I document it in a src before?
    ## how to search and get a url of an image; how to download with an url; how to view an image;
    ### how to create folders using path; how to search and download images in folders; how to resize images 
    ## Train my model
    ### How to find and unlink images not properly downloaded
    ### How to create a DataLoaders with DataBlock; how to view data with it
    ### How to build my model with dataloaders and pretrained model; how to train my model
    ### How to predict with my model; how to avoid running cells in nbdev_prepare
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0002_fastai_Saving_Model_fastai.md
    ## what to import to handle vision problems in fastai
    ## how to download and decompress datasets prepared by fastai
    ## how to tell it is a cat by reading filename
    ## how to create dataloaders with `from_name_func`
    ## how to create a pretrained model with resnet18 and error_rate; how to fine tune it 3 epochs
    ## how to export model to a pickle file and download it from Kaggle
    ## how to convert ipynb to md
    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0003_fastai_which_image_model_best.md
    ## timm
    ## how to git clone TIMM analysis data; how to enter a directory with %cd
    ## how to read a csv file with pandas
    ## how to merge data with pandas; how to create new column with pandas; how to string extract with regex expression; how to select columns up to a particular column with pandas; how to do loc in pandas; how to select a group of columns using str.contains and regex
    ## Inference results
    ### how to scatterplot with plotly.express; how to set the plot's width, height, size, title, x, y, log_x, color, hover_name, hover_data; 
    ### how to scatterplot on a subgroup of data using regex and plotly
    ## Training results
    ### convert ipynb to md
    
    /Users/Natsume/Documents/fastdebug/mds/lib/utils.md
    ## setup for exporting to a module
    ## how to get current notebook's name, path and url
    ## how to convert ipynb to md automatically; how to run commands in python
    ## Autoreload for every notebook
    ## Expand cells
    ## Import fastcore env
    ## to inspect a class
    ### get the docs for each function of a class
    ## is it a metaclass?
    ## is it a decorator
    ### handle all kinds of exceptions for evaluating retn 
    ## whatinside a module of a library
    ### show the type of objects inside `__all__`
    ### working for fastdebug.core
    ### to show Fastdb methods
    ## whichversion of a library
    ## fastview
    ## fastscrs
    ## getrootport
    ## jn_link
    ## get_all_nbs
    ### get all nbs path for both md and ipynb
    ### add index.ipynb
    ## openNB
    ## highlight
    ## display_md
    ## display_block
    ### handle both file path and file content at the same time
    ## fastnbs
    ## fastcodes
    ## fastnotes
    ### multiple folders
    ## fastlistnbs
    ## fastlistsrcs
    ## Best practice of fastdebug.core
    ## Best practice of fastdebug.utils
    ## Export
    
    /Users/Natsume/Documents/fastdebug/mds/lib/00_core.md
    ## make life easier with defaults  
    ## globals() and locals()
    ## Execute strings
    ### new variable or updated variable by exec will only be accessible from locals()
    ### eval can override its own globals() and locals()
    ### when exec update existing functions
    ### when the func to be udpated involve other libraries
    ### inside a function, exec() allow won't give you necessary env from function namespace
    ### magic of `exec(b, globals().update(locals()))`
    ### Bring variables from a func namespace to the sideout world
    ### globals() in a cell vs globals() in a func
    ## make a colorful string
    ## align text to the most right
    ## printsrcwithidx
    ### print the entire source code with idx from 0
    ### print the whole src with idx or print them in parts
    ### use cmts from dbprint to print out src with comments
    ### no more update for printsrcwithidx, for the latest see Fastdb.print
    ## print out src code
    ### basic version
    ### print src with specific number of lines
    ### make the naming more sensible
    ### Allow a dbline occur more than once
    ### adding idx to the selected srclines
    #### printsrclinewithidx
    ### dblines can be string of code or idx number
    ### avoid multi-occurrance of the same srcline
    ## dbprint on expression
    ### basic version
    ### insert dbcode and make a new dbfunc
    ### Bring outside namespace variables into exec()
    ### Bring what inside the func namespace variables to the outside world
    ### Adding g = locals() to dbprintinsert to avoid adding env individually
    ### enable srclines to be either string or int 
    ### enable = to be used as assignment in codes
    ### avoid adding "env=g" for dbprintinsert
    ### collect cmt for later printsrcwithidx to print comments together
    ### make sure only one line with correct idx is debugged
    ### avoid typing "" when there is no codes
    ### no more update for dbprint, for the latest see Fastdb.dbprint
    ### use dbprint to override the original official code without changing its own pyfile
    ## dbprintinsert
    ### Run and display the inserted dbcodes 
    ### use locals() inside the dbsrc code to avoid adding env individually
    ### enable dbprintinsert to do exec on a block of code
    ## printrunsrclines() 
    ### Examples
    #### simple example
    #### complex example
    ### insert a line after each srcline to add idx
    ### add correct indentation to each inserted line
    #### count the indentation for each srcline
    ### indentation special case: if, else, for, def
    ### remove pure comments or docs from dbsrc
    ### print out the srclines which get run
    ### Make sure all if, else, for get printed
    ### Put all together into the function printrunsrclines()
    #### no more renaming of foo
    #### add example as a param into the function
    #### improve on search for `if`, else, for, def to avoid errors for more examples
    #### remove an empty line with indentation
    ### make it work
    ### more difficult examples to test printrunsrc()
    ## Make fastdebug a class
    ### improve on the line idx readability
    ### collect cmt from dbprint and print
    ### make sure only the line with correct idx is debugged
    ### having "" or "   " inside codes without causing error
    ### replace Fastdb.printsrcwithdix with Fastdb.print
    ### add idx to dbsrc when showdbsrc=True
    ### not load the inner locals() to outenv can prevent mysterious printing of previous db messages
    ### using @patch to enable docs for instance methods like `dbprint` and `print`
    ### move param env into `__init__`
    ### Add example to the obj
    ### Take not only function but also class
    ### To remove the necessity of self.takExample()
    ### Try to remove g = locals()
    ### Make sure `showdbsrc=True` give us the line starting with 'dbprintinsert'
    ### Make sure `showdbsrc=True` give us info on changes in g or outenv
    ### exit and print a warning message: idx has to be int
    ### handle errors by codes with trailing spaces 
    ### showdbsrc=True, check whether Fastdb.dbprint and fdb.dbprint are same object using `is`
    ### remove unnecessary db printout when showdbsrc=True and add printout to display sections
    ### raise TypeError when decode are not integer and showdbsrc=true working on both method and function
    ### when debugging dbprint, make sure dbsrc is printed with the same idx as original
    ### update dbsrc to the global env
    ### go back to normal before running dbprint again
    ### auto print src with cmt and idx as the ending part of dbprint
    ### to mark my explorations (expressions to evaluate) to stand out
    ### Add the print of src with idx and comments at the end of dbsrc
    ### embed example and autoprint to shorten the code to type
    ### Make title for dbprint
    ### Adding self.eg info and color group into dbprint and print
    #### todo: make the comments with same self.eg have the same color
    ### make dbsrc print idx right
    ### add self.eg to a dict with keys are idxsrc
    ### handle both function and class as src
    ### documenting on Fastdb.dbprint itself
    ## mk_dbsrc
    ## Turn mk_dbsrc into docsrc 
    ### handle when no codes are given
    ## create_dbsrc_from_string
    ## replaceWithDbsrc
    ### handle class and metaclass
    ### improve on handling function as decorator
    ### Handling `inspect._signature_from_callable` to become `self.dbsrc`
    ### handling usage of `@delegates`
    ### handling `@delegates` with indentation before it
    ### handling classes by inspect.isclass() rather than == type and add more class situations
    ### handling `class _T(_TestA, metaclass=BypassNewMeta): `
    ## run_example
    ### `exec(self.eg, globals().update(self.egEnv), locals())` works better than `...update(locals()), self.egEnv)
    ### no more env cells run before `fdb.eg` to make `fdb.run_example` work
    ## Autoprint
    ## Take an example and its env into Fastdb obj
    ## print src with idx and cmt in whole or parts
    ### print self.eg after each comment and colorize comments
    ### color examples and cmts separately and make the func simpler
    ### split each cmt and colorize parts randomly
    ### printcmts1 while saving into a file
    ## goback
    ## Fastdb.explore
    ### adding one breakpoint with comment
    ### Adding multiple breakpoints by multiple set_trace()
    ### Go back to normal before running explore again
    ### enable fdb.takExample("whatinside(fu), ...) without using `fu.whatinside`
    ### refactory explore
    ## snoop
    ### snoop on both function and class
    ### snoop on class and method and all???
    ### snoop
    ### simplify adding @snoop for both normal function and decorator
    ### handling classes
    ### add watch
    ## Snoop
    ### add watch
    ### use guide on Fastdb.dbprint
    ## reliveonce
    ## Fastdb.debug
    ## Export
    ## Send to Obsidian
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0001_fastcore_meta_delegates.md
    ## Import
    ## Initiate Fastdb and example in str
    ## Example
    ## docsrc
    ## Snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0003_Explore_document_FixSigMeta_PrePostInitMeta_AutoInit.md
    ## Initialize fastdebug objects
    ## class FixSigMeta(type) vs class Foo(type)
    ## class Foo()
    ## class PrePostInitMeta(FixSigMeta)
    ## class Foo(metaclass=FixSigMeta)
    ## class AutoInit(metaclass=PrePostInitMeta)
    ## Prepare examples for FixSigMeta, PrePostInitMeta, AutoInit 
    ## Snoop them together in one go
    ### embed the dbsrc of FixSigMeta into PrePostInitMeta
    ### embed dbsrc of PrePostInitMeta into AutoInit
    ## Explore and Document on them together 
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0004_fastcore.meta._rm_self.md
    ## imports
    ## set up
    ## document
    ## snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0005_fastcore.meta.test_sig.md
    ## imports
    ## setups
    ## documents
    ## snoop
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0007_fastcore.meta.BypassNewMeta.md
    ## Reading official docs
    ## Inspecting class
    ## Initiating with examples
    ## Snoop
    ## Document
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0002_signature_from_callable.md
    ## Expand cell
    ## Imports and initiate
    ## Examples
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0008_use_kwargs_dict.md
    ## Imports
    ## Reading official docs
    ## empty2none
    ## `_mk_param`
    ## use_kwargs_dict
    ### Reading docs
    ## use_kwargs
    ### Reading docs
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0006_fastcore.meta.NewChkMeta.md
    ## Import and Initalization
    ## Official docs
    ## Prepare Example
    ## Inspect classes
    ## Snoop
    ## Document
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0000_tour.md
    ### Documentation
    ### Testing
    ### Foundations
    ### L
    ### Transforms
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0012_fastcore_foundation_L.md
    ## Document `L` with fastdebug
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0011_Fastdb.md
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0009_funcs_kwargs.md
    ## fastcore.meta.method
    ### Reading Docs
    ### Running codes
    ### Document
    ### snoop
    ## funcs_kwargs
    ### Official docs
    ### snoop: from _funcs_kwargs to funcs_kwargs
    ### snoop only '_funcs_kwargs' by breaking up 'funcs_kwargs'
    
    /Users/Natsume/Documents/fastdebug/mds/demos/0010_fastcore_meta_summary.md
    ## import
    ## fastcore and fastcore.meta
    ### What's inside fastcore.meta
    ## Review individual funcs and classes
    ### What is fastcore.meta all about? 
    ### What can these metaclasses do for me?
    #### FixSigMeta
    #### PrePostInitMeta
    #### AutoInit
    #### NewChkMeta
    #### BypassNewMeta
    ### What can those decorators do for me?
    #### use_kwargs_dict
    #### use_kwargs
    #### delegates
    #### funcs_kwargs
    ### The remaining functions
    ## What is fastcore.meta all about
    
    /Users/Natsume/Documents/fastdebug/mds/questions/00_question_anno_dict.md
    ## `anno_dict` docs
    ## Dive in
    ## `anno_dict` seems not add anything new to `__annotations__`
    ## use fastdebug to double check
    ## Does fastcore want anno_dict to include params with no annos?
    ## Jeremy's response



```

```
