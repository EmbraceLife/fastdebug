# 0001_Is it a bird? Creating a model from your own data

## Useful Course sites

**Official course site**:  for lesson [1](https://course.fast.ai/Lessons/lesson1.html)    

**Official notebooks** [repo](https://github.com/fastai/course22), on [nbviewer](https://nbviewer.org/github/fastai/course22/tree/master/)

Official **Is it a bird** [notebook](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data) on kaggle     



```
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


```
# !mamba update -q -y fastai
```


```
# !pip install -Uqq duckduckgo_search
```

## Know a little about the libraries


```
from fastdebug.utils import *
from fastdebug.core import *
```


<style>.container { width:100% !important; }</style>


### what is fastai


```
import fastai
```


```
whichversion("fastai")
```

    fastai: 2.7.9 
    fastai simplifies training fast and accurate neural nets using modern best practices    
    Jeremy Howard, Sylvain Gugger, and contributors 
    https://github.com/fastai/fastai/tree/master/     
    python_version: >=3.7     
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/fastai



```
whatinside(fastai, lib=True)
```

    The library has 24 modules
    ['_modidx',
     '_nbdev',
     '_pytorch_doc',
     'basics',
     'callback',
     'collab',
     'data',
     'distributed',
     'fp16_utils',
     'imports',
     'interpret',
     'layers',
     'learner',
     'losses',
     'medical',
     'metrics',
     'optimizer',
     'tabular',
     'test_utils',
     'text',
     'torch_basics',
     'torch_core',
     'torch_imports',
     'vision']



```
import fastai.losses as fl
```


```
whatinside(fl, dun=True)
```

    fastai.losses has: 
    11 items in its __all__, and 
    334 user defined functions, 
    178 classes or class objects, 
    4 builtin funcs and methods, and
    535 callables.
    
    BaseLoss:                           class, type    Same as `loss_cls`, but flattens input and target.
    CrossEntropyLossFlat:               class, type    Same as `nn.CrossEntropyLoss`, but flattens input and target.
    FocalLoss:                          class, PrePostInitMeta    Same as `nn.Module`, but no need for subclasses to call `super().__init__`
    FocalLossFlat:                      class, type    Same as CrossEntropyLossFlat but with focal paramter, `gamma`. Focal loss is introduced by Lin et al. 
    https://arxiv.org/pdf/1708.02002.pdf. Note the class weighting factor in the paper, alpha, can be 
    implemented through pytorch `weight` argument passed through to F.cross_entropy.
    BCEWithLogitsLossFlat:              class, type    Same as `nn.BCEWithLogitsLoss`, but flattens input and target.
    BCELossFlat:                        function    Same as `nn.BCELoss`, but flattens input and target.
    MSELossFlat:                        function    Same as `nn.MSELoss`, but flattens input and target.
    L1LossFlat:                         function    Same as `nn.L1Loss`, but flattens input and target.
    LabelSmoothingCrossEntropy:         class, PrePostInitMeta    Same as `nn.Module`, but no need for subclasses to call `super().__init__`
    LabelSmoothingCrossEntropyFlat:     class, type    Same as `LabelSmoothingCrossEntropy`, but flattens input and target.
    DiceLoss:                           class, type    Dice loss for segmentation


### what is duckduckgo


```
import duckduckgo_search
```


```
whichversion("duckduckgo_search")
```

    duckduckgo-search: 2.1.3 
    Search for words, documents, images, news, maps and text translation using the DuckDuckGo.com search engine.    
    deedy5 
    https://github.com/deedy5/duckduckgo_search     
    python_version: >=3.7     
    /Users/Natsume/mambaforge/lib/python3.9/site-packages/duckduckgo_search



```
whatinside(duckduckgo_search)
```

    duckduckgo_search has: 
    0 items in its __all__, and 
    6 user defined functions, 
    0 classes or class objects, 
    0 builtin funcs and methods, and
    6 callables.
    



```
whatinside(duckduckgo_search, func=True)
```

    duckduckgo_search has: 
    0 items in its __all__, and 
    6 user defined functions, 
    0 classes or class objects, 
    0 builtin funcs and methods, and
    6 callables.
    
    The user defined functions are:
    ddg:               function    (keywords, region='wt-wt', safesearch='Moderate', time=None, max_results=25, output=None)
    ddg_images:        function    (keywords, region='wt-wt', safesearch='Moderate', time=None, size=None, color=None, type_image=None, layout=None, license_image=None, max_results=100, output=None, download=False)
    ddg_maps:          function    (keywords, place=None, street=None, city=None, county=None, state=None, country=None, postalcode=None, latitude=None, longitude=None, radius=0, max_results=None, output=None)
    ddg_news:          function    (keywords, region='wt-wt', safesearch='Moderate', time=None, max_results=25, output=None)
    ddg_translate:     function    (keywords, from_=None, to='en', output=None)
    ddg_videos:        function    (keywords, region='wt-wt', safesearch='Moderate', time=None, resolution=None, duration=None, license_videos=None, max_results=50, output=None)



```

```


```

```


```

```

## How to use fastdebug with fastai notebooks

### what is fastdebug


```
from fastdebug.utils import *
from fastdebug.core import *
import fastdebug.utils as fu
import fastdebug.core as core
```


```
whatinside(fu,dun=True)
```

    fastdebug.utils has: 
    25 items in its __all__, and 
    38 user defined functions, 
    3 classes or class objects, 
    0 builtin funcs and methods, and
    41 callables.
    
    expand:NoneType
    test_eq:           function    `test` that `a==b`
    test_is:           function    `test` that `a is b`
    FunctionType:      class, type    Create a function object.
    
    code
      a code object
    globals
      the globals dictionary
    name
      a string that overrides the name from the code object
    argdefs
      a tuple that specifies the default argument values
    closure
      a tuple that supplies the bindings for free variables
    MethodType:        class, type    method(function, instance)
    
    Create a bound instance method object.
    expandcell:        function    expand cells of the current notebook to its full width
    inspect_class:     function    examine the details of a class
    ismetaclass:       function    check whether a class is a metaclass or not
    isdecorator:       decorator, function    check whether a function is a decorator
    whatinside:        function    Check what inside a module: `__all__`, functions, classes, builtins, and callables
    whichversion:      function    Give you library version and other basic info.
    fastview:          function    to view the commented src code in color print and with examples
    fastsrcs:          function    to list all commented src files
    getrootport:       function    get the local port and notebook dir
    jn_link:           function    Get a link to the notebook at `path` on Jupyter Notebook
    get_all_nbs:       function    return paths for all nbs both in md and ipynb format into lists
    openNB:            function    Get a link to the notebook at by searching keyword or notebook name
    highlight:         function    highlight a string with yellow background
    display_md:        function    Get a link to the notebook at `path` on Jupyter Notebook
    display_block:     function    `line` is a section title, find all subsequent lines which belongs to the same section and display them together
    fastnbs:           function    using keywords to search learning points (a section title and a section itself) from my documented fastai notebooks
    fastcodes:         function    using keywords to search learning points from commented sources files
    fastnotes:         function    using key words to search notes and display the found line and lines surround it
    fastlistnbs:       function    display all my commented notebooks subheadings in a long list
    fastlistsrcs:      function    display all my commented src codes learning comments in a long list



```
whatinside(core, dun=True)
```

    fastdebug.core has: 
    14 items in its __all__, and 
    117 user defined functions, 
    18 classes or class objects, 
    1 builtin funcs and methods, and
    138 callables.
    
    defaults:SimpleNamespace
    pprint:                       function    Pretty-print a Python object to a stream [default is sys.stdout].
    inspect:module
    dbcolors:                     class, type    None
    randomColor:                  function    create a random color by return a random dbcolor from dbcolors
    colorize:                     function    return the string with dbcolors
    strip_ansi:                   function    to make printright work using regex
    printright:                   function    print a block of text to the right of the cell
    printsrclinewithidx:          function    add idx number to a srcline
    printsrc:                     function    print the seleted srcline with comment, idx and specified num of expanding srclines
    dbprintinsert:                function    insert arbitary code expressions into source code for evaluation
    Fastdb:                       class, type    None
    randomize_cmtparts_color:     function    give each comment a different color for easy viewing
    reliveonce:                   function    Replace current version of srcode with older version, and back to normal



```
inspect_class(Fastdb)
```

    
    is Fastdb a metaclass: False
    is Fastdb created by a metaclass: False
    Fastdb is created by <class 'type'>
    Fastdb.__new__ is object.__new__: True
    Fastdb.__new__ is type.__new__: False
    Fastdb.__new__: <built-in method __new__ of type object>
    Fastdb.__init__ is object.__init__: False
    Fastdb.__init__ is type.__init__: False
    Fastdb.__init__: <function Fastdb.__init__>
    Fastdb.__call__ is object.__call__: False
    Fastdb.__call__ is type.__call__: False
    Fastdb.__call__: <method-wrapper '__call__' of type object>
    Fastdb.__class__: <class 'type'>
    Fastdb.__bases__: (<class 'object'>,)
    Fastdb.__mro__: (<class 'fastdebug.core.Fastdb'>, <class 'object'>)
    
    Fastdb's function members are:
    __init__: Create a Fastdebug class which has two functionalities: dbprint and print.
    autoprint: print srcode with appropriate number of lines automatically
    create_dbsrc_from_string: create dbsrc from a string
    create_dbsrc_string: create the dbsrc string
    create_explore_from_string: evaluate the explore dbsrc from string
    create_explore_str: create the explore dbsrc string
    create_snoop_from_string: evaluate the snoop dbsrc from string
    create_snoop_str: creat the snoop dbsrc string
    debug: to quickly check for clues of errors
    docsrc: create dbsrc the string and turn the string into actual dbsrc function, we have self.dbsrcstr and self.dbsrc available from now on.
    explore: insert 'import ipdb; ipdb.set_trace()' above srcline of idx to create dbsrc, and exec on dbsrc
    goback: Return src back to original state.
    print: Print the source code in whole or parts with idx and comments you added with dbprint along the way.
    printcmts1: print the entire srcode and save it to a file if save=True
    printcmts2: print the srcodes in parts
    printtitle: print title which includes src name, line number under investigation, example.
    replaceWithDbsrc: to replace self.orisrc.__name__ with 'self.dbsrc' and assign this new self.eg to self.eg
    run_example: run self.eg with self.dbsrc
    snoop: run snoop on the func or class under investigation only when example is available
    takeoutExample: get the line of example code with srcode name in it
    
    Fastdb's method members are:
    {}
    
    Fastdb's class members are:
    {'__class__': <class 'type'>}
    
    Fastdb's namespace are:
    mappingproxy({'__dict__': <attribute '__dict__' of 'Fastdb' objects>,
                  '__doc__': None,
                  '__init__': <function Fastdb.__init__>,
                  '__module__': 'fastdebug.core',
                  '__weakref__': <attribute '__weakref__' of 'Fastdb' objects>,
                  'autoprint': <function Fastdb.autoprint>,
                  'create_dbsrc_from_string': <function Fastdb.create_dbsrc_from_string>,
                  'create_dbsrc_string': <function Fastdb.create_dbsrc_string>,
                  'create_explore_from_string': <function Fastdb.create_explore_from_string>,
                  'create_explore_str': <function Fastdb.create_explore_str>,
                  'create_snoop_from_string': <function Fastdb.create_snoop_from_string>,
                  'create_snoop_str': <function Fastdb.create_snoop_str>,
                  'debug': <function Fastdb.debug>,
                  'docsrc': <function Fastdb.docsrc>,
                  'explore': <function Fastdb.explore>,
                  'goback': <function Fastdb.goback>,
                  'print': <function Fastdb.print>,
                  'printcmts1': <function Fastdb.printcmts1>,
                  'printcmts2': <function Fastdb.printcmts2>,
                  'printtitle': <function Fastdb.printtitle>,
                  'replaceWithDbsrc': <function Fastdb.replaceWithDbsrc>,
                  'run_example': <function Fastdb.run_example>,
                  'snoop': <function Fastdb.snoop>,
                  'takeoutExample': <function Fastdb.takeoutExample>})



```

```

### Did I document it in a notebook before?

run `push-code-new` in teminal to convert all current notebooks into mds     

so that the followign search will get me the latest result if I did document similar things


```
fastnbs("what is fastdebug")
```


###  <mark style="background-color: #ffff00">what</mark>   <mark style="background-color: #ffff00">is</mark>   <mark style="background-color: #FFFF00">fastdebug</mark> 





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





[Open `0001_is_it_a_bird` in Jupyter Notebook](http://localhost:8888/tree/nbs/2022part1/0001_is_it_a_bird.ipynb)


I can also extract all the notebook subheadings with the function below    
and to check whether I have documented something similar by `cmd + f` and search keywords there


```
fastlistnbs()
```

    
    /Users/Natsume/Documents/fastdebug/mds/2022part1/0001_is_it_a_bird.md
    ## Useful Course sites
    ## How to use autoreload
    ## How to install and update libraries
    ## Know a little about the libraries
    ### what is fastai
    ### what is duckduckgo
    ## How to use fastdebug with fastai notebooks
    ### what is fastdebug
    ### Did I document it in a notebook before?
    ### Did I document it in a src before?
    ## how to download images
    ### how to create folders using path; how to search and download images in folders; how to resize images 
    ## Train my model
    ### How to find and unlink images not properly downloaded
    ### How to create a DataLoaders with DataBlock; how to view data with it
    ### How to build my model with dataloaders and pretrained model; how to train my model
    ### How to predict with my model
    
    /Users/Natsume/Documents/fastdebug/mds/lib/utils.md
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
    ## openNB
    ## highlight
    ## display_md
    display_md("#### heading level 4")
    ## display_block
            lochead2 = belowline.find("##")
            lochead3 = belowline.find("###")
            lochead4 = belowline.find("####")
            lochead2 = belowline.find("##")
            lochead3 = belowline.find("###")
            lochead4 = belowline.find("####")
    ## fastnbs
                    if pct >= accu and (l.startswith("##") or l.startswith("###") or l.startswith("####")):
    ## fastcodes
    ## fastnotes
    ### multiple folders
    ## fastlistnbs
                    if "##" in l:
    ## fastlistsrcs
    ## Examples
    ## Export
    ## Send to Obsidian
    
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
    # overriding the original official source with our dbsrc, even though rewriting _signature_from_callable inside inspect.py ######################
    ## dbprintinsert
    ### Run and display the inserted dbcodes 
    ### use locals() inside the dbsrc code to avoid adding env individually
    ### enable dbprintinsert to do exec on a block of code
                ### Note: we shall not use the expression like `for k, v in abc print(abc)`
                ### Note: we shall not use the expression like `for k, v in abc if k == def`
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


### Did I document it in a src before?


```
fastcodes("how to access parameters")
```


keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">_rm_self.py</mark> 


        sigd = dict(sig.parameters)===========================================================(1) # [92;1mhow to access parameters from a signature[0m; [36;1mhow is parameters stored in sig[0m; [36;1mhow to turn parameters into a dict;[0m; 
    



the entire source code in  <mark style="background-color: #FFFF00">_rm_self.py</mark> 


    
    class Foo:
        def __init__(self, a, b:int=1): pass
    pprint(inspect.signature(Foo.__init__))
    pprint(_rm_self(inspect.signature(Foo.__init__)))
    
    def _rm_self(sig):========================================================================(0) # [92;1mremove parameter self from a signature which has self;[0m; 
        sigd = dict(sig.parameters)===========================================================(1) # [92;1mhow to access parameters from a signature[0m; [36;1mhow is parameters stored in sig[0m; [36;1mhow to turn parameters into a dict;[0m; 
        sigd.pop('self')======================================================================(2) # [92;1mhow to remove the self parameter from the dict of sig;[0m; 
        return sig.replace(parameters=sigd.values())==========================================(3) # [35;1mhow to update a sig using a updated dict of sig's parameters[0m; 
                                                                                                                                                            (4)
    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">delegates.py</mark> 


            sigd = dict(sig.parameters)=======================================================(12) # [35;1mHow to access parameters of a signature?[0m; [37;1mHow to turn parameters into a dict?[0m; 
    



the entire source code in  <mark style="background-color: #FFFF00">delegates.py</mark> 


    
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [35;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [35;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [34;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [91;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [36;1mIs classmethod callable[0m; [93;1mdoes classmethod has __func__[0m; [36;1mcan we do inspect.signature(clsmethod)[0m; [93;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10) # [91;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [93;1mhasattr(obj, '__delwrap__')[0m; 
            sig = inspect.signature(from_f)===================================================(11) # [35;1mhow to get signature obj of B[0m; [91;1mwhat does a signature look like[0m; [91;1mwhat is the type[0m; 
            sigd = dict(sig.parameters)=======================================================(12) # [35;1mHow to access parameters of a signature?[0m; [37;1mHow to turn parameters into a dict?[0m; 
            k = sigd.pop('kwargs')============================================================(13) # [92;1mHow to remove an item from a dict?[0m; [92;1mHow to get the removed item from a dict?[0m; [35;1mHow to add the removed item back to the dict?[0m; [91;1mwhen writing expressions, as they share environment, so they may affect the following code[0m; 
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [36;1mHow to access a signature's parameters as a dict?[0m; [36;1mHow to replace the kind of a parameter with a different kind?[0m; [93;1mhow to check whether a parameter has a default value?[0m; [37;1mHow to check whether a string is in a dict and a list?[0m; [34;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [92;1mHow to get A's __annotations__?[0m; [34;1mHow to access it as a dict?[0m; [92;1mHow to select annotations of the right params with names?[0m; [37;1mHow to put them into a dict?[0m; [35;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [35;1mHow to add the selected params from A's signature to B's signature[0m; [37;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [93;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [91;1mHow to create a new attr for a function or obj;[0m; 
            from_f.__signature__ = sig.replace(parameters=sigd.values())======================(20) # [93;1mHow to update a signature with a new set of parameters;[0m; 
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)========(21) # [92;1mHow to check whether a func has __annotations__[0m; [92;1mHow add selected params' annotations from A to B's annotations;[0m; 
            return f==========================================================================(22)      
        return _f=============================================================================(23)      
                                                                                                                                                            (24)
    



keyword match is  <mark style="background-color: #ffff00">1.0</mark> , found a line: in  <mark style="background-color: #FFFF00">delegates.py</mark> 


            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [36;1mHow to access a signature's parameters as a dict?[0m; [36;1mHow to replace the kind of a parameter with a different kind?[0m; [93;1mhow to check whether a parameter has a default value?[0m; [37;1mHow to check whether a string is in a dict and a list?[0m; [34;1mhow dict.items() and dict.values() differ[0m;  (14)
    



the entire source code in  <mark style="background-color: #FFFF00">delegates.py</mark> 


    
    def low(a, b:int=1): pass
    @delegates(low)
    def mid(c, d:list=None, **kwargs): pass
    pprint(inspect.signature(mid)) # pprint and inspect is loaded from fastdebug
    
    def delegates(to:FunctionType=None, # Delegatee===========================================(0) # [35;1mhow to make delegates(to) to have to as FunctionType and default as None[0m; 
                  keep=False, # Keep `kwargs` in decorated function?==========================(1)       
                  but:list=None): # Exclude these parameters from signature===================(2) # [35;1mhow to make delegates(to, but) to have 'but' as list and default as None[0m; 
        "Decorator: replace `**kwargs` in signature with params from `to`"====================(3)       
        if but is None: but = []==============================================================(4)       
        def _f(f):============================================================================(5)       
            if to is None: to_f,from_f = f.__base__.__init__,f.__init__=======================(6) # [34;1mhow to write 2 ifs and elses in 2 lines[0m; 
            else:          to_f,from_f = to.__init__ if isinstance(to,type) else to,f=========(7) # [91;1mhow to assign a,b together with if and else[0m; 
            from_f = getattr(from_f,'__func__',from_f)========================================(8) # [36;1mIs classmethod callable[0m; [93;1mdoes classmethod has __func__[0m; [36;1mcan we do inspect.signature(clsmethod)[0m; [93;1mhow to use getattr(obj, attr, default)[0m; 
            to_f = getattr(to_f,'__func__',to_f)==============================================(9)       
            if hasattr(from_f,'__delwrap__'): return f========================================(10) # [91;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [93;1mhasattr(obj, '__delwrap__')[0m; 
            sig = inspect.signature(from_f)===================================================(11) # [35;1mhow to get signature obj of B[0m; [91;1mwhat does a signature look like[0m; [91;1mwhat is the type[0m; 
            sigd = dict(sig.parameters)=======================================================(12) # [35;1mHow to access parameters of a signature?[0m; [37;1mHow to turn parameters into a dict?[0m; 
            k = sigd.pop('kwargs')============================================================(13) # [92;1mHow to remove an item from a dict?[0m; [92;1mHow to get the removed item from a dict?[0m; [35;1mHow to add the removed item back to the dict?[0m; [91;1mwhen writing expressions, as they share environment, so they may affect the following code[0m; 
            s2 = {k:v.replace(kind=inspect.Parameter.KEYWORD_ONLY) for k,v in inspect.signature(to_f).parameters.items() # [36;1mHow to access a signature's parameters as a dict?[0m; [36;1mHow to replace the kind of a parameter with a different kind?[0m; [93;1mhow to check whether a parameter has a default value?[0m; [37;1mHow to check whether a string is in a dict and a list?[0m; [34;1mhow dict.items() and dict.values() differ[0m;  (14)
                  if v.default != inspect.Parameter.empty and k not in sigd and k not in but}=(15)      
            anno = {k:v for k,v in getattr(to_f, "__annotations__", {}).items() if k not in sigd and k not in but} # [92;1mHow to get A's __annotations__?[0m; [34;1mHow to access it as a dict?[0m; [92;1mHow to select annotations of the right params with names?[0m; [37;1mHow to put them into a dict?[0m; [35;1mHow to do it all in a single line[0m;  (16)
            sigd.update(s2)===================================================================(17) # [35;1mHow to add the selected params from A's signature to B's signature[0m; [37;1mHow to add items into a dict;[0m; 
            if keep: sigd['kwargs'] = k=======================================================(18) # [93;1mHow to add a new item into a dict;[0m; 
            else: from_f.__delwrap__ = to_f===================================================(19) # [91;1mHow to create a new attr for a function or obj;[0m; 
            from_f.__signature__ = sig.replace(parameters=sigd.values())======================(20) # [93;1mHow to update a signature with a new set of parameters;[0m; 
            if hasattr(from_f, '__annotations__'): from_f.__annotations__.update(anno)========(21) # [92;1mHow to check whether a func has __annotations__[0m; [92;1mHow add selected params' annotations from A to B's annotations;[0m; 
            return f==========================================================================(22)      
        return _f=============================================================================(23)      
                                                                                                                                                            (24)
    


I can check all the commented src files.


```
fastsrcs()
```

    test_sig.py
    BypassNewMeta.py
    snoop.py
    FixSigMeta.py
    fastnbs.py
    funcs_kwargs.py
    NewChkMeta.py
    printtitle.py
    AutoInit.py
    method.py
    _rm_self.py
    delegates.py
    create_explore_str.py
    PrePostInitMeta.py
    _funcs_kwargs.py
    whatinside.py


I can print out all the learning points as comments inside each src file    

However, I need to figure out a way to extract them nicely from the files    

Todos: how to comment src for list extraction


```
fastlistsrcs()
```

     [93;1mtest_sig(f:FunctionType or ClassType, b:str)[0m; [92;1mtest_sig will get f's signature as a string[0m; [92;1mb is a signature in string provided by the user[0m; [34;1min fact, test_sig is to compare two strings[0m; 
     [34;1mtest_sig is to test two strings with test_eq[0m; [92;1mhow to turn a signature into a string;[0m; 
     since t2 just references t these will be the same
     likewise, chaning an attribute on t will also affect t2 because they both point to the same object.
     both t and t2's __class__ is _T
     [36;1mBypassNewMeta allows its instance class e.g., _T to choose a specific class e.g., _TestB and change `__class__` of an object e.g., t of _TestB to _T without creating a new object[0m; 
     [36;1mIf the instance class like _T has attr '_new_meta', then run it with param x;[0m; 
     [37;1mwhen x is not an instance of _T's _bypass_type[0m; [35;1mor when a positional param is given[0m; [93;1mor when a keyword arg is given[0m; [34;1mlet's run _T's super's __call__ function with x as param[0m; [37;1mand assign the result to x[0m;  (4)
     [37;1mIf x.__class__ is not cls or _T, then make it so[0m; 
     learn about /tmp folder https://www.fosslinux.com/41739/linux-tmp-directory-everything-you-need-to-know.htm                                       (1)
                 exec(dbsrc, locals(), self.egEnv)                ===========================(6)       
         exec(code, globals().update(self.outenv), locals())  when dbsrc is a method, it will update as part of a class                                               (8)
     store dbsrc func inside Fastdb obj==================================================(9)       
     using __new__ of  FixSigMeta instead of type
     Base
     [92;1mAny class having FixSigMeta as metaclass will have its own __init__ func stored in its attr __signature__;FixSigMeta uses its __new__ to create a class instance[0m; [93;1mthen check whether its class instance has its own __init__;if so, remove self from the sig of __init__[0m; [36;1mthen assign this new sig to __signature__ for the class instance;[0m; 
     [34;1mhow does a metaclass create a class instance[0m; [91;1mwhat does super().__new__() do here;[0m; 
     [93;1mhow to remove self from a signature[0m; [36;1mhow to check whether a class' __init__ is inherited from object or not;[0m;  (4)
     allows you to add method b upon instantiation
     don't forget to include **kwargs in __init__
     the attempt to add a is ignored and uses the original method instead.
     access the num attribute from the instance
     adds method b
     self.num + 5 = 10
    multiply instead of add 
     add method b from the super class
     3 * 5 = 15
     [34;1mhow funcs_kwargs works[0m; [93;1mit is a wrapper around _funcs_kwargs[0m; [91;1mit offers two ways of running _funcs_kwargs[0m; [36;1mthe first, default way, is to add a func to a class without using self[0m; [93;1msecond way is to add func to class enabling self use;[0m; 
     [91;1mhow to check whether an object is callable[0m; [36;1mhow to return a result of running a func[0m; [35;1m[0m; 
     [36;1mhow to custom the params of `_funcs_kwargs` for a particular use with partial[0m; 
     if `o` is not an object without an attribute `foo`, set foo = 1
     1 was not of type _T, so foo = 1
     t2 will now reference t
     t and t2 are the same object
     this will also change t.foo to 5 because it is the same object
     without any arguments the constructor will return a reference to the same object
     [34;1mNewChkMeta is a metaclass inherited from FixSigMea[0m; [91;1mit makes its own __call__[0m; [37;1mwhen its class instance, e.g., _T, create object instances (e.g, t) without args nor kwargs but only x, and x is an object of the instance class, then return x[0m; [93;1motherwise, create and return a new object created by the instance class's super class' __call__ method with x as param[0m; [91;1mIn other words, t = _T(3) will create a new obj[0m; [92;1m_T(t) will return t[0m; [35;1m_T(t, 1) or _T(t, b=1) will also return a new obj[0m; 
     [37;1mhow to create a __call__ method with param cls, x, *args, **kwargs;[0m; 
     [36;1mhow to express no args and no kwargs and x is an instance of cls?[0m; 
     [34;1mhow to call __call__ of super class with x and consider all possible situations of args and kwargs[0m; 
     make sure self.orieg has no self inside===================(4)       
     [93;1mhow to use :=<, :=>, :=^ with format to align text to left, right, and middle[0m;  (5)
     h=10 is initialized in the parent class
     [36;1mAutoInit inherit __new__ and __init__ from object to create and initialize object instances[0m; [93;1mAutoInit uses PrePostInitMeta.__new__ or in fact FixSigMeta.__new__ to create its own class instance, which can have __signature__[0m; [36;1mAutoInit uses PrePostInitMeta.__call__ to specify how its object instance to be created and initialized (with pre_init, init, post_init))[0m; [35;1mAutoInit as a normal or non-metaclass, it writes its own __pre_init__ method[0m; 
     [92;1mhow to run superclass' __init__ function[0m; 
     how to test on the type of function or method
     `1` is a dummy instance since Py3 doesn't allow `None` any more=====================(2)       
     [92;1mremove parameter self from a signature which has self;[0m; 
     [92;1mhow to access parameters from a signature[0m; [36;1mhow is parameters stored in sig[0m; [36;1mhow to turn parameters into a dict;[0m; 
     [92;1mhow to remove the self parameter from the dict of sig;[0m; 
     [35;1mhow to update a sig using a updated dict of sig's parameters[0m; 
     pprint and inspect is loaded from fastdebug
     Delegatee===========================================(0)  Keep `kwargs` in decorated function?==========================(1)       
     Exclude these parameters from signature===================(2)  [34;1mhow to write 2 ifs and elses in 2 lines[0m; 
     [91;1mhow to assign a,b together with if and else[0m; 
     [36;1mIs classmethod callable[0m; [93;1mdoes classmethod has __func__[0m; [36;1mcan we do inspect.signature(clsmethod)[0m; [93;1mhow to use getattr(obj, attr, default)[0m; 
     [91;1mif B has __delwrap__, can we do delegates(A)(B) again?[0m; [93;1mhasattr(obj, '__delwrap__')[0m; 
     [35;1mhow to get signature obj of B[0m; [91;1mwhat does a signature look like[0m; [91;1mwhat is the type[0m; 
     [35;1mHow to access parameters of a signature?[0m; [37;1mHow to turn parameters into a dict?[0m; 
     [92;1mHow to remove an item from a dict?[0m; [92;1mHow to get the removed item from a dict?[0m; [35;1mHow to add the removed item back to the dict?[0m; [91;1mwhen writing expressions, as they share environment, so they may affect the following code[0m; 
     [36;1mHow to access a signature's parameters as a dict?[0m; [36;1mHow to replace the kind of a parameter with a different kind?[0m; [93;1mhow to check whether a parameter has a default value?[0m; [37;1mHow to check whether a string is in a dict and a list?[0m; [34;1mhow dict.items() and dict.values() differ[0m;  (14)
     [92;1mHow to get A's __annotations__?[0m; [34;1mHow to access it as a dict?[0m; [92;1mHow to select annotations of the right params with names?[0m; [37;1mHow to put them into a dict?[0m; [35;1mHow to do it all in a single line[0m;  (16)
     [35;1mHow to add the selected params from A's signature to B's signature[0m; [37;1mHow to add items into a dict;[0m; 
     [93;1mHow to add a new item into a dict;[0m; 
     [91;1mHow to create a new attr for a function or obj;[0m; 
     [93;1mHow to update a signature with a new set of parameters;[0m; 
     [92;1mHow to check whether a func has __annotations__[0m; [92;1mHow add selected params' annotations from A to B's annotations;[0m; 
     set with __pre_init__
     set with __init__
     set with __post_init__
     [36;1mPrePostInitMeta inherit __new__ and __init__ from FixSigMeta as a metaclass (a different type)[0m; [37;1mnot from type, nor from object[0m; [92;1mPrePostInitMeta is itself a metaclass, which is used to create class instance not object instance[0m; [36;1mPrePostInitMeta writes its own __call__ which regulates how its class instance create and initialize object instance[0m; 
     [91;1mhow to create an object instance with a cls[0m; [34;1mhow to check the type of an object is cls[0m; [93;1mhow to run a function without knowing its params;[0m; 
     [92;1mhow to run __init__ without knowing its params[0m; 
     allows you to add method b upon instantiation
     don't forget to include **kwargs in __init__
     the attempt to add a is ignored and uses the original method instead.
     [93;1mhow does _funcs_kwargs work: _funcs_kwargs is a decorator[0m; [35;1mit helps class e.g., T to add more methods[0m; [34;1mI need to give the method a name, and put the name e.g., 'b' inside a list called _methods=['b'] inside class T[0m; [37;1mthen after writing a func e.g., _new_func, I can add it by T(b = _new_func)[0m; [93;1mif I want the func added to class to use self, I shall write @funcs_kwargs(as_method=True)[0m; 
     [37;1mhow to define a method which can use self and accept any parameters[0m; 
     [36;1mhow to pop out the value of an item in a dict (with None as default), and if the item name is not found, pop out None instead[0m; [92;1m[0m; 
     [34;1mhow to turn a func into a method[0m; 
     [91;1mhow to give a method a different instance, like self[0m; 
     [36;1mhow to add a method to a class as an attribute[0m; 
     [34;1mhow to wrap `_init` around `old_init`, so that `_init` can use `old_init` inside itself[0m; 
     [37;1mhow to add a list of names with None as default value to function `_init` to repalce its kwargs param[0m; 
     [36;1mhow to make a class.`__init__` signature to be the signature of the class using `__signature__` and `_rm_self`[0m;  (12)
     module, e.g., `import fastcore.all as fa`, use `fa` here=============(0)       
     print all items in __all__===============================(1)       
     print all user defined functions========================(2)       
     print all class objects=================================(3)       
     print all builtin funcs or methods=====================(4)       
     print all the modules of the library it belongs to=======(5)       
     print all callables=======================================(6)       
     [93;1mhow many items inside mo.__all__?[0m; 
     [37;1mget all funcs of a module[0m; 
     [37;1mget all classes from the module[0m; 
     [92;1mget the file path of the module[0m; 
     [92;1mget names of all modules of a lib[0m; 
                 print(f"{i[0]}: {kind}")  ==================================================(44)      
                 print(f"{i[0]}: {kind}")  ==================================================(56)      


## how to download images


```
from duckduckgo_search import ddg_images
from fastcore.all import *
```


```
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
```


```
#|eval: false
#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('bird photos', max_images=1)
urls[0]
```

    Searching for 'bird photos'





    'https://amazinganimalphotos.com/wp-content/uploads/2016/11/beautiful-birds.jpeg'




```
#|eval: false
from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```




    
![png](0001_is_it_a_bird_files/0001_is_it_a_bird_49_0.png)
    




```
#|eval: false
download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```

    Searching for 'forest photos'





    
![png](0001_is_it_a_bird_files/0001_is_it_a_bird_50_1.png)
    



### how to create folders using path; how to search and download images in folders; how to resize images 

Our searches seem to be giving reasonable results, so let's grab 200 examples of each of "bird" and "forest" photos, and save each group of photos to a different folder:


```
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


```
#|eval: false
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
```


```

```

### How to create a DataLoaders with DataBlock; how to view data with it

To train a model, we'll need DataLoaders:     

1) a training set (the images used to create a model) and 

2) a validation set (the images used to check the accuracy of a model -- not used during training). 

We can view sample images from it:


```
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


```
#|eval: false
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```


```

```

### How to predict with my model; how to avoid running cells in nbdev_prepare


```
#|eval: false
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
```


```

```
