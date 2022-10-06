# 0012_fastai_using_nbdev_export_in_kaggle_notebook
---
skip_exec: true
---
### how to install nbdev in Kaggle notebook

It can be handy to create script files from notebooks, using nbdev's `notebook2script`. But since Kaggle doesn't actually save the notebook to the file-system, we have to do some workarounds to make this happen. Here's all the steps needed to export a notebook to a script:


```
# nbdev requires jupyter, but we're already in a notebook environment, so we can install without dependencies
!pip install -U nbdev
```

    Collecting nbdev
      Downloading nbdev-2.0.7-py3-none-any.whl (49 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.2/49.2 KB[0m [31m379.8 kB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: astunparse in /opt/conda/lib/python3.7/site-packages (from nbdev) (1.6.3)
    Collecting fastcore>=1.5.11
      Downloading fastcore-1.5.11-py3-none-any.whl (69 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m69.1/69.1 KB[0m [31m1.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting execnb
      Downloading execnb-0.0.9-py3-none-any.whl (13 kB)
    Collecting ghapi
      Downloading ghapi-1.0.0-py3-none-any.whl (55 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m55.3/55.3 KB[0m [31m1.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting asttokens
      Downloading asttokens-2.0.5-py2.py3-none-any.whl (20 kB)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.11->nbdev) (21.3)
    Requirement already satisfied: pip in /opt/conda/lib/python3.7/site-packages (from fastcore>=1.5.11->nbdev) (22.0.4)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from asttokens->nbdev) (1.16.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.7/site-packages (from astunparse->nbdev) (0.37.1)
    Requirement already satisfied: ipython in /opt/conda/lib/python3.7/site-packages (from execnb->nbdev) (7.32.0)
    Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (4.8.0)
    Requirement already satisfied: matplotlib-inline in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (0.1.3)
    Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (0.18.1)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (3.0.27)
    Requirement already satisfied: setuptools>=18.5 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (59.8.0)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (2.11.2)
    Requirement already satisfied: traitlets>=4.2 in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (5.1.1)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (5.1.1)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (0.2.0)
    Requirement already satisfied: pickleshare in /opt/conda/lib/python3.7/site-packages (from ipython->execnb->nbdev) (0.7.5)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.7/site-packages (from packaging->fastcore>=1.5.11->nbdev) (3.0.7)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/conda/lib/python3.7/site-packages (from jedi>=0.16->ipython->execnb->nbdev) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.7/site-packages (from pexpect>4.3->ipython->execnb->nbdev) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython->execnb->nbdev) (0.2.5)
    Installing collected packages: asttokens, fastcore, ghapi, execnb, nbdev
      Attempting uninstall: fastcore
        Found existing installation: fastcore 1.4.2
        Uninstalling fastcore-1.4.2:
          Successfully uninstalled fastcore-1.4.2
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    fastai 2.6.0 requires fastcore<1.5,>=1.3.27, but you have fastcore 1.5.11 which is incompatible.[0m[31m
    [0mSuccessfully installed asttokens-2.0.5 execnb-0.0.9 fastcore-1.5.11 ghapi-1.0.0 nbdev-2.0.7
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

### which pyfile I am export the notebook to


```
#|default_exp app
```

### what to export from the notebook to the pyfile 


```
#|export
a=1
```

### how to export the current current IPython history to a notebook file using `%notebook`


```
# NB: This only works if you run all the cells in order - click "Save Version" to do this automatically
%notebook -e testnbdev.ipynb
```

### how to check all the jupyter magic commands


```
%lsmagic
```




    Available line magics:
    %aimport  %alias  %alias_magic  %autoawait  %autocall  %automagic  %autoreload  %autosave  %bookmark  %cat  %cd  %clear  %colors  %conda  %config  %connect_info  %cp  %debug  %dhist  %dirs  %doctest_mode  %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %lf  %lk  %ll  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %lx  %macro  %magic  %man  %matplotlib  %mkdir  %more  %mv  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %pip  %popd  %pprint  %precision  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %rep  %rerun  %reset  %reset_selective  %rm  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%debug  %%file  %%html  %%javascript  %%js  %%latex  %%markdown  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.



### how to export the specified kaggle notebook to the pyfile module


```
from nbdev.export import nb_export
```


```
nb_export('testnbdev.ipynb', '.')
```

### how to check the pyfile/module on kaggle


```
!cat app.py
```

    # AUTOGENERATED! DO NOT EDIT! File to edit: testnbdev.ipynb.
    
    # %% auto 0
    __all__ = ['a']
    
    # %% testnbdev.ipynb 2
    a=1

