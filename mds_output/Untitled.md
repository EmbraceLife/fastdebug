```
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```
from fastdebug.utils import *
import inspect
```


```
nb_name()
```




    'Untitled.ipynb'




```
nb_path()
```




    '/Users/Natsume/Documents/fastdebug/nbs/2022part1/Untitled.ipynb'




```
nb_url()
```




    'http://localhost:8888/notebooks/nbs/2022part1/Untitled.ipynb?kernel_name=python3'




```
print(inspect.getsource(nb_name))
```

    def nb_name():
        "run this func to get nb_path of this current notebook"
        import ipyparams
        return eval("ipyparams.notebook_name")
    



```
print(inspect.getsource(nb_path))
```

    def nb_path():
        "run this func to get nb_path of this current notebook"
        import ipyparams
        return eval("os.path.join(os.getcwd(), ipyparams.notebook_name)")
    



```
print(inspect.getsource(nb_url))
```

    def nb_url():
        "run this func to get nb_url of this current notebook"
        import ipyparams
        return eval("ipyparams.raw_url")
    



```
print(inspect.getsource(ipy2md))
```

    def ipy2md(db=True):
        "convert the current notebook to md"
        import ipyparams
        import os
        path = nb_path()
        name = nb_name()
        url = nb_url()
        obs_path = "/Users/Natsume/Documents/divefastai/Debuggable/jupytext"
        obs_output_path = "/Users/Natsume/Documents/divefastai/Debuggable/nbconvert"    
        mds_path = path.replace("nbs", "mds").split(name)[0]
        mds_output = "/Users/Natsume/Documents/fastdebug/mds_output"
        # https://stackabuse.com/executing-shell-commands-with-python/
        os.system(f"jupytext --to md {path}")
        os.system(f"cp {path.split('.ipynb')[0]+'.md'} {obs_path}")
        if db: print(f'cp to : {obs_path}')
        os.system(f"mv {path.split('.ipynb')[0]+'.md'} {mds_path}")
        if db: print(f'move to : {mds_path}')
        os.system(f"jupyter nbconvert --to markdown {path}")
        os.system(f"cp {path.split('.ipynb')[0]+'.md'} {mds_output}")
        os.system(f"mv {path.split('.ipynb')[0]+'.md'} {mds_output}")
        if db: print(f'move to : {mds_output}')
    



```
ipy2md()
```

    [jupytext] Reading /Users/Natsume/Documents/fastdebug/nbs/2022part1/Untitled.ipynb in format ipynb
    [jupytext] Writing /Users/Natsume/Documents/fastdebug/nbs/2022part1/Untitled.md
    cp to : /Users/Natsume/Documents/divefastai/Debuggable/jupytext
    move to : /Users/Natsume/Documents/fastdebug/mds/2022part1/


    [NbConvertApp] Converting notebook /Users/Natsume/Documents/fastdebug/nbs/2022part1/Untitled.ipynb to markdown
    [NbConvertApp] Writing 2700 bytes to /Users/Natsume/Documents/fastdebug/nbs/2022part1/Untitled.md


    move to : /Users/Natsume/Documents/fastdebug/mds_output



```

```
