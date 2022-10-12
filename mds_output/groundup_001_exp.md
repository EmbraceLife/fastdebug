# groundup_001_exp
how to use nbdev export 


```
#| default_exp delete0000
```


```
#| export
TEST = "test"
```


```
#| export groundup
test1 = "from test_export module"
```

## nb_name


```
from fastdebug.utils import *
```


<style>.container { width:100% !important; }</style>



```
nbname = nb_name()
```


    <IPython.core.display.Javascript object>



```
nbname
```




    ''



## notebook as json


```
import json
```


```
if bool(nbname):
    d = json.load(open(nbname,'r'))
else: 
    d = None
```


```
d
```


```
all_src = []
if bool(d):
    for dct in d['cells']:
        all_src = all_src + dct['source']
all_src
```




    []




```
#|hide
import nbdev; nbdev.nbdev_export()
```


```

```
