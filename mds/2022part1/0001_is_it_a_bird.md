---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 0001_Is it a bird? Creating a model from your own data


## Install and update fastai and duckduckgo_search

```python
# !mamba update -q -y fastai
```

```python
# !pip install -Uqq duckduckgo_search
```

## Know a little about the libraries

```python
from fastdebug.utils import *
from fastdebug.core import *
```

### what is fastai

```python
import fastai
```

```python
whichversion("fastai")
```

```python
whatinside(fastai, lib=True)
```

```python
import fastai.losses as fl
```

```python
whatinside(fl, dun=True)
```

### what is duckduckgo

```python
import duckduckgo_search
```

```python
whichversion("duckduckgo_search")
```

```python
whatinside(duckduckgo_search)
```

```python
whatinside(duckduckgo_search, func=True)
```

```python

```

## Download images

```python
from duckduckgo_search import ddg_images
from fastcore.all import *
```

```python
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
```

```python
#NB: `search_images` depends on duckduckgo.com, which doesn't always return correct responses.
#    If you get a JSON error, just try running it again (it may take a couple of tries).
urls = search_images('bird photos', max_images=1)
urls[0]
```

```python
from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)
```

```python
download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)
```

```python

```

```python

```

```python
from fastdebug.utils import *
from fastdebug.core import *
```

```python
get_all_nbs()

fastnbs("what is fastai", nb=True, db=False)

fastnbs("download images", nb=True, db=False)
```

```python

```
