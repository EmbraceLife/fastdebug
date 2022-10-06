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

# Learning fastai with joy

```python
#| hide
from fastdebug.utils import *
```

## Search notes and notebooks


### search notebooks
The first step to learn fastai with joy is to make revision easier. I would like to be able to search learning points in fastai notebooks with ease.

If I want to read or run the notebook, I could click the second link to run the notebook on Kaggle.

```python
#| column: page
fastnbs("how gradient accumulation work")
```

If `fastnbs` doesn't return anything to your query, it is because the search ability of `fastnbs` is minimum, I need to learn to improve it.  But don't worry the next function below `fastlistnbs` will assist you to continue searching. 


### list all notebook learning points
I would also like to view all the learning points (in the form of questions) of all the fastai notebooks I have studied. This is a long list, so press `cmd + f` and search keywords e.g., "ensemble" to find the relevant questions, and then use `fastnbs` to search and display the details like above.

press `cmd + o` to view them all without scrolling inside a small window

```python
#| column: page
fastlistnbs() 
```

### Search notes


I would also like to search my own fastai notes with ease. The `fastnotes` can search but very rough at the moment, and the notes need a lot of rewrite.

```python
# fastnotes("how random forest work")
```

```python

```
