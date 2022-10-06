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

# 0012_fastai_using_nbdev_export_in_kaggle_notebook

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->

### how to install nbdev in Kaggle notebook


It can be handy to create script files from notebooks, using nbdev's `notebook2script`. But since Kaggle doesn't actually save the notebook to the file-system, we have to do some workarounds to make this happen. Here's all the steps needed to export a notebook to a script:

```python
# nbdev requires jupyter, but we're already in a notebook environment, so we can install without dependencies
!pip install -U nbdev
```

### which pyfile I am export the notebook to

```python
#|default_exp app
```

### what to export from the notebook to the pyfile 

```python
#|export
a=1
```

### how to export the current current IPython history to a notebook file using `%notebook`

```python
# NB: This only works if you run all the cells in order - click "Save Version" to do this automatically
%notebook -e testnbdev.ipynb
```

### how to check all the jupyter magic commands

```python
%lsmagic
```

### how to export the specified kaggle notebook to the pyfile module

```python
from nbdev.export import nb_export
```

```python
nb_export('testnbdev.ipynb', '.')
```

### how to check the pyfile/module on kaggle

```python
!cat app.py
```
