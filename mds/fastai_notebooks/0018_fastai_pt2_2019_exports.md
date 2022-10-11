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

# 0018_fastai_pt2_2019_exports

<!-- #raw -->
---
skip_exec: true
---
<!-- #endraw -->




### [28:09](https://youtu.be/4u8FxNEDUeg?list=PLfYUBJiXbdtTIdtE1U8qgyxo4Jy2Y91uj&t=1689) - how to build a library with jupyter notebook with export and notebook2script.py (fastforward to 2022, we use `#| export`, `nbdev_export`, `#| default_exp`) Jupyter [notebook](https://nbviewer.org/github/fastai/course-v3/blob/7fceebfd14d4f3bc7e0ec649834309b8cb786e40/nbs/dl2/00_exports.ipynb) is just a json data file



```python
#export
TEST = 'test'
```

Export

```python
!python notebook2script.py 00_exports.ipynb
```

How it works:

```python
import json
d = json.load(open('00_exports.ipynb','r'))['cells']
```

```python
d[0]
```



