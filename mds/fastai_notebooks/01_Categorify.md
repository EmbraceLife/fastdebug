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

```python
import cudf
import nvtabular as nvt

import warnings
warnings.filterwarnings("ignore")
```

# Basic `Categorify` workflow

```python
gdf = cudf.DataFrame(data=[['apple'], ['apple'], ['orange']], columns=['item'])
gdf
```

```python
output = ['item'] >> nvt.ops.Categorify()
```

```python
workflow = nvt.Workflow(output)
dataset = nvt.Dataset(gdf)
```

```python
workflow.fit_transform(dataset).to_ddf().compute()
```

```python
cudf.read_parquet(output.op.categories['item'])
```

# Categorify with merging of other columns

```python
gdf = cudf.DataFrame(data=[['apple', 2.5], ['apple', 2.5], ['orange', 3]], columns=['item', 'price'])
gdf
```

```python
cats = ['item'] >> nvt.ops.Categorify()
output = ['price'] + cats
```

```python
output.graph
```

```python
workflow = nvt.Workflow(output)
dataset = nvt.Dataset(gdf)
```

```python
workflow.fit_transform(dataset).to_ddf().compute()
```

# Combining columns using `Categorify`


As Kaggle GMs share in the outstanding [Recsys2020 Tutorial](https://github.com/rapidsai/deeplearning/blob/main/RecSys2020Tutorial/03_1_CombineCategories.ipynb), combining categories can be a very powerful move.

By doing so, we can expose valuable splits to our tree based model that it wouldn't be able to identify otherwise. This technique allows us to tap into the powerful Cross Column or Cross Product features and its applicability goes beyond just tree based models.

`nvtabular` supports this operation natively!

```python
gdf = cudf.DataFrame(data=[['apple', 'red'], ['apple', 'red'], ['apple', 'green'], ['orange', 'red']], columns=['item', 'color'])
gdf
```

```python
output = [['item', 'color']] >> nvt.ops.Categorify(encode_type='combo', ) # note the column selector to the left of `>>`
                                                                          # it is a list of lists!
                                                                          # the nested list denotes a column group
```

```python
workflow = nvt.Workflow(output)
dataset = nvt.Dataset(gdf)
```

```python
workflow.fit_transform(dataset).to_ddf().compute()
```

```python
cudf.read_parquet(output.op.categories['item_color'])
```

What makes this technique particularly valuable is a built in mechanism for controlling overfitting.

Combining categories with only a couple of examples can prove detrimental -- we might not have enough data to learn anything that would generalize well to unseen data.

`Categorify` supports setting a threshold above which combining categories will take place. Categories of lesser frequency will all be placed in a single bin (that of &lt;NA>).
    
To demonstrate this functionality, let us only combine categories with minimal threshold of 2 examples.

```python
gdf
```

```python
output = [['item', 'color']] >> nvt.ops.Categorify(encode_type='combo', freq_threshold=2)
```

```python
workflow = nvt.Workflow(output)
dataset = nvt.Dataset(gdf)
```

```python
workflow.fit_transform(dataset).to_ddf().compute()
```

```python
cudf.read_parquet(output.op.categories['item_color'])
```

# Benchmarking against pandas

```python
import pandas as pd
import numpy as np
```

```python
colors = pd.DataFrame(data={'color': np.random.choice(['red', 'green', 'blue'], 100_000_000)})
```

```python
%%time

c = pd.Categorical(colors['color'])
```

```python
gcolors = cudf.from_pandas(colors)
```

```python
%%time

output = ['color'] >> nvt.ops.Categorify()
workflow = nvt.Workflow(output)
dataset = nvt.Dataset(gcolors)
gdf = workflow.fit_transform(dataset).to_ddf().compute()
gdf.head()
```
