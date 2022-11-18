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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
    </tr>
  </tbody>
</table>
</div>




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
cudf.read_parquet(output.op.categories['item'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>item_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Categorify with merging of other columns


```python
gdf = cudf.DataFrame(data=[['apple', 2.5], ['apple', 2.5], ['orange', 3]], columns=['item', 'price'])
gdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>orange</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cats = ['item'] >> nvt.ops.Categorify()
output = ['price'] + cats
```


```python
output.graph
```




    
![svg](01_Categorify_files/01_Categorify_10_0.svg)
    




```python
workflow = nvt.Workflow(output)
dataset = nvt.Dataset(gdf)
```


```python
workflow.fit_transform(dataset).to_ddf().compute()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



# Combining columns using `Categorify`

As Kaggle GMs share in the outstanding [Recsys2020 Tutorial](https://github.com/rapidsai/deeplearning/blob/main/RecSys2020Tutorial/03_1_CombineCategories.ipynb), combining categories can be a very powerful move.

By doing so, we can expose valuable splits to our tree based model that it wouldn't be able to identify otherwise. This technique allows us to tap into the powerful Cross Column or Cross Product features and its applicability goes beyond just tree based models.

`nvtabular` supports this operation natively!


```python
gdf = cudf.DataFrame(data=[['apple', 'red'], ['apple', 'red'], ['apple', 'green'], ['orange', 'red']], columns=['item', 'color'])
gdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>green</td>
    </tr>
    <tr>
      <th>3</th>
      <td>orange</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
cudf.read_parquet(output.op.categories['item_color'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>green</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>orange</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>



What makes this technique particularly valuable is a built in mechanism for controlling overfitting.

Combining categories with only a couple of examples can prove detrimental -- we might not have enough data to learn anything that would generalize well to unseen data.

`Categorify` supports setting a threshold above which combining categories will take place. Categories of lesser frequency will all be placed in a single bin (that of &lt;NA>).
    
To demonstrate this functionality, let us only combine categories with minimal threshold of 2 examples.


```python
gdf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>apple</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>apple</td>
      <td>green</td>
    </tr>
    <tr>
      <th>3</th>
      <td>orange</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cudf.read_parquet(output.op.categories['item_color'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>apple</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>



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

    CPU times: user 4.48 s, sys: 475 ms, total: 4.96 s
    Wall time: 4.94 s



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

    CPU times: user 820 ms, sys: 147 ms, total: 966 ms
    Wall time: 966 ms





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


