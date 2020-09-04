---
title: "Dataframe"
author: "Palaniappan S"
date: 2020-09-04
description: "-"
type: technical_note
draft: false
---

```python
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
```


```python
 a=(            [ [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a
```




    [[2, 3, 1], [4, 9, 2], [8, 27, 4], [16, 1, 1]]




```python
row_names = ['first', 'second', 'third', 'fourth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame( a,index=row_names, columns=col_names)
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>second</th>
      <td>4</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>third</th>
      <td>8</td>
      <td>27</td>
      <td>4</td>
    </tr>
    <tr>
      <th>fourth</th>
      <td>16</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean()
```




    A     7.5
    B    10.0
    C     2.0
    dtype: float64




```python
df.var()
```




    A     38.333333
    B    140.000000
    C      2.000000
    dtype: float64




```python
df.mean(axis=1)
```




    first      2.0
    second     5.0
    third     13.0
    fourth     6.0
    dtype: float64




```python
df.var(axis=1)
```




    first       1.0
    second     13.0
    third     151.0
    fourth     75.0
    dtype: float64




```python
df['A']
```




    first      2
    second     4
    third      8
    fourth    16
    Name: A, dtype: int64




```python
df['A'].mean()
```




    7.5




```python
df['A'].var()
```




    38.333333333333336




```python
df.values
```




    array([[ 2,  3,  1],
           [ 4,  9,  2],
           [ 8, 27,  4],
           [16,  1,  1]])




```python
df.to_numpy()
```




    array([[ 2,  3,  1],
           [ 4,  9,  2],
           [ 8, 27,  4],
           [16,  1,  1]])




```python
df.describe()
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.000000</td>
      <td>4.00000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.500000</td>
      <td>10.00000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.191392</td>
      <td>11.83216</td>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.500000</td>
      <td>2.50000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>6.00000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10.000000</td>
      <td>13.50000</td>
      <td>2.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.000000</td>
      <td>27.00000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe().at['mean', 'A']

```




    7.5




```python
df.describe().at['50%', 'B']
```




    6.0


