---
title: "Working-with-text-data"
author: "Palaniappan S"
date: 2020-09-04
description: "-"
type: technical_note
draft: false
---

```python
import numpy as np
import scipy.stats
import pandas as pd
import sklearn
```


```python
s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveSmith'])

s
```




    0             Tom
    1    William Rick
    2            John
    3         Alber@t
    4             NaN
    5            1234
    6      SteveSmith
    dtype: object




```python
s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveSmith'])

s.str.lower()
```




    0             tom
    1    william rick
    2            john
    3         alber@t
    4             NaN
    5            1234
    6      stevesmith
    dtype: object




```python
s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveSmith'])

s.str.upper()
```




    0             TOM
    1    WILLIAM RICK
    2            JOHN
    3         ALBER@T
    4             NaN
    5            1234
    6      STEVESMITH
    dtype: object




```python
s = pd.Series(['Tom', 'William Rick', 'John', 'Alber@t', np.nan, '1234','SteveSmith'])
s.str.len()
```




    0     3.0
    1    12.0
    2     4.0
    3     7.0
    4     NaN
    5     4.0
    6    10.0
    dtype: float64




```python
s = pd.Series(['Tom ', ' William Rick', 'John', 'Alber@t'])
s
```




    0             Tom 
    1     William Rick
    2             John
    3          Alber@t
    dtype: object




```python
s.str.strip()
```




    0             Tom
    1    William Rick
    2            John
    3         Alber@t
    dtype: object




```python
s.str.split(' ')
```




    0              [Tom, ]
    1    [, William, Rick]
    2               [John]
    3            [Alber@t]
    dtype: object




```python
s.str.cat(sep='_')
```




    'Tom _ William Rick_John_Alber@t'




```python
s.str.get_dummies()
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
      <th>William Rick</th>
      <th>Alber@t</th>
      <th>John</th>
      <th>Tom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
s.str.contains(' ')
```




    0     True
    1     True
    2    False
    3    False
    dtype: bool




```python
s.str.replace('@','$')
```




    0             Tom 
    1     William Rick
    2             John
    3          Alber$t
    dtype: object




```python
s.str.repeat(2)
```




    0                      Tom Tom 
    1     William Rick William Rick
    2                      JohnJohn
    3                Alber@tAlber@t
    dtype: object




```python
s.str.find('e')
```




    0   -1
    1   -1
    2   -1
    3    3
    dtype: int64




```python
s.str.swapcase()
```




    0             tOM 
    1     wILLIAM rICK
    2             jOHN
    3          aLBER@T
    dtype: object




```python
 s.str.isnumeric()
```




    0    False
    1    False
    2    False
    3    False
    dtype: bool


