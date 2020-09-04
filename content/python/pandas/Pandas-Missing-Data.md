---
title: "Pandas-Missing-Data"
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
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f',
'h'],columns=['one', 'two', 'three'])

df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

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
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-1.594997</td>
      <td>-0.870257</td>
      <td>-0.040415</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.242981</td>
      <td>-0.829525</td>
      <td>0.167767</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-0.243680</td>
      <td>0.341354</td>
      <td>0.342675</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.443074</td>
      <td>-0.915286</td>
      <td>0.799884</td>
    </tr>
    <tr>
      <th>g</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.692233</td>
      <td>1.093079</td>
      <td>-0.116146</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Check for Missing Values
df['one'].isnull()
```




    a    False
    b     True
    c    False
    d     True
    e    False
    f    False
    g     True
    h    False
    Name: one, dtype: bool




```python
#Calculations with Missing Data
df['one'].sum()
```




    -2.7310031100055356




```python
df['one'].sum()
```




    -2.7310031100055356




```python
#Cleaning / Filling Missing Data
#Replace NaN with a Scalar Value
df.fillna(0)
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-1.594997</td>
      <td>-0.870257</td>
      <td>-0.040415</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.242981</td>
      <td>-0.829525</td>
      <td>0.167767</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-0.243680</td>
      <td>0.341354</td>
      <td>0.342675</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.443074</td>
      <td>-0.915286</td>
      <td>0.799884</td>
    </tr>
    <tr>
      <th>g</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.692233</td>
      <td>1.093079</td>
      <td>-0.116146</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Fill NA Forward and Backward
df.fillna(method='pad')
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-1.594997</td>
      <td>-0.870257</td>
      <td>-0.040415</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.594997</td>
      <td>-0.870257</td>
      <td>-0.040415</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.242981</td>
      <td>-0.829525</td>
      <td>0.167767</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.242981</td>
      <td>-0.829525</td>
      <td>0.167767</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-0.243680</td>
      <td>0.341354</td>
      <td>0.342675</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.443074</td>
      <td>-0.915286</td>
      <td>0.799884</td>
    </tr>
    <tr>
      <th>g</th>
      <td>-0.443074</td>
      <td>-0.915286</td>
      <td>0.799884</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.692233</td>
      <td>1.093079</td>
      <td>-0.116146</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Drop Missing Values
df.dropna()
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
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>-1.594997</td>
      <td>-0.870257</td>
      <td>-0.040415</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.242981</td>
      <td>-0.829525</td>
      <td>0.167767</td>
    </tr>
    <tr>
      <th>e</th>
      <td>-0.243680</td>
      <td>0.341354</td>
      <td>0.342675</td>
    </tr>
    <tr>
      <th>f</th>
      <td>-0.443074</td>
      <td>-0.915286</td>
      <td>0.799884</td>
    </tr>
    <tr>
      <th>h</th>
      <td>-0.692233</td>
      <td>1.093079</td>
      <td>-0.116146</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Replace Missing (or) Generic Values

df = pd.DataFrame({'one':[10,20,30,40,50,2000], 'two':[1000,0,30,40,50,60]})

df.replace({1000:10,2000:60})
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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>


