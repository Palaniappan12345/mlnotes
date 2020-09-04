---
title: "Pandas-Aggregations"
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
#Applying Aggregations on DataFrame
df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B', 'C', 'D'])

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
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>0.744195</td>
      <td>0.496521</td>
      <td>-0.134224</td>
      <td>-0.409785</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>-0.566781</td>
      <td>1.073489</td>
      <td>-0.366540</td>
      <td>-0.438647</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>0.116797</td>
      <td>2.020010</td>
      <td>1.350292</td>
      <td>1.059606</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>-1.307016</td>
      <td>-0.376845</td>
      <td>-0.255637</td>
      <td>-1.209852</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>-0.211576</td>
      <td>0.353178</td>
      <td>1.508859</td>
      <td>0.414203</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>0.916487</td>
      <td>-0.745630</td>
      <td>1.668101</td>
      <td>0.212629</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>-0.742183</td>
      <td>0.420636</td>
      <td>0.256069</td>
      <td>0.322182</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>-0.107362</td>
      <td>-0.337345</td>
      <td>-0.406309</td>
      <td>0.897112</td>
    </tr>
    <tr>
      <th>2000-01-09</th>
      <td>0.799588</td>
      <td>-0.381785</td>
      <td>1.757035</td>
      <td>0.031658</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>-1.582323</td>
      <td>-1.925102</td>
      <td>-0.361671</td>
      <td>1.229835</td>
    </tr>
  </tbody>
</table>
</div>




```python
r = df.rolling(window=3,min_periods=1)
r
```




    Rolling [window=3,min_periods=1,center=False,axis=0]




```python
#Apply Aggregation on a Whole Dataframe
r = df.rolling(window=3,min_periods=1)
r.aggregate(np.sum)
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
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>0.744195</td>
      <td>0.496521</td>
      <td>-0.134224</td>
      <td>-0.409785</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>0.177414</td>
      <td>1.570010</td>
      <td>-0.500764</td>
      <td>-0.848433</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>0.294211</td>
      <td>3.590020</td>
      <td>0.849527</td>
      <td>0.211173</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>-1.757000</td>
      <td>2.716654</td>
      <td>0.728115</td>
      <td>-0.588893</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>-1.401794</td>
      <td>1.996343</td>
      <td>2.603514</td>
      <td>0.263957</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>-0.602105</td>
      <td>-0.769297</td>
      <td>2.921323</td>
      <td>-0.583020</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>-0.037272</td>
      <td>0.028183</td>
      <td>3.433029</td>
      <td>0.949014</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>0.066942</td>
      <td>-0.662339</td>
      <td>1.517861</td>
      <td>1.431924</td>
    </tr>
    <tr>
      <th>2000-01-09</th>
      <td>-0.049957</td>
      <td>-0.298494</td>
      <td>1.606795</td>
      <td>1.250953</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>-0.890096</td>
      <td>-2.644232</td>
      <td>0.989055</td>
      <td>2.158605</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Apply Aggregation on a Single Column of a Dataframe
r = df.rolling(window=3,min_periods=1)
r['A'].aggregate(np.sum)
```




    2000-01-01    0.744195
    2000-01-02    0.177414
    2000-01-03    0.294211
    2000-01-04   -1.757000
    2000-01-05   -1.401794
    2000-01-06   -0.602105
    2000-01-07   -0.037272
    2000-01-08    0.066942
    2000-01-09   -0.049957
    2000-01-10   -0.890096
    Freq: D, Name: A, dtype: float64




```python
#Apply Aggregation on Multiple Columns of a DataFrame
r = df.rolling(window=3,min_periods=1)
r[['A','B']].aggregate(np.sum)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>0.744195</td>
      <td>0.496521</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>0.177414</td>
      <td>1.570010</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>0.294211</td>
      <td>3.590020</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>-1.757000</td>
      <td>2.716654</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>-1.401794</td>
      <td>1.996343</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>-0.602105</td>
      <td>-0.769297</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>-0.037272</td>
      <td>0.028183</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>0.066942</td>
      <td>-0.662339</td>
    </tr>
    <tr>
      <th>2000-01-09</th>
      <td>-0.049957</td>
      <td>-0.298494</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>-0.890096</td>
      <td>-2.644232</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Apply Multiple Functions on Multiple Columns of a DataFrame
r = df.rolling(window=3,min_periods=1)
r[['A','B']].aggregate([np.sum,np.mean])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">A</th>
      <th colspan="2" halign="left">B</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>sum</th>
      <th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>0.744195</td>
      <td>0.744195</td>
      <td>0.496521</td>
      <td>0.496521</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>0.177414</td>
      <td>0.088707</td>
      <td>1.570010</td>
      <td>0.785005</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>0.294211</td>
      <td>0.098070</td>
      <td>3.590020</td>
      <td>1.196673</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>-1.757000</td>
      <td>-0.585667</td>
      <td>2.716654</td>
      <td>0.905551</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>-1.401794</td>
      <td>-0.467265</td>
      <td>1.996343</td>
      <td>0.665448</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>-0.602105</td>
      <td>-0.200702</td>
      <td>-0.769297</td>
      <td>-0.256432</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>-0.037272</td>
      <td>-0.012424</td>
      <td>0.028183</td>
      <td>0.009394</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>0.066942</td>
      <td>0.022314</td>
      <td>-0.662339</td>
      <td>-0.220780</td>
    </tr>
    <tr>
      <th>2000-01-09</th>
      <td>-0.049957</td>
      <td>-0.016652</td>
      <td>-0.298494</td>
      <td>-0.099498</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>-0.890096</td>
      <td>-0.296699</td>
      <td>-2.644232</td>
      <td>-0.881411</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Apply Different Functions to Different Columns of a Dataframe
r = df.rolling(window=3,min_periods=1)
r.aggregate({'A' : np.sum,'B' : np.mean})
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>0.744195</td>
      <td>0.496521</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>0.177414</td>
      <td>0.785005</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>0.294211</td>
      <td>1.196673</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>-1.757000</td>
      <td>0.905551</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>-1.401794</td>
      <td>0.665448</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>-0.602105</td>
      <td>-0.256432</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>-0.037272</td>
      <td>0.009394</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>0.066942</td>
      <td>-0.220780</td>
    </tr>
    <tr>
      <th>2000-01-09</th>
      <td>-0.049957</td>
      <td>-0.099498</td>
    </tr>
    <tr>
      <th>2000-01-10</th>
      <td>-0.890096</td>
      <td>-0.881411</td>
    </tr>
  </tbody>
</table>
</div>


