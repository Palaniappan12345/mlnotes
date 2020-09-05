---
title: "Pandas-Sorting"
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
unsorted_df=pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns=['col2','col1'])
(unsorted_df)
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
      <th>col2</th>
      <th>col1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.219984</td>
      <td>-0.464369</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.300591</td>
      <td>-0.076174</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.840346</td>
      <td>-1.347094</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.221474</td>
      <td>0.442064</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.017141</td>
      <td>-0.480142</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.388411</td>
      <td>0.548868</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.254729</td>
      <td>0.538852</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.466358</td>
      <td>1.224428</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-0.829130</td>
      <td>0.832262</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.341993</td>
      <td>-2.400047</td>
    </tr>
  </tbody>
</table>
</div>




```python
sorted_df=unsorted_df.sort_index()
print (sorted_df)
```

           col2      col1
    0 -0.829130  0.832262
    1 -0.219984 -0.464369
    2  0.221474  0.442064
    3 -0.017141 -0.480142
    4 -1.300591 -0.076174
    5 -0.388411  0.548868
    6  0.840346 -1.347094
    7 -0.341993 -2.400047
    8 -0.466358  1.224428
    9  0.254729  0.538852



```python
sorted_df = unsorted_df.sort_index(ascending=False)
print (sorted_df)
```

           col2      col1
    9  0.254729  0.538852
    8 -0.466358  1.224428
    7 -0.341993 -2.400047
    6  0.840346 -1.347094
    5 -0.388411  0.548868
    4 -1.300591 -0.076174
    3 -0.017141 -0.480142
    2  0.221474  0.442064
    1 -0.219984 -0.464369
    0 -0.829130  0.832262



```python
sorted_df=unsorted_df.sort_index(axis=1)

print (sorted_df)
```

           col1      col2
    1 -0.464369 -0.219984
    4 -0.076174 -1.300591
    6 -1.347094  0.840346
    2  0.442064  0.221474
    3 -0.480142 -0.017141
    5  0.548868 -0.388411
    9  0.538852  0.254729
    8  1.224428 -0.466358
    0  0.832262 -0.829130
    7 -2.400047 -0.341993



```python
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1')

print (sorted_df)
```

       col1  col2
    1     1     3
    2     1     2
    3     1     4
    0     2     1



```python
unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})
sorted_df = unsorted_df.sort_values(by='col1' ,kind='mergesort')

print (sorted_df)
```

       col1  col2
    1     1     3
    2     1     2
    3     1     4
    0     2     1

