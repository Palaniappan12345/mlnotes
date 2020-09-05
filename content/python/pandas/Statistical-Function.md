---
title: "Statistical-Function"
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
s = pd.Series([1,2,3,4,5,4])
print (s.pct_change())
```

    0         NaN
    1    1.000000
    2    0.500000
    3    0.333333
    4    0.250000
    5   -0.200000
    dtype: float64



```python
df = pd.DataFrame(np.random.randn(5, 2))
df.pct_change()
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.288289</td>
      <td>-43.839522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.513995</td>
      <td>-0.874372</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.261274</td>
      <td>12.072874</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.888029</td>
      <td>-1.649426</td>
    </tr>
  </tbody>
</table>
</div>




```python
s1 = pd.Series(np.random.randn(10))
s2 = pd.Series(np.random.randn(10))
print (s1.cov(s2))
```

    0.2525814828324182



```python
frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
print (frame['a'].cov(frame['b']))
print (frame.cov())
```

    -0.2619811929124795
              a         b         c         d         e
    a  0.942075 -0.261981 -0.117061 -0.434967  0.263974
    b -0.261981  2.521208 -0.307216  0.337377  0.122552
    c -0.117061 -0.307216  0.825823  0.358778 -0.032586
    d -0.434967  0.337377  0.358778  1.235682  0.099762
    e  0.263974  0.122552 -0.032586  0.099762  0.979503



```python
frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])

print (frame['a'].corr(frame['b']))
print (frame.corr())
```

    0.04652251241079996
              a         b         c         d         e
    a  1.000000  0.046523 -0.150956  0.668061  0.164873
    b  0.046523  1.000000 -0.218552 -0.096044  0.151963
    c -0.150956 -0.218552  1.000000 -0.084925 -0.456316
    d  0.668061 -0.096044 -0.084925  1.000000  0.230871
    e  0.164873  0.151963 -0.456316  0.230871  1.000000



```python
s = pd.Series(np.random.randn(5), index=list('abcde'))
s['d'] = s['b'] # so there's a tie
s.rank()
```




    a    5.0
    b    2.5
    c    1.0
    d    2.5
    e    4.0
    dtype: float64


