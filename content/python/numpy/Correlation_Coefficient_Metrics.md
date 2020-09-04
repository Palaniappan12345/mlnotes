---
title: "Correlation-Coefficient-Metrics"
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
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
```




    [8.0, 1, 2.5, 4, 28.0]




```python
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y
```




    array([ 8. ,  1. ,  2.5,  4. , 28. ])




```python
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)
```


```python
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy
```




    19.95




```python
cov_matrix = np.cov(x_, y_)
cov_matrix
```




    array([[38.5       , 19.95      ],
           [19.95      , 13.91428571]])




```python
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r
```




    0.861950005631606




```python
r, p = scipy.stats.pearsonr(x_, y_)
r
```




    0.8619500056316061




```python
p
```




    5.122760847201135e-07




```python
corr_matrix = np.corrcoef(x_, y_)
corr_matrix
```




    array([[1.        , 0.86195001],
           [0.86195001, 1.        ]])




```python
r = corr_matrix[0, 1]
r
```




    0.8619500056316061




```python
r = corr_matrix[1, 0]
r
```




    0.861950005631606




```python
scipy.stats.linregress(x_, y_)
```




    LinregressResult(slope=0.5181818181818181, intercept=5.714285714285714, rvalue=0.861950005631606, pvalue=5.122760847201164e-07, stderr=0.06992387660074979)




```python
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r
```




    0.861950005631606




```python
r = x__.corr(y__)
r
```




    0.8619500056316061




```python
r = y__.corr(x__)
r
```




    0.861950005631606


