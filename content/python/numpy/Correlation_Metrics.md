---
title: "Correlation-Metrics"
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
x_.var(ddof=1)
```




    38.5




```python
y_.var(ddof=1)
```




    13.914285714285711




```python
cov_xy = cov_matrix[0, 1]
cov_xy

```




    19.95




```python
cov_xy = cov_matrix[1, 0]
cov_xy

```




    19.95




```python
cov_xy = x__.cov(y__)
cov_xy
```




    19.95




```python
cov_xy = y__.cov(x__)
cov_xy
```




    19.95


