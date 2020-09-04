---
title: "Mean-Metrics"
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
x_with_nan
```




    [8.0, 1, 2.5, nan, 4, 28.0]




```python
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y
```




    array([ 8. ,  1. ,  2.5,  4. , 28. ])




```python
y_with_nan

z
```




    0     8.0
    1     1.0
    2     2.5
    3     4.0
    4    28.0
    dtype: float64




```python
z_with_nan
```




    0     8.0
    1     1.0
    2     2.5
    3     NaN
    4     4.0
    5    28.0
    dtype: float64




```python
mean_ = sum(x) / len(x)
mean_

```




    8.7




```python
mean_ = statistics.mean(x)
mean_

```




    8.7




```python
mean_ = statistics.fmean(x)
mean_

```




    8.7




```python
mean_ = statistics.mean(x_with_nan)
mean_
```




    nan




```python
mean_ = statistics.fmean(x_with_nan)
mean_
```




    nan




```python
mean_ = np.mean(y)
mean_
```




    8.7




```python
mean_ = y.mean()
mean_
```




    8.7




```python
np.mean(y_with_nan)
```




    nan




```python
y_with_nan.mean()
```




    nan




```python
np.nanmean(y_with_nan)
```




    8.7




```python
mean_ = z.mean()
mean_

```




    8.7




```python
z_with_nan.mean()
```




    8.7




```python

```
