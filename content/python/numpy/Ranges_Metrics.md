---
title: "Ranges(Min/Max)-Metrics"
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
np.ptp(y)
```




    27.0




```python
np.ptp(y_with_nan)
```




    nan




```python
np.amax(y) - np.amin(y)
```




    27.0




```python
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
```




    27.0




```python
y.max() - y.min()
```




    27.0




```python
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]
```




    5.5




```python
result = scipy.stats.describe(y, ddof=1, bias=False)
result
```




    DescribeResult(nobs=5, minmax=(1.0, 28.0), mean=8.7, variance=123.19999999999999, skewness=1.9470432273905927, kurtosis=3.878019618875446)




```python
result.nobs
```




    5




```python
result.minmax[0]  # Min
```




    1.0




```python
result.minmax[1]  # Max
```




    28.0




```python
result.mean
```




    8.7




```python
result.variance
```




    123.19999999999999




```python
result.skewness
```




    1.9470432273905927




```python
result.kurtosis
```




    3.878019618875446




```python
result = z.describe()
result
```




    count     5.00000
    mean      8.70000
    std      11.09955
    min       1.00000
    25%       2.50000
    50%       4.00000
    75%       8.00000
    max      28.00000
    dtype: float64


