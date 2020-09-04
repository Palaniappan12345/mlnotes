---
title: "Median-Metrics"
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
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])

median_
```




    4




```python
median_ = statistics.median(x)
median_
```




    4




```python
median_ = statistics.median(x[:-1])
median_
```




    3.25




```python
statistics.median_low(x[:-1])
```




    2.5




```python
statistics.median_high(x[:-1])
```




    4




```python
statistics.median(x_with_nan)
```




    6.0




```python
statistics.median_low(x_with_nan)
```




    4




```python
statistics.median_high(x_with_nan)
```




    8.0




```python
median_ = np.median(y)
median_
```




    4.0




```python
median_ = np.median(y[:-1])
median_
```




    3.25




```python
np.nanmedian(y_with_nan)
```




    4.0




```python
np.nanmedian(y_with_nan[:-1])
```




    3.25


