---
title: "Standard-Deviation-Metrics"
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
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_
```




    123.19999999999999




```python
var_ = statistics.variance(x)
var_
```




    123.2




```python
std_ = var_ ** 0.5
std_

```




    11.099549540409287




```python
std_ = statistics.stdev(x)
std_

```




    11.099549540409287




```python
np.std(y, ddof=1)
```




    11.099549540409285




```python
y.std(ddof=1)
```




    11.099549540409285




```python
np.std(y_with_nan, ddof=1)
```




    nan




```python
y_with_nan.std(ddof=1)
```




    nan




```python
np.nanstd(y_with_nan, ddof=1)
```




    11.099549540409285


