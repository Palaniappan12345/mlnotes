---
title: "Mode-Metrics"
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
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_
```




    2




```python
mode_ = statistics.mode(u)
mode_
```




    2




```python
mode_ = statistics.multimode(u)
mode_
```




    [2]




```python
v = [12, 15, 12, 15, 21, 15, 12]
```


```python
statistics.multimode(v)
```




    [12, 15]




```python
statistics.mode([2, math.nan, 2])
```




    2




```python
statistics.multimode([2, math.nan, 2])
```




    [2]




```python
statistics.mode([2, math.nan, 0, math.nan, 5])
```




    nan




```python
statistics.multimode([2, math.nan, 0, math.nan, 5])
```




    [nan]




```python
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_
```




    ModeResult(mode=array([2]), count=array([2]))




```python
mode_ = scipy.stats.mode(v)
mode_

```




    ModeResult(mode=array([12]), count=array([3]))




```python
mode_.mode
```




    array([12])




```python
mode_.count
```




    array([3])




```python
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
```




    0    2
    dtype: int64




```python
v.mode()

```




    0    12
    1    15
    dtype: int64




```python
w.mode()
```




    0    2.0
    dtype: float64


