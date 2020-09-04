---
title: "Weighted-Mean-Metrics"
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
0.2 * 2 + 0.5 * 4 + 0.3 * 8
```




    4.8




```python
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean
```




    6.95




```python
wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean
```




    6.95




```python
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean
```




    6.95




```python
wmean = np.average(z, weights=w)
wmean
```




    6.95




```python
(w * y).sum() / w.sum()

```




    6.95




```python
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()
```




    nan




```python
np.average(y_with_nan, weights=w)

```




    nan




```python

```
