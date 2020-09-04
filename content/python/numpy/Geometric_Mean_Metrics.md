---
title: "Geometric-Mean-Metrics"
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
gmean = 1
for item in x:
    gmean *= item
```


```python
gmean **= 1 / len(x)
gmean
```




    4.677885674856041




```python
gmean = statistics.geometric_mean(x)
gmean
```




    4.67788567485604




```python
gmean = statistics.geometric_mean(x_with_nan)
gmean
```




    nan




```python
scipy.stats.gmean(y)
```




    4.67788567485604


