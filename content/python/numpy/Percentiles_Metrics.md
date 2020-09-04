---
title: "Percentiles-Metrics"
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
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)
```




    [8.0]




```python
statistics.quantiles(x, n=4, method='inclusive')
```




    [0.1, 8.0, 21.0]




```python
y = np.array(x)
np.percentile(y, 5)
```




    -3.44




```python
np.percentile(y, 95)
```




    34.919999999999995




```python
np.percentile(y, [25, 50, 75])
```




    array([ 0.1,  8. , 21. ])




```python
np.median(y)
```




    8.0




```python
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan
```




    array([-5. , -1.1,  nan,  0.1,  2. ,  8. , 12.8, 21. , 25.8, 41. ])




```python
np.nanpercentile(y_with_nan, [25, 50, 75])
```




    array([ 0.1,  8. , 21. ])




```python
np.quantile(y, 0.05)
```




    -3.44




```python
np.quantile(y, 0.95)
```




    34.919999999999995




```python
np.quantile(y, [0.25, 0.5, 0.75])
```




    array([ 0.1,  8. , 21. ])




```python
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])
```




    array([ 0.1,  8. , 21. ])


