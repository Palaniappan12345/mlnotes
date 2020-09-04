---
title: "Axes-Metrics"
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
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a
```




    array([[ 1,  1,  1],
           [ 2,  3,  1],
           [ 4,  9,  2],
           [ 8, 27,  4],
           [16,  1,  1]])




```python
np.mean(a)
```




    5.4




```python
a.mean()
```




    5.4




```python
np.median(a)
```




    2.0




```python
a.var(ddof=1)
```




    53.40000000000001




```python
np.mean(a, axis=0)
```




    array([6.2, 8.2, 1.8])




```python
a.mean(axis=0)
```




    array([6.2, 8.2, 1.8])




```python
np.mean(a, axis=1)
```




    array([ 1.,  2.,  5., 13.,  6.])




```python
a.mean(axis=1)
```




    array([ 1.,  2.,  5., 13.,  6.])




```python
 np.median(a, axis=0)
```




    array([4., 3., 1.])




```python
np.median(a, axis=1)
```




    array([1., 2., 4., 8., 1.])




```python
a.var(axis=0, ddof=1)
```




    array([ 37.2, 121.2,   1.7])




```python
a.var(axis=1, ddof=1)
```




    array([  0.,   1.,  13., 151.,  75.])




```python
scipy.stats.gmean(a)  # Default: axis=0
```




    array([4.        , 3.73719282, 1.51571657])




```python
scipy.stats.gmean(a, axis=1)
```




    array([1.        , 1.81712059, 4.16016765, 9.52440631, 2.5198421 ])




```python
scipy.stats.gmean(a, axis=None)
```




    2.829705017016332




```python
scipy.stats.describe(a, axis=None, ddof=1, bias=False)
```




    DescribeResult(nobs=15, minmax=(1, 27), mean=5.4, variance=53.40000000000001, skewness=2.264965290423389, kurtosis=5.212690982795767)




```python
scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0
```




    DescribeResult(nobs=5, minmax=(array([1, 1, 1]), array([16, 27,  4])), mean=array([6.2, 8.2, 1.8]), variance=array([ 37.2, 121.2,   1.7]), skewness=array([1.32531471, 1.79809454, 1.71439233]), kurtosis=array([1.30376344, 3.14969121, 2.66435986]))




```python
scipy.stats.describe(a, axis=1, ddof=1, bias=False)
```




    DescribeResult(nobs=3, minmax=(array([1, 1, 2, 4, 1]), array([ 1,  3,  9, 27, 16])), mean=array([ 1.,  2.,  5., 13.,  6.]), variance=array([  0.,   1.,  13., 151.,  75.]), skewness=array([0.        , 0.        , 1.15206964, 1.52787436, 1.73205081]), kurtosis=array([-3. , -1.5, -1.5, -1.5, -1.5]))




```python
result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean
```




    array([ 1.,  2.,  5., 13.,  6.])


