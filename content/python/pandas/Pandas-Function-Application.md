---
title: "Pandas-Function-Application"
author: "Palaniappan S"
date: 2020-09-04
description: "-"
type: technical_note
draft: false
---

```python
import numpy as np
import scipy.stats
import pandas as pd
import sklearn
```


```python
#Table-wise Function Application
def adder(ele1,ele2):
   return ele1+ele2

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
df.apply(np.mean)
```




    col1   -0.165957
    col2   -0.488008
    col3   -0.516572
    dtype: float64




```python
#Row or Column Wise Function Application
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.apply(np.mean)
df.apply(np.mean)
```




    col1    0.129294
    col2    0.310800
    col3    0.410874
    dtype: float64




```python
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.apply(lambda x: x.max() - x.min())
df.apply(np.mean)
```




    col1    0.284628
    col2   -0.365105
    col3    0.449976
    dtype: float64




```python
#Element Wise Function Application
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])

# My custom function
df['col1'].map(lambda x:x*100)
df.apply(np.mean)
```




    col1   -0.095422
    col2    0.272017
    col3    0.163533
    dtype: float64




```python
# My custom function
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.applymap(lambda x:x*100)
df.apply(np.mean)
```




    col1   -0.483925
    col2   -0.246296
    col3   -1.356933
    dtype: float64


