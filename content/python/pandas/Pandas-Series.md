---
title: "Pandas-Series"
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
#Creating empty series
#import the pandas library and aliasing as pd
import pandas as pd
s = pd.Series()
s
```

    <ipython-input-4-4eaadbf67d2e>:4: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
      s = pd.Series()





    Series([], dtype: float64)




```python
#Create a Series from ndarray
data = np.array(['a','b','c','d'])
s = pd.Series(data)
s
```




    0    a
    1    b
    2    c
    3    d
    dtype: object




```python
data = np.array(['a','b','c','d'])
s = pd.Series(data,index=[100,101,102,103])
s
```




    100    a
    101    b
    102    c
    103    d
    dtype: object




```python
#Create a Series from dict
data = {'a' : 0., 'b' : 1., 'c' : 2.}
s = pd.Series(data)
s
```




    a    0.0
    b    1.0
    c    2.0
    dtype: float64




```python
#Create a Series from Scalar
s = pd.Series(5, index=[0, 1, 2, 3])
s
```




    0    5
    1    5
    2    5
    3    5
    dtype: int64




```python
#Accessing Data from Series with Position
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve the first element
s[0]
```




    1




```python
#Retrieve Data Using Label (Index)
s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])

#retrieve multiple elements
s[['a','c','d']]
```




    a    1
    c    3
    d    4
    dtype: int64


