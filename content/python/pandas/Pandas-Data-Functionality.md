---
title: "Pandas-Data-Functionality"
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
#Create a Range of Dates
pd.date_range('1/1/2011', periods=5)
```




    DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04',
                   '2011-01-05'],
                  dtype='datetime64[ns]', freq='D')




```python
#Change the Date Frequency
pd.date_range('1/1/2011', periods=5,freq='M')
```




    DatetimeIndex(['2011-01-31', '2011-02-28', '2011-03-31', '2011-04-30',
                   '2011-05-31'],
                  dtype='datetime64[ns]', freq='M')




```python
#bdate_range
pd.date_range('1/1/2011', periods=5)
```




    DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04',
                   '2011-01-05'],
                  dtype='datetime64[ns]', freq='D')




```python
start = pd.datetime(2011, 1, 1)
end = pd.datetime(2011, 1, 5)

pd.date_range(start, end)
```

    <ipython-input-7-dd7f9190c50e>:1: FutureWarning: The pandas.datetime class is deprecated and will be removed from pandas in a future version. Import from datetime module instead.
      start = pd.datetime(2011, 1, 1)
    <ipython-input-7-dd7f9190c50e>:2: FutureWarning: The pandas.datetime class is deprecated and will be removed from pandas in a future version. Import from datetime module instead.
      end = pd.datetime(2011, 1, 5)





    DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03', '2011-01-04',
                   '2011-01-05'],
                  dtype='datetime64[ns]', freq='D')


