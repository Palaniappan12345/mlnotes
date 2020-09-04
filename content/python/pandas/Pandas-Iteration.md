---
title: "Pandas-Iteration"
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
N=20
df = pd.DataFrame({
   'A': pd.date_range(start='2016-01-01',periods=N,freq='D'),
   'x': np.linspace(0,stop=N-1,num=N),
   'y': np.random.rand(N),
   'C': np.random.choice(['Low','Medium','High'],N).tolist(),
   'D': np.random.normal(100, 10, size=(N)).tolist()
   })

for col in df:
   print(col)
```

    A
    x
    y
    C
    D



```python
#iteritems()
df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])
for key,value in df.iteritems():
   print (key,value)
```

    col1 0    1.383371
    1   -0.752669
    2   -0.989764
    3    0.184005
    Name: col1, dtype: float64
    col2 0   -1.432349
    1    1.804341
    2    1.730258
    3   -0.071981
    Name: col2, dtype: float64
    col3 0    1.673208
    1   -0.460951
    2    0.117168
    3    1.432042
    Name: col3, dtype: float64



```python
#iterrows()
df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])
for row_index,row in df.iterrows():
   print (row_index,row)
```

    0 col1   -1.598730
    col2    0.150081
    col3    2.090892
    Name: 0, dtype: float64
    1 col1   -1.005536
    col2   -0.702690
    col3   -0.177421
    Name: 1, dtype: float64
    2 col1   -1.456623
    col2    0.005965
    col3    1.387848
    Name: 2, dtype: float64
    3 col1   -0.860579
    col2    0.687143
    col3    1.130771
    Name: 3, dtype: float64



```python
#itertuples()

df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])
for row in df.itertuples():
    print (row)
```

    Pandas(Index=0, col1=0.4995751442133796, col2=-1.8822488559989277, col3=0.7296904097725797)
    Pandas(Index=1, col1=0.9220072487989778, col2=1.2647163095295109, col3=1.4179312382655798)
    Pandas(Index=2, col1=-0.49672528772288904, col2=1.5933363825396494, col3=-1.2392169822844235)
    Pandas(Index=3, col1=1.2404854643637526, col2=1.1245862721867765, col3=-0.5181051363165968)



```python
df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])

for index, row in df.iterrows():
   row['a'] = 10
print (df)
```

           col1      col2      col3
    0  0.320707  1.975114  1.901456
    1 -0.316004  0.445194  1.563622
    2  1.344503  0.175026  0.084224
    3 -1.469521 -0.147534  1.003356

