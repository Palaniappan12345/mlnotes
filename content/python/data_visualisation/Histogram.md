---
title: "Histogram"
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
import matplotlib.pyplot as plt
plt.style.use('ggplot')
```


```python
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)
```


```python
hist, bin_edges = np.histogram(x, bins=10)
hist
```




    array([  9,  20,  70, 146, 217, 239, 160,  86,  38,  15])




```python
bin_edges
```




    array([-3.04614305, -2.46559324, -1.88504342, -1.3044936 , -0.72394379,
           -0.14339397,  0.43715585,  1.01770566,  1.59825548,  2.1788053 ,
            2.75935511])




```python
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()
```


![png](Histogram_6_0.png)



```python
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()
```


![png](Histogram_7_0.png)

