---
title: "Bar-chart"
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
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)
```


```python
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```


![png](Bar_Chart_4_0.png)

