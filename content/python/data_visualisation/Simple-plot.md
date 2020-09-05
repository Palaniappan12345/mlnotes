---
title: "Simple-plot"
author: "Palaniappan S"
date: 2020-09-05
description: "-"
type: technical_note
draft: false
---

```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```


```python
# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)
```




    [<matplotlib.lines.Line2D at 0x7fcfbdfa4400>]




![png](Simple-plot_2_1.png)



```python
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
print(ax.grid())
```

    None



```python
fig.savefig("test.png")
print(plt.show())
```

    None

