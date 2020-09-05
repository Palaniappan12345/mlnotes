---
title: "Sub-plots"
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
x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
```




    Text(0, 0.5, 'Damped oscillation')




![png](Sub-plots_2_1.png)



```python

plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.show()
```


![png](Sub-plots_3_0.png)

