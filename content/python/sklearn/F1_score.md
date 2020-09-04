---
title: "F1-score"
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
import sklearn
```


```python
from sklearn.metrics import f1_score
true = [0, 1, 2, 0, 1, 2]
pred = [0, 2, 1, 0, 0, 1]
```


```python
f1_score(true, pred,average=None).mean()
```




    0.26666666666666666




```python
f1_score(true, pred,average="macro")
```




    0.26666666666666666




```python
f1_score(true, pred,average="micro")
```




    0.3333333333333333




```python
f1_score(true, pred,average="weighted")
```




    0.26666666666666666


