---
title: "Confusion-matrix"
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
from sklearn.metrics import confusion_matrix
```


```python
data = {'y_Actual':    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        'y_Predicted': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
        }
```


```python
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
```


```python
f1_score(true, pred,average="micro")
```




    0.3333333333333333




```python
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)
```

    Predicted  0  1
    Actual         
    0          5  2
    1          1  4

