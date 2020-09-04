---
title: "Label-encoding"
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
data = pd.read_csv("grocery.csv")
print(data) 

```

               Item  Price
    0         Onion    100
    1           Egg     10
    2        Tomato     60
    3        Carrot     60
    4       Cabbage     20
    5          Milk     30
    6        Potato     50
    7  Mosquito Bat    200
    8       Scissor     75
    9       Shampoo      3



```python
# label encoding the data 
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 
```


```python
data['Item']= le.fit_transform(data['Item']) 
print(data)
```

       Item  Price
    0     5    100
    1     2     10
    2     9     60
    3     1     60
    4     0     20
    5     3     30
    6     6     50
    7     4    200
    8     7     75
    9     8      3

