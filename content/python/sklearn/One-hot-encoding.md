---
title: "One-hot-encoding"
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



```python
# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 
ohe = OneHotEncoder() 

```


```python
# one hot encoding the data
from sklearn.compose import ColumnTransformer 
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') 
```


```python
datum = np.array(columnTransformer.fit_transform(data).toarray()) 
datum = pd.DataFrame(datum)
print(datum)
```

         0    1    2    3    4    5    6    7    8    9     10
    0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  100.0
    1  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   10.0
    2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   60.0
    3  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   60.0
    4  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   20.0
    5  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0   30.0
    6  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0   50.0
    7  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  200.0
    8  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0   75.0
    9  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0    3.0



```python
data = ohe.fit_transform(data).toarray()
data = pd.DataFrame(data)
print(data)
```

         0    1    2    3    4    5    6    7    8    9   10   11   12   13   14  \
    0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    1  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0   
    2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0   
    3  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    4  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0   
    5  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0   
    6  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0   
    7  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    8  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   
    9  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  1.0  0.0  0.0  0.0  0.0   
    
        15   16   17   18  
    0  0.0  0.0  1.0  0.0  
    1  0.0  0.0  0.0  0.0  
    2  1.0  0.0  0.0  0.0  
    3  1.0  0.0  0.0  0.0  
    4  0.0  0.0  0.0  0.0  
    5  0.0  0.0  0.0  0.0  
    6  0.0  0.0  0.0  0.0  
    7  0.0  0.0  0.0  1.0  
    8  0.0  1.0  0.0  0.0  
    9  0.0  0.0  0.0  0.0  

