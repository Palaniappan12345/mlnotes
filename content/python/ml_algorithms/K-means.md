---
title: "K-means"
author: "Palaniappan S"
date: 2020-09-07
description: "-"
type: technical_note
draft: false
---

```python
# importing required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
```


```python

# read the train and test dataset
train_data = pd.read_csv('train1-data.csv')
test_data = pd.read_csv('test1-data.csv')

# shape of the dataset
print('Shape of training data :',train_data.shape)
print('Shape of testing data :',test_data.shape)

```

    Shape of training data : (100, 5)
    Shape of testing data : (100, 5)



```python
model = KMeans() 

# fit the model with the training data
model.fit(train_data,train_data)
```




    KMeans()




```python
# Number of Clusters
print('\nDefault number of Clusters : ',model.n_clusters)

# predict the clusters on the train dataset
predict_train = model.predict(train_data)
print('\nCLusters on train data',predict_train) 
```

    
    Default number of Clusters :  8
    
    CLusters on train data [4 6 2 6 4 5 5 6 6 3 0 0 3 2 6 0 2 1 5 4 1 3 3 2 1 2 0 0 2 3 1 3 3 2 2 0 7
     0 1 3 2 7 0 0 2 3 3 2 6 0 3 2 5 0 2 0 5 7 4 1 3 4 5 2 3 2 1 3 3 0 5 0 4 5
     6 6 4 3 5 3 5 3 0 5 7 5 2 3 7 3 1 6 0 2 0 5 3 4 0 4]



```python
# predict the target on the test dataset
predict_test = model.predict(test_data)
print('Clusters on test data',predict_test) 

# Now, we will train a model with n_cluster = 3
model_n3 = KMeans(n_clusters=3)

# fit the model with the training data
model_n3.fit(train_data)
```

    Clusters on test data [3 4 7 2 5 4 2 3 5 1 3 1 5 5 5 3 3 5 5 6 2 2 5 5 3 5 2 4 5 2 0 4 3 5 4 2 0
     6 3 2 2 4 7 2 5 3 5 6 3 3 1 4 3 0 4 3 0 3 3 7 3 3 5 0 6 5 0 5 3 3 5 7 2 0
     5 2 3 2 3 4 3 5 1 2 7 4 3 5 4 2 4 1 3 5 2 4 4 4 0 2]





    KMeans(n_clusters=3)




```python
# Number of Clusters
print('\nNumber of Clusters : ',model_n3.n_clusters)

# predict the clusters on the train dataset
predict_train_3 = model_n3.predict(train_data)
print('\nCLusters on train data',predict_train_3) 
```

    
    Number of Clusters :  3
    
    CLusters on train data [2 0 1 0 2 1 2 0 0 2 0 0 2 1 0 0 1 2 2 2 2 2 2 1 2 1 0 0 1 2 2 2 2 1 1 0 2
     0 2 2 1 2 0 0 1 2 2 1 0 0 2 1 2 0 1 0 2 2 2 2 2 2 2 1 2 1 2 2 2 0 1 0 2 2
     0 0 0 2 0 2 2 2 0 2 2 2 1 2 2 2 2 0 0 1 0 2 2 2 0 2]



```python
# predict the target on the test dataset
predict_test_3 = model_n3.predict(test_data)
print('Clusters on test data',predict_test_3) 
```

    Clusters on test data [2 2 2 1 2 2 1 2 2 2 2 2 2 1 1 2 2 2 2 0 1 1 2 2 2 2 1 2 2 1 0 2 2 2 2 1 0
     0 2 1 1 2 2 1 2 2 2 0 2 2 2 2 2 0 2 2 0 2 2 2 2 2 2 0 0 2 0 2 2 2 0 2 1 0
     2 1 2 1 2 0 2 2 2 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 0 1]

