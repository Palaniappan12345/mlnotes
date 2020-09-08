---
title: "Gradient-Boosting-Algorithms"
author: "Palaniappan S"
date: 2020-09-07
description: "-"
type: technical_note
draft: false
---

```python
# importing required libraries
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
```


```python

# read the train and test dataset
train_data = pd.read_csv('train-data.csv')
test_data = pd.read_csv('test-data.csv')

# shape of the dataset
print('Shape of training data :',train_data.shape)
print('Shape of testing data :',test_data.shape)

```

    Shape of training data : (712, 25)
    Shape of testing data : (179, 25)



```python
train_x = train_data.drop(columns=['Survived'],axis=1)
train_y = train_data['Survived']

# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['Survived'],axis=1)
test_y = test_data['Survived']
```


```python
model = GradientBoostingClassifier(n_estimators=100,max_depth=5)

# fit the model with the training data
model.fit(train_x,train_y)
```




    GradientBoostingClassifier(max_depth=5)




```python
# predict the target on the train dataset
predict_train = model.predict(train_x)
print('\nTarget on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)
```

    
    Target on train data [0 1 1 0 0 0 0 0 0 0 0 1 1 1 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 0 0
     1 0 0 0 1 0 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 1 0 1 1 1 1 0 1 0 0 1 0 0 0 0 0
     0 1 1 0 0 1 0 1 1 1 1 0 0 0 1 0 1 0 0 1 0 0 0 1 1 0 0 1 0 1 1 1 0 1 0 0 0
     0 0 0 0 1 0 0 1 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 1 0 1 1 0
     0 0 0 1 1 0 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
     0 1 1 1 1 0 0 1 0 1 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 1 1 1 0 1 0 0 0 1 0
     0 1 1 0 1 1 1 0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0
     0 0 1 1 0 1 1 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 1
     1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 0 1 0 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0
     0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1
     0 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 0 0 0 0 1 0 0
     0 1 0 0 0 0 0 0 1 0 0 1 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
     0 0 1 1 1 0 0 1 0 1 1 0 1 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
     1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0 1 0 1 0 0 1
     0 0 0 1 1 0 0 1 0 0 1 0 1 0 0 1 0 1 0 1 0 0 1 0 0 1 0 0 0 0 1 1 0 1 1 1 0
     1 0 1 0 0 1 0 1 0 1 0 0 1 0 0 1 0 1 1 0 1 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1
     0 0 0 1 0 1 1 1 0 0 1 1 0 0 1 0 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0 1 0 1 0 1 0
     0 0 0 0 1 0 0 1 0 0 1 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0
     1 0 1 1 1 0 0 1 0]
    
    accuracy_score on train dataset :  0.9550561797752809



```python
# predict the target on the test dataset
predict_test = model.predict(test_x)
print('\nTarget on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)
```

    
    Target on test data [0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1
     0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 1 1 0 0 1 0 1 0 0 0 1 1 0 1 1 1 1
     0 1 0 0 0 0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 1 0 1 0 0 0 0 0 0 0
     0 0 0 1 1 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 1 0 1 1 0 0 0 0 0 0 0]
    
    accuracy_score on test dataset :  0.8324022346368715

