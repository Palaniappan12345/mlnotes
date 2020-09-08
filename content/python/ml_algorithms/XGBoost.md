---
title: "XGBoost"
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
from xgboost import XGBClassifier
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
model = XGBClassifier()

# fit the model with the training data
model.fit(train_x,train_y)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)




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
     0 0 0 0 1 0 0 1 0 1 0 1 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 1 0 1 1 0
     0 0 0 1 1 0 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1
     0 1 1 1 1 0 0 1 0 1 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 1 1 1 0 1 0 0 0 1 0
     0 1 1 0 1 1 1 0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0
     0 0 1 1 0 1 1 0 1 0 1 1 1 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 1
     1 0 0 1 1 0 1 0 0 0 1 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0
     0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1
     0 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 1 0 0 0 1 0 0
     0 1 0 0 0 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
     0 0 1 1 1 0 0 0 0 1 1 0 1 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
     1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0 1 0 1 0 0 1
     0 0 0 1 1 0 0 1 0 0 1 0 1 0 0 1 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 1 0 1 1 1 0
     1 0 1 0 0 1 0 1 0 1 0 0 1 0 0 1 0 1 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1
     0 0 0 1 0 1 1 1 1 0 1 1 0 0 1 0 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0 1 0 1 1 1 0
     0 0 0 0 1 0 0 1 0 0 1 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0
     1 0 1 1 1 0 0 1 0]
    
    accuracy_score on train dataset :  0.9691011235955056



```python
# predict the target on the test dataset
predict_test = model.predict(test_x)
print('\nTarget on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)
```

    
    Target on test data [0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 1 0 1 1 0 1 1 0 0 0 1 0 0
     0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 1 1 1
     0 1 0 0 0 1 1 0 0 1 0 1 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 1 0 1 0 1 0 0 0 0 0
     0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 1 1 0 0 0 0 0 0 0]
    
    accuracy_score on test dataset :  0.8491620111731844

