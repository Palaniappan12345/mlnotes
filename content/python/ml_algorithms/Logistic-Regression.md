---
title: "Logistic-Regression"
author: "Palaniappan S"
date: 2020-09-05
description: "-"
type: technical_note
draft: false
---

```python
# importing required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```


```python
# read the train and test dataset
train_data = pd.read_csv('train-data.csv')
test_data = pd.read_csv('test-data.csv')


print(train_data.head())
```

       Survived        Age     Fare  Pclass_1  Pclass_2  Pclass_3  Sex_female  \
    0         0  28.500000   7.2292         0         0         1           0   
    1         1  27.000000  10.5000         0         1         0           1   
    2         1  29.699118  16.1000         0         0         1           1   
    3         0  29.699118   0.0000         1         0         0           0   
    4         0  17.000000   8.6625         0         0         1           0   
    
       Sex_male  SibSp_0  SibSp_1  ...  Parch_0  Parch_1  Parch_2  Parch_3  \
    0         1        1        0  ...        1        0        0        0   
    1         0        1        0  ...        1        0        0        0   
    2         0        0        1  ...        1        0        0        0   
    3         1        1        0  ...        1        0        0        0   
    4         1        1        0  ...        1        0        0        0   
    
       Parch_4  Parch_5  Parch_6  Embarked_C  Embarked_Q  Embarked_S  
    0        0        0        0           1           0           0  
    1        0        0        0           0           0           1  
    2        0        0        0           0           0           1  
    3        0        0        0           0           0           1  
    4        0        0        0           0           0           1  
    
    [5 rows x 25 columns]



```python
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
model = LogisticRegression()

# fit the model with the training data
model.fit(train_x,train_y)
```

    /home/palaniappan/miniconda3/envs/kagglevil_/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    LogisticRegression()




```python
# coefficeints of the trained model
print('Coefficient of model :', model.coef_)

# intercept of the model
print('Intercept of model',model.intercept_)
```

    Coefficient of model : [[-0.031126    0.00155626  0.93299804  0.08451933 -1.02556693  1.2454209
      -1.25347047  1.05047825  0.97898909  0.61562408 -1.14084285 -0.78091599
      -0.28356148 -0.44782067  0.16173042  0.63398083 -0.04705201  0.2046181
      -0.45766536 -0.33677633 -0.1668852   0.07947994  0.28574002 -0.37326953]]
    Intercept of model [0.07227509]



```python
# predict the target on the train dataset
predict_train = model.predict(train_x)
print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)
```

    Target on train data [0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1
     1 0 0 0 1 0 1 1 1 1 0 0 0 1 0 0 1 0 0 0 0 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0
     0 1 1 0 0 1 0 0 0 1 1 0 0 1 1 0 1 0 0 0 0 1 0 1 1 0 0 1 0 1 0 1 0 1 1 0 1
     0 0 0 0 1 0 0 1 1 1 0 0 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 1 0 1 1 0
     0 0 1 1 0 0 0 1 0 1 0 0 1 1 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
     0 1 1 1 1 0 0 1 0 1 0 0 1 1 1 1 1 0 0 0 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 1 0 1 0 1 0 0 1 0 1 0 0 0 1 0 1 0 1 1 0 0 0 1 1 0 1 0 0 1 1 1
     0 0 1 0 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0
     0 0 1 1 0 1 1 0 1 0 1 0 0 0 0 0 1 0 1 1 1 0 0 0 1 0 1 0 0 0 1 0 0 1 0 1 1
     1 0 0 1 1 0 1 1 0 0 0 0 0 1 0 1 1 0 1 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0
     0 0 0 1 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0 1 1 1 0 1 0 1 1 0 1
     0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 1 1 1 1 0 0 0 0 1 0 0
     0 1 0 0 0 0 0 0 1 0 0 1 1 0 1 1 1 0 0 0 0 0 0 1 0 0 1 1 1 0 1 0 0 0 0 0 0
     0 0 1 0 0 0 0 0 0 1 1 0 1 1 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1
     1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 0 0 0 0 1 0 1 1 1 0 0 1 1 0 1 1 0 0 0 0 1
     0 1 0 1 1 0 0 1 0 0 1 0 1 1 0 1 0 1 0 1 0 0 0 0 1 1 0 1 0 0 1 1 0 1 1 0 0
     1 0 1 0 0 1 0 1 0 1 0 1 1 0 0 1 0 1 1 0 1 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 1
     0 0 0 0 0 0 1 1 0 0 1 1 0 0 0 1 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 1 0 0 0
     0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 1 0
     1 0 1 1 1 0 1 0 0]
    accuracy_score on train dataset :  0.8047752808988764



```python
# predict the target on the test dataset
predict_test = model.predict(test_x)
print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)
```

    Target on test data [0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 1 0 1 0 1 0 1
     0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0
     1 0 0 0 1 0 0 0 0 1 1 0 1 1 0 1 0 0 0 0 1 1 1 0 1 0 0 0 0 0 1 1 0 1 1 1 1
     0 1 0 0 0 0 1 1 1 1 0 1 1 0 1 1 0 0 1 1 0 0 1 1 1 0 1 1 0 1 0 0 0 0 0 0 0
     0 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 1 0 1 0 0 0 0 1]
    accuracy_score on test dataset :  0.8324022346368715

