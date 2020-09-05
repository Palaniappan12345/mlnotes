---
title: "Random-Forest-Classifier"
author: "Palaniappan S"
date: 2020-09-05
description: "-"
type: technical_note
draft: false
---

```python
# importing required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```


```python
# read the train and test dataset
train_data = pd.read_csv('train-data.csv')
test_data = pd.read_csv('test-data.csv')
```


```python
# view the top 3 rows of the dataset
print(train_data.head(3))

# shape of the dataset
print('\nShape of training data :',train_data.shape)
print('\nShape of testing data :',test_data.shape)
```

       Survived        Age     Fare  Pclass_1  Pclass_2  Pclass_3  Sex_female  \
    0         0  28.500000   7.2292         0         0         1           0   
    1         1  27.000000  10.5000         0         1         0           1   
    2         1  29.699118  16.1000         0         0         1           1   
    
       Sex_male  SibSp_0  SibSp_1  ...  Parch_0  Parch_1  Parch_2  Parch_3  \
    0         1        1        0  ...        1        0        0        0   
    1         0        1        0  ...        1        0        0        0   
    2         0        0        1  ...        1        0        0        0   
    
       Parch_4  Parch_5  Parch_6  Embarked_C  Embarked_Q  Embarked_S  
    0        0        0        0           1           0           0  
    1        0        0        0           0           0           1  
    2        0        0        0           0           0           1  
    
    [3 rows x 25 columns]
    
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
model = RandomForestClassifier()

# fit the model with the training data
model.fit(train_x,train_y)
```




    RandomForestClassifier()




```python
print('Number of Trees used : ', model.n_estimators)

```

    Number of Trees used :  100



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
     0 1 1 0 0 1 0 1 1 1 1 0 0 0 1 0 1 0 0 1 0 0 0 1 1 0 0 1 0 1 1 1 0 0 0 0 0
     0 0 0 1 1 0 0 1 0 1 0 1 1 0 0 0 1 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 1 0 1 1 0
     0 0 0 1 1 0 0 1 0 0 1 0 1 1 0 1 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1
     0 1 1 1 1 0 0 1 0 1 0 0 1 1 1 1 0 0 0 1 1 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0
     0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 1 1 1 0 1 0 0 0 1 0
     0 1 1 0 1 1 1 0 1 1 0 0 1 0 1 1 1 1 1 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 0
     0 0 1 1 0 1 1 0 1 0 1 1 1 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 1
     1 0 0 1 1 0 1 0 0 0 1 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0
     0 0 0 0 0 0 1 0 0 1 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0 1 1 1 1 0 1 1 0 1 1 1
     0 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 1 0 0 1 0 1 0 1 1 1 1 0 0 1 0 0
     0 1 0 0 0 0 0 1 1 0 0 1 1 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0
     0 0 1 1 1 0 0 1 0 1 1 0 1 0 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
     1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 1 1 1 0 0 1 1 1 0 1 0 1 0 0 1
     0 0 0 1 1 0 0 1 0 0 1 0 1 0 0 1 0 0 0 1 0 0 1 1 0 1 0 0 0 0 1 1 0 1 1 1 0
     1 0 1 0 1 1 0 1 0 1 0 0 1 0 0 1 0 1 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1
     0 0 0 1 0 1 1 1 1 0 1 1 0 0 1 0 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0 1 0 1 1 1 0
     1 0 0 0 1 0 0 1 0 0 1 0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0
     1 0 1 1 1 0 0 1 0]
    
    accuracy_score on train dataset :  0.9859550561797753



```python
# predict the target on the test dataset
predict_test = model.predict(test_x)
print('\nTarget on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)
```

    
    Target on test data [0 0 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 1 1 0 1 0 1 1 0
     1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0
     1 0 0 0 0 0 0 0 0 1 0 0 1 1 0 1 0 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1 0 1 1 0 1
     0 1 0 0 0 1 1 1 1 1 0 1 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 1 0 1 0 1 0 0 0 0 0
     0 0 0 1 1 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 1 0 1 1 0 0 0 0 0 0 0]
    
    accuracy_score on test dataset :  0.8100558659217877

