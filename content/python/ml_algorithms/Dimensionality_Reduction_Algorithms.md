---
title: "Dimensionality-Reduction-Algorithms"
author: "Palaniappan S"
date: 2020-09-07
description: "-"
type: technical_note
draft: false
---

```python
# importing required libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
```


```python
# read the train and test dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# view the top 3 rows of the dataset
print(train_data.head(3))
```

       Item_Weight  Item_Visibility  Item_MRP  Outlet_Establishment_Year  \
    0     6.800000         0.037490   48.6034                       2004   
    1    15.600000         0.172597  114.8518                       1997   
    2    12.911575         0.054721  107.8254                       1985   
    
       Item_Outlet_Sales  Item_Fat_Content_LF  Item_Fat_Content_Low Fat  \
    0           291.6204                    0                         1   
    1          2163.1842                    0                         1   
    2          2387.5588                    0                         1   
    
       Item_Fat_Content_Regular  Item_Fat_Content_low fat  Item_Fat_Content_reg  \
    0                         0                         0                     0   
    1                         0                         0                     0   
    2                         0                         0                     0   
    
       ...  Outlet_Size_High  Outlet_Size_Medium  Outlet_Size_Small  \
    0  ...                 0                   0                  1   
    1  ...                 0                   0                  1   
    2  ...                 0                   1                  0   
    
       Outlet_Location_Type_Tier 1  Outlet_Location_Type_Tier 2  \
    0                            0                            1   
    1                            1                            0   
    2                            0                            0   
    
       Outlet_Location_Type_Tier 3  Outlet_Type_Grocery Store  \
    0                            0                          0   
    1                            0                          0   
    2                            1                          0   
    
       Outlet_Type_Supermarket Type1  Outlet_Type_Supermarket Type2  \
    0                              1                              0   
    1                              1                              0   
    2                              0                              0   
    
       Outlet_Type_Supermarket Type3  
    0                              0  
    1                              0  
    2                              1  
    
    [3 rows x 36 columns]



```python
# shape of the dataset
print('\nShape of training data :',train_data.shape)
print('\nShape of testing data :',test_data.shape)
```

    
    Shape of training data : (1364, 36)
    
    Shape of testing data : (341, 36)



```python
# target variable - Item_Outlet_Sales
train_x = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y = train_data['Item_Outlet_Sales']

# seperate the independent and target variable on testing data
test_x = test_data.drop(columns=['Item_Outlet_Sales'],axis=1)
test_y = test_data['Item_Outlet_Sales']
```


```python

print('\nTraining model with {} dimensions.'.format(train_x.shape[1]))
```

    
    Training model with 35 dimensions.



```python
# create object of model
model = LinearRegression()

# fit the model with the training data
model.fit(train_x,train_y)
```




    LinearRegression()




```python
# predict the target on the train dataset
predict_train = model.predict(train_x)

# Accuray Score on train dataset
rmse_train = mean_squared_error(train_y,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)
```

    
    RMSE on train dataset :  1135.8159344155245



```python
# predict the target on the test dataset
predict_test = model.predict(test_x)

# Accuracy Score on test dataset
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
print('\nRMSE on test dataset : ', rmse_test)
```

    
    RMSE on test dataset :  1009.2517232209669



```python
model_pca = PCA(n_components=12)

new_train = model_pca.fit_transform(train_x)
new_test  = model_pca.fit_transform(test_x)

print('\nTraining model with {} dimensions.'.format(new_train.shape[1]))
```

    
    Training model with 12 dimensions.



```python
# create object of model
model_new = LinearRegression()

# fit the model with the training data
model_new.fit(new_train,train_y)
```




    LinearRegression()




```python
# predict the target on the new train dataset
predict_train_pca = model_new.predict(new_train)

# Accuray Score on train dataset
rmse_train_pca = mean_squared_error(train_y,predict_train_pca)**(0.5)
print('\nRMSE on new train dataset : ', rmse_train_pca)
```

    
    RMSE on new train dataset :  1159.9625320934565



```python
# predict the target on the new test dataset
predict_test_pca = model_new.predict(new_test)

# Accuracy Score on test dataset
rmse_test_pca = mean_squared_error(test_y,predict_test_pca)**(0.5)
print('\nRMSE on new test dataset : ', rmse_test_pca)
```

    
    RMSE on new test dataset :  1014.4129003671715

