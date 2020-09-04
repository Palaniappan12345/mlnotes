---
title: "Pandas-Categorial-Data"
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
#Object Creation
s = pd.Series(["a","b","c","a"], dtype="category")
s
```




    0    a
    1    b
    2    c
    3    a
    dtype: category
    Categories (3, object): ['a', 'b', 'c']




```python
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
cat
```




    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']




```python

cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
df = pd.DataFrame({"cat":cat, "s":["a", "c", "c", np.nan]})

df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cat</th>
      <th>s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>c</td>
      <td>c</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["cat"].describe()
```




    count     3
    unique    2
    top       c
    freq      2
    Name: cat, dtype: object




```python
#Get the Properties of the Category
s = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
s.categories
```




    Index(['b', 'a', 'c'], dtype='object')




```python
cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
cat.ordered
```




    False




```python
#Renaming Categories
s = pd.Series(["a","b","c","a"], dtype="category")
s.cat.categories = ["Group %s" % g for g in s.cat.categories]
s.cat.categories
```




    Index(['Group a', 'Group b', 'Group c'], dtype='object')




```python
#Appending New Categories
s = pd.Series(["a","b","c","a"], dtype="category")
s = s.cat.add_categories([4])
s.cat.categories
```




    Index(['a', 'b', 'c', 4], dtype='object')




```python
#Comparison of Categorical Data
cat = pd.Series([1,2,3]).astype("category", categories=[1,2,3], ordered=True)
cat1 = pd.Series([2,2,2]).astype("category", categories=[1,2,3], ordered=True)

cat>cat1
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-22-13d397a1aa38> in <module>
          1 #Comparison of Categorical Data
    ----> 2 cat = pd.Series([1,2,3]).astype("category", categories=[1,2,3], ordered=True)
          3 cat1 = pd.Series([2,2,2]).astype("category", categories=[1,2,3], ordered=True)
          4 
          5 cat>cat1


    TypeError: astype() got an unexpected keyword argument 'categories'

