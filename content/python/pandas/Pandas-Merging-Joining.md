---
title: "Pandas-Merging/Joining"
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
left = pd.DataFrame({
   'id':[1,2,3,4,5],
   'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
   'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame(
   {'id':[1,2,3,4,5],
   'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
   'subject_id':['sub2','sub4','sub3','sub6','sub5']})
left
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
      <th>id</th>
      <th>Name</th>
      <th>subject_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Alex</td>
      <td>sub1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Amy</td>
      <td>sub2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Allen</td>
      <td>sub4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>sub6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>sub5</td>
    </tr>
  </tbody>
</table>
</div>




```python
right
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
      <th>id</th>
      <th>Name</th>
      <th>subject_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Billy</td>
      <td>sub2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Brian</td>
      <td>sub4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Bran</td>
      <td>sub3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Bryce</td>
      <td>sub6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Betty</td>
      <td>sub5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merge Two DataFrames on a Key
pd.merge(left,right,on='id')
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
      <th>id</th>
      <th>Name_x</th>
      <th>subject_id_x</th>
      <th>Name_y</th>
      <th>subject_id_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Alex</td>
      <td>sub1</td>
      <td>Billy</td>
      <td>sub2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Amy</td>
      <td>sub2</td>
      <td>Brian</td>
      <td>sub4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Allen</td>
      <td>sub4</td>
      <td>Bran</td>
      <td>sub3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>sub6</td>
      <td>Bryce</td>
      <td>sub6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>Betty</td>
      <td>sub5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merge Two DataFrames on Multiple Keys
pd.merge(left,right,on=['id','subject_id'])
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
      <th>id</th>
      <th>Name_x</th>
      <th>subject_id</th>
      <th>Name_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>Alice</td>
      <td>sub6</td>
      <td>Bryce</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>Betty</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Merge Using 'how' Argument
#Left Join
pd.merge(left, right, on='subject_id', how='left')
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
      <th>id_x</th>
      <th>Name_x</th>
      <th>subject_id</th>
      <th>id_y</th>
      <th>Name_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Alex</td>
      <td>sub1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Amy</td>
      <td>sub2</td>
      <td>1.0</td>
      <td>Billy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Allen</td>
      <td>sub4</td>
      <td>2.0</td>
      <td>Brian</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alice</td>
      <td>sub6</td>
      <td>4.0</td>
      <td>Bryce</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>5.0</td>
      <td>Betty</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Right Join
pd.merge(left, right, on='subject_id', how='right')
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
      <th>id_x</th>
      <th>Name_x</th>
      <th>subject_id</th>
      <th>id_y</th>
      <th>Name_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>Amy</td>
      <td>sub2</td>
      <td>1</td>
      <td>Billy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>Allen</td>
      <td>sub4</td>
      <td>2</td>
      <td>Brian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub3</td>
      <td>3</td>
      <td>Bran</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>Alice</td>
      <td>sub6</td>
      <td>4</td>
      <td>Bryce</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>5</td>
      <td>Betty</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Outer Join
pd.merge(left, right, how='outer', on='subject_id')
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
      <th>id_x</th>
      <th>Name_x</th>
      <th>subject_id</th>
      <th>id_y</th>
      <th>Name_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>Alex</td>
      <td>sub1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>Amy</td>
      <td>sub2</td>
      <td>1.0</td>
      <td>Billy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>Allen</td>
      <td>sub4</td>
      <td>2.0</td>
      <td>Brian</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>Alice</td>
      <td>sub6</td>
      <td>4.0</td>
      <td>Bryce</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>5.0</td>
      <td>Betty</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>sub3</td>
      <td>3.0</td>
      <td>Bran</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Inner Join
pd.merge(left, right, on='subject_id', how='inner')
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
      <th>id_x</th>
      <th>Name_x</th>
      <th>subject_id</th>
      <th>id_y</th>
      <th>Name_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>Amy</td>
      <td>sub2</td>
      <td>1</td>
      <td>Billy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>Allen</td>
      <td>sub4</td>
      <td>2</td>
      <td>Brian</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>Alice</td>
      <td>sub6</td>
      <td>4</td>
      <td>Bryce</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>5</td>
      <td>Betty</td>
    </tr>
  </tbody>
</table>
</div>


