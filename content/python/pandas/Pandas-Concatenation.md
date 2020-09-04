---
title: "Pandas-Concatenation"
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
one = pd.DataFrame({
   'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
   'subject_id':['sub1','sub2','sub4','sub6','sub5'],
   'Marks_scored':[98,90,87,69,78]},
   index=[1,2,3,4,5])

two = pd.DataFrame({
   'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
   'subject_id':['sub2','sub4','sub3','sub6','sub5'],
   'Marks_scored':[89,80,79,97,88]},
   index=[1,2,3,4,5])
pd.concat([one,two])
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
      <th>Name</th>
      <th>subject_id</th>
      <th>Marks_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Alex</td>
      <td>sub1</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amy</td>
      <td>sub2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Allen</td>
      <td>sub4</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alice</td>
      <td>sub6</td>
      <td>69</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Billy</td>
      <td>sub2</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brian</td>
      <td>sub4</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bran</td>
      <td>sub3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bryce</td>
      <td>sub6</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Betty</td>
      <td>sub5</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([one,two],keys=['x','y'])
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
      <th></th>
      <th>Name</th>
      <th>subject_id</th>
      <th>Marks_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">x</th>
      <th>1</th>
      <td>Alex</td>
      <td>sub1</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amy</td>
      <td>sub2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Allen</td>
      <td>sub4</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alice</td>
      <td>sub6</td>
      <td>69</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>78</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">y</th>
      <th>1</th>
      <td>Billy</td>
      <td>sub2</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brian</td>
      <td>sub4</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bran</td>
      <td>sub3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bryce</td>
      <td>sub6</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Betty</td>
      <td>sub5</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([one,two],keys=['x','y'],ignore_index=True)
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
      <th>Name</th>
      <th>subject_id</th>
      <th>Marks_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alex</td>
      <td>sub1</td>
      <td>98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Amy</td>
      <td>sub2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Allen</td>
      <td>sub4</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alice</td>
      <td>sub6</td>
      <td>69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>78</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Billy</td>
      <td>sub2</td>
      <td>89</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Brian</td>
      <td>sub4</td>
      <td>80</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Bran</td>
      <td>sub3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bryce</td>
      <td>sub6</td>
      <td>97</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Betty</td>
      <td>sub5</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([one,two],axis=1)
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
      <th>Name</th>
      <th>subject_id</th>
      <th>Marks_scored</th>
      <th>Name</th>
      <th>subject_id</th>
      <th>Marks_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Alex</td>
      <td>sub1</td>
      <td>98</td>
      <td>Billy</td>
      <td>sub2</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amy</td>
      <td>sub2</td>
      <td>90</td>
      <td>Brian</td>
      <td>sub4</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Allen</td>
      <td>sub4</td>
      <td>87</td>
      <td>Bran</td>
      <td>sub3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alice</td>
      <td>sub6</td>
      <td>69</td>
      <td>Bryce</td>
      <td>sub6</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>78</td>
      <td>Betty</td>
      <td>sub5</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Concatenating Using append
one.append(two)
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
      <th>Name</th>
      <th>subject_id</th>
      <th>Marks_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Alex</td>
      <td>sub1</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amy</td>
      <td>sub2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Allen</td>
      <td>sub4</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alice</td>
      <td>sub6</td>
      <td>69</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Billy</td>
      <td>sub2</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brian</td>
      <td>sub4</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bran</td>
      <td>sub3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bryce</td>
      <td>sub6</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Betty</td>
      <td>sub5</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
one.append([two,one,two])
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
      <th>Name</th>
      <th>subject_id</th>
      <th>Marks_scored</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Alex</td>
      <td>sub1</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amy</td>
      <td>sub2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Allen</td>
      <td>sub4</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alice</td>
      <td>sub6</td>
      <td>69</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Billy</td>
      <td>sub2</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brian</td>
      <td>sub4</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bran</td>
      <td>sub3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bryce</td>
      <td>sub6</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Betty</td>
      <td>sub5</td>
      <td>88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alex</td>
      <td>sub1</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amy</td>
      <td>sub2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Allen</td>
      <td>sub4</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alice</td>
      <td>sub6</td>
      <td>69</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ayoung</td>
      <td>sub5</td>
      <td>78</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Billy</td>
      <td>sub2</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brian</td>
      <td>sub4</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bran</td>
      <td>sub3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bryce</td>
      <td>sub6</td>
      <td>97</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Betty</td>
      <td>sub5</td>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Time Series
pd.datetime.now()
```

    <ipython-input-9-e7159963b6ce>:2: FutureWarning: The pandas.datetime class is deprecated and will be removed from pandas in a future version. Import from datetime module instead.
      pd.datetime.now()





    datetime.datetime(2020, 9, 4, 20, 9, 29, 477135)




```python
pd.Timestamp('2017-03-01')
```




    Timestamp('2017-03-01 00:00:00')




```python
pd.Timestamp(1587687255,unit='s')
```




    Timestamp('2020-04-24 00:14:15')




```python
pd.date_range("11:00", "13:30", freq="30min").time
```




    array([datetime.time(11, 0), datetime.time(11, 30), datetime.time(12, 0),
           datetime.time(12, 30), datetime.time(13, 0), datetime.time(13, 30)],
          dtype=object)


