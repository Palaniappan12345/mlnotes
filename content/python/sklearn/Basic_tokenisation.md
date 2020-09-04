---
title: "Basic-tokenisation"
author: "Palaniappan S"
date: 2020-09-04
description: "-"
type: technical_note
draft: false
---

```python
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
import sklearn
import re
```


```python
Sms_content=['What is going on','How is your life','oh! god what is happening']
df=pd.DataFrame(Sms_content,columns={'sms'})
df
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
      <th>sms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is going on</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How is your life</td>
    </tr>
    <tr>
      <th>2</th>
      <td>oh! god what is happening</td>
    </tr>
  </tbody>
</table>
</div>




```python
def tokenize(text):
    tokens=re.split('\W+',text)
    return tokens

```


```python
df['tokenized_text']=df['sms'].apply(lambda row : tokenize(row.lower()))
df.head()
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
      <th>sms</th>
      <th>tokenized_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is going on</td>
      <td>[what, is, going, on]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>How is your life</td>
      <td>[how, is, your, life]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>oh! god what is happening</td>
      <td>[oh, god, what, is, happening]</td>
    </tr>
  </tbody>
</table>
</div>


