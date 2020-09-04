---
title: "Indexing-DataFrames"
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
dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }
```


```python
df = pd.DataFrame(dict)
```


```python
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
      <th>country</th>
      <th>capital</th>
      <th>area</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>8.516</td>
      <td>200.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Russia</td>
      <td>Moscow</td>
      <td>17.100</td>
      <td>143.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>India</td>
      <td>New Dehli</td>
      <td>3.286</td>
      <td>1252.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>China</td>
      <td>Beijing</td>
      <td>9.597</td>
      <td>1357.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>South Africa</td>
      <td>Pretoria</td>
      <td>1.221</td>
      <td>52.98</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.read_csv('glass.csv', index_col = 0)
```


```python
data
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
      <th>Na</th>
      <th>Mg</th>
      <th>Al</th>
      <th>Si</th>
      <th>K</th>
      <th>Ca</th>
      <th>Ba</th>
      <th>Fe</th>
      <th>Type</th>
    </tr>
    <tr>
      <th>RI</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.52101</th>
      <td>13.64</td>
      <td>4.49</td>
      <td>1.10</td>
      <td>71.78</td>
      <td>0.06</td>
      <td>8.75</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1.51761</th>
      <td>13.89</td>
      <td>3.60</td>
      <td>1.36</td>
      <td>72.73</td>
      <td>0.48</td>
      <td>7.83</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1.51618</th>
      <td>13.53</td>
      <td>3.55</td>
      <td>1.54</td>
      <td>72.99</td>
      <td>0.39</td>
      <td>7.78</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1.51766</th>
      <td>13.21</td>
      <td>3.69</td>
      <td>1.29</td>
      <td>72.61</td>
      <td>0.57</td>
      <td>8.22</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1.51742</th>
      <td>13.27</td>
      <td>3.62</td>
      <td>1.24</td>
      <td>73.08</td>
      <td>0.55</td>
      <td>8.07</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1.51623</th>
      <td>14.14</td>
      <td>0.00</td>
      <td>2.88</td>
      <td>72.61</td>
      <td>0.08</td>
      <td>9.18</td>
      <td>1.06</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1.51685</th>
      <td>14.92</td>
      <td>0.00</td>
      <td>1.99</td>
      <td>73.06</td>
      <td>0.00</td>
      <td>8.40</td>
      <td>1.59</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1.52065</th>
      <td>14.36</td>
      <td>0.00</td>
      <td>2.02</td>
      <td>73.42</td>
      <td>0.00</td>
      <td>8.44</td>
      <td>1.64</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1.51651</th>
      <td>14.38</td>
      <td>0.00</td>
      <td>1.94</td>
      <td>73.61</td>
      <td>0.00</td>
      <td>8.48</td>
      <td>1.57</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1.51711</th>
      <td>14.23</td>
      <td>0.00</td>
      <td>2.08</td>
      <td>73.36</td>
      <td>0.00</td>
      <td>8.62</td>
      <td>1.67</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 9 columns</p>
</div>




```python
data['K']
```




    RI
    1.52101    0.06
    1.51761    0.48
    1.51618    0.39
    1.51766    0.57
    1.51742    0.55
               ... 
    1.51623    0.08
    1.51685    0.00
    1.52065    0.00
    1.51651    0.00
    1.51711    0.00
    Name: K, Length: 214, dtype: float64




```python
data[['Si']]
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
      <th>Si</th>
    </tr>
    <tr>
      <th>RI</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.52101</th>
      <td>71.78</td>
    </tr>
    <tr>
      <th>1.51761</th>
      <td>72.73</td>
    </tr>
    <tr>
      <th>1.51618</th>
      <td>72.99</td>
    </tr>
    <tr>
      <th>1.51766</th>
      <td>72.61</td>
    </tr>
    <tr>
      <th>1.51742</th>
      <td>73.08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1.51623</th>
      <td>72.61</td>
    </tr>
    <tr>
      <th>1.51685</th>
      <td>73.06</td>
    </tr>
    <tr>
      <th>1.52065</th>
      <td>73.42</td>
    </tr>
    <tr>
      <th>1.51651</th>
      <td>73.61</td>
    </tr>
    <tr>
      <th>1.51711</th>
      <td>73.36</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 1 columns</p>
</div>




```python
data[['Na','Mg']]
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
      <th>Na</th>
      <th>Mg</th>
    </tr>
    <tr>
      <th>RI</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.52101</th>
      <td>13.64</td>
      <td>4.49</td>
    </tr>
    <tr>
      <th>1.51761</th>
      <td>13.89</td>
      <td>3.60</td>
    </tr>
    <tr>
      <th>1.51618</th>
      <td>13.53</td>
      <td>3.55</td>
    </tr>
    <tr>
      <th>1.51766</th>
      <td>13.21</td>
      <td>3.69</td>
    </tr>
    <tr>
      <th>1.51742</th>
      <td>13.27</td>
      <td>3.62</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1.51623</th>
      <td>14.14</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1.51685</th>
      <td>14.92</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1.52065</th>
      <td>14.36</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1.51651</th>
      <td>14.38</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1.51711</th>
      <td>14.23</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 2 columns</p>
</div>




```python
data.iloc[2]
```




    Na      13.53
    Mg       3.55
    Al       1.54
    Si      72.99
    K        0.39
    Ca       7.78
    Ba       0.00
    Fe       0.00
    Type     1.00
    Name: 1.5161799999999999, dtype: float64


