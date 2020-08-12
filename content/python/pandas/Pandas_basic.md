---
title: "Pandas"
author: "Palaniappan S"
date: 2020-08-11
description: "-"
type: technical_note
draft: false
---

```python
import pandas as pd
```


```python
df = pd.read_csv('glass.csv')
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
      <th>RI</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.52101</td>
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
      <th>1</th>
      <td>1.51761</td>
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
      <th>2</th>
      <td>1.51618</td>
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
      <th>3</th>
      <td>1.51766</td>
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
      <th>4</th>
      <td>1.51742</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>209</th>
      <td>1.51623</td>
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
      <th>210</th>
      <td>1.51685</td>
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
      <th>211</th>
      <td>1.52065</td>
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
      <th>212</th>
      <td>1.51651</td>
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
      <th>213</th>
      <td>1.51711</td>
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
<p>214 rows × 10 columns</p>
</div>




```python
df.head(5)
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
      <th>RI</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.52101</td>
      <td>13.64</td>
      <td>4.49</td>
      <td>1.10</td>
      <td>71.78</td>
      <td>0.06</td>
      <td>8.75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.51761</td>
      <td>13.89</td>
      <td>3.60</td>
      <td>1.36</td>
      <td>72.73</td>
      <td>0.48</td>
      <td>7.83</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.51618</td>
      <td>13.53</td>
      <td>3.55</td>
      <td>1.54</td>
      <td>72.99</td>
      <td>0.39</td>
      <td>7.78</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.51766</td>
      <td>13.21</td>
      <td>3.69</td>
      <td>1.29</td>
      <td>72.61</td>
      <td>0.57</td>
      <td>8.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.51742</td>
      <td>13.27</td>
      <td>3.62</td>
      <td>1.24</td>
      <td>73.08</td>
      <td>0.55</td>
      <td>8.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(5)
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
      <th>RI</th>
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
  </thead>
  <tbody>
    <tr>
      <th>209</th>
      <td>1.51623</td>
      <td>14.14</td>
      <td>0.0</td>
      <td>2.88</td>
      <td>72.61</td>
      <td>0.08</td>
      <td>9.18</td>
      <td>1.06</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>210</th>
      <td>1.51685</td>
      <td>14.92</td>
      <td>0.0</td>
      <td>1.99</td>
      <td>73.06</td>
      <td>0.00</td>
      <td>8.40</td>
      <td>1.59</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>211</th>
      <td>1.52065</td>
      <td>14.36</td>
      <td>0.0</td>
      <td>2.02</td>
      <td>73.42</td>
      <td>0.00</td>
      <td>8.44</td>
      <td>1.64</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>212</th>
      <td>1.51651</td>
      <td>14.38</td>
      <td>0.0</td>
      <td>1.94</td>
      <td>73.61</td>
      <td>0.00</td>
      <td>8.48</td>
      <td>1.57</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>213</th>
      <td>1.51711</td>
      <td>14.23</td>
      <td>0.0</td>
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
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 214 entries, 0 to 213
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   RI      214 non-null    float64
     1   Na      214 non-null    float64
     2   Mg      214 non-null    float64
     3   Al      214 non-null    float64
     4   Si      214 non-null    float64
     5   K       214 non-null    float64
     6   Ca      214 non-null    float64
     7   Ba      214 non-null    float64
     8   Fe      214 non-null    float64
     9   Type    214 non-null    int64  
    dtypes: float64(9), int64(1)
    memory usage: 16.8 KB



```python
df.shape
```




    (214, 10)




```python
df.columns
```




    Index(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'], dtype='object')




```python
df.isnull()
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
      <th>RI</th>
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
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>209</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>210</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>211</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>212</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>213</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 10 columns</p>
</div>




```python
df.isnull().sum()
```




    RI      0
    Na      0
    Mg      0
    Al      0
    Si      0
    K       0
    Ca      0
    Ba      0
    Fe      0
    Type    0
    dtype: int64




```python

```
