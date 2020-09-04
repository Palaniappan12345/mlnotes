---
title: "Data-importing-and-exploring"
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
df.head(n=2)
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
  </tbody>
</table>
</div>




```python
df.tail()
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
df.tail(n=2)
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
      <th>212</th>
      <td>1.51651</td>
      <td>14.38</td>
      <td>0.0</td>
      <td>1.94</td>
      <td>73.61</td>
      <td>0.0</td>
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
      <td>0.0</td>
      <td>8.62</td>
      <td>1.67</td>
      <td>0.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    RI      float64
    Na      float64
    Mg      float64
    Al      float64
    Si      float64
    K       float64
    Ca      float64
    Ba      float64
    Fe      float64
    Type      int64
    dtype: object


