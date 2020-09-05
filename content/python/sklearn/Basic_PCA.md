---
title: "Basic-PCA"
author: "Palaniappan S"
date: 2020-09-04
description: "-"
type: technical_note
draft: false
---

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
```


```python
iris = load_iris()
```


```python
X, y = iris.data, iris.target
```


```python
pca = PCA(n_components=2)
```


```python
# Maybe some original features where good, too?
selection = SelectKBest(k=1)
# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)
svm = SVC(kernel="linear")
```


```python

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])
```


```python
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1, score=0.867, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=1, score=0.900, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=1, score=0.867, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=10, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=10, score=0.900, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=10, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=1, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=10, score=0.967, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=10, score=0.967, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=10, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=10, score=0.933, total=   0.0s
    [CV] features__pca__n_components=1, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=1, features__univ_select__k=2, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1, score=0.867, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=10 


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.1s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   6 out of   6 | elapsed:    0.1s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   7 out of   7 | elapsed:    0.1s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   8 out of   8 | elapsed:    0.1s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   9 out of   9 | elapsed:    0.1s remaining:    0.0s


    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=10, score=0.967, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=10, score=0.967, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=10, score=0.900, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=10, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=1, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=10, score=0.967, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=10, score=0.900, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=10, score=0.933, total=   0.0s
    [CV] features__pca__n_components=2, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=2, features__univ_select__k=2, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=10, score=0.933, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=10, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=1, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=1, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1, score=0.933, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=0.1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=1, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=1 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=1, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=10, score=1.000, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=10, score=0.900, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=10, score=0.967, total=   0.0s
    [CV] features__pca__n_components=3, features__univ_select__k=2, svm__C=10 
    [CV]  features__pca__n_components=3, features__univ_select__k=2, svm__C=10, score=1.000, total=   0.0s
    Pipeline(steps=[('features',
                     FeatureUnion(transformer_list=[('pca', PCA(n_components=3)),
                                                    ('univ_select',
                                                     SelectKBest(k=1))])),
                    ('svm', SVC(C=10, kernel='linear'))])


    [Parallel(n_jobs=1)]: Done  90 out of  90 | elapsed:    0.5s finished



```python

```
