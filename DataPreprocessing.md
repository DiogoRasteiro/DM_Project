---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from pandas_profiling import ProfileReport
```

```python
data=pd.read_csv('data/donors.csv')
```

```python
data_backup=data.copy()
```

```python
data=data_backup.copy()
```

```python
sample=data.sample(frac=0.01)
```

```python
profile = ProfileReport(
    data, 
    title='Tugas Customer Data',
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": False},
        "kendall": {"calculate": False},
        "phi_k": {"calculate": False},
        "cramers": {"calculate": False},
    },minimal=True
)



profile.to_file(os.path.join('.', "donor_data.html"))
```

```python
features_to_delete = [
    'OSOURCE', # Does not contain information pertaining to the Donor's characteristics
    'TCODE', # Title does not contain information
    'MAILCODE', # Does not help characterize a donor
    'PVASTATE', # Does not add information because we can calculate it using STATE, and most values are empty
    'NOEXCH', # Could not analyze, and does not contain pertinent information
    'RECPGVG', # Not pertinent
    'RECSWEEP', # Not pertinent
    'CHILD03', 'CHILD07', 'CHILD12', 'CHILD18', # Values are mostly empty, and don't give much info
    'GEOCODE', # Not pertinent
    'HPHONE_D', 
    'CONTROLN'
    
]
data.drop(features_to_delete, axis=1)
```

```python
print('Jupytext')
```
