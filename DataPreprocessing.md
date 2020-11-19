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
#profile = ProfileReport(
 #   data, 
  #  title='Tugas Customer Data',
    #correlations={
     #   "pearson": {"calculate": True},
      #  "spearman": {"calculate": False},
       # "kendall": {"calculate": False},
        #"phi_k": {"calculate": False},
        #"cramers": {"calculate": False},
   # },minimal=True
#)



#profile.to_file(os.path.join('.', "donor_data.html"))
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
    'CONTROLN',
    'MDMAUD', #ALREADY HAVE VARIABLES THAT KEEP THE INDIVIDUAL BYTES OF THIS VAR
    'RFA_2','RFA_3','RFA_4','RFA_5','RFA_6','RFA_7','RFA_8','RFA_9','RFA_10','RFA_11','RFA_12','RFA_13','RFA_14','RFA_15',
    'RFA_16','RFA_17','RFA_18','RFA_19','RFA_20','RFA_21','RFA_22','RFA_23','RFA_24',
    'ADATE_2','ADATE_3','ADATE_4','ADATE_5','ADATE_6','ADATE_7','ADATE_8','ADATE_9','ADATE_10','ADATE_11','ADATE_12','ADATE_13',
    'ADATE_14','ADATE_15','ADATE_16','ADATE_17','ADATE_18','ADATE_19','ADATE_20','ADATE_21','ADATE_22','ADATE_23','ADATE_24',]
data.drop(features_to_delete,inplace=True, axis=1)
```
```python
data['RFA_2R'].value_counts()
```

```python
data['MDMAUD_R'].value_counts()
```

```python
cor_matrix=data.corr(method='spearman').abs()
```

```python
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
```

```python
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
```

```python
data.drop(columns=to_drop, inplace=True ,axis=1)
```

```python
data.columns.to_list()
```

```python
sns.set_theme(style="darkgrid")
gender=data['GENDER'].map(lambda x: 'U' if x==" " else x)
perc_gender=round(gender.value_counts()/len(data['GENDER'])*100, 2)
perc_gender.plot(kind='pie', colors=['fuchsia','royalblue','forestgreen','black'])
```

```python
sns.histplot(data['NUMCHLD'])

```

```python
data.describe()
```

```python
data=data.loc[:, data.isnull().mean() <= .1]
```

```python
data.columns.to_list()
```

```python
datetimecol=[True if 'DATE' in column else False for column in data.columns]
```

```python
datetimecol
```

```python
data.loc[datetimecol].astype('datetime64[ns]')
```

```python

```
