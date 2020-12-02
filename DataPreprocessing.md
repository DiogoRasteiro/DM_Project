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

# Donor Segmentation - PVA


## Library Importing

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
from datetime import datetime
%matplotlib inline
```

## Data Importing

```python
data=pd.read_csv('data/donors.csv')
```

```python
data
```

```python
data_backup=data.copy()
```

```python
data=data_backup.copy()
```

### Initial Data Transformation


Since there are some columns that are flags(essentially binary) but are represented as categorical variables, we will detect them and change them to a binary column.

```python
# Detect all variables that are binary(have only two unique values)
flag_col=[]
for col in data.columns:
    if len(data[col].unique())<=2:
        flag_col.append(col)
# By reading the metadata, we specify 'a priori' the positive values. Then, we specify that they're 1 if they
# are one of those values and 0 otherwise
for col in flag_col:
    data[col]=data[col].apply(lambda x: 1 if x in ['X', 'Y', 'H']  else 0)
```

### Missing Value Treatment

```python
data = data.apply(lambda x: x.replace(' ',np.nan ))
data['GEOCODE2'].fillna('Other', inplace=True)
data['GEOCODE2'].replace(' ', 'Other', inplace=True)
```

```python
data=data.loc[:, data.isnull().mean() <= .1]
```

### Pandas-Profiling Report

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
   # },minimal=True
#)



#profile.to_file(os.path.join('.', "donor_data.html"))
```

## Feature Extraction

```python
# First we convert all columns involving dates to a datetime format
datetimecol=[True if 'DATE' in column else False for column in data.columns]
for col in data.loc[:, datetimecol].columns:
    try:
        data[col] = data[col].astype('datetime64[ns]')
    except Exception as e:
        pass

# Then, for every Promotion, we calculate the difference between receiving the solicitation
# and answering it
for i in range(3, 25):
    try:
        data['DIF_' + str(i)] = (data['RDATE_' + str(i)] - data['ADATE_' + str(i)])
        data['DIF_' + str(i)] = data['DIF_' + str(i)].map(lambda x: x.days)
    except Exception as e:
        pass
    
# For each promotion the donor responded to, we calculate the average amount of days to respond
difcol=[True if 'DIF_' in column else False for column in data.columns]
data['AVG_DIF']=data.loc[:,difcol].mean(axis=1)

# Afterwards we drop the columns with the differences since we do not need them for further analysis
difcol=[True if 'DIF_' in column else False for column in data.columns]
data.drop(data.loc[:,difcol].columns, inplace=True, axis=1)

# Then we convert several dates to 'Amount of Day since' format
data['ODATE'] = data['ODATEDW'].apply(lambda x: (datetime.now()-x).days)
data['LASTDATE_DAYS'] = data['LASTDATE'].apply(lambda x: (datetime.now()-x).days)
data['FISTDATE_DAYS'] = data['FISTDATE'].apply(lambda x: (datetime.now()-x).days)
data['MAXRDATE_DAYS'] = data['MAXRDATE'].apply(lambda x: (datetime.now()-x).days)
data['NEXTDATE_DIF'] = data['FISTDATE'] - data['NEXTDATE']

# Detect all columns regarding amounts given
AMNTcol=[True if 'RAMNT_' in column else False for column in data.columns]
# Make an average out of them
data['AVG_AMNT']=data.loc[:,AMNTcol].mean(axis=1)

# Detect if the user has a limit for the 'In House' campaign
data['HAS_LIMIT_SOLIH'] = data['SOLIH'].apply(lambda x: 1 if float(x)>=0 else 0)

# By dividing the time period between First and Last gifts by the total amount of gifts,
# we obtain the average period between gifts
data['DAYS_PER_GIFT']=(data['FISTDATE_DAYS']-data['LASTDATE_DAYS'])/data['NGIFTALL']
```

```python
data['DAYS_PER_GIFT']=(data['FISTDATE_DAYS']-data['LASTDATE_DAYS'])/data['NGIFTALL']
```
```python
data['DAYS_PER_GIFT']
```
# Data Cleaning and Missing Values Treatment

```python
features_to_delete = [
    'Unnamed: 0',
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
    'MSA',
    'ADI',
    'DMA',
    'MDMAUD',#ALREADY HAVE VARIABLES THAT KEEP THE INDIVIDUAL BYTES OF THIS VAR
    'RFA_2R',
    'RFA_2','RFA_3','RFA_4','RFA_5','RFA_6','RFA_7','RFA_8','RFA_9','RFA_10','RFA_11','RFA_12','RFA_13','RFA_14','RFA_15',
    'RFA_16','RFA_17','RFA_18','RFA_19','RFA_20','RFA_21','RFA_22','RFA_23','RFA_24',
    'ADATE_2','ADATE_3','ADATE_4','ADATE_5','ADATE_6','ADATE_7','ADATE_8','ADATE_9','ADATE_10','ADATE_11','ADATE_12','ADATE_13',
    'ADATE_14','ADATE_15','ADATE_16','ADATE_17','ADATE_18','ADATE_19','ADATE_20','ADATE_21','ADATE_22','ADATE_23','ADATE_24',
    'RDATE_3','RDATE_4','RDATE_5','RDATE_6','RDATE_7','RDATE_8','RDATE_9','RDATE_10','RDATE_11','RDATE_12','RDATE_13',
    'RDATE_14','RDATE_15','RDATE_16','RDATE_17','RDATE_18','RDATE_19','RDATE_20','RDATE_21','RDATE_22','RDATE_23','RDATE_24',
    'RAMNT_3' ,'RAMNT_4' ,'RAMNT_5' ,'RAMNT_6' ,'RAMNT_7' ,'RAMNT_8' ,'RAMNT_9' ,'RAMNT_10' ,'RAMNT_11' ,'RAMNT_12' ,
    'RAMNT_13' ,'RAMNT_14' ,'RAMNT_15' ,'RAMNT_16' ,'RAMNT_17' ,'RAMNT_18' ,'RAMNT_19' ,'RAMNT_20' ,'RAMNT_21' ,
    'RAMNT_22' ,'RAMNT_23' ,'RAMNT_24',
    'FISTDATE',
    'LASTDATE',
    'MINRDATE',
    'MAXRDATE',
    'NEXTDATE'
    
]

data.drop(features_to_delete,inplace=True, axis=1)
```
```python
flag_col=[]
for col in data.columns:
    if len(data[col].unique())<=2:
        flag_col.append(col)
for col in flag_col:
    data[col]=data[col].apply(lambda x: 1 if x in ['X', 'Y', 'H']  else 0)
```

```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

metric_feat = data.select_dtypes(include=numerics).columns
```

```python
non_metric_feat = data.columns.drop(metric_feat).to_list()
```

```python
data[non_metric_feat]
```

```python
data['SOLIH'].value_counts()
```

```python
data=data.loc[:, data.isnull().mean() <= .1]
```

```python
data['GEOCODE2'].value_counts()
```

```python
data['GEOCODE2'].fillna('Other', inplace=True)
data['GEOCODE2'].replace(' ', 'Other', inplace=True)
```

# Correlation Analysis

```python
cor_matrix=data.corr(method='spearman').abs()
```

```python
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
```

```python
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

```

```python
data.drop(columns=to_drop,inplace=True,axis=1)
```

## Population Characteristics

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
plt.figure(figsize=(10,5))
maxrdate_month=data['MAXRDATE'].apply(lambda x: x.month)
maxrdate_month.value_counts().plot(kind='bar', color='b')

```

```python
plt.figure(figsize=(10,5))
maxrdate_month=data['MINRDATE'].apply(lambda x: x.month)
maxrdate_month.value_counts().plot(kind='bar', color='b')
```

```python
plt.figure(figsize=(10,5))
maxrdate_month=data['FISTDATE'].apply(lambda x: x.month)
maxrdate_month.value_counts().plot(kind='bar', color='b')
```

```python
plt.figure(figsize=(10,5))
maxrdate_month=data['LASTDATE'].apply(lambda x: x.month)
maxrdate_month.value_counts().plot(kind='bar', color='b')
```

```python

```
