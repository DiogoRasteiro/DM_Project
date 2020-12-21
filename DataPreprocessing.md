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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from pandas_profiling import ProfileReport
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances

%matplotlib inline
pd.set_option('display.max_rows', 350)
```

## Data Importing

```python
data=pd.read_csv('data/donors.csv')
```

```python
data_backup=pd.read_csv('data/donors.csv')
```

```python
data
```

```python
data_backup = data.copy()
```

```python
data = data_backup.copy()
```

### Pandas-Profiling Report


To get an impression of the dataset in its initial form, we used the library 'pandas-profiling' to automatically generate a report.

```python
'Generates a Report using pandas-profiling for the dataset at the designated location'
def generate_profile_report(profile_location):
    profile = ProfileReport(data,
                            title='Tugas Customer Data',
                            correlations={
                                "pearson": {
                                    "calculate": True
                                },
                                "spearman": {
                                    "calculate": False
                                },
                                "kendall": {
                                    "calculate": False
                                },
                                "phi_k": {
                                    "calculate": False
                                },
                                "cramers": {
                                    "calculate": False
                                },
                            },
                            minimal=True)
    profile.to_file(os.path.join('.', profile_location))
    
# generate_profile_report('donor_data.html')
```

### Unnecessary Variables


After browsing through the provided metadata and looking at the generated report, we decided the following variables were unnecessary for our analysis, and thus removed them from the dataset. For each one, we provided an explanation as to why.

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
]

data.drop(features_to_delete,inplace=True, axis=1)
```
### Initial Data Transformation


After browsing the data, we noticed several variables that had a weird format and thus decided to perform some changes to them, in order to get a better quality dataset. This is done at this stage, before a thorough treatment of the data, in order to facilitate treatment or because certain variables would be excluded after.


Since there are some columns that are flags(essentially binary) but are represented as categorical variables, we will detect them and change them to a binary column.

```python
# Detect all variables that are binary(have only two unique values)
flag_col = []
for col in data.columns:
    if len(data[col].unique()) <= 2:
        flag_col.append(col)

# By reading the metadata, we specify 'a priori' the positive values. Then, we specify that they're 1 if they
# are one of those values and 0 otherwise
for col in flag_col:
    data[col] = data[col].apply(lambda x: 1 if x in ['X', 'Y', 'H'] else 0)
```

We noticed that the 'RDATE' series of columns had a lot of NaN values. We assumed this to be because, when a donor does not respond to a campaign, they will be registered as NaN for the 'Response Date'. This means that the 'RDATE's will most certainly be excluded when performing a Missing Values Treatment. However, although there is a lot of missing information in this group of columns, we can also consider the lack of information(donors that didn't respond) to be data itself. Therefore we counted the amount of dates that weren't NaN to obtain the amount of campaigns a donor has responded to.

```python
# This variable will count the number of times each individual replied to a promotion
# by counting the amount of 'RDATE's that aren't null
RDATEcol = [True if 'RDATE_' in column else False for column in data.columns]
data['NREPLIES'] = data.loc[:, RDATEcol].count(axis=1)
```

```python
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

# Detect all columns regarding amounts given
AMNTcol=[True if 'RAMNT_' in column else False for column in data.columns]
# Make an average out of them
data['AVG_AMNT']=data.loc[:,AMNTcol].mean(axis=1)
```

```python
lista = []
for column in data.columns:
    if 'RAMNT_' in column:
        lista.append(column)
```

```python
data.drop(columns = lista, inplace = True)
```

### Missing Values Treatment


First, through observation of the values and reading the Metadata, we detected several missing values that weren't directly identified as such. Thus, we will fix them.

```python
data = data.apply(lambda x: x.replace(' ', np.nan))
data['GEOCODE2'].fillna('Other', inplace=True)
data['GEOCODE2'].replace(' ', 'Other', inplace=True)
```

Then, we begin dealing with the missing values themselves. 

Due to the large amount of variables, it is impossible to look at each column individually. Therefore, we decided that any column with a high amount of missing values(over 10% of the observations) should be dropped, because such features don't provide enough information and trying to fill such a large amount of missing values would worsen the data quality too much.

```python
# Through data.isnull().mean(), we calculate the % of missing values for each column
# We then alter the dataframe, keeping only those with less than 10% missing values.
data=data.loc[:, data.isnull().mean() <= .1]
```

Finally, with the largest of missing values out of the way, we can deal with the remaining on a case-by-case basis.

```python
data.isna().sum().sort_values(ascending=False).to_frame()
```

For 'FISTDATE', since there are only 2 rows with missing values, we decided to exclude them since they represent a very small loss in quantity.

```python
# Removing the 2 rows with missing values in fistdate
data = data[~(data.FISTDATE.isna())]
```

For the 'ADATE' group, many of them were eliminated. Furthermore, we concluded that the only information we could extract from them was in conjunction with the 'RDATE' group, of which none were kept(although we already extracted information from this group into the 'NREPLIES' variable). Therefore, we exclude them.

```python
# Detect all the columns with ADATE in their name and drop them
ADATEcol=[True if 'ADATE' in column else False for column in data.columns]
data.drop(data.loc[:,ADATEcol].columns, inplace=True, axis=1)
```

For 'GENDER', we will treat the missing values. First, we detected some errors in this column. Namely, 4 observations with values not detailed in the Metadata. Due to this, we changed their value to 'U', in other words, donors whose gender we don't know.

```python
data['GENDER'] = data['GENDER'].apply(lambda x: 'U' if x in ['A', 'C', np.nan] else x)
data['GENDER'].value_counts()
```

For 'DOMAIN', we will also impute the missing values.

Since this variable is actually a combination of two characteristics, we decided to split it in two and treat each part separately.

```python
# 'DOMAIN' is composed of two bytes, so we simply slice each byte into its own column, while respecting NaNs 
data['URB_LVL']=data['DOMAIN'].apply(lambda x: str(x)[0] if x is not np.nan else np.nan)
data['SOCIO_ECO']=data['DOMAIN'].apply(lambda x: str(x)[1] if x is not np.nan else np.nan)
```

```python
data[['DOMAIN', 'URB_LVL', 'SOCIO_ECO']]
```

```python
data.drop(columns='DOMAIN', inplace=True)
```

## Feature Extraction


Here, we will take the features we've kept and try to extract as much information as we can from them, creating new variables.

```python
# First we convert all columns involving dates to a datetime format
datetimecol=[True if 'DATE' in column else False for column in data.columns]
for col in data.loc[:, datetimecol].columns:
    try:
        data[col] = data[col].astype('datetime64[ns]')
    except Exception as e:
        pass

# Then we convert several dates to 'Amount of Day since' format
data['ODATE'] = data['ODATEDW'].apply(lambda x: (datetime.now()-x).days)
data['LASTDATE_DAYS'] = data['LASTDATE'].apply(lambda x: (datetime.now()-x).days)
data['FISTDATE_DAYS'] = data['FISTDATE'].apply(lambda x: (datetime.now()-x).days)
data['MAXRDATE_DAYS'] = data['MAXRDATE'].apply(lambda x: (datetime.now()-x).days)

# By dividing the time period between First and Last gifts by the total amount of gifts,
# we obtain the average period between gifts
data['DAYS_PER_GIFT']=(data['FISTDATE_DAYS']-data['LASTDATE_DAYS'])/data['NGIFTALL']
```

# Data Cleaning and Missing Values Treatment

```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

metric_feat = data.select_dtypes(include=np.number).columns

non_metric_feat = data.columns.drop(metric_feat).to_list()
```

```python
data[non_metric_feat]
```

```python
data = data.drop(columns=[
    'ODATEDW', 'MINRDATE', 'MAXRDATE', 'LASTDATE', 'FISTDATE', 'ZIP',
    'GEOCODE2', 'STATE'
])
```

```python
data = pd.concat([
    data,
    pd.get_dummies(data['MDMAUD_R'], prefix='MDMAUD_Recency').iloc[:, :-1]
],
                 axis=1)
data = pd.concat([
    data,
    pd.get_dummies(data['MDMAUD_F'], prefix='MDMAUD_Frequency').iloc[:, :-1]
],
                 axis=1)
data = pd.concat([
    data,
    pd.get_dummies(data['MDMAUD_A'], prefix='MDMAUD_Amount').iloc[:, :-1]
],
                 axis=1)
data = pd.concat([
    data,
    pd.get_dummies(data['RFA_2A'], prefix='RFA_2_Amount', drop_first=True)
],
                 axis=1)
```

```python

```

```python
data['is_male'] = data['GENDER'].map(lambda x: 1 if x == 'M' else 0)
```

```python
data['URB_LVL_S'] = data['URB_LVL'].astype('str').apply(lambda x: 1 if x == 'S' else (np.nan if x=='nan' else 0))
data['URB_LVL_R'] = data['URB_LVL'].astype('str').apply(lambda x: 1 if x == 'R' else (np.nan if x=='nan' else 0))
data['URB_LVL_C'] = data['URB_LVL'].astype('str').apply(lambda x: 1 if x == 'C' else (np.nan if x=='nan' else 0))
data['URB_LVL_T'] = data['URB_LVL'].astype('str').apply(lambda x: 1 if x == 'T' else (np.nan if x=='nan' else 0))
data['URB_LVL_U'] = data['URB_LVL'].astype('str').apply(lambda x: 1 if x == 'U' else (np.nan if x=='nan' else 0))
```

```python
data['SOCIO_ECO_1'] = data['SOCIO_ECO'].astype('str').apply(lambda x: 1 if x == '1' else(np.nan if x=='nan' else 0))
data['SOCIO_ECO_2'] = data['SOCIO_ECO'].astype('str').apply(lambda x: 1 if x == '2' else(np.nan if x=='nan' else 0))
data['SOCIO_ECO_3'] = data['SOCIO_ECO'].astype('str').apply(lambda x: 1 if x == '3' else(np.nan if x=='nan' else 0))
data['SOCIO_ECO_4'] = data['SOCIO_ECO'].astype('str').apply(lambda x: 1 if x == '4' else(np.nan if x=='nan' else 0))
```

```python
data.iloc[:,-5:].isna().sum()
```

```python
data.drop([
    'GENDER', 'RFA_2A', 'MDMAUD_R', 'MDMAUD_F',
    'MDMAUD_A', 'GENDER', 'URB_LVL', 'SOCIO_ECO'
],
          axis=1,
          inplace=True)
```

```python
data.isna().sum()
```

```python
imputer = KNNImputer(n_neighbors=1)
KNN=imputer.fit_transform(data)
KNN
####
####
#Há valores diferentes de 0 e 1
```

```python
data1=pd.DataFrame(KNN, index=data.index, columns=data.columns)
```

```python
data1['URB_LVL_S'].value_counts()
```

```python
data['URB_LVL_S'].value_counts()
```

```python
data1_backup=data1.copy()
```

```python
data_backup=data.copy()
```

```python
data=data1.copy()
```

# Coherence Check

```python
data['URB_COH']=data1['URB_LVL_S']+data1['URB_LVL_R']+data1['URB_LVL_C']+data1['URB_LVL_T']+data1['URB_LVL_U']
data['URB_COH'].value_counts()
```

```python
data['SOCIO_COH']=data1['SOCIO_ECO_1']+data1['SOCIO_ECO_2']+data1['SOCIO_ECO_3']+data1['SOCIO_ECO_4']
data['SOCIO_COH'].value_counts()
```

```python
data.drop(columns=['SOCIO_COH','URB_COH'], inplace=True)
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

### Data Partition- Cluster Perspectives

```python
preferences=['is_male', 'COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 
             'PETS', 'CDPLAY', 'STEREO', 'PCOWNERS', 'PHOTO', 'CRAFTS', 
             'FISHER', 'GARDENIN', 'BOATS', 'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES']

demography=['is_male', 'MALEMILI', 'MALEVET', 'VIETVETS', 'WWIIVETS', 'LOCALGOV', 
            'STATEGOV', 'FEDGOV', 'POP901', 'POP90C1', 'POP90C2', 'POP90C3', 'POP90C4', 
            'ETH1', 'ETH2', 'ETH3', 'ETH4', 'ETH5', 'ETH6', 'ETH7', 'ETH8', 'ETH9', 'ETH10', 
            'ETH11', 'ETH12', 'ETH13', 'ETH14', 'ETH15', 'ETH16', 'AGE901', 'AGE907', 'CHIL1', 
            'CHIL2', 'CHIL3', 'AGEC1', 'AGEC2', 'AGEC3', 'AGEC4', 'AGEC5', 'AGEC7', 'CHILC1', 'CHILC2', 
            'CHILC3', 'CHILC4', 'CHILC5', 'HHAGE2', 'HHN1', 'HHN2', 'HHN3', 'HHN6', 'MARR1', 'MARR2', 'MARR4', 
            'DW1', 'DW3', 'DW4', 'DW7', 'DW8', 'DW9', 'HV1', 'HV3', 'HU1', 'HU3', 'HU5', 'HHD4', 'HHD7', 'HHD8', 
            'HHD10', 'HHD12', 'ETHC1', 'ETHC2', 'ETHC4', 'ETHC6', 'HVP1', 'HVP6', 'HUR1', 'HUR2', 'RHP4', 'HUPA1', 
            'HUPA3', 'HUPA4', 'IC1', 'IC6', 'IC7', 'IC8', 'IC9', 'IC10', 'IC11', 'IC12', 'IC13', 'IC14', 'IC15', 'IC16', 
            'IC17', 'IC18', 'HHAS2', 'HHAS3', 'HHAS4', 'MC1', 'MC3', 'TPE1', 'TPE2', 'TPE3', 'TPE4', 'TPE5', 'TPE6', 'TPE7', 
            'TPE8', 'TPE9', 'PEC1', 'PEC2', 'TPE10', 'TPE11', 'TPE12', 'TPE13', 'LFC1', 'LFC6', 'LFC7', 'LFC8', 'LFC9', 'LFC10', 
            'OCC1', 'OCC2', 'OCC3', 'OCC4', 'OCC5', 'OCC6', 'OCC7', 'OCC8', 'OCC9', 'OCC10', 'OCC11', 'OCC12', 'OCC13', 'EIC1', 'EIC2',
            'EIC3', 'EIC4', 'EIC5', 'EIC6', 'EIC7', 'EIC8', 'EIC9', 'EIC10', 'EIC11', 'EIC12', 'EIC13', 'EIC14', 'EIC15', 'EIC16',
            'OEDC1', 'OEDC2', 'OEDC3', 'OEDC4', 'OEDC5', 'OEDC6', 'OEDC7', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6', 'EC7',
            'EC8', 'SEC1', 'SEC2', 'SEC3', 'SEC4', 'SEC5', 'AFC1', 'AFC2', 'AFC3', 'AFC4', 'AFC6', 'VC1', 'VC2', 'VC3',
            'VC4', 'ANC1', 'ANC2', 'ANC3', 'ANC4', 'ANC5', 'ANC6', 'ANC7', 'ANC8', 'ANC9', 'ANC10', 'ANC11', 'ANC12',
            'ANC13', 'ANC14', 'ANC15', 'POBC1', 'POBC2', 'LSC1', 'LSC2', 'LSC3', 'LSC4', 'VOC1', 'VOC2', 'VOC3',
            'HC1', 'HC2', 'HC3', 'HC4', 'HC6', 'HC7', 'HC9', 'HC10', 'HC11', 'HC12', 'HC13', 'HC14', 'HC15',
            'HC16', 'HC17', 'HC19', 'HC20','HC21', 'MHUC1', 'MHUC2', 'AC1', 'AC2', 'URB_LVL_S','URB_LVL_R','URB_LVL_C','URB_LVL_T',
            'URB_LVL_U','SOCIO_ECO_1','SOCIO_ECO_2','SOCIO_ECO_3','SOCIO_ECO_4']

value=['RECINHSE', 'RECP3', 'GENDER', 'HIT', 'MAJOR', 'PEPSTRFL', 'CARDPROM', 'CARDPM12', 
       'NUMPRM12', 'RAMNTALL', 'NGIFTALL', 'MINRAMNT', 'MAXRAMNT', 'LASTGIFT', 'AVGGIFT',
       'RFA_2F', 'RFA_2_Amount', 'MDMAUD_Recency', 'MDMAUD_Frequency', 'MDMAUD_Amount', 'NREPLIES', 'AVG_AMNT', 'LASTDATE_DAYS',
       'MAXRDATE_DAYS', 'DAYS_PER_GIFT']
```

# Outlier Analysis


Preferences

```python
binary_cols=data.apply(lambda x: max(x)==1, 0)
binary_cols=data.loc[:, binary_cols].columns
```

```python
# All Numeric Variables' Box Plots in one figure
sns.set()
# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(4, int(len(preferences) / 4), figsize=(20, 20))
# Plot data# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), preferences): 
# Notice the zip() function and flatten() method
    sns.histplot(x=data[feat], ax=ax)
# Layout# Add a centered title to the figure:
title = "Numeric Variables' Box Plots"
plt.suptitle(title)
        
plt.show()
```

Demography

```python
# All Numeric Variables' Box Plots in one figure
sns.set()
# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(20, int(len(demography) / 10), figsize=(20, 20))
# Plot data# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), preferences): 
# Notice the zip() function and flatten() method
    sns.histplot(x=data[feat], ax=ax)
# Layout# Add a centered title to the figure:
title = "Numeric Variables' Box Plots"
plt.suptitle(title)
        
plt.show()
```

## Feature Selection - Continuation

```python
# Prepare figure
fig = plt.figure(figsize=(20, 20))
# Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
corr = np.round(data[preferences].corr(method="spearman"), decimals=2)
# Build annotation matrix (values above |0.5| will appear annotated in the plot)
mask_annot = np.absolute(corr.values) >= 0.5
annot = np.where(mask_annot, corr.values, np.full(corr.shape,"")) # Try to understand what this np.where() does
# Plot heatmap of the correlation matrix
sns.heatmap(data=corr, annot=annot, cmap=sns.diverging_palette(220, 10, as_cmap=True), 
            fmt='s', vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
# Layout
fig.subplots_adjust(top=0.95)
fig.suptitle("Correlation Matrix", fontsize=20)
plt.show()
```

```python
# preferences=['COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 
#              'PETS', 'CDPLAY', 'STEREO', 'PCOWNERS', 'PHOTO', 'CRAFTS', 
#              'FISHER', 'GARDENIN', 'BOATS', 'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES']
preferences=['VETERANS', 'BIBLE', 
             'PETS', 'CDPLAY', 'STEREO', 'PCOWNERS', 
             'GARDENIN', 'WALKER']
```

```python
## K Means
inertia=[]
k=range(2, 20)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(data[preferences])
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
kmeans=KMeans(n_clusters=7, random_state=45).fit(data[preferences])
```

```python
centroids=pd.DataFrame(kmeans.cluster_centers_, columns=data[preferences].columns)
```

```python
centroids=np.round(centroids, 4)
```

```python
centroids
```

```python
data['Preferences']=kmeans.labels_
```

```python
data['Preferences'].value_counts()
```

```python
distance_inter_cluster = euclidean_distances(centroids)
average_inter_cluters_distance = (sum(sum(distance_inter_cluster)) / 2) / 7
print('Average distance inter_cluster:', average_inter_cluters_distance)

clusters_preferences = pd.DataFrame(kmeans.labels_, columns=['Centroids'])
clusters_preferences['ID'] = data[preferences].index
```

```python
clusters_preferences
```

## Population Characteristics

```python
sns.set_style(style="darkgrid")
gender=data['GENDER'].map(lambda x: 'U' if x==" " else x)
perc_gender=round(gender.value_counts()/len(data['GENDER'])*100, 2)
perc_gender.plot(kind='pie', colors=['fuchsia','royalblue','forestgreen','black'])
```
