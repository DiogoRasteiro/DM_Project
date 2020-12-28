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
data=
```

### Data Partition- Cluster Perspectives

```python
preferences=['COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 
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
    sns.distplot(a=data[feat], ax=ax)
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
    sns.distplot(a=data[feat], ax=ax)
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

## Clustering - Preferences

```python
def get_ss(df):
    """Computes the sum of squares for all variables given a dataset
    """
    ss = np.sum(df.var() * (df.count() - 1))
    return ss # return sum of sum of squares of each df variable



def r2(df, labels):
    sst = get_ss(df)
    ssw = np.sum(df.groupby(labels).apply(get_ss))
    return 1 - ssw/sst

def get_r2_scores(df, clusterer, min_k=2, max_k=10):
    """
    Loop over different values of k. To be used with sklearn clusterers.
    """
    r2_clust = {}
    for n in range(min_k, max_k):
        clust = clone(clusterer).set_params(n_clusters=n)
        labels = clust.fit_predict(df)
        r2_clust[n] = r2(df, labels)
    
    return r2_clust
```

### K-means

```python
preferences=['COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 
             'PETS', 'CDPLAY', 'STEREO', 'PCOWNERS', 'PHOTO', 'CRAFTS', 
             'FISHER', 'GARDENIN', 'BOATS', 'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES']
```

```python
## K Means
inertia=[]
k=range(2, 10)
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
kmeans=KMeans(n_clusters=6, random_state=45).fit(data[preferences])
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
data['Preferences_Kmeans'] = kmeans.labels_
```

```python
Kmeans = KMeans()
get_r2_scores(data[preferences], Kmeans)
```

# K-Means Followed by Hierarchical Clustering


## K-Means

```python
k = 500
```

```python
k_means_preferences = KMeans(n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 500).fit(data[preferences])
```

```python
centroids_preferences = k_means_preferences.cluster_centers_
centroids_preferences = pd.DataFrame(centroids_preferences, columns = data[preferences].columns)
```

```python
clusters_preferences = pd.DataFrame(k_means_preferences.labels_, columns = ['Centroids'])
clusters_preferences['ID'] = data[preferences].index
```

```python
centroids_preferences=np.round(centroids_preferences, 4)
centroids_preferences
```

```python
clusters_preferences
```

## Hierarchical Clustering on Top of K-Means

```python
def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt. 
    
    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".
    
    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable
    
    sst = get_ss(df)  # get total sum of squares
    
    r2 = []  # where we will store the R2 metrics for each cluster solution
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, affinity=dist, linkage=link_method)
        hclabels = cluster.fit_predict(df) #get cluster labels
        df_concat = pd.concat((df, pd.Series(hclabels, name='labels')), axis=1)  # concat df with labels
        ssw_labels = df_concat.groupby(by='labels').apply(get_ss)  # compute ssw for each cluster labels
        ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
        r2.append(ssb / sst)  # save the R2 of the given cluster solution
        
    return np.array(r2)
```

```python
# Prepare input
hc_methods = ["ward", "complete", "average", "single"]
# Call function defined above to obtain the R2 statistic for each hc_method
max_nclus = 10
r2_hc_methods = np.vstack([get_r2_hc(df=centroids_preferences, link_method = link, max_nclus=max_nclus) for link in hc_methods]).T
r2_hc_methods = pd.DataFrame(r2_hc_methods, index=range(1, max_nclus + 1), columns=hc_methods)

sns.set()
# Plot data
fig = plt.figure(figsize=(11,5))
sns.lineplot(data=r2_hc_methods, linewidth=2.5, markers=["o"]*4)

# Finalize the plot
fig.suptitle("R2 plot for various hierarchical methods", fontsize=21)
plt.gca().invert_xaxis()  # invert x axis
plt.legend(title="HC methods", title_fontsize=11)
plt.xticks(range(1, max_nclus + 1))
plt.xlabel("Number of clusters", fontsize=13)
plt.ylabel("R2 metric", fontsize=13)

plt.show()
```

```python
linkage = linkage(centroids_preferences, method = 'ward')
```

```python
dendo = dendrogram(linkage)
```

```python
Hierarchical = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
HC = Hierarchical.fit(centroids_preferences)
labels = pd.DataFrame(HC.labels_).reset_index()
labels.columns = ['Centroids', 'Cluster']
```

```python
count_centroids = labels.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')
```

```python
KMeans_HC = clusters_preferences.merge(labels, how = 'inner', on = 'Centroids')
KMeans_HC = data.merge(KMeans_HC[['ID','Cluster']], how = 'inner', left_on = data.index, right_on = 'ID')
KMeans_HC.drop(columns = 'ID', inplace = True)
KMeans_HC.rename(columns = {'Cluster': 'Preferences_K_Hierarchical'}, inplace=True)
```

```python
KMeans_HC
```

```python
centroids_KMHC = KMeans_HC.groupby('Preferences_K_Hierarchical')[preferences].mean()
```

```python
count_KHC = KMeans_HC.Preferences_K_Hierarchical.value_counts()
count_KHC = KMeans_HC.groupby(by='Preferences_K_Hierarchical')['Preferences_K_Hierarchical'].count().reset_index(name='N')
```

```python
count_KHC
```

# T-SNE

```python
variavel_engracosa = TSNE(random_state = 5).fit_transform(KMeans_HC[preferences.append('Preferences_K_Hierarchical')])
```

```python
pd.DataFrame(variavel_engracosa).plot.scatter(x = 0, y = 1, c = KMeans_HC['Preferences_K_Hierarchical'], colormap = 'tab10', fisixe = (15,10))
```

# SOM


# Hierarchical Clustering on top of SOM


# K-Means Clustering on top of SOM


# KMODES


# DBSCAN


# Gaussian Mixture


# Principal Components Analysis

```python
# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['A','B','C','D'],
    'var1': [38, 1.5, 30, 4],
    'var2': [29, 10, 9, 34],
    'var3': [8, 39, 23, 24],
    'var4': [7, 31, 33, 14],
    'var5': [28, 15, 32, 14]
    })

# number of variable
categories=list(df)[1:]
N = len(categories)

# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
values

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
plt.ylim(0,40)

# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')

# Fill area
ax.fill(angles, values, 'b', alpha=0.1)
```

## Population Characteristics

```python
sns.set_style(style="darkgrid")
gender=data['GENDER'].map(lambda x: 'U' if x==" " else x)
perc_gender=round(gender.value_counts()/len(data['GENDER'])*100, 2)
perc_gender.plot(kind='pie', colors=['fuchsia','royalblue','forestgreen','black'])
```
