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
import matplotlib.cm as cm
from scipy import stats
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from pandas_profiling import ProfileReport
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import clone
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import math
from sklearn.mixture import GaussianMixture


from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import sompy
from scipy.spatial import distance
from kmodes.kmodes import KModes
from sompy.visualization.mapview import View2D
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.hitmap import HitMapView
from math import pi

%matplotlib inline
pd.set_option('display.max_rows', 350)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
```

Data Preprocessed Importing

```python
data=pd.read_csv('data/donorsPreprocessed.csv')
```

```python
data
data.head()
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
            'URB_LVL_U','SOCIO_ECO']

value=['RECINHSE', 'RECP3', 'HIT', 'MAJOR', 'PEPSTRFL', 'CARDPROM', 'CARDPM12', 'NUMPRM12', 'RAMNTALL', 'NGIFTALL', 'MINRAMNT', 
       'MAXRAMNT', 'LASTGIFT', 'AVGGIFT', 'RFA_2F', 'NREPLIES', 'AVG_AMNT', 'LASTDATE_DAYS', 'MAXRDATE_DAYS', 'DAYS_PER_GIFT']
```

```python
# Adapted from:
# https://matplotlib.org/3.1.1/gallery/specialty_plots/radar_chart.html

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def visualize_clusters(data):
    N = len(data.columns)
    theta = radar_factory(N, frame='polygon')

    spoke_labels = data.columns

    fig, axes = plt.subplots(figsize=(20, 20), nrows=int(np.ceil(len(data)/2)), ncols=2,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    viridis = cm.get_cmap('viridis', 19)
    colors =  viridis(range(len(data.columns)))
    # Plot the four cases from the example data on separate axes
    for ax, centroid in zip(axes.flat, range(len(data))):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for col, color in zip(data.columns, range(len(colors))):
            ax.plot(theta, data.loc[centroid], color=colors[color])
            # ax.fill(theta, data.loc[centroid],alpha=0.55, facecolor=colors[color])
        ax.set_varlabels(spoke_labels)
        ax.set_ylim(0,1)
    # add legend relative to top-left plot
    ax = axes[0, 0]
    labels = data.columns
    legend = ax.legend(labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, 'Cluster Profiles',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()

```

# Feature Selection


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
    sns.countplot(x=data[feat], ax=ax)
# Layout# Add a centered title to the figure:
title = "Preferences Vars"
plt.suptitle(title)
        
plt.show()
```

```python
preferences=data[preferences].drop(columns=['KIDSTUFF', 'BOATS', 'HOMEE']).columns
```

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
kmeans=KMeans(n_clusters=3, random_state=45,).fit(data[preferences])
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
visualize_clusters(centroids)
```

```python
data['Preferences_Kmeans'] = kmeans.labels_
```

```python
Kmeans = KMeans(random_state=45)
get_r2_scores(data[preferences], Kmeans)
```

```python
data['Preferences_Kmeans'].value_counts()
```

```python
# Adapted from:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
range_clusters=[3,6]
# Storing average silhouette metric
avg_silhouette = []
for nclus in range_clusters:
    # Skip nclus == 1
    if nclus == 1:
        continue

    # Create a figure
    fig = plt.figure(figsize=(13, 7))
 
    # Initialize the KMeans object with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    kmclust = KMeans(n_clusters=nclus,random_state=45)
    cluster_labels = kmclust.fit_predict(data[preferences])
 
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(data[preferences], cluster_labels)
    avg_silhouette.append(silhouette_avg)
    print(f"For n_clusters = {nclus}, the average silhouette_score is : {silhouette_avg}")
 
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data[preferences], cluster_labels)
 
    y_lower = 10
    for i in range(nclus):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        # Get y_upper to demarcate silhouette y range size
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Filling the silhouette
        color = cm.nipy_spectral(float(i) / nclus)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
 
        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
 
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
 
    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    # The silhouette coefficient can range from -1, 1
    xmin, xmax = np.round(sample_silhouette_values.min() -0.1, 2), np.round(sample_silhouette_values.max() + 0.1, 2)
    plt.xlim([xmin, xmax])

    # The (nclus+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(data[preferences]) + (nclus + 1) * 10])
 
    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks(np.arange(xmin, xmax, 0.1))
```

# K-Means Followed by Hierarchical Clustering


### K-Means

```python
k = 500
```

```python
k_means_preferences = KMeans(random_state=10, n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 500).fit(data[preferences])
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

```python
visualize_clusterslize_clusters(centroids_preferences)
```

### Hierarchical Clustering

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
r2_hc_methods = np.vstack([get_r2_hc(df=centroids_preferences.copy(), link_method = link, max_nclus=max_nclus) for link in hc_methods]).T
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
link =linkage(centroids_preferences, method = 'ward')
```

```python
dendo = dendrogram(link, color_threshold=7.1)
plt.axhline(7.1, linestyle='--')
plt.show()
```

```python
Hierarchical = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'average')
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

```python
visualize_clusters(centroids_KMHC)
```

# SOM

```python
np.random.seed(42)

sm = sompy.SOMFactory().build(
    data[preferences].values, 
    mapsize=(20,20),
    initialization='random', 
    neighborhood='gaussian',
    training='batch',
    lattice='hexa',
    component_names=preferences
)
sm.train(n_job=4, verbose='info', train_rough_len=100, train_finetune_len=100)
```

```python
sm.get_node_vectors()

# Component planes on the 50x50 grid
sns.set()
view2D = View2D(12,12,"", text_size=10)
view2D.show(sm, col_sz=3, what='codebook')
plt.subplots_adjust(top=0.90)
plt.suptitle("Component Planes", fontsize=20)
plt.show()
```

```python
# U-matrix of the 50x50 grid
u = sompy.umatrix.UMatrixView(12, 12, 'umatrix', show_axis=True, text_size=8, show_text=True)

UMAT = u.show(
    sm, 
    distance2=1, 
    row_normalized=False, 
    show_data=False, 
    contooor=True # Visualize isomorphic curves
)
```

```python
som_clusters = pd.DataFrame(sm._data, columns=preferences).set_index(data.index)
som_labels = pd.DataFrame(sm._bmu[0], columns=['SOM_Preferences']).set_index(data.index)
som_clusters = pd.concat([som_clusters, som_labels], axis=1)
```

```python
som_clusters
```

# K-Means Clustering on top of SOM

```python
# Perform K-Means clustering on top of the 2500 untis (sm.get_node_vectors() output)
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=20, random_state=42)
nodeclus_labels = sm.cluster(kmeans)

 

hits  = HitMapView(12, 12,"Clustering", text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=preferences)
df_nodes['label'] = nodeclus_labels
df_nodes
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(data[preferences])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((data[preferences], np.expand_dims(bmus_map,1)), axis=1),
    index=data.index, columns=np.append(preferences,"BMU")
)
df_bmus
```

```python
cent_k_som = df_bmus.merge(df_nodes['label'], 'left', left_on="BMU", right_index=True)
cent_k_som
```

```python
cent_k_som.drop(columns='BMU').groupby('label').mean()
```

```python
def get_ss(df):
    ss = np.sum(df.var() * (df.count() - 1))
    return ss  # return sum of sum of squares of each df variable

sst = get_ss(cent_k_som[preferences])  # get total sum of squares
ssw_labels = cent_k_som[preferences.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_k_som = ssb / sst
r2_score_k_som
```

# Hierarchical Clustering on top of SOM

```python
hierclust = AgglomerativeClustering(n_clusters=6, linkage='ward')
nodeclus_labels = sm.cluster(hierclust)

 

hits  = HitMapView(12, 12,"Clustering",text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=preferences)
df_nodes['label'] = nodeclus_labels
df_nodes
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(data[preferences])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((data[preferences], np.expand_dims(bmus_map,1)), axis=1),
    index=data.index, columns=np.append(preferences,"BMU")
)
df_bmus
```

```python
cent_hc_som = df_bmus.merge(df_nodes['label'], 'left', left_on="BMU", right_index=True)
cent_hc_som
```

```python
cent_hc_som.drop(columns='BMU').groupby('label').mean()
```

```python
sst = get_ss(cent_hc_som[preferences])  # get total sum of squares
ssw_labels = cent_hc_som[preferences.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_hc = ssb / sst
r2_score_hc
```

# KMODES

```python
## K Modes
costs=[]
k=range(2, 7)
for i in k:
        kM = KModes(n_clusters=i, init='Huang')
        cls = kM.fit_predict(data[preferences])
        costs.append(kM.cost_)
        
plt.plot(k, costs, 'bx-')
plt.xlabel('k')
plt.ylabel('Cost')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
km = KModes(n_clusters=5, random_state=45, init='Huang', verbose=1)

clusters = km.fit_predict(data[preferences])

# Print the cluster centroids
print(km.cluster_centroids_)
```

```python
kmodes_cent=pd.DataFrame(km.cluster_centroids_, columns=preferences)
kmodes_cent
```

```python
data['Preferences_KModes']=clusters
```

```python
data['Preferences_KModes'].value_counts()
```

```python
r2(data[preferences],data['Preferences_KModes'])
```

# K Means on top of Principal Components Analysis

```python
data_pca=data[preferences].copy()
```

```python
pca=PCA(n_components='mle', random_state=45).fit(data_pca)
```

```python
pd.DataFrame(
    {"Eigenvalue": pca.explained_variance_,
     "Difference": np.insert(np.diff(pca.explained_variance_), 0, 0),
     "Proportion": pca.explained_variance_ratio_,
     "Cumulative": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, pca.n_components_ + 1)
)
```

```python
pca=PCA(n_components=9, random_state=45)
pca_feat = pca.fit_transform(data_pca)
pca_feat_names = [f"PC{i}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_feat, index=data_pca.index, columns=pca_feat_names)  # remember index=df_pca.index
pca_df
```

```python
data_pca = pd.concat([data_pca, pca_df], axis=1)
data_pca.head()
```

```python
def _color_red_or_green(val):
    if val < -0.45:
        color = 'background-color: red'
    elif val > 0.45:
        color = 'background-color: green'
    else:
        color = ''
    return color

# Interpreting each Principal Component
loadings = data_pca.corr().loc[preferences, pca_feat_names]
loadings.style.applymap(_color_red_or_green)
```

```python
data_pca.drop(columns=['PC1','PC4', 'PC5','PC6'], inplace=True)
```

```python
principal_components=['PC0','PC2','PC3','PC7','PC8']
```

```python
inertia=[]
k=range(2, 10)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(data_pca[principal_components])
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
kmeans=KMeans(n_clusters=5, random_state=45).fit(data_pca[principal_components])
clusters_PCA=pd.DataFrame(kmeans.cluster_centers_, columns=principal_components)
```

```python
labels=kmeans.predict(data_pca[principal_components])
```

```python
data_pca['PCA_Clusters']=labels
```

```python
data_pca.groupby('PCA_Clusters').mean()
```

```python
data_pca['PCA_Clusters'].value_counts()
```

```python
r2_pca=r2(data_pca,'PCA_Clusters')
r2_pca
```

# T-SNE

```python
# This is step can be quite time consuming
two_dim = TSNE(random_state=42).fit_transform(data[preferences])
```

```python
# t-SNE visualization
pd.DataFrame(two_dim).plot.scatter(x=0, y=1, c=data['Preferences_Kmeans'], colormap='tab10', figsize=(15,10))
plt.show()
```

```python
tsne = TSNE(random_state = 5).fit_transform(KMeans_HC[preferences.append('Preferences_K_Hierarchical')])
```

```python
pd.DataFrame(tsne).plot.scatter(x = 0, y = 1, c = KMeans_HC['Preferences_K_Hierarchical'], colormap = 'tab10', fisixe = (15,10))
```

# Demographics

```python
data[demography]
```

## Feature Selection

```python
binary_cols=data[demography].apply(lambda x: max(x)==1, 0)
binary_cols=data[demography].loc[:, binary_cols].columns
```

```python
# All Numeric Variables' Box Plots in one figure
sns.set()
# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(4, int(len(binary_cols) / 4), figsize=(20, 20))
# Plot data# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), binary_cols): 
# Notice the zip() function and flatten() method
    sns.countplot(x=data[feat], ax=ax)
# Layout# Add a centered title to the figure:
title = "Preferences Vars"
plt.suptitle(title)
        
plt.show()
```

```python
data['PERCGOV']=data['LOCALGOV']+data['STATEGOV']+data['FEDGOV']
```

```python
data['PERCMINORITY']=100-data['ETH1']
```

```python
demography_kept=['is_male','MALEMILI', 'MALEVET', 'VIETVETS', 'WWIIVETS','PERCGOV',
                 'POP901','PERCMINORITY','AGE901','MARR1','HV1','HU3','HHD4','IC1','LFC1',
                 'EC1','SEC1','AFC1','POBC1','URB_LVL_S','URB_LVL_R','URB_LVL_C','URB_LVL_T','SOCIO_ECO']
```

```python
# Prepare figure
fig = plt.figure(figsize=(20, 20))
# Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
corr = np.round(data[demography_kept].corr(method="spearman"), decimals=2)
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
fea_to_del=['HV1', 'AFC1','EC1', 'SOCIO_ECO', 'POBC1', 'MARR1', 'SEC1', 'LFC1']
```

```python
demography_kept=data[demography_kept].drop(columns=fea_to_del).columns
```

```python
metric_cols=data[demography_kept].apply(lambda x: max(x)>1, 0)
metric_cols=data[demography_kept].loc[:, metric_cols].columns
```

```python
pair=sns.pairplot(data[metric_cols])
plt.show()
```

```python
len(demography_kept)
```

## Outliers

```python
metric_features = ['MALEMILI', 'MALEVET', 'VIETVETS', 'WWIIVETS', 'PERCGOV', 'POP901', 'PERCMINORITY', 'AGE901', 'HU3', 'HHD4', 'IC1']
```

```python
# All Numeric Variables' Box Plots in one figure
sns.set()

# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(2, math.ceil(len(metric_features) / 2), figsize=(20, 11))

# Plot data
# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), metric_features): # Notice the zip() function and flatten() method
    sns.boxplot(data[feat], ax=ax)
    
# Layout
# Add a centered title to the figure:
title = "Numeric Variables' Box Plots"

plt.suptitle(title)

plt.show()
```

```python
# All Numeric Variables' Histograms in one figure
sns.set() #setting the sns. A way to have some preconfigured graphs in our visualizations.

# Prepare figure. Create individual axes where each histogram will be placed
fig, axes = plt.subplots(2, math.ceil(len(metric_features) / 2), figsize=(20, 11))
#a figure with squares where we are going to add stuff into.
#We are going to add stuff into the axis and the figure is where we are going to see the information.
#we want two rows and the next number of columns according to the number of features that we have.
#The number of metric features in divided by to and is upper rounded.

# Plot data
# Iterate across axes objects and associate each histogram (hint: use the ax.hist() instead of plt.hist()):
for ax, feat in zip(axes.flatten(), metric_features): # Notice the zip() function and flatten() method
    ax.hist(data[feat], bins = 10)
    ax.set_title(feat)
    
# Layout
# Add a centered title to the figure:
title = "Numeric Variables' Histograms"

plt.suptitle(title)

plt.show()
```

```python
# This may vary from session to session, and is prone to varying interpretations.
# A simple example is provided below:

filters = (
    (data['MALEMILI']<=50)
    &
    (data['PERCGOV']<=70)
    &
    (data['POP901']<=75000)
    &
    (data['HU3']>= 25)
    &
    (data['IC1']<= 1250)
)

df_outliers = data[filters]
```

```python
print('Percentage of data kept after removing outliers:', (np.round(df_outliers.shape[0] / data.shape[0], 4))*100)
```

## Data Normalization

```python
df_standard = df_outliers.copy()
```

```python
scaler = StandardScaler()
scaled_feat = scaler.fit_transform(df_outliers[demography_kept])
scaled_feat
```

```python
df_standard[demography_kept] = scaled_feat
df_standard.head()
```

```python
df_standard[demography_kept]
```

```python
# Prepare figure
fig = plt.figure(figsize=(20, 20))
# Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
corr = np.round(df_standard[demography_kept].corr(method="spearman"), decimals=2)
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

## Principal Components Analysis

```python
data_pca=df_standard[demography_kept].copy()
```

```python
pca=PCA(n_components='mle', random_state=45).fit(data_pca)
```

```python
pca.explained_variance_ratio_
```

```python
pd.DataFrame(
    {"Eigenvalue": pca.explained_variance_,
     "Difference": np.insert(np.diff(pca.explained_variance_), 0, 0),
     "Proportion": pca.explained_variance_ratio_,
     "Cumulative": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, pca.n_components_ + 1)
)
```

```python
pca=PCA(n_components=9, random_state=45)
pca_feat = pca.fit_transform(data_pca)
pca_feat_names = [f"PC{i}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_feat, index=data_pca.index, columns=pca_feat_names)  # remember index=df_pca.index
pca_df
```

```python
data_pca = pd.concat([data_pca, pca_df], axis=1)
data_pca.head()
```

```python
def _color_red_or_green(val):
    if val < -0.45:
        color = 'background-color: red'
    elif val > 0.45:
        color = 'background-color: green'
    else:
        color = ''
    return color

# Interpreting each Principal Component
loadings = data_pca.corr().loc[demography_kept, pca_feat_names]
loadings.style.applymap(_color_red_or_green)
```

```python
demography_kept=df_standard[demography_kept].drop(columns=['URB_LVL_T']).columns
```

Since we dropped a column, we will normalize our data again


## Data Normalization

```python
df_standard = df_outliers[demography_kept].copy()
```

```python
scaler = StandardScaler()
scaled_feat = scaler.fit_transform(df_outliers[demography_kept])
scaled_feat
```

```python
df_standard[demography_kept] = scaled_feat
df_standard.head()
```

```python
df_standard[demography_kept]
```

## K-means

```python
## K Means
inertia=[]
k=range(2, 10)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(df_standard[demography_kept])
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
kmeans=KMeans(n_clusters=4, random_state=45,).fit(df_standard[demography_kept])
```

```python
centroids=pd.DataFrame(kmeans.cluster_centers_, columns=df_standard[demography_kept].columns)
```

```python
centroids=np.round(centroids, 4)
```

```python
pd.DataFrame(scaler.inverse_transform(centroids), columns = demography_kept)
```

```python
df_outliers[demography_kept].describe(include="all").T
```

```python
df_outliers['Demography_Kmeans'] = kmeans.labels_
```

```python
Kmeans = KMeans(random_state=45)
get_r2_scores(df_standard[demography_kept], Kmeans)
```

```python
df_outliers['Demography_Kmeans'].value_counts()
```

```python
# Adapted from:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
range_clusters=[3,4]
# Storing average silhouette metric
avg_silhouette = []
for nclus in range_clusters:
    # Skip nclus == 1
    if nclus == 1:
        continue

    # Create a figure
    fig = plt.figure(figsize=(13, 7))
 
    # Initialize the KMeans object with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    kmclust = KMeans(n_clusters=nclus,random_state=45)
    cluster_labels = kmclust.fit_predict(df_standard[demography_kept])
 
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(df_standard[demography_kept], cluster_labels)
    avg_silhouette.append(silhouette_avg)
    print(f"For n_clusters = {nclus}, the average silhouette_score is : {silhouette_avg}")
 
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df_standard[demography_kept], cluster_labels)
 
    y_lower = 10
    for i in range(nclus):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        # Get y_upper to demarcate silhouette y range size
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Filling the silhouette
        color = cm.nipy_spectral(float(i) / nclus)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
 
        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
 
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
 
    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    # The silhouette coefficient can range from -1, 1
    xmin, xmax = np.round(sample_silhouette_values.min() -0.1, 2), np.round(sample_silhouette_values.max() + 0.1, 2)
    plt.xlim([xmin, xmax])

    # The (nclus+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(data[demography_kept]) + (nclus + 1) * 10])
 
    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks(np.arange(xmin, xmax, 0.1))
```

### K-Means

```python
k = 500
```

```python
k_means_demography = KMeans(random_state=10, n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 500).fit(df_standard[demography_kept])
```

```python
centroids_demography = k_means_demography.cluster_centers_
centroids_demography = pd.DataFrame(centroids_demography, columns = df_standard[demography_kept].columns)
```

```python
clusters_demography = pd.DataFrame(k_means_demography.labels_, columns = ['Centroids'])
clusters_demography['ID'] = df_standard[demography_kept].index
```

```python
centroids_demography=np.round(centroids_demography, 4)
```

```python
pd.DataFrame(scaler.inverse_transform(centroids_demography), columns = demography_kept)
```

```python
clusters_demography
```

### Hierarchical Clustering

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
r2_hc_methods = np.vstack([get_r2_hc(df=centroids_demography.copy(), link_method = link, max_nclus=max_nclus) for link in hc_methods]).T
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
link =linkage(centroids_demography, method = 'ward')
```

```python
dendo = dendrogram(link, color_threshold=7.1)
plt.axhline(7.1, linestyle='--')
plt.show()
```

```python
Hierarchical = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'average')
HC = Hierarchical.fit(centroids_demography)
labels = pd.DataFrame(HC.labels_).reset_index()
labels.columns = ['Centroids', 'Cluster']
```

```python
count_centroids = labels.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')
```

```python
KMeans_HC = clusters_demography.merge(labels, how = 'inner', on = 'Centroids')
KMeans_HC = df_outliers.merge(KMeans_HC[['ID','Cluster']], how = 'inner', left_on = df_outliers.index, right_on = 'ID')
KMeans_HC.drop(columns = 'ID', inplace = True)
KMeans_HC.rename(columns = {'Cluster': 'demography_K_Hierarchical'}, inplace=True)
```

```python
KMeans_HC
```

```python
centroids_KMHC = KMeans_HC.groupby('demography_K_Hierarchical')[demography_kept].mean()
```

```python
centroids_KMHC
```

```python
count_KHC = KMeans_HC.demography_K_Hierarchical.value_counts()
count_KHC = KMeans_HC.groupby(by='demography_K_Hierarchical')['demography_K_Hierarchical'].count().reset_index(name='N')
```

```python
count_KHC
```

# SOM

```python
np.random.seed(42)

sm = sompy.SOMFactory().build(
    df_standard[demography_kept].values, 
    mapsize=(20,20),
    initialization='random', 
    neighborhood='gaussian',
    training='batch',
    lattice='hexa',
    component_names=demography_kept
)
sm.train(n_job=4, verbose='info', train_rough_len=100, train_finetune_len=100)
```

```python
sm.get_node_vectors()

# Component planes on the 50x50 grid
sns.set()
view2D = View2D(12,12,"", text_size=10)
view2D.show(sm, col_sz=3, what='codebook')
plt.subplots_adjust(top=0.90)
plt.suptitle("Component Planes", fontsize=20)
plt.show()
```

```python
# U-matrix of the 50x50 grid
u = sompy.umatrix.UMatrixView(12, 12, 'umatrix', show_axis=True, text_size=8, show_text=True)

UMAT = u.show(
    sm, 
    distance2=1, 
    row_normalized=False, 
    show_data=False, 
    contooor=True # Visualize isomorphic curves
)
```

```python
som_clusters = pd.DataFrame(sm._data, columns=demography_kept).set_index(df_outliers.index)
som_labels = pd.DataFrame(sm._bmu[0], columns=['SOM_demography']).set_index(df_outliers.index)
som_clusters = pd.concat([som_clusters, som_labels], axis=1)
```

```python
som_clusters
```

# K-Means Clustering on top of SOM

```python
## K Means
inertia=[]
k=range(2, 10)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(som_clusters)
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
# Perform K-Means clustering on top of the 2500 untis (sm.get_node_vectors() output)
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=20, random_state=42)
nodeclus_labels = sm.cluster(kmeans)

 

hits  = HitMapView(12, 12,"Clustering", text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=demography_kept)
df_nodes['label'] = nodeclus_labels
df_nodes
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(df_standard[demography_kept])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((df_standard[demography_kept], np.expand_dims(bmus_map,1)), axis=1),
    index=df_outliers.index, columns=np.append(demography_kept,"BMU")
)
df_bmus
```

```python
cent_k_som = df_bmus.merge(df_nodes['label'], 'left', left_on="BMU", right_index=True)
cent_k_som
```

```python
centroids_k_som = cent_k_som.drop(columns='BMU').groupby('label').mean()
```

```python
pd.DataFrame(scaler.inverse_transform(centroids_k_som), columns = demography_kept)
```

```python
def get_ss(df):
    ss = np.sum(df.var() * (df.count() - 1))
    return ss  # return sum of sum of squares of each df variable

sst = get_ss(cent_k_som[demography_kept])  # get total sum of squares
ssw_labels = cent_k_som[demography_kept.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_k_som = ssb / sst
r2_score_k_som
```

# Hierarchical Clustering on top of SOM

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=demography_kept)
df_nodes['label'] = nodeclus_labels
df_nodes
```

```python
# Prepare input
hc_methods = ["ward", "complete", "average", "single"]
# Call function defined above to obtain the R2 statistic for each hc_method
max_nclus = 10
r2_hc_methods = np.vstack([get_r2_hc(df=df_nodes.copy(), link_method = link, max_nclus=max_nclus) for link in hc_methods]).T
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
link =linkage(centroids_demography, method = 'ward')
```

```python
dendo = dendrogram(link, color_threshold=7.1)
plt.axhline(7.1, linestyle='--')
plt.show()
```

```python
hierclust = AgglomerativeClustering(n_clusters=4, linkage='ward')
nodeclus_labels = sm.cluster(hierclust)

 

hits  = HitMapView(12, 12,"Clustering",text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(df_standard[demography_kept])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((df_standard[demography_kept], np.expand_dims(bmus_map,1)), axis=1),
    index=df_standard.index, columns=np.append(demography_kept,"BMU")
)
df_bmus
```

```python
cent_hc_som = df_bmus.merge(df_nodes['label'], 'left', left_on="BMU", right_index=True)
cent_hc_som
```

```python
cent_hc_som.drop(columns='BMU').groupby('label').mean()
```

```python
sst = get_ss(cent_hc_som[demography_kept])  # get total sum of squares
ssw_labels = cent_hc_som[demography_kept.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_hc = ssb / sst
r2_score_hc
```

# K Means on top of Principal Components Analysis

```python
data_pca=df_standard[demography_kept].copy()
```

```python
pca=PCA(n_components='mle', random_state=45).fit(data_pca)
```

```python
pd.DataFrame(
    {"Eigenvalue": pca.explained_variance_,
     "Difference": np.insert(np.diff(pca.explained_variance_), 0, 0),
     "Proportion": pca.explained_variance_ratio_,
     "Cumulative": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, pca.n_components_ + 1)
)
```

```python
pca=PCA(n_components=9, random_state=45)
pca_feat = pca.fit_transform(data_pca)
pca_feat_names = [f"PC{i}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_feat, index=data_pca.index, columns=pca_feat_names)  # remember index=df_pca.index
pca_df
```

```python
data_pca = pd.concat([data_pca, pca_df], axis=1)
data_pca.head()
```

```python
def _color_red_or_green(val):
    if val < -0.45:
        color = 'background-color: red'
    elif val > 0.45:
        color = 'background-color: green'
    else:
        color = ''
    return color

# Interpreting each Principal Component
loadings = data_pca.corr().loc[demography_kept, pca_feat_names]
loadings.style.applymap(_color_red_or_green)
```

```python
principal_components = ['PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']
```

```python
inertia=[]
k=range(2, 10)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(data_pca[principal_components])
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
kmeans=KMeans(n_clusters=6, random_state=45).fit(data_pca[principal_components])
clusters_PCA=pd.DataFrame(kmeans.cluster_centers_, columns=principal_components)
```

```python
labels=kmeans.predict(data_pca[principal_components])
```

```python
data_pca['PCA_Clusters']=labels
```

```python
data_pca.groupby('PCA_Clusters').mean()
```

```python
data_pca['PCA_Clusters'].value_counts()
```

```python
r2_pca=r2(data_pca,'PCA_Clusters')
r2_pca
```

# Gaussian Mixture

```python
# Selecting number of components based on AIC and BIC
n_components = np.arange(1, 16)
models = [GaussianMixture(n, covariance_type='full', n_init=10, random_state=1).fit(df_standard[demography_kept])
          for n in n_components]

bic_values = [m.bic(df_standard[demography_kept]) for m in models]
aic_values = [m.aic(df_standard[demography_kept]) for m in models]
plt.plot(n_components, bic_values, label='BIC')
plt.plot(n_components, aic_values, label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.xticks(n_components)
plt.show()
```

```python
# Performing GMM clustering
gmm = GaussianMixture(n_components=5, covariance_type='full', n_init=10, init_params='kmeans', random_state=1)
gmm_labels = gmm.fit_predict(df_standard[demography_kept])
labels_proba = gmm.predict_proba(df_standard[demography_kept])
```

```python
# The estimated component weights
gmm.weights_
```

```python
# The estimated mean vectors of the Components
gmm.means_
```

```python
# The estimated covariance matrices of the Components
gmm.covariances_.shape
```

```python
# Concatenating the labels to df
df_concat = pd.concat([df_standard[demography_kept], pd.Series(gmm_labels, index=df_outliers.index, name="gmm_labels")], axis=1)
df_concat.head()
```

```python
# Computing the R^2 of the cluster solution
sst = get_ss(df_standard[demography_kept])  # get total sum of squares
ssw_labels = df_concat.groupby(by='gmm_labels').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score = ssb / sst
print("Cluster solution with R^2 of %0.4f" % r2)
```

# T-SNE

```python
# This is step can be quite time consuming
two_dim = TSNE(random_state=42).fit_transform(df_standard[demography_kept])
```

```python
# t-SNE visualization
pd.DataFrame(two_dim).plot.scatter(x=0, y=1, c=df_outliers['Demography_Kmeans'], colormap='tab10', figsize=(15,10))
plt.show()
```

# Value

```python
data[value]
```

## Feature Selection

```python
binary_cols=data[value].apply(lambda x: max(x)==1, 0)
binary_cols=data[value].loc[:, binary_cols].columns
```

```python
# All Numeric Variables' Box Plots in one figure
sns.set()
# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(4, int(len(binary_cols) / 4), figsize=(20, 20))
# Plot data# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), binary_cols): 
# Notice the zip() function and flatten() method
    sns.countplot(x=data[feat], ax=ax)
# Layout# Add a centered title to the figure:
title = "Preferences Vars"
plt.suptitle(title)
        
plt.show()
```

```python
value_kept=['HIT', 'PEPSTRFL', 'CARDPROM', 'CARDPM12', 'NUMPRM12', 'RAMNTALL', 'NGIFTALL', 'MINRAMNT', 
       'MAXRAMNT', 'LASTGIFT', 'AVGGIFT', 'RFA_2F', 'NREPLIES', 'AVG_AMNT', 'LASTDATE_DAYS', 'MAXRDATE_DAYS', 'DAYS_PER_GIFT']
```

```python
# Prepare figure
fig = plt.figure(figsize=(20, 20))
# Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
corr = np.round(data[value_kept].corr(method="spearman"), decimals=2)
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
fea_to_del=['NGIFTALL', 'CARDPROM', 'CARDPM12', 'AVG_AMNT', 'MAXRAMNT', 'LASTGIFT', 'MINRAMNT']
```

```python
value_kept=data[value_kept].drop(columns=fea_to_del).columns
```

```python
metric_cols=data[value_kept].apply(lambda x: max(x)>1, 0)
metric_cols=data[value_kept].loc[:, metric_cols].columns
```

```python
pair=sns.pairplot(data[metric_cols])
plt.show()
```

```python
len(value_kept)
```

## Outliers

```python
# All Numeric Variables' Box Plots in one figure
sns.set()

# Prepare figure. Create individual axes where each box plot will be placed
fig, axes = plt.subplots(2, math.ceil(len(metric_cols) / 2), figsize=(20, 11))

# Plot data
# Iterate across axes objects and associate each box plot (hint: use the ax argument):
for ax, feat in zip(axes.flatten(), metric_cols): # Notice the zip() function and flatten() method
    sns.boxplot(data[feat], ax=ax)
    
# Layout
# Add a centered title to the figure:
title = "Numeric Variables' Box Plots"

plt.suptitle(title)

plt.show()
```

```python
# All Numeric Variables' Histograms in one figure
sns.set() #setting the sns. A way to have some preconfigured graphs in our visualizations.

# Prepare figure. Create individual axes where each histogram will be placed
fig, axes = plt.subplots(2, math.ceil(len(metric_cols) / 2), figsize=(20, 11))
#a figure with squares where we are going to add stuff into.
#We are going to add stuff into the axis and the figure is where we are going to see the information.
#we want two rows and the next number of columns according to the number of features that we have.
#The number of metric features in divided by to and is upper rounded.

# Plot data
# Iterate across axes objects and associate each histogram (hint: use the ax.hist() instead of plt.hist()):
for ax, feat in zip(axes.flatten(), metric_cols): # Notice the zip() function and flatten() method
    ax.hist(data[feat], bins = 10)
    ax.set_title(feat)
    
# Layout
# Add a centered title to the figure:
title = "Numeric Variables' Histograms"

plt.suptitle(title)

plt.show()
```

```python
# This may vary from session to session, and is prone to varying interpretations.
# A simple example is provided below:

filters = (
    (data['HIT']<=50)
    &
    (data['NUMPRM12']<=50)
    &
    (data['MAXRDATE_DAYS']<=6000)
    &
    (data['DAYS_PER_GIFT']<= 600)
)

df_outliers = data[filters]
```

```python
print('Percentage of data kept after removing outliers:', (np.round(df_outliers.shape[0] / data.shape[0], 4))*100)
```

## Data Normalization

```python
df_standard = df_outliers.copy()
```

```python
scaler = StandardScaler()
scaled_feat = scaler.fit_transform(df_outliers[value_kept])
scaled_feat
```

```python
df_standard[value_kept] = scaled_feat
df_standard.head()
```

```python
df_standard[value_kept]
```

```python
# Prepare figure
fig = plt.figure(figsize=(20, 20))
# Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
corr = np.round(df_standard[value_kept].corr(method="spearman"), decimals=2)
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

## Principal Components Analysis

```python
data_pca=df_standard[value_kept].copy()
```

```python
pca=PCA(n_components='mle', random_state=45).fit(data_pca)
```

```python
pca.explained_variance_ratio_
```

```python
pd.DataFrame(
    {"Eigenvalue": pca.explained_variance_,
     "Difference": np.insert(np.diff(pca.explained_variance_), 0, 0),
     "Proportion": pca.explained_variance_ratio_,
     "Cumulative": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, pca.n_components_ + 1)
)
```

```python
pca=PCA(n_components=6, random_state=45)
pca_feat = pca.fit_transform(data_pca)
pca_feat_names = [f"PC{i}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_feat, index=data_pca.index, columns=pca_feat_names)  # remember index=df_pca.index
pca_df
```

```python
data_pca = pd.concat([data_pca, pca_df], axis=1)
data_pca.head()
```

```python
def _color_red_or_green(val):
    if val < -0.45:
        color = 'background-color: red'
    elif val > 0.45:
        color = 'background-color: green'
    else:
        color = ''
    return color

# Interpreting each Principal Component
loadings = data_pca.corr().loc[value_kept, pca_feat_names]
loadings.style.applymap(_color_red_or_green)
```

## Data Normalization

```python
df_standard = df_outliers[value_kept].copy()
```

```python
scaler = StandardScaler()
scaled_feat = scaler.fit_transform(df_outliers[value_kept])
scaled_feat
```

```python
df_standard[value_kept] = scaled_feat
df_standard.head()
```

```python
df_standard[value_kept]
```

## K-means

```python
## K Means
inertia=[]
k=range(2, 10)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(df_standard[value_kept])
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
kmeans=KMeans(n_clusters=3, random_state=45,).fit(df_standard[value_kept])
```

```python
centroids=pd.DataFrame(kmeans.cluster_centers_, columns=df_standard[value_kept].columns)
```

```python
centroids=np.round(centroids, 4)
```

```python
pd.DataFrame(scaler.inverse_transform(centroids), columns = value_kept)
```

```python
df_outliers[value_kept].describe(include="all").T
```

```python
df_outliers['value_Kmeans'] = kmeans.labels_
```

```python
Kmeans = KMeans(random_state=45)
get_r2_scores(df_standard[value_kept], Kmeans)
```

```python
df_outliers['value_Kmeans'].value_counts()
```

```python
# Adapted from:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
range_clusters=[3,4]
# Storing average silhouette metric
avg_silhouette = []
for nclus in range_clusters:
    # Skip nclus == 1
    if nclus == 1:
        continue

    # Create a figure
    fig = plt.figure(figsize=(13, 7))
 
    # Initialize the KMeans object with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    kmclust = KMeans(n_clusters=nclus,random_state=45)
    cluster_labels = kmclust.fit_predict(df_standard[value_kept])
 
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(df_standard[value_kept], cluster_labels)
    avg_silhouette.append(silhouette_avg)
    print(f"For n_clusters = {nclus}, the average silhouette_score is : {silhouette_avg}")
 
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df_standard[value_kept], cluster_labels)
 
    y_lower = 10
    for i in range(nclus):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        # Get y_upper to demarcate silhouette y range size
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # Filling the silhouette
        color = cm.nipy_spectral(float(i) / nclus)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
 
        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
 
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
 
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
 
    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    # The silhouette coefficient can range from -1, 1
    xmin, xmax = np.round(sample_silhouette_values.min() -0.1, 2), np.round(sample_silhouette_values.max() + 0.1, 2)
    plt.xlim([xmin, xmax])

    # The (nclus+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(data[value_kept]) + (nclus + 1) * 10])
 
    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks(np.arange(xmin, xmax, 0.1))
```

### K-Means

```python
k = 500
```

```python
k_means_value = KMeans(random_state=10, n_clusters = k, init = 'k-means++', n_init = 10, max_iter = 500).fit(df_standard[value_kept])
```

```python
centroids_value = k_means_value.cluster_centers_
centroids_value = pd.DataFrame(centroids_value, columns = df_standard[value_kept].columns)
```

```python
clusters_value = pd.DataFrame(k_means_value.labels_, columns = ['Centroids'])
clusters_value['ID'] = df_standard[value_kept].index
```

```python
centroids_demography=np.round(centroids_value, 4)
```

```python
pd.DataFrame(scaler.inverse_transform(centroids_value), columns = value_kept)
```

```python
clusters_value
```

# FICÁMOS AQUI CONTINUAR DAQUI O SANGUE DE JESUS TEM PODER


### Hierarchical Clustering

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
r2_hc_methods = np.vstack([get_r2_hc(df=centroids_demography.copy(), link_method = link, max_nclus=max_nclus) for link in hc_methods]).T
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
link =linkage(centroids_demography, method = 'ward')
```

```python
dendo = dendrogram(link, color_threshold=7.1)
plt.axhline(7.1, linestyle='--')
plt.show()
```

```python
Hierarchical = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'average')
HC = Hierarchical.fit(centroids_demography)
labels = pd.DataFrame(HC.labels_).reset_index()
labels.columns = ['Centroids', 'Cluster']
```

```python
count_centroids = labels.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')
```

```python
KMeans_HC = clusters_demography.merge(labels, how = 'inner', on = 'Centroids')
KMeans_HC = df_outliers.merge(KMeans_HC[['ID','Cluster']], how = 'inner', left_on = df_outliers.index, right_on = 'ID')
KMeans_HC.drop(columns = 'ID', inplace = True)
KMeans_HC.rename(columns = {'Cluster': 'demography_K_Hierarchical'}, inplace=True)
```

```python
KMeans_HC
```

```python
centroids_KMHC = KMeans_HC.groupby('demography_K_Hierarchical')[demography_kept].mean()
```

```python
centroids_KMHC
```

```python
count_KHC = KMeans_HC.demography_K_Hierarchical.value_counts()
count_KHC = KMeans_HC.groupby(by='demography_K_Hierarchical')['demography_K_Hierarchical'].count().reset_index(name='N')
```

```python
count_KHC
```

# SOM

```python
np.random.seed(42)

sm = sompy.SOMFactory().build(
    df_standard[demography_kept].values, 
    mapsize=(20,20),
    initialization='random', 
    neighborhood='gaussian',
    training='batch',
    lattice='hexa',
    component_names=demography_kept
)
sm.train(n_job=4, verbose='info', train_rough_len=100, train_finetune_len=100)
```

```python
sm.get_node_vectors()

# Component planes on the 50x50 grid
sns.set()
view2D = View2D(12,12,"", text_size=10)
view2D.show(sm, col_sz=3, what='codebook')
plt.subplots_adjust(top=0.90)
plt.suptitle("Component Planes", fontsize=20)
plt.show()
```

```python
# U-matrix of the 50x50 grid
u = sompy.umatrix.UMatrixView(12, 12, 'umatrix', show_axis=True, text_size=8, show_text=True)

UMAT = u.show(
    sm, 
    distance2=1, 
    row_normalized=False, 
    show_data=False, 
    contooor=True # Visualize isomorphic curves
)
```

```python
som_clusters = pd.DataFrame(sm._data, columns=demography_kept).set_index(df_outliers.index)
som_labels = pd.DataFrame(sm._bmu[0], columns=['SOM_demography']).set_index(df_outliers.index)
som_clusters = pd.concat([som_clusters, som_labels], axis=1)
```

```python
som_clusters
```

# K-Means Clustering on top of SOM

```python
## K Means
inertia=[]
k=range(2, 10)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(som_clusters)
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
# Perform K-Means clustering on top of the 2500 untis (sm.get_node_vectors() output)
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=20, random_state=42)
nodeclus_labels = sm.cluster(kmeans)

 

hits  = HitMapView(12, 12,"Clustering", text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=demography_kept)
df_nodes['label'] = nodeclus_labels
df_nodes
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(df_standard[demography_kept])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((df_standard[demography_kept], np.expand_dims(bmus_map,1)), axis=1),
    index=df_outliers.index, columns=np.append(demography_kept,"BMU")
)
df_bmus
```

```python
cent_k_som = df_bmus.merge(df_nodes['label'], 'left', left_on="BMU", right_index=True)
cent_k_som
```

```python
centroids_k_som = cent_k_som.drop(columns='BMU').groupby('label').mean()
```

```python
pd.DataFrame(scaler.inverse_transform(centroids_k_som), columns = demography_kept)
```

```python
def get_ss(df):
    ss = np.sum(df.var() * (df.count() - 1))
    return ss  # return sum of sum of squares of each df variable

sst = get_ss(cent_k_som[demography_kept])  # get total sum of squares
ssw_labels = cent_k_som[demography_kept.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_k_som = ssb / sst
r2_score_k_som
```

# Hierarchical Clustering on top of SOM

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=demography_kept)
df_nodes['label'] = nodeclus_labels
df_nodes
```

```python
# Prepare input
hc_methods = ["ward", "complete", "average", "single"]
# Call function defined above to obtain the R2 statistic for each hc_method
max_nclus = 10
r2_hc_methods = np.vstack([get_r2_hc(df=df_nodes.copy(), link_method = link, max_nclus=max_nclus) for link in hc_methods]).T
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
link =linkage(centroids_demography, method = 'ward')
```

```python
dendo = dendrogram(link, color_threshold=7.1)
plt.axhline(7.1, linestyle='--')
plt.show()
```

```python
hierclust = AgglomerativeClustering(n_clusters=4, linkage='ward')
nodeclus_labels = sm.cluster(hierclust)

 

hits  = HitMapView(12, 12,"Clustering",text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(df_standard[demography_kept])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((df_standard[demography_kept], np.expand_dims(bmus_map,1)), axis=1),
    index=df_standard.index, columns=np.append(demography_kept,"BMU")
)
df_bmus
```

```python
cent_hc_som = df_bmus.merge(df_nodes['label'], 'left', left_on="BMU", right_index=True)
cent_hc_som
```

```python
cent_hc_som.drop(columns='BMU').groupby('label').mean()
```

```python
sst = get_ss(cent_hc_som[demography_kept])  # get total sum of squares
ssw_labels = cent_hc_som[demography_kept.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_hc = ssb / sst
r2_score_hc
```

# K Means on top of Principal Components Analysis

```python
data_pca=df_standard[demography_kept].copy()
```

```python
pca=PCA(n_components='mle', random_state=45).fit(data_pca)
```

```python
pd.DataFrame(
    {"Eigenvalue": pca.explained_variance_,
     "Difference": np.insert(np.diff(pca.explained_variance_), 0, 0),
     "Proportion": pca.explained_variance_ratio_,
     "Cumulative": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, pca.n_components_ + 1)
)
```

```python
pca=PCA(n_components=9, random_state=45)
pca_feat = pca.fit_transform(data_pca)
pca_feat_names = [f"PC{i}" for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_feat, index=data_pca.index, columns=pca_feat_names)  # remember index=df_pca.index
pca_df
```

```python
data_pca = pd.concat([data_pca, pca_df], axis=1)
data_pca.head()
```

```python
def _color_red_or_green(val):
    if val < -0.45:
        color = 'background-color: red'
    elif val > 0.45:
        color = 'background-color: green'
    else:
        color = ''
    return color

# Interpreting each Principal Component
loadings = data_pca.corr().loc[demography_kept, pca_feat_names]
loadings.style.applymap(_color_red_or_green)
```

```python
principal_components = ['PC0', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']
```

```python
inertia=[]
k=range(2, 10)
for i in k:
        kmeans=KMeans(n_clusters=i, random_state=45).fit(data_pca[principal_components])
        inertia.append(kmeans.inertia_)
        
plt.plot(k, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
kmeans=KMeans(n_clusters=6, random_state=45).fit(data_pca[principal_components])
clusters_PCA=pd.DataFrame(kmeans.cluster_centers_, columns=principal_components)
```

```python
labels=kmeans.predict(data_pca[principal_components])
```

```python
data_pca['PCA_Clusters']=labels
```

```python
data_pca.groupby('PCA_Clusters').mean()
```

```python
data_pca['PCA_Clusters'].value_counts()
```

```python
r2_pca=r2(data_pca,'PCA_Clusters')
r2_pca
```

# Gaussian Mixture

```python
# Selecting number of components based on AIC and BIC
n_components = np.arange(1, 16)
models = [GaussianMixture(n, covariance_type='full', n_init=10, random_state=1).fit(df_standard[demography_kept])
          for n in n_components]

bic_values = [m.bic(df_standard[demography_kept]) for m in models]
aic_values = [m.aic(df_standard[demography_kept]) for m in models]
plt.plot(n_components, bic_values, label='BIC')
plt.plot(n_components, aic_values, label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.xticks(n_components)
plt.show()
```

```python
# Performing GMM clustering
gmm = GaussianMixture(n_components=5, covariance_type='full', n_init=10, init_params='kmeans', random_state=1)
gmm_labels = gmm.fit_predict(df_standard[demography_kept])
labels_proba = gmm.predict_proba(df_standard[demography_kept])
```

```python
# The estimated component weights
gmm.weights_
```

```python
# The estimated mean vectors of the Components
gmm.means_
```

```python
# The estimated covariance matrices of the Components
gmm.covariances_.shape
```

```python
# Concatenating the labels to df
df_concat = pd.concat([df_standard[demography_kept], pd.Series(gmm_labels, index=df_outliers.index, name="gmm_labels")], axis=1)
df_concat.head()
```

```python
# Computing the R^2 of the cluster solution
sst = get_ss(df_standard[demography_kept])  # get total sum of squares
ssw_labels = df_concat.groupby(by='gmm_labels').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score = ssb / sst
print("Cluster solution with R^2 of %0.4f" % r2)
```

# T-SNE

```python
# This is step can be quite time consuming
two_dim = TSNE(random_state=42).fit_transform(df_standard[demography_kept])
```

```python
# t-SNE visualization
pd.DataFrame(two_dim).plot.scatter(x=0, y=1, c=df_outliers['Demography_Kmeans'], colormap='tab10', figsize=(15,10))
plt.show()
```

## Population Characteristics

```python
sns.set_style(style="darkgrid")
gender=data['GENDER'].map(lambda x: 'U' if x==" " else x)
perc_gender=round(gender.value_counts()/len(data['GENDER'])*100, 2)
perc_gender.plot(kind='pie', colors=['fuchsia','royalblue','forestgreen','black'])
```
