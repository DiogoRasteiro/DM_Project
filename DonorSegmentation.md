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
sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.impute import KNNImputer
from pandas_profiling import ProfileReport
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.base import clone
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import math
from sklearn.mixture import GaussianMixture
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report
import umap
from sklearn.tree import DecisionTreeClassifier


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
data=pd.read_csv('data/donorsPreprocessed.csv',index_col='CONTROLN')
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

# Functions

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

def visualize_clusters(data, title='Cluster Visualization'):
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

```python
def generate_corr_matrix(df):
    # Prepare figure
    fig = plt.figure(figsize=(20, 20))
    # Obtain correlation matrix. Round the values to 2 decimal cases. Use the DataFrame corr() and round() method.
    corr = np.round(df.corr(method="spearman"), decimals=2)
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
def get_ss(df):
    """Computes the sum of squares for all variables given a dataset
    """
    ss = np.sum(df.var() * (df.count() - 1))
    return ss # return sum of sum of squares of each df variable



def r2_calculator(df, labels):
    sst = get_ss(df)
    ssw = np.sum(df.groupby(labels).apply(get_ss))
    return 1 - ssw/sst

def get_r2_scores(df, clusterer, min_k=2, max_k=10, labels=None):
    """
    Loop over different values of k. To be used with sklearn clusterers.
    """
    r2_clust = {}
    for n in range(min_k, max_k):
        if labels is None:
            clust = clone(clusterer).set_params(n_clusters=n)
            labels = clust.fit_predict(df)
        r2_clust[n] = r2_calculator(df, labels)
    
    return r2_clust
```

```python
def plot_inertia(df, clusterer, n_start, n_stop, verbose=True):
    ## K Means
    inertia=[]
    k=range(n_start, n_stop)
    for i in k:
        print(f'Running for k = {i}')
        clusters=clusterer(n_clusters=i, random_state=45).fit(df)
        try:
            inertia.append(clusters.inertia_)
        except:
            inertia.append(clusters.cost_)
        if(verbose):
            print(f'Inertia for k = {i} is {inertia[-1]}')

    plt.plot(k, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
```

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
    sst = get_ss_no_label(df)  # get total sum of squares
    
    r2 = []  # where we will store the R2 metrics for each cluster solution
    
    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, affinity=dist, linkage=link_method)
        hclabels = cluster.fit_predict(df) #get cluster labels
        df_concat = pd.concat((df, pd.Series(hclabels, name='labels')), axis=1)  # concat df with labels
        ssw_labels = df_concat.groupby(by='labels').apply(get_ss_no_label)  # compute ssw for each cluster labels
        ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
        r2.append(ssb / sst)  # save the R2 of the given cluster solution
        
    return np.array(r2)
```

```python
def get_ss_no_label(df):
    ss = np.sum(df.var() * (df.count() - 1))
    return ss  # return sum of sum of squares of each df variable
```

```python
def r2_calc_label(cluster_data, cols, label='label'):
    sst = get_ss_no_label(cluster_data[cols])  # get total sum of squares
    ssw_labels = cluster_data[cols.to_list() + [label]].groupby(
        by=label).apply(get_ss)  # compute ssw for each cluster labels
    ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
    return ssb / sst
```

```python
def generate_count_plots(df, title='Count Plots'):
    # Prepare figure. Create individual axes where each plot will be placed
    fig, axes = plt.subplots(4, int(len(df.columns) / 4), figsize=(20, 20))
    
    # Plot data# Iterate across axes objects and associate each box plot
    for ax, feat in zip(axes.flatten(), df.columns):
        # Notice the zip() function and flatten() method
        sns.countplot(x=df[feat], ax=ax)

    # Layout# Add a centered title to the figure:
    plt.suptitle(title)
    plt.show()
```

```python
def generate_box_plots(df, title='Box Plots'):
    # Prepare figure. Create individual axes where each box plot will be placed
    fig, axes = plt.subplots(2, math.ceil(len(df.columns) / 2), figsize=(20, 11))
    
    # Iterate across axes objects and associate each box plot (hint: use the ax argument):
    for ax, feat in zip(axes.flatten(), df.columns): # Notice the zip() function and flatten() method
        sns.boxplot(df[feat], ax=ax)
        
    # Add a centered title to the figure:
    plt.suptitle(title)
    plt.show()
```

```python
def generate_silhouette_plots(df, clusterer, range_clusters):
    # Adapted from:
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis

    # Storing average silhouette metric
    avg_silhouette = []
    for nclus in range_clusters:
        # Create a figure
        fig = plt.figure(figsize=(13, 7))

        curr_clusterer = clone(clusterer).set_params(n_clusters=nclus)
        cluster_labels = curr_clusterer.fit_predict(data[preferences])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(data[preferences], cluster_labels)
        avg_silhouette.append(silhouette_avg)
        print(
            f"For n_clusters = {nclus}, the average silhouette_score is : {silhouette_avg}"
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, cluster_labels)

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
                              0,
                              ith_cluster_silhouette_values,
                              facecolor=color,
                              edgecolor=color,
                              alpha=0.7)

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
        xmin, xmax = np.round(sample_silhouette_values.min() - 0.1,
                              2), np.round(
                                  sample_silhouette_values.max() + 0.1, 2)
        plt.xlim([xmin, xmax])

        # The (nclus+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        plt.ylim([0, len(data[preferences]) + (nclus + 1) * 10])

        plt.yticks([])  # Clear the yaxis labels / ticks
        plt.xticks(np.arange(xmin, xmax, 0.1))
```

```python
def generate_hc_methods_plot(df):
    # Prepare input
    hc_methods = ["ward", "complete", "average", "single"]
    # Call function defined above to obtain the R2 statistic for each hc_method
    max_nclus = 10
    r2_hc_methods = np.vstack([
        get_r2_hc(df=df.copy(), link_method=link,
                  max_nclus=max_nclus) for link in hc_methods
    ]).T
    r2_hc_methods = pd.DataFrame(r2_hc_methods,
                                 index=range(1, max_nclus + 1),
                                 columns=hc_methods)

    sns.set()
    # Plot data
    fig = plt.figure(figsize=(11, 5))
    sns.lineplot(data=r2_hc_methods, linewidth=2.5, markers=["o"] * 4)

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
def generate_dendrogram(df, hc_method):
    link = linkage(df, method=hc_method)
    dendo = dendrogram(link, color_threshold=7.1)
    plt.axhline(7.1, linestyle='--')
    plt.show()
```

```python
def generate_hit_map(sm):
    hits  = HitMapView(12, 12,"Clustering", text_size=10)
    hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")
    plt.show()
```

```python
def generate_component_planes(sm):
    sm.get_node_vectors()
    # Component planes on the SOM grid
    view2D = View2D(12,12,"", text_size=10)
    view2D.show(sm, col_sz=3, what='codebook')
    plt.subplots_adjust(top=0.90)
    plt.suptitle("Component Planes", fontsize=20)
    plt.show()
```

```python
def cluster_profiles(df, columns, label_columns, figsize, compar_titles=None):
    """
    Pass df with labels columns of one or multiple clustering labels. 
    Then specify this label columns to perform the cluster profile according to them.
    """
    if compar_titles == None:
        compar_titles = [""]*len(label_columns)

    sns.set()
    fig, axes = plt.subplots(nrows=len(label_columns), ncols=2, figsize=figsize, squeeze=False)
    for ax, cols_to_use, label, titl in zip(axes, columns, label_columns, compar_titles):
        # Filtering df
        drop_cols = [i for i in label_columns if i!=label]
        dfax = df.drop(drop_cols, axis=1)
        dfax = dfax[cols_to_use]
        dfax[label] = df[label]

        # Getting the cluster centroids and counts
        centroids = dfax.groupby(by=label, as_index=False).mean()
        counts = dfax.groupby(by=label, as_index=False).count().iloc[:,[0,1]]
        counts.columns = [label, "counts"]

        # Setting Data
        pd.plotting.parallel_coordinates(centroids, label, color=sns.color_palette(), ax=ax[0],linewidth=10)
        sns.barplot(x=label, y="counts", data=counts, ax=ax[1])
 
        #Setting Layout
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
        ax[0].annotate(s=titl, xy=(0.95,1.1), xycoords='axes fraction', fontsize=13, fontweight = 'heavy') 
        ax[0].legend(handles, cluster_labels) # Adaptable to number of clusters
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=13)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=-20)
        ax[1].set_xticklabels(cluster_labels)
        ax[1].set_xlabel("")
        ax[1].set_ylabel("Absolute Frequency")
        ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=13)

    plt.subplots_adjust(hspace=0.4, top=0.90)
    plt.suptitle("Cluster Simple Profilling", fontsize=23)
    plt.show()
```

# Preferences


## Feature Selection

```python
binary_cols = data.apply(lambda x: max(x) == 1, 0)
binary_cols = data.loc[:, binary_cols].columns
```

```python
generate_count_plots(data[preferences], 'Preference Perspective')
```

```python
preferences=data[preferences].drop(columns=['KIDSTUFF', 'BOATS', 'HOMEE']).columns
```

# KMODES

```python
plot_inertia(data[preferences], KModes, 2, 7)
```


```python
km = KModes(n_clusters=3, random_state=45, init='Huang', verbose=1)

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
r2_calc_label(data, preferences, 'Preferences_KModes')
```

```python
cluster_profiles(data, [preferences], ['Preferences_KModes'], (28,10))
```

# T-SNE

```python
# This is step can be quite time consuming
two_dim = TSNE(random_state=42).fit_transform(data[preferences])
# t-SNE visualization
pd.DataFrame(two_dim).plot.scatter(x=0, y=1, c=data['Preferences_KModes'], colormap='tab10', figsize=(15,10))
plt.show()
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

We started by doing some feature selection on the binary variables and then we made also some feature engineering. The 'PERCGOV' and 'PERCMINORITY' variables were created here in order to reduce the input space.

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
sns.distplot(data['WWIIVETS'])
```

```python
sns.distplot(data['VIETVETS'])
```

```python
fea_to_del=['HV1', 'AFC1','EC1', 'POBC1', 'MARR1', 'SEC1','is_male', 'LFC1', 'WWIIVETS',
           'HU3', 'HHD4']
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

filters_demo = (
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

demo_no_out = data[filters_demo]
```

```python
print('Percentage of data kept after removing outliers:', (np.round(demo_no_out.shape[0] / data.shape[0], 4))*100)
```

```python
numerical = demo_no_out[demography_kept].loc[:,demo_no_out[demography_kept].apply(lambda x: x.max()>1, axis=0)]

for c in numerical.columns:
    pt = PowerTransformer()
    numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))

##preprocessing categorical
categorical = demo_no_out[demography_kept].drop(columns=numerical.columns)

#Percentage of columns which are categorical is used as weight parameter in embeddings later
categorical_weight = len(categorical.columns) / demo_no_out[demography_kept].shape[1]

#Embedding numerical & categorical
fit1 = umap.UMAP(metric='l2').fit(numerical)
fit2 = umap.UMAP(metric='dice').fit(categorical)

#Augmenting the numerical embedding with categorical
intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
intersection = umap.umap_.reset_local_connectivity(intersection)
embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components,
fit1._initial_alpha, fit1._a, fit1._b,
fit1.repulsion_strength, fit1.negative_sample_rate,
200, 'random', np.random, fit1.metric,
fit1._metric_kwds, False)

plt.figure(figsize=(20, 10))
plt.scatter(*embedding.T, s=2, cmap='Spectral', alpha=1.0)
plt.show()
```

```python
numerical = demo_no_out[demography_kept].loc[:,demo_no_out[demography_kept].apply(lambda x: x.max()>1, axis=0)].columns
categorical = demo_no_out[demography_kept].drop(columns=numerical).columns
```

## Data Normalization

```python
demo_no_out_minmax = demo_no_out.copy()
```

```python
scaler = StandardScaler()
scaled_feat = scaler.fit_transform(demo_no_out_minmax[numerical])
scaled_feat
```

```python
demo_no_out_minmax[numerical] = scaled_feat
demo_no_out_minmax.head()
```

```python
demo_no_out_minmax[demography_kept]
```

## K-Prototypes

```python
categorical_columns = [8, 9, 10, 11]
```

```python
costs=[]
k=range(2, 10)
for i in k:
        kproto=KPrototypes(n_clusters=i, random_state=45).fit(demo_no_out_minmax[plot], categorical=categorical_columns)
        costs.append(kproto.cost_)
        
plt.plot(k, costs, 'bx-')
plt.xlabel('k')
plt.ylabel('Cost')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

```python
kproto = KPrototypes(n_clusters= 4, init='Cao', random_state=25, n_jobs = -1)
clusters = kproto.fit_predict(demo_no_out_minmax[demography_kept], categorical=categorical_columns)
```

```python
demo_no_out_minmax['demography_KPrototypes'] = clusters
```

```python
plot = ['is_male', 'PERCMINORITY','IC1', 'URB_LVL_S',
       'URB_LVL_R', 'URB_LVL_C', 'URB_LVL_T']
```

```python
cluster_profiles(demo_no_out_minmax, [demography_kept] , ['demography_KPrototypes'], (28, 10))
```

```python
demo_no_out_minmax['demography_KPrototypes'].value_counts()
```

# Outliers Prediction

```python
demo_out=data[~filters_demo]
```

```python
test=demo_out[demography_kept]

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(test[numerical])
scaled_feat

test[numerical] = scaled_feat
```

```python
test
```

```python
y=demo_no_out_minmax['demography_KPrototypes']
X=demo_no_out_minmax[demography_kept]
X_train, X_val, y_train, y_val=train_test_split(X,
                                                y,
                                                stratify=y, 
                                                test_size=0.25, 
                                                random_state=10)
```

```python
dt=DecisionTreeClassifier(random_state=10)

dt.fit(X_train,y_train)

y_train_pred=dt.predict(X_train)
y_pred=dt.predict(X_val)

print(classification_report(y_train,y_train_pred))
print(classification_report(y_val,y_pred))
```

```python
test['demography_KPrototypes']=dt.predict(test)
```

```python
test['demography_KPrototypes'].value_counts()
```

```python
demo_labels=pd.concat([demo_no_out_minmax,test], )
```

```python
demo_labels
```

```python
data
```

```python
data=data.join(demo_labels['demography_KPrototypes'], how='left')
```

```python
data['demography_KPrototypes']
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
    cluster_labels = kmclust.fit_predict(demo_no_out_minmax[demography_kept])
 
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(demo_no_out_minmax[demography_kept], cluster_labels)
    avg_silhouette.append(silhouette_avg)
    print(f"For n_clusters = {nclus}, the average silhouette_score is : {silhouette_avg}")
 
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(demo_no_out_minmax[demography_kept], cluster_labels)
 
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
    plt.ylim([0, len(demo_no_out_minmax[demography_kept]) + (nclus + 1) * 10])
 
    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks(np.arange(xmin, xmax, 0.1))
```

# T-SNE

```python
# This is step can be quite time consuming
two_dim = TSNE(random_state=42).fit_transform(demo_no_out_minmax[demography_kept])
```

```python
# t-SNE visualization
pd.DataFrame(two_dim).plot.scatter(x=0, y=1, c=demo_no_out_minmax['Demography_Kmeans'], colormap='tab10', figsize=(15,10))
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
value_kept=['HIT', 'CARDPROM', 'CARDPM12', 'NUMPRM12', 'RAMNTALL', 'NGIFTALL', 'MINRAMNT', 
       'MAXRAMNT', 'LASTGIFT', 'AVGGIFT', 'RFA_2F', 'NREPLIES', 'AVG_AMNT', 'LASTDATE_DAYS', 'MAXRDATE_DAYS', 'DAYS_PER_GIFT']
```

```python
data[value_kept].describe()
```

```python
value_kept = data[value_kept].drop(columns = 'LASTDATE_DAYS').columns
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

value_no_out = data[filters]
```

```python
print('Percentage of data kept after removing outliers:', (np.round(value_no_out.shape[0] / data.shape[0], 4))*100)
```

## Data Normalization

```python
df_standard = value_no_out.copy()
```

```python
scaler = StandardScaler()
scaled_feat = scaler.fit_transform(value_no_out[value_kept])
scaled_feat
```

```python
df_standard[value_kept] = scaled_feat
df_standard.head()
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

## K-means

```python
## K Means
inertia=[]
k=range(2, 8)
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
kmeans=KMeans(n_clusters=4, random_state=45,).fit(df_standard[value_kept])
```

```python
centroids=pd.DataFrame(kmeans.cluster_centers_, columns=df_standard[value_kept].columns)
```

```python
centroids=np.round(centroids, 4)
```

```python
centroids=pd.DataFrame(scaler.inverse_transform(centroids), columns = value_kept)
```

```python
df_standard['value_Kmeans'] = kmeans.labels_
```

```python
Kmeans = KMeans(random_state=45)
get_r2_scores(df_standard[value_kept], Kmeans)
```

```python
df_standard['value_Kmeans'].value_counts()
```

```python
# Adapted from:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
range_clusters=[6]
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

```python
cluster_profiles(df_standard, [value_kept], ['value_Kmeans'], (28, 10))
```

```python
r2_calc_label(df_standard,value_kept,label='value_Kmeans')
```

# Outliers Prediction

```python
value_out=data[~filters]
```

```python
test=value_out[value_kept]

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(test[value_kept])
scaled_feat

test[value_kept] = scaled_feat
```

```python
test
```

```python
y=df_standard['value_Kmeans']
X=df_standard[value_kept]
X_train, X_val, y_train, y_val=train_test_split(X,
                                                y,
                                                stratify=y, 
                                                test_size=0.25, 
                                                random_state=10)
```

```python
dt=DecisionTreeClassifier(random_state=10)

dt.fit(X_train,y_train)

y_train_pred=dt.predict(X_train)
y_pred=dt.predict(X_val)

print(classification_report(y_train,y_train_pred))
print(classification_report(y_val,y_pred))
```

```python
test['value_Kmeans']=dt.predict(test)
```

```python
test['value_Kmeans'].value_counts()
```

```python
value_labels=pd.concat([df_standard,test], )
```

```python
value_labels
```

```python
data
```

```python
data=data.join(value_labels['value_Kmeans'], how='left')
```

```python

```

## Hierarchical Clustering on Top of K-Means


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
centroids_value=np.round(centroids_value, 4)
```

```python
pd.DataFrame(scaler.inverse_transform(centroids_value), columns = value_kept)
```

```python
clusters_value
```

### Hierarchical Clustering

```python
generate_hc_methods_plot(centroids_value)
```

```python
link =linkage(centroids_value, method = 'ward')
```

```python
dendo = dendrogram(link, color_threshold=5)
plt.axhline(5, linestyle='--')
plt.show()
```

```python
Hierarchical = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
HC = Hierarchical.fit(centroids_value)
labels = pd.DataFrame(HC.labels_).reset_index()
labels.columns = ['Centroids', 'Cluster']
```

```python
count_centroids = labels.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')
```

```python
count_centroids
```

```python
KMeans_HC = clusters_value.merge(labels, how = 'inner', on = 'Centroids')
KMeans_HC = df_outliers.merge(KMeans_HC[['ID','Cluster']], how = 'inner', left_on = df_outliers.index, right_on = 'ID')
KMeans_HC.drop(columns = 'ID', inplace = True)
KMeans_HC.rename(columns = {'Cluster': 'value_K_Hierarchical'}, inplace=True)
```

```python
KMeans_HC
```

```python
centroids_KMHC = KMeans_HC.groupby('value_K_Hierarchical')[value_kept].mean()
```

```python
centroids_KMHC
```

```python
count_KHC = KMeans_HC.value_K_Hierarchical.value_counts()
count_KHC = KMeans_HC.groupby(by='value_K_Hierarchical')['value_K_Hierarchical'].count().reset_index(name='N')
```

```python
count_KHC
```

```python
r2_calc_label(KMeans_HC,value_kept,label='value_K_Hierarchical')
```

# SOM

```python
np.random.seed(42)

sm = sompy.SOMFactory().build(
    df_standard[value_kept].values, 
    mapsize=(10,10),
    initialization='random', 
    neighborhood='gaussian',
    training='batch',
    lattice='hexa',
    component_names=value_kept
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
som_clusters = pd.DataFrame(sm._data, columns=value_kept).set_index(df_standard.index)
som_labels = pd.DataFrame(sm._bmu[0], columns=['SOM_demography']).set_index(df_standard.index)
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
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
nodeclus_labels = sm.cluster(kmeans)

 

hits  = HitMapView(12, 12,"Clustering", text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=value_kept)
df_nodes['label'] = nodeclus_labels
df_nodes
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(df_standard[value_kept])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((df_standard[value_kept], np.expand_dims(bmus_map,1)), axis=1),
    index=df_standard.index, columns=np.append(value_kept,"BMU")
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
cent_k_som['label'].value_counts()
```

```python
pd.DataFrame(scaler.inverse_transform(centroids_k_som), columns = value_kept)
```

```python
sst = get_ss(cent_k_som[value_kept])  # get total sum of squares
ssw_labels = cent_k_som[value_kept.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_k_som = ssb / sst
r2_score_k_som
```

```python

```

# Hierarchical Clustering on top of SOM

```python
# Check the nodes and and respective clusters
nodes = sm.get_node_vectors()

df_nodes = pd.DataFrame(nodes, columns=value_kept)
```

```python
# Prepare input
hc_methods = ["ward", "complete", "average", "single"]
# Call function defined above to obtain the R2 statistic for each hc_method
max_nclus = 10
r2_hc_methods = np.vstack([get_r2_hc(df=df_nodes, link_method = link, max_nclus=max_nclus) for link in hc_methods]).T
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
link =linkage(nodes, method = 'average')
```

```python
dendo = dendrogram(link, color_threshold=7.1)
plt.axhline(7.1, linestyle='--')
plt.show()
```

```python
hierclust = AgglomerativeClustering(n_clusters=2, linkage='ward')
nodeclus_labels = sm.cluster(hierclust)

df_nodes['label'] = nodeclus_labels

 

hits  = HitMapView(12, 12,"Clustering",text_size=10)
hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")

 

plt.show()
```

```python
# Obtaining SOM's BMUs labels
bmus_map = sm.find_bmu(df_standard[value_kept])[0]  # get bmus for each observation in df

df_bmus = pd.DataFrame(
    np.concatenate((df_standard[value_kept], np.expand_dims(bmus_map,1)), axis=1),
    index=df_standard.index, columns=np.append(value_kept,"BMU")
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
cent_hc_som['label'].value_counts()
```

```python
sst = get_ss(cent_hc_som[value_kept])  # get total sum of squares
ssw_labels = cent_hc_som[value_kept.to_list() + ["label"]].groupby(by='label').apply(get_ss)  # compute ssw for each cluster labels
ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB
r2_score_hc = ssb / sst
r2_score_hc
```

# K Means on top of Principal Components Analysis

```python
data_pca=df_standard[value_kept].copy()
```

```python
data_pca
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
pca=PCA(n_components=5, random_state=45)
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
pca_metrics=pd.DataFrame(
    {"Eigenvalue": pca.explained_variance_,
     "Difference": np.insert(np.diff(pca.explained_variance_), 0, 0),
     "Proportion": pca.explained_variance_ratio_,
     "Cumulative": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, pca.n_components_ + 1)
)
pca_metrics['Proportion'].plot(kind='bar')
plt.xlabel('PCA')
plt.xticks(ticks=[0,1,2,3,4],labels=[0,1,2,3,4])
plt.ylabel('Variance Explained')
plt.show()
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

```python
principal_components = ['PC0', 'PC1', 'PC2','PC4']
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
data_pca['PCA_Clusters'].value_counts()
```

```python
data_pca[value_kept]
```

```python
r2_calc_label(data_pca, data_pca[principal_components].columns, label='PCA_Clusters')
```

```python
cluster_profiles(data_pca,[principal_components],['PCA_Clusters'],(28,10))
```

PC0: Quanto maior PC0, Menos doações e menos Frequencia <br>
PC1: Quanto maior PC1, Mais Valor doado total e mais valor medio por doacao <br>
PC2: QUANTO MAIOR PC2, maior o valor medio por doaçao menor o intervalo entre doacoes e doacao maxima feitas há menos tempo <br>
PC4: QUANTO MAIOR, menor o numero de promocoes recebidas, menor o intervalo entre doacoes e maxima doação feita há mais tempo

```python
Azul: - Doações, - Menos Freq, AVG Valor doado Total e - Valor por Doação, + tempo Doacao Max
Vermelho:- Doações, Avg Menos Freq, +Valor doado Total, + Valor por Doação
Verde
Amarelo
Roxo
```

```python
data_pca.groupby('PCA_Clusters')[value_kept].mean()
```

# T-SNE

```python
# This is step can be quite time consuming
two_dim = TSNE(random_state=42).fit_transform(df_standard[value_kept])
```

```python
# t-SNE visualization
pd.DataFrame(two_dim).plot.scatter(x=0, y=1, c=df_outliers2['value_Kmeans'], colormap='tab10', figsize=(15,10))
plt.show()
```

# Cluster Analysis

```python
# data.to_csv('data/data_labels.csv', index=True)
```

```python
data = pd.read_csv('data/data_labels.csv', index_col='CONTROLN')
data.head()
```

```python
final_keep = [
    'COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'HOMEE', 'PETS', 'CDPLAY',
    'STEREO', 'PCOWNERS', 'PHOTO', 'CRAFTS', 'FISHER', 'GARDENIN', 'BOATS',
    'WALKER', 'KIDSTUFF', 'CARDS', 'PLATES', 'MALEMILI', 'MALEVET', 'VIETVETS',
    'PERCGOV', 'POP901', 'PERCMINORITY', 'AGE901', 'IC1', 'URB_LVL_S',
    'URB_LVL_R', 'URB_LVL_C', 'URB_LVL_T', 'SOCIO_ECO', 'HIT', 'NUMPRM12',
    'RAMNTALL', 'AVGGIFT', 'RFA_2F', 'NREPLIES', 'MAXRDATE_DAYS',
    'DAYS_PER_GIFT'
]
value_kept = [
    'HIT', 'NUMPRM12', 'RAMNTALL', 'AVGGIFT', 'RFA_2F', 'NREPLIES',
    'MAXRDATE_DAYS', 'DAYS_PER_GIFT'
]
preferences_kept = [
    'COLLECT1', 'VETERANS', 'BIBLE', 'CATLG', 'PETS', 'CDPLAY', 'STEREO',
    'PCOWNERS', 'PHOTO', 'CRAFTS', 'FISHER', 'GARDENIN', 'WALKER', 'CARDS',
    'PLATES'
]
demography_kept = [
    'MALEMILI', 'MALEVET', 'VIETVETS', 'PERCGOV', 'POP901', 'PERCMINORITY',
    'AGE901', 'IC1', 'URB_LVL_S', 'URB_LVL_R', 'URB_LVL_C', 'URB_LVL_T',
    'SOCIO_ECO'
]
```

```python
numerical = []

for col in data[final_keep].columns:
    if max(data[col] > 1):
        numerical.append(col)
numerical
```

```python
data_minmax=data.copy()

scaler = MinMaxScaler()
scaled_feat = scaler.fit_transform(data_minmax[numerical])

data_minmax[numerical] = scaled_feat
data_minmax.head()
```

```python
data_std=data.copy()

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(data_std[numerical])

data_std[numerical] = scaled_feat
data_std.head()
```

```python
def label_cluster_preferences(label):
    if(label==0):
        return "No Preferences"
    elif(label==1):
        return "Pets & Gardening"
    elif(label==2):
        return 'Veteran Lovers'
```

```python
def label_cluster_demography(label):
    if(label==0):
        return "High Income Families"
    elif(label==1):
        return "Military and Government Families"
    elif(label==2):
        return 'Low Income and Minority Families'
    elif(label==3):
        return 'Rural Average Families'
```

```python
def label_cluster_value(label):
    if(label==0):
        return "Average Donors"
    elif(label==1):
        return "High Potential Donors"
    elif(label==2):
        return 'Active Low-spending Donors'
    elif(label==3):
        return 'High Value Donors'
```

```python
cluster_profiles(data_minmax, [preferences_kept], ['Preferences_KModes'], (20, 5))
```

```python
cluster_profiles(data_std, [demography_kept], ['demography_KPrototypes'], (20,5))
```

```python
cluster_profiles(data_std, [value_kept], ['value_Kmeans'], (20,5))
```

# Contingency Tables

```python
data.groupby(['Preferences_KModes','demography_KPrototypes'])\
.size()\
.to_frame()\
.reset_index()\
.pivot('Preferences_KModes','demography_KPrototypes' , 0)
```

## Cluster Merging First Round

```python
# Clusters with low frequency to be merged:
to_merge = [(2,1), (1,1), (2,2)]
df_centroids = data_std.groupby(['Preferences_KModes','demography_KPrototypes'])\
    [final_keep].mean()

# Computing the euclidean distance matrix between the centroids
euclidean = pairwise_distances(df_centroids[numerical])
df_dists = pd.DataFrame(
    euclidean, columns=df_centroids.index, index=df_centroids.index
)

# Merging each low frequency clustering (source) to the closest cluster (target)
source_target = {}
for clus in to_merge:
    if clus not in source_target.values():
        source_target[clus] = df_dists.loc[clus].sort_values().index[1]

source_target
```

```python
for source, target in source_target.items():
    mask = (data['Preferences_KModes']==source[0]) & (data['demography_KPrototypes']==source[1])
    data.loc[mask, 'Preferences_KModes'] = target[0]
    data.loc[mask, 'demography_KPrototypes'] = target[1]# New contigency table
    
data.groupby(['Preferences_KModes','demography_KPrototypes'])\
    .size()\
    .to_frame()\
    .reset_index()\
    .pivot('Preferences_KModes','demography_KPrototypes', 0)
```

```python
generate_dendrogram(df_centroids[numerical], 'ward')
```

```python
# Re-running the Hierarchical clustering based on the correct number of clusters
hclust = AgglomerativeClustering(
    linkage='ward', 
    affinity='euclidean', 
    n_clusters=4
)
hclust_labels = hclust.fit_predict(df_centroids[numerical])
df_centroids['hclust_labels'] = hclust_labels

df_centroids
```

```python
# Mapper between concatenated clusters and hierarchical clusters
cluster_mapper = df_centroids['hclust_labels'].to_dict()

data_ = data.copy()

# Mapping the hierarchical clusters on the centroids to the observations
data_['merged_labels'] = data_.apply(
    lambda row: cluster_mapper[
        (row['Preferences_KModes'], row['demography_KPrototypes'])
    ], axis=1
)

# Merged cluster centroids
data_.groupby('merged_labels').mean()[final_keep]
```

## Cluster Merging Second Round

```python
data_std=data_.copy()

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(data_std[numerical])

data_std[numerical] = scaled_feat
data_std.head()
```

```python
data_.groupby(['value_Kmeans','merged_labels'])\
.size()\
.to_frame()\
.reset_index()\
.pivot('value_Kmeans','merged_labels' , 0)
```

```python
# Clusters with low frequency to be merged:
to_merge = [(1,0), (2,0), (3,2), (3,0)]
df_centroids = data_std.groupby(['value_Kmeans','merged_labels'])\
    [final_keep].mean()

# Computing the euclidean distance matrix between the centroids
euclidean = pairwise_distances(df_centroids[numerical])
df_dists = pd.DataFrame(
    euclidean, columns=df_centroids.index, index=df_centroids.index
)

# Merging each low frequency clustering (source) to the closest cluster (target)
source_target = {}
for clus in to_merge:
    if clus not in source_target.values():
        source_target[clus] = df_dists.loc[clus].sort_values().index[1]

source_target
```

```python
for source, target in source_target.items():
    mask = (data_['value_Kmeans']==source[0]) & (data_['merged_labels']==source[1])
    data_.loc[mask, 'value_Kmeans'] = target[0]
    data_.loc[mask, 'merged_labels'] = target[1]# New contigency table
    
data_.groupby(['value_Kmeans','merged_labels'])\
    .size()\
    .to_frame()\
    .reset_index()\
    .pivot('value_Kmeans','merged_labels', 0)
```

```python
generate_dendrogram(df_centroids[numerical], 'ward')
```

```python
# Re-running the Hierarchical clustering based on the correct number of clusters
hclust = AgglomerativeClustering(
    linkage='ward', 
    affinity='euclidean', 
    n_clusters=4
)
hclust_labels = hclust.fit_predict(df_centroids[numerical])
df_centroids['hclust_labels'] = hclust_labels

df_centroids
```

```python
# Mapper between concatenated clusters and hierarchical clusters
cluster_mapper = df_centroids['hclust_labels'].to_dict()


# Mapping the hierarchical clusters on the centroids to the observations
data_['final_labels'] = data_.apply(
    lambda row: cluster_mapper[
        (row['value_Kmeans'], row['merged_labels'])
    ], axis=1
)

# Merged cluster centroids
data_.groupby('final_labels').mean()[final_keep]
```

```python
data['final_labels'] = data_['final_labels']
```

```python
data_minmax=data.copy()

scaler = MinMaxScaler()
scaled_feat = scaler.fit_transform(data_minmax[final_keep])

data_minmax[final_keep] = scaled_feat
data_minmax.head()
```

```python
data_std=data.copy()

scaler = StandardScaler()
scaled_feat = scaler.fit_transform(data_std[numerical])

data_std[numerical] = scaled_feat
data_std.head()
```

## Population Characteristics

```python
cluster_profiles(data, [preferences_kept], ['final_labels'], figsize=(28,10))
```

```python
cluster_profiles(data_std, [demography_kept], ['final_labels'], figsize=(28,10))
```

```python
cluster_profiles(data_std, [value_kept], ['final_labels'], figsize=(28,10))
```

```python

```
