import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.analys import get_data
from bioinfokit.visuz import cluster


data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_concat_no_outliers.csv'
data = pd.read_csv(data_path)
X = data.iloc[:, 2:]
target = data['Diagnosis'].to_numpy()
print(X.head(2))
print(target)
X_st = StandardScaler().fit_transform(X)
pca_out = PCA(26).fit(X_st)

loadings = pca_out.components_
print(loadings)
# get eigenvalues (variance explained by each pc)
print(pca_out.explained_variance_)

print(X.columns.values)
print(loadings.shape)
pca_scores = PCA(30).fit_transform(X_st)
print(pca_scores.shape)
# cluster.biplot(cscore=pca_scores, loadings=loadings, var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
#     var2=round(pca_out.explained_variance_ratio_[1]*100, 2), colorlist=target)
#cluster.biplot(cscore=pca_scores, loadings=loadings, labels=X.columns.values, var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
  #  var2=round(pca_out.explained_variance_ratio_[1]*100, 2), colorlist=target)
import umap.umap_ as umap

embedding = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='correlation').fit_transform(pca_scores)

label_colour_dict = {'plexiform': 'orange', 'healthy': 'green', 'mpnst': 'red'}

#color vector creation

cvec = [label_colour_dict[label] for label in target]

plt.figure(figsize=(20, 20))
plt.scatter(embedding[:,0], embedding[:,1], c=cvec,
            edgecolor=['none'],
            alpha=0.50)
plt.show()