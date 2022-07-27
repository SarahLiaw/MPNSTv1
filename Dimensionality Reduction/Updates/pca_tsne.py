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


time_start = time.time()
tsne = TSNE(n_components=3, verbose=0, perplexity=30, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_scores)
tsne_df = pd.DataFrame({'X' : tsne_pca_results [:,0],
                       'Y' : tsne_pca_results [:,1],
                        'classification': target
                       })

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

sns.scatterplot(
    x="X", y="Y",
    hue='classification',
    data=tsne_df,
    legend="full"
)
plt.show()