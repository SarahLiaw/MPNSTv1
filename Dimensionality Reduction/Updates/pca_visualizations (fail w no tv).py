#https://www.reneshbedre.com/blog/principal-component-analysis.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_concat_no_outliers.csv'
data = pd.read_csv(data_path)
data = data.iloc[:, 1:-1]
print(data.head(2))
x = data.iloc[:, 1:]


from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

pca_out = PCA().fit(x)
print(pca_out.explained_variance_ratio_)
print(np.cumsum(pca_out.explained_variance_ratio_))
loadings = pca_out.components_
num_pc = pca_out.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = x.columns
loadings_df = loadings_df.set_index('variable')
print(loadings_df)

import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()


# get scree plot (for scree or elbow test)
from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca_out.explained_variance_ratio_])
cluster.pcaplot(x=loadings[0], y=loadings[1], labels=x.columns,
    var1=round(pca_out.explained_variance_ratio_[0]*100, 2),
    var2=round(pca_out.explained_variance_ratio_[1]*100, 2))

cluster.pcaplot(x=loadings[0], y=loadings[1], z=loadings[2],  labels=x.columns,
    var1=round(pca_out.explained_variance_ratio_[0]*100, 2), var2=round(pca_out.explained_variance_ratio_[1]*100, 2),
    var3=round(pca_out.explained_variance_ratio_[2]*100, 2))

