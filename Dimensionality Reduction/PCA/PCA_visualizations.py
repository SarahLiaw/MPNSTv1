# variance explained graph.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv'
data = pd.read_csv(data_path)
data = data.iloc[:, 1:-1]
print(data.keys())
print(data.head())
print(data.columns)
print(data.isnull().sum())
print(data.describe())
print(data.corr())

sns.countplot(x='Diagnosis', data=data)
# plt.show(block=True)
# plt.interactive(False)
freq = data['Diagnosis'].value_counts()
print(freq)

y = data.Diagnosis
x = data.iloc[:, 1:]
print(y)
print(y.describe())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape)
print(y_train.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_ps = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)

#find covariance matrix
from scipy.linalg import eigh
cov = np.matmul(x_train_ps.T, x_train_ps)
eg = eigh(cov, eigvals_only=True)

print(np.mean(x_train_ps))
print(np.std(x_train_ps))
print(eg)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(x_train_ps)

percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
# calc of variation % of each PC
labels = ['PC'+str(p) for p in range(1,len(percent_var)+1)]
f, ax = plt.subplots(figsize=(23, 5))
plt.bar(x=range(1,len(percent_var)+1), height=percent_var, tick_label=labels)
plt.xlabel('Principal Components')
plt.ylabel('Variation %')
plt.title('Scree Plot: All n PCs')
plt.show()


pca = PCA(n_components=29)
pca.fit_transform(x_train_ps)

percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC'+str(p) for p in range(1,len(percent_var)+1)]
f, ax = plt.subplots(figsize=(8,5))
plt.bar(x=range(1,len(percent_var)+1), height=percent_var, tick_label=labels)

plt.xlabel('Principal Components')
plt.ylabel('Variation %')
plt.title('Scree Plot: n = 29 PC')
plt.show()




explained = pca.explained_variance_ratio_
print(explained)

cum = np.cumsum(np.round(explained, decimals = 3))
cum_perc = cum*100
pc_df = pd.DataFrame(['PC1','PC2', 'PC3'], columns=['PC'])
explained_df = pd.DataFrame(explained, columns=['Explained variance'])
cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)
print(total_explained)
pca_data = pca.fit_transform(x_train_ps)
print('The reduced data is of the dimension: ', pca_data.shape)
pca_1 = PCA(n_components=2)
data_1 = pca_1.fit_transform(x_train_ps)
data_1 = np.vstack((data_1.T, y_train)).T

#creating a new dataframe for plotting the labeled points
pca_df = pd.DataFrame(data=data_1, columns=('1st Principal', '2nd Principal', 'Labels'))
print(pca_df.head())

#visualizing the 2D points
sns.FacetGrid(pca_df, hue='Labels', height=6).map(plt.scatter, '1st Principal', '2nd Principal').add_legend()
plt.show()