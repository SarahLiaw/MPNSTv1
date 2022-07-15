import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_concat.csv'
data = pd.read_csv(data_path)
data = data.iloc[:, 1:-1]
print(data.keys())
print(data.head())
print(data.columns)
print(data.isnull().sum())
print(data.describe())
print(data.corr())
# f, ax = plt.subplots(figsize=(20, 20))
# sns.set(font_scale=1.25)
# sns.heatmap(data.corr(), annot=True, fmt='.1f')
# plt.show(block=True)
# plt.interactive(False)

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
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)
print(np.mean(x_train_std))
print(np.std(x_train_std))

# #Finding the covariance matrix which is M^T*M
# cov_matrix = np.matmul(x_train_std.T, x_train_std)
# print('The dimension of the covariance matrix is: ', cov_matrix.shape)
#
# from scipy.linalg import eigh
# ev = eigh(cov_matrix, eigvals_only=True)
# print(ev)
#

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(x_train_std)
percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
#calculating the percentage of variation that each principal component acco)
labels = ['PC'+str(p) for p in range(1,len(percent_var)+1)]

f, ax = plt.subplots(figsize=(23, 5))
plt.bar(x=range(1,len(percent_var)+1), height=percent_var, tick_label=labels)
plt.xlabel('Principal Components')
plt.ylabel('Percentage of Variation Explained')
plt.title('Scree Plot showing all components')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit_transform(x_train_std)

percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC'+str(p) for p in range(1,len(percent_var)+1)]
f, ax = plt.subplots(figsize=(8,5))
plt.bar(x=range(1,len(percent_var)+1), height=percent_var, tick_label=labels)
plt.xlabel('Principal Components')
plt.ylabel('Percentage of Variation explained')
plt.title('Scree Plot showing first 3 components')
plt.show()

explained_var = pca.explained_variance_ratio_
print(explained_var)

cum_var = np.cumsum(np.round(explained_var, decimals = 3))
cum_var_perc = cum_var*100
pc_df = pd.DataFrame(['PC1','PC2', 'PC3'], columns=['PC'])
explained_var_df = pd.DataFrame(explained_var, columns=['Explained variance'])
cum_var_df = pd.DataFrame(cum_var_perc, columns=['Cumulative variance (in %)'])
total_var_explained = pd.concat([pc_df, explained_var_df, cum_var_df], axis=1)
print(total_var_explained)
pca_data = pca.fit_transform(x_train_std)
print('The reduced data is of the dimension: ', pca_data.shape)
pca_1 = PCA(n_components=2)
pca_data_1 = pca_1.fit_transform(x_train_std)
pca_data_1 = np.vstack((pca_data_1.T, y_train)).T

#creating a new dataframe for plotting the labeled points
pca_df = pd.DataFrame(data=pca_data_1, columns=('1st Principal', '2nd Principal', 'Labels'))
print(pca_df.head())

#visualizing the 2D points
sns.FacetGrid(pca_df, hue='Labels', size=6).map(plt.scatter, '1st Principal', '2nd Principal').add_legend()
plt.show()