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

#Finding the covariance matrix which is M^T*M
cov_matrix = np.matmul(x_train_std.T, x_train_std)
print('The dimension of the covariance matrix is: ', cov_matrix.shape)

from scipy.linalg import eigh
ev = eigh(cov_matrix, eigvals_only=True)
print(ev)