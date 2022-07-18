from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler

# Change path if not running locally.
data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_concat.csv'
data = pd.read_csv(data_path)

data = data.iloc[:, 1:-1]
# Cross-checking some stuff here.

print(data.dtypes)
print(data.isnull().sum())

y = data["Diagnosis"]
print(y.value_counts())

le = LabelEncoder()
binary_encoded_y = pd.Series(le.fit_transform(y))
print(y.shape)

X = data.iloc[:, 1:]
print(X.head())

scaler = StandardScaler()
scaler.fit(X)
scaled_data=scaler.transform(X)

print(scaled_data)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

print(scaled_data)
print(x_pca)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.show(block=True)
plt.interactive(False)

# add labels and colours: https://stackoverflow.com/questions/45333733/plotting-pca-output-in-scatter-plot-whilst-colouring-according-to-to-label-pytho
