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

sxc = StandardScaler()
X = sxc.fit_transform(X)

embedding = umap.UMAP(n_neighbors=10, min_dist=0.6, metric='correlation').fit_transform(X)
import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 6.28, 100)
#
# plt.plot(x, x**0.5, label='square root')
# plt.plot(x, np.sin(x), label='sinc')
#
# plt.xlabel('x label')
# plt.ylabel('y label')
#
# plt.title("test plot")
#
# plt.legend()
#


plt.figure(figsize=(10,10))
plt.scatter(embedding[:,0], embedding[:,1],

            edgecolor='none',
            alpha=0.80,
            s=56)
plt.axis('off')
plt.show(block=True)
plt.interactive(False)