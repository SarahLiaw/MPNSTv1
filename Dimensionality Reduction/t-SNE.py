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
sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Change path if not running locally.
data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_concat_no_outliers.csv'
data = pd.read_csv(data_path)

sns.countplot(x='Diagnosis', data=data)
#plt.show()
data.drop(['ID'], axis = 1, inplace = True)
print(data.head())
y = data.Diagnosis.values
x = data.iloc[:, 1:].values
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=3, random_state=0)
tsne_obj = tsne.fit_transform(X_train)
tsne_df = pd.DataFrame({'X' : tsne_obj[:,0],
                       'Y' : tsne_obj[:,1],
                        'classification' : y_train
                       })

print(tsne_df.head())
print(tsne_df['classification'].value_counts())

label_colour_dict = {'plexiform': 'orange', 'healthy': 'green', 'mpnst': 'red'}

#color vector creation

cvec = [label_colour_dict[label] for label in y]

plt.figure(figsize=(10, 10))
sns.scatterplot(x="X", y="Y", hue='classification', legend = 'full', data=tsne_df)
plt.show()