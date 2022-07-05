# For this, I used delfi data and the samplelist.csv file.
# I will continue to update this with the ichor data, and the tumor fraction (need to check if the tumor fraction
# is for the diagnosis or whether it gives us more data to add to).

# Will do the above soon!


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
data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi+diagnosis.csv'
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
print(X.shape)

embedding = umap.UMAP(n_neighbors=8, min_dist=0.3, metric='correlation').fit_transform(X)
# I need to figure out a way to plot how it looks dimensionally.


train_X, test_X, train_y, test_y = train_test_split(X, binary_encoded_y, random_state=1)

classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
)
classifier.fit(train_X, train_y)

predictions = classifier.predict(test_X)
check_confusion = confusion_matrix(test_y, predictions)
print(check_confusion)


