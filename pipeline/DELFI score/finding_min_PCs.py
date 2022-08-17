import pandas as pd

import numpy as np

from sklearn.model_selection import cross_validate, KFold
import matplotlib.pyplot as plt

data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi_ratio_w_diagnosis.csv'
data = pd.read_csv(data_path)
data = data.iloc[:, 1:-1]
y = data.Diagnosis
x = data.iloc[:, 1:]
# print(x.shape)
# print(.shape)
# print(x.keys())
print(x)
print(y)

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")

x_array = np.array(x)
x_array = x_array.T
print(x_array)
sc = StandardScaler()
x_scale = sc.fit_transform(x_array)
x_scale = x_scale.T

kf = KFold(n_splits=5)

minimum_PC = 200
for train_index, test_index in kf.split(x_scale, y):
     print("TRAIN:", train_index, "TEST:", test_index)

     X_train, X_test = x_scale[train_index], x_scale[test_index]

     pca = PCA()
     pca.fit_transform(X_train)

     explained = pca.explained_variance_ratio_

     cum_perc = np.cumsum(np.round(explained, decimals=3)) * 100
     pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3'], columns=['PC'])
     explained_df = pd.DataFrame(explained, columns=['Explained variance'])
     cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
     total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)

     for i in range(len(cum_perc)):
         if cum_perc[i] >= 90.0:
             minimum_PC = min(minimum_PC, i + 1)
             break

     percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
     # calc of variation % of each PC
     labels = ['PC' + str(p) for p in range(1, len(percent_var) + 1)]
     f, ax = plt.subplots(figsize=(23, 5))
     plt.bar(x=range(1, len(percent_var) + 1), height=percent_var, tick_label=labels)
     plt.xlabel('Principal Components')
     plt.ylabel('Variation %')
     plt.title('Scree Plot: All n PCs')
     plt.show()

print("Therefore, the minimum number of PCs that will be used for log reg after k fold is", minimum_PC)


# your input features to the logistic regression should be the combination of z-scores and
# that minimal set of PCs. You should have something on the order of 60-80ish features if I had
# to guess: 39 from the z-scores, and the then 20-40ish PCs. No worries if it ends up being a bit more.
# The idea here is that we’re drastically reducing the number of input features to the model: if we used
# the raw delfi+ichor data, we’d have a few thousand features, which will make it hard for models to converge
# with our limited cohort size.
