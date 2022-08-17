import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_validate
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate, KFold
#Generic Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#SK Learn Libraries
import sklearn
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier   #1vs1 & 1vsRest Classifiers
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import gc

z_score_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/arm_frag_edited_rc.csv'
ratio_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi_ratio_w_diagnosis_rm2otlr.csv'

z_df = pd.read_csv(z_score_path)
ratio_df = pd.read_csv(ratio_path)


# print(sorted(list(z_df['library'])))
# print(sorted(list(ratio_df['ID'])))
# print(sorted(list(z_df['library'])) == sorted(list(ratio_df['ID'])))

z_df = z_df.iloc[:, 1:]

ratio_df_no_index = ratio_df.iloc[:, 1:]
print(z_df.shape)
print(ratio_df.shape)
print(z_df.head())
y = ratio_df_no_index.Diagnosis
x = ratio_df_no_index.iloc[:, 1:]

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")

x_array = np.array(x)
x_array = x_array.T
sc = StandardScaler()
x_scale = sc.fit_transform(x_array)
x_scale = x_scale.T

n_comp = 24
pca = PCA(n_components=n_comp)
x_pca = pca.fit_transform(x_scale)

explained = pca.explained_variance_ratio_

cum = np.cumsum(np.round(explained, decimals=3))
cum_perc = cum * 100
pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3'], columns=['PC'])
explained_df = pd.DataFrame(explained, columns=['Explained variance'])
cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)


print(x_pca)

columns = ['pca_%i' % i for i in range(n_comp)]
df_pca = pd.DataFrame(x_pca, columns=columns)
print(df_pca.head())

log_reg_df = pd.concat([df_pca, z_df], axis=1, join='inner')


model = SVC(decision_function_shape='ovo')
model.fit(log_reg_df, y)
yhat = model.predict(log_reg_df)
print(list(y))
print(yhat)

diagnosis_binary = pd.DataFrame(yhat, columns=['Binary_Diagnosis'])

log_reg_df = pd.concat([diagnosis_binary, log_reg_df], axis=1, join='inner')
print(log_reg_df.head())

# log = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=42)
# cv = KFold(n_splits=10, random_state=1)

#segment datasets, only look at these 3 comparisons.



#save segmentation of datasets

#take datasets and filter into 2 classes wanna look at, then do 50 cross validations.
# then do PCA on the 4 folds. 