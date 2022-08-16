import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder



z_score_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/arm_frag_edited_rc.csv'
ratio_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi_ratio_w_diagnosis_rm2otlr.csv'

#clf = sklearn.linear_model.LogisticRegression(penalty='l1',n_jobs =-1,solver='liblinear',C=1).fit(X, y).

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
#print(explained)

cum = np.cumsum(np.round(explained, decimals=3))
cum_perc = cum * 100
pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3'], columns=['PC'])
explained_df = pd.DataFrame(explained, columns=['Explained variance'])
cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)
#print(total_explained)

print(x_pca)

columns = ['pca_%i' % i for i in range(n_comp)]
df_pca = pd.DataFrame(x_pca, columns=columns)
print(df_pca.head())

log_reg_df = pd.concat([df_pca, z_df], axis=1, join='inner')
print(log_reg_df)
print(log_reg_df.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedKFold
from sklearn.model_selection import cross_val_score

log = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=42)
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
scores = cross_val_score(log, log_reg_df, encoded_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = abs(scores)
print(scores)
print(scores.shape)