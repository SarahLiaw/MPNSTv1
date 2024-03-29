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

print(x_pca.shape)

columns = ['pca_%i' % i for i in range(n_comp)]
df_pca = pd.DataFrame(x_pca, columns=columns)
print(df_pca.head())

log_reg_df = pd.concat([df_pca, z_df], axis=1, join='inner')
print(log_reg_df)
print(log_reg_df.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_validate
from sklearn.model_selection import cross_val_score

log = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=42)
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)

cv_results = cross_validate(log, log_reg_df, encoded_y, cv=cv, return_estimator=True)
counter = 1
for model in cv_results['estimator']:
    print(model.coef_)
    print (counter)
    print(np.average(model.coef_, axis=1))

    counter += 1









import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold

#concat delfi_ratio_w_diagnosis

z_score_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/arm_frag_edited_rc.csv'
ratio_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi_ratio_w_diagnosis_rm2otlr.csv'

#clf = sklearn.linear_model.LogisticRegression(penalty='l1',n_jobs =-1,solver='liblinear',C=1).fit(X, y).

z_df = pd.read_csv(z_score_path)  # Grab the first 1:40 (40 not inclusive, no lib ID)
ratio_df = pd.read_csv(ratio_path)

# Check if this is true: print(sorted(list(z_df['library'])) == sorted(list(ratio_df['ID'])))

log_reg_df = pd.concat([z_df, ratio_df.iloc[:, 2:], ratio_df.Diagnosis], axis=1, join='inner')

y = log_reg_df.Diagnosis
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")

# split log_reg_df into MPNST vs plexiform

mpnst_plex_df = log_reg_df[(log_reg_df['Diagnosis'] == 'plexiform') | (log_reg_df['Diagnosis'] == 'mpnst')]
x_mplx_z = mpnst_plex_df.iloc[:, 1:40]
mplx_id = mpnst_plex_df.iloc[:, 0:1]
x_mplx_pca = mpnst_plex_df.iloc[:, 40:-1]
y_mplx = mpnst_plex_df['Diagnosis']
x_array = np.array(x_mplx_pca)
x_array = x_array.T
sc = StandardScaler()
x_mplx_sc = sc.fit_transform(x_array)
x_mplx_sc = x_mplx_sc.T
x_mplx_sc_df = pd.DataFrame(x_mplx_sc, columns=list(mpnst_plex_df.columns[40:-1]))

#combine this with the z_score
#final_mplx_df = pd.concat([mplx_id, y_mplx, x_mplx_z, x_mplx_sc_df], axis=1, join='inner')
# so 2:41 gives you z score
final_mplx_df = pd.concat([y_mplx, x_mplx_z, x_mplx_sc_df], axis=1, join='inner')

#split a dataframe and use the pca components
#for i in range(10):
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(x_mplx_sc):
    X_train, X_test = x_mplx_sc[train_index], x_mplx_sc[test_index]
    pca = PCA()
    x_pca = pca.fit_transform(X_train[:, 40:-1])

    explained = pca.explained_variance_ratio_

    cum_perc = np.cumsum(np.round(explained, decimals=3)) * 100
    pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3'], columns=['PC'])
    explained_df = pd.DataFrame(explained, columns=['Explained variance'])
    cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
    total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)
    print(total_explained)
    minimum_PC = 100
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

    print(minimum_PC)
    # print(x_pca)
    # print(x_pca[:, :minimum_PC].shape)
    columns = ['pca_%i' % i for i in range(minimum_PC)]
    df_pca = pd.DataFrame(x_pca[:, :minimum_PC], columns=columns)
    print(df_pca.head())



#log_reg_df into plexiform vs healthy

healthy_plex_df = log_reg_df[(log_reg_df['Diagnosis'] == 'plexiform') | (log_reg_df['Diagnosis'] == 'healthy')]


# first split into MPNST and