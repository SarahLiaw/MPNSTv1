import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from abstractions import *

import csv

z_score_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/arm_frag_edited_rc.csv'
ratio_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi_ratio_w_diagnosis_rm2otlr.csv'

z_df = pd.read_csv(z_score_path)  # Includes sorted library index and z-scores (total 39 features).
ratio_df = pd.read_csv(ratio_path)  # Includes sorted library index, diagnosis, delfi values.

mpnst_plex_df = pd.concat([z_df, ratio_df.iloc[:, 2:], ratio_df.Diagnosis], axis=1, join='inner')

mpnst_plex_df = mpnst_plex_df[(mpnst_plex_df['Diagnosis'] == 'plexiform') | (mpnst_plex_df['Diagnosis'] == 'healthy')]
print(mpnst_plex_df)
mpnst_plex_df.set_index('library', inplace=True)

mpnst_index = list(mpnst_plex_df.index)

x_mplx_z = mpnst_plex_df.iloc[:, :39]
x_mplx_pca = mpnst_plex_df.iloc[:, 39:-1]

y_mplx_df = pd.DataFrame(encode_target(mpnst_plex_df['Diagnosis']),
                         columns=['Diagnosis'], index=mpnst_index)
x_mplx_sc_df = pd.DataFrame(standard_scaling_delfi(x_mplx_pca),
                            columns=list(mpnst_plex_df.columns[39:-1]), index=mpnst_index)

final_mplx_df = pd.concat([y_mplx_df, x_mplx_z, x_mplx_sc_df],
                          axis=1, join='inner')  # so iloc 2:41 gives you z score

kf = KFold(n_splits=5)

binary_classification_dict = {}

kf_train_idx = {}
kf_test_idx = {}

kf_idx = 1

for _ in range(10):
    for train_index, test_index in kf.split(final_mplx_df):

        kf_train_idx[str(kf_idx) + "th iteration"] = train_index
        kf_test_idx[str(kf_idx) + "th iteration"] = test_index
        kf_idx += 1

        X_pca_parts = final_mplx_df.iloc[:, 40:]
        others_index = final_mplx_df.iloc[:, 0:40]

        X_pca_train, X_pca_test = X_pca_parts.iloc[train_index], X_pca_parts.iloc[test_index]
        others_index_train, others_index_test = others_index.iloc[train_index], others_index.iloc[test_index]
        training_pca_index = list(others_index_train.index)
        testing_pca_index = list(others_index_test.index)

        pca = PCA()
        pca.fit(X_pca_train)
        x_pca_transform_train = pca.transform(X_pca_train)
        x_pca_transform_test = pca.transform(X_pca_test)

        explained = pca.explained_variance_ratio_
        cum_perc = np.cumsum(np.round(explained, decimals=3)) * 100
        pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3'], columns=['PC'])
        explained_df = pd.DataFrame(explained, columns=['Explained variance'])
        cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
        total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)

        minimum_PC = find_min_PCs(cum_perc)

        columns = ['pca_%i' % i for i in range(minimum_PC)]
        df_pca = pd.DataFrame(x_pca_transform_train[:, :minimum_PC], columns=columns, index=training_pca_index)

        log_reg_train = pd.concat([others_index_train, df_pca], axis=1, join='inner')

        pca_test = pd.DataFrame(x_pca_transform_test[:, :minimum_PC], columns=columns, index=testing_pca_index)
        log_reg_test = pd.concat([others_index_test, pca_test], axis=1, join='inner')

        log = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=42)
        log.fit(log_reg_train.iloc[:, 1:], log_reg_train['Diagnosis'])
        y_pred = log.predict(log_reg_test.iloc[:, 1:])

        y_pred = list(y_pred)

        train_index_list = list(log_reg_train.index)
        test_index_list = list(log_reg_test.index)

        for i in range(len(train_index_list)):
            if train_index_list[i] not in binary_classification_dict:
                binary_classification_dict[train_index_list[i]] = [log_reg_train._get_value(train_index_list[i], 'Diagnosis')]
            else:
                binary_classification_dict[train_index_list[i]].append(log_reg_train._get_value(train_index_list[i], 'Diagnosis'))

        for i in range(len(test_index_list)):
            if test_index_list[i] not in binary_classification_dict:
                binary_classification_dict[test_index_list[i]] = [y_pred[i]]
            else:
                binary_classification_dict[test_index_list[i]].append(y_pred[i])


delfi_score = {}

for i in binary_classification_dict:
    delfi_score[i] = sum(binary_classification_dict[i]) / len(binary_classification_dict[i])

print(len(list(delfi_score.keys())))

print(delfi_score)

with open('plxhealthy_delfi.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(delfi_score.items())
