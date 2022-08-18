
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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

# split log_reg_df into MPNST vs plexiform

mpnst_plex_df = log_reg_df[(log_reg_df['Diagnosis'] == 'plexiform') | (log_reg_df['Diagnosis'] == 'mpnst')]
x_mplx_z = mpnst_plex_df.iloc[:, 1:40]
mplx_id = mpnst_plex_df.iloc[:, 0:1]
x_mplx_pca = mpnst_plex_df.iloc[:, 40:-1]

y_mplx = mpnst_plex_df['Diagnosis']
label_encoder = LabelEncoder()
y_mplx = label_encoder.fit_transform(y_mplx)
y_mplx_df = pd.DataFrame(y_mplx, columns=['Diagnosis'])
label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")

x_array = np.array(x_mplx_pca)
x_array = x_array.T
sc = StandardScaler()
x_mplx_sc = sc.fit_transform(x_array)
x_mplx_sc = x_mplx_sc.T
x_mplx_sc_df = pd.DataFrame(x_mplx_sc, columns=list(mpnst_plex_df.columns[40:-1]))


# so 2:41 gives you z score
final_mplx_df = pd.concat([mplx_id, y_mplx_df, x_mplx_z, x_mplx_sc_df], axis=1, join='inner')

#split a dataframe and use the pca components
#for i in range(10):
kf = KFold(n_splits=5)
# X_np = np.array(final_mplx_df.iloc[:, 1:])
# y_np = np.array(final_mplx_df.oloc[:, 0:1])

final_dictionary = {}
for train_index, test_index in kf.split(final_mplx_df.iloc[2:]):
    X_pca_parts = final_mplx_df.iloc[:, 41:]

    # y_parts = final_mplx_df.iloc[:, 1:2]
    # X_zscore = final_mplx_df.iloc[:, 2:41]
    # idx = final_mplx_df.iloc[:, 0:1]

    others_index = final_mplx_df.iloc[:, 0:41]

    X_pca_train, X_pca_test = X_pca_parts.iloc[train_index], X_pca_parts.iloc[test_index]
    # y_train, y_test = y_parts.iloc[train_index], y_parts.iloc[test_index]
    # X_zscore_train, X_zscore_test = X_zscore.iloc[train_index], X_zscore.iloc[test_index]
    # idx_train, idx_test = idx.iloc[train_index], idx.iloc[test_index]
    others_index_train, others_index_test = others_index.iloc[train_index], others_index.iloc[test_index]

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

    minimum_PC = 100
    for i in range(len(cum_perc)):
        if cum_perc[i] >= 90.0:
            minimum_PC = min(minimum_PC, i + 1)
            break

    print(minimum_PC)
    percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    # calc of variation % of each PC
    labels = ['PC' + str(p) for p in range(1, len(percent_var) + 1)]
    f, ax = plt.subplots(figsize=(23, 5))
    plt.bar(x=range(1, len(percent_var) + 1), height=percent_var, tick_label=labels)
    plt.xlabel('Principal Components')
    plt.ylabel('Variation %')
    plt.title('Scree Plot: All n PCs')
    #plt.show()

    columns = ['pca_%i' % i for i in range(minimum_PC)]
    df_pca = pd.DataFrame(x_pca_transform_train[:, :minimum_PC], columns=columns, index=others_index_train['library'].tolist())

    others_index_train.set_index('library', inplace=True)
    log_reg = pd.concat([others_index_train, df_pca], axis=1, join='inner')
    df_pca_test = pd.DataFrame(x_pca_transform_test[:, :minimum_PC], columns=columns, index=others_index_test['library'].tolist())
    others_index_test.set_index('library', inplace=True)
    X_concat_train = pd.concat([others_index_test, df_pca_test], axis=1, join='inner')

    log = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=42)
    log.fit(log_reg.iloc[:, 1:], log_reg['Diagnosis'])
    y_pred = log.predict(X_concat_train.iloc[:, 1:])

    y_pred = list(y_pred)
    train_index_list = list(log_reg.index)
    test_index_list = list(X_concat_train.index)

    for i in range(len(train_index_list)):
        if train_index_list[i] not in final_dictionary:
            final_dictionary[train_index_list[i]] = [log_reg._get_value(train_index_list[i], 'Diagnosis')]
        else:
            final_dictionary[train_index_list[i]].append(log_reg._get_value(train_index_list[i], 'Diagnosis'))

    for i in range(len(test_index_list)):
        if test_index_list[i] not in final_dictionary:
            final_dictionary[test_index_list[i]] = [y_pred[i]]
        else:
            final_dictionary[test_index_list[i]].append(y_pred[i])
    print(final_dictionary)








#log_reg_df into plexiform vs healthy

healthy_plex_df = log_reg_df[(log_reg_df['Diagnosis'] == 'plexiform') | (log_reg_df['Diagnosis'] == 'healthy')]


# first split into MPNST and