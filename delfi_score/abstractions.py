import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def variance_explained_graph(pca):
    percent_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # calc of variation % of each PC
    labels = ['PC' + str(p) for p in range(1, len(percent_var) + 1)]
    f, ax = plt.subplots(figsize=(23, 5))
    plt.bar(x=range(1, len(percent_var) + 1), height=percent_var, tick_label=labels)
    plt.xlabel('Principal Components')
    plt.ylabel('Variation %')
    plt.title('Scree Plot: All n PCs')
    plt.show()


def check_z_delfi_index(z_df, ratio_df):
    """
    :param z_df: Dataframe of z-scores.
    :param ratio_df: Dataframe of delfi_ratios.
    :return: Should return true.
    """
    return sorted(list(z_df['library'])) == sorted(list(ratio_df['ID']))


def encode_target(y):
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y)
    label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")
    return encoded_y


def standard_scaling_delfi(x_delfi):
    x_array = np.array(x_delfi)
    x_array = x_array.T
    sc = StandardScaler()
    x_mplx_sc = sc.fit_transform(x_array)
    return x_mplx_sc.T


def find_min_PCs(cum_perc):
    minimum_PC = 100
    for per in range(len(cum_perc)):
        if cum_perc[per] >= 90.0:
            minimum_PC = min(minimum_PC, per + 1)
            break

    return minimum_PC