import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from bioinfokit.visuz import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def heatmap(pca, hm_label, data):
    """
    :type hm_label: bool
    :param pca: Fitted PCA from pca = PCA().
    :param hm_label: True if annot=true for labelling on hm.
    :param data: Data returned from read_data, where it is iloced.
    :return: None. Draws a heatmap based on eigenvectors.
    """
    # pca_components = pca.components_
    plt.figure(figsize=(12, 6))
    df_comp = pd.DataFrame(pca.components_, columns=list(data.keys())[1:])
    sns.heatmap(df_comp, annot=hm_label, cmap='plasma')
    plt.show(block=True)
    plt.interactive(False)


def scree_plot_all_PCs(pca):
    """
    :param pca:Fitted PCA from pca = PCA().
    :return: Scree Plot of all principal components.
    """
    percent_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # calc of variation % of each PC
    labels = ['PC' + str(p) for p in range(1, len(percent_var) + 1)]
    f, ax = plt.subplots(figsize=(23, 5))
    plt.bar(x=range(1, len(percent_var) + 1), height=percent_var, tick_label=labels)
    plt.xlabel('Principal Components')
    plt.ylabel('Variation %')
    plt.title('Scree Plot: All n PCs')
    plt.show(block=True)
    plt.interactive(False)


def scree_plot_some_PCs(pca):
    return None


def biplot_cluster(X, target, n):
    """
    :param X: Call get_X from data path/data_transformation.py.
    :param target: Get from data.diagnosis or call get_y from data path/data_transformation.py
    :param n: Number of PCA components.
    :return: none. Saves biplot in file.
    """
    X_st = StandardScaler().fit_transform(X)
    pca_out = PCA(n).fit(X_st)
    pca_scores = PCA().fit_transform(X_st)
    cluster.biplot(cscore=pca_scores, loadings=pca_out.components_, labels=X.columns.values,
                   var1=round(pca_out.explained_variance_ratio_[0] * 100, 2),
                   var2=round(pca_out.explained_variance_ratio_[1] * 100, 2), colorlist=target)


def diagnosis_count_chart(data):
    """
    :param data: Data returned from read_data, where it is iloced.
    :return: Bar chart of MPNST, healthy and PN count in sample.
    """
    sns.countplot(x='Diagnosis', data=data)
    plt.show(block=True)
    plt.interactive(False)


def pc_scatter(data_path, X):
    """
    Plot all samples with labels in scatter plot of PC2 against PC1.
    :param data_path: Path of where it is/directory.
    :param X: From get_X.
    :return: None. Scatter plot of samples along PCs.
    """
    data = pd.read_csv(data_path)
    id = list(data['ID'])
    id_concat = [i[3:] for i in id]
    label_colour_dict = {'plexiform': 'orange', 'healthy': 'green', 'mpnst': 'red'}

    scaler = StandardScaler()
    scaler.fit(X)
    scaled_data = scaler.transform(X)
    cvec = [label_colour_dict[label] for label in data["Diagnosis"]]
    pca = PCA(n_components=26)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cvec, edgecolor=['none'], alpha=0.5)

    # can comment this out if don't want labels
    for lib in id_concat:
        i = id_concat.index(lib)
        labelpad = 0.01
        plt.text(x_pca[i, 0] + labelpad, x_pca[i, 1] + labelpad, lib, fontsize=7)

    plt.xlabel('First principle component')
    plt.ylabel('Second principle component')
    plt.show(block=True)
    plt.interactive(False)


def myplot(score, target, coeff, labels=None):
    """
    Reference material: https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
    scores: projected features and coeff:  elements of the eigenvetors
    :param score: x_pca first 2 projected features.
    :param target: encoded y.
    :param coeff: elements of the eigenvectors.
    :param labels:... (should be column names)
    :return:
    """
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=target)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        # if labels is None:
        #     plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        # else:
        #     plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show(block=True)
    plt.interactive(False)


def tsne_visuals(tsne_df):
    """
    :param tsne_df: the tsne pd.dataframe.
    :return: None. Scatter plot of sample distribution.
    """
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="X", y="Y", hue='classification', legend='full', data=tsne_df)
    plt.show()
