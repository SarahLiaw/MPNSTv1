# "The role of PCA is to find such highly correlated or duplicate features and to come up
# with a new feature set where there is minimum correlation between the features or in other words
# feature set with maximum variance between the features."


import sys
from pipeline.data_path.data_transformation import *
# from pipeline.visualizations.def_visualizations import *
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/data_path')
# sys.path.insert(1, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/visualizations')

path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv'
data = read_data(path)

X = get_X(data)
y = get_y(data)
encode_y = encode_target(y)


n_comp = 26

# typically before train test split and do standard scaling on X_test with fit_transform via standard_scaling
# and do standard scaling on y_test with fit.

scaled_data = standard_scaling(X)

pca = PCA(n_components=n_comp)
x_pca = pca.fit_transform(scaled_data)
