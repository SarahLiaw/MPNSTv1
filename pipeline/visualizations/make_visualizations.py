import sys
from pipeline.data_path.data_transformation import *
from def_visualizations import *

sys.path.insert(0, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/data_path')

path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv'
data = read_data(path)

X = get_X(data)
y = get_y(data)

n_comp = 26

scaled_data = standard_scaling(X)

pca = PCA(n_components=n_comp)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

diagnosis_count_chart(data)

heatmap(pca, False, data)

pc_scatter(path, X)

biplot_cluster(X, y, n_comp)

