import sys
from pipeline.data_path.data_transformation import *
from def_visualizations import *

sys.path.insert(0, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/data_path')

path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv'
data = read_data(path)

X = get_X(data)
y = get_y(data)
encode_y = encode_target(y)

n_comp = 29

scaled_data = standard_scaling(X)

pca = PCA()
x_pca = pca.fit_transform(scaled_data)
pca.fit_transform(scaled_data)

diagnosis_count_chart(data)

heatmap(pca, False, data)

pc_scatter(path, X)

scree_plot_all_PCs(pca)

#biplot_cluster(X, y, n_comp)

myplot(x_pca[:,0:2],encode_y, np.transpose(pca.components_[0:2, :]))