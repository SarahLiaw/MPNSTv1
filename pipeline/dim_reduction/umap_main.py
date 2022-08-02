import umap.umap_ as umap
# for some reason, umap doesn't work here but works in other directories.

import sys
from pipeline.data_path.data_transformation import *
from pipeline.visualizations.def_visualizations import *

sys.path.insert(0, '/pipeline/data_path')
sys.path.insert(1, '/pipeline/visualizations')

path = '/data_v1/MPNST_v1_rm2otlr.csv'
data = read_data(path)

X = get_X(data)
y = get_y(data)
encode_y = encode_target(y)
scaled_X = standard_scaling(X)
embedding = umap.UMAP(n_neighbors=10, min_dist=0.6, metric='correlation').fit_transform(scaled_X)

umap_visuals(y, embedding)
