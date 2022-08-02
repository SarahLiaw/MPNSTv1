import sys
from pipeline.data_path.data_transformation import *
from pipeline.visualizations.def_visualizations import *

sys.path.insert(0, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/data_path')
sys.path.insert(1, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/visualizations')

from sklearn.manifold import TSNE
path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv'
data = read_data(path)

X = get_X(data)
y = get_y(data)
encode_y = encode_target(y)

scaled_data = standard_scaling(X)
tsne = TSNE(n_components=3, random_state=0)
tsne_obj = tsne.fit_transform(scaled_data)
tsne_df = pd.DataFrame({'X' : tsne_obj[:,0],
                       'Y' : tsne_obj[:,1],
                        'classification':encode_y
                       })

tsne_visuals(tsne_df)