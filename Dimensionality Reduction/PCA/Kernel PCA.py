import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# using removed outliers, heatmap, n comp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Change path if not running locally.
data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_concat_no_outliers.csv'
data = pd.read_csv(data_path)
id = list(data['ID'])
id_concat = [i[3:] for i in id]
print(id)
data = data.iloc[:, 1:-1]
# Cross-checking some stuff here.


