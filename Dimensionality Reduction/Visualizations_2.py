import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_concat.csv'
data = pd.read_csv(data_path)
data = data.iloc[:, 1:-1]