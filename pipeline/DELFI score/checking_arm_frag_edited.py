import pandas as pd

import numpy as np

from sklearn.model_selection import cross_validate, KFold


arm_df = pd.read_csv('/data_v1/data_provided/arm_frag_edited_rc.csv')
print(arm_df.shape)
