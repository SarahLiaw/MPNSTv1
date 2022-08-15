import pandas as pd

import numpy as np

from sklearn.model_selection import cross_validate, KFold


arm_df = pd.read_csv('/home/sarahl/PycharmProjects/MPNST_v1/data_v1/arm_frag.tsv', sep='\t')

library_id_set = set(arm_df['library'])


mpnst_df = pd.read_csv('/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv')
mpnst_ids = sorted(set(mpnst_df['ID']))

for id in library_id_set:
    if id not in mpnst_ids:
        arm_df.drop(arm_df.index[arm_df['library'] == id], inplace=True)
       # print(id)

check = sorted(set(arm_df['library']))

print(check)
print(mpnst_ids)
print(check == mpnst_ids)
arm_df['chr'] = arm_df['chr'] + ' ' + arm_df['arm']

print(arm_df.columns)

arm_df.drop('arm', axis=1, inplace=True)
arm_df.drop('raw_frag_cnt', axis=1, inplace=True)
arm_df.to_csv('removed_irrelevant_index_arm_frag.csv', index=False)

# here you can now read removed irrelevant index for arm-frag