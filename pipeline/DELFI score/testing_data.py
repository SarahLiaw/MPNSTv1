import pandas as pd

arm_df = pd.read_csv('/home/sarahl/PycharmProjects/MPNST_v1/data_v1/arm_frag.tsv', sep='\t')

library_id_set = set(arm_df['library'])
print(library_id_set)

mpnst_df = pd.read_csv('/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv')
mpnst_ids = sorted(set(mpnst_df['ID']))

for id in library_id_set:
    if id not in mpnst_ids:
        arm_df.drop(arm_df.index[arm_df['library'] == id], inplace=True)
        print(id)

print(arm_df)
check = sorted(set(arm_df['library']))

print(check)
print(mpnst_ids)
print(check == mpnst_ids)

# basically removed all the rows of the data for the library-ids that are not relevant in the above.

# You should be performing PCA over the just delfi ratios for each cross-validation fold,
# and then take the minimum number of PC’s needed for 90% of the variance. (This technically
# falls under the umbrellas of feature engineering/reduction as well.) Then, like you said, your
# input features to the logistic regression should be the combination of z-scores and that minimal
# set of PCs. You should have something on the order of 60-80ish features if I had to guess: 39
# from the z-scores, and the then 20-40ish PCs. No worries if it ends up being a bit more. The
# idea here is that we’re drastically reducing the number of input features to the model: if we
# used the raw delfi+ichor data, we’d have a few thousand features, which will make it hard
# for models to converge with our limited cohort size.

# if chr are the same in the library, find the average for each chrom
