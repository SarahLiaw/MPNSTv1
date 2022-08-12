import pandas as pd

import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold, KFold


import matplotlib.pyplot as plt
arm_df = pd.read_csv('/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi_ratio.csv')

library_id_set = set(arm_df['library'])
#print(library_id_set)

mpnst_df = pd.read_csv('/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv')
mpnst_ids = sorted(set(mpnst_df['ID']))

for id in library_id_set:
    if id not in mpnst_ids:
        arm_df.drop(arm_df.index[arm_df['library'] == id], inplace=True)
       # print(id)

#print(arm_df)
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

# for each fold:
#    split data
#    conduct PCA on the 90% used for training
#    pick the number of components
#    fit linear regression
#    predict the 10% held out
# end:


# code for PCA over the delfi fragment ratios computed and select the minimum number of PCs to get 90% of the variance.

data_path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/delfi_ratio_w_diagnosis.csv'
data = pd.read_csv(data_path)
data = data.iloc[:, 1:-1]
y = data.Diagnosis
x = data.iloc[:, 1:]
# print(x.shape)
# print(.shape)
# print(x.keys())
# #print(x)
# #print(y)
# # print(data.columns)
# print(arm_df.keys())
# print(data.head())
# print(data.corr())


def cross_validation(model, _X, _y, _cv=5):
    '''Function to perform 5 Folds Cross-Validation
     Parameters
     ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
         This is the matrix of features.
    _y: array
         This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
     Returns
     -------
     The function returns a dictionary containing the metrics 'accuracy', 'precision',
     'recall', 'f1' for both training set and validation set.
    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean() * 100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean() * 100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }

    def plot_result(x_label, y_label, plot_title, train_data, val_data):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str,
            Name of the algorithm used for training e.g 'Decision Tree'

         y_label: str,
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str,
            This is the title of the plot e.g 'Accuracy Plot'

         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.

         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''

        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")
#print(encoded_y)

sc = StandardScaler()
x_scale = sc.fit_transform(x)
print(x_scale)
print(x_scale.shape)
#kf =KFold(n_splits=5, shuffle=True, random_state=1)
kf = KFold(n_splits=5)

for train_index, test_index in kf.split(x_scale, y):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = x_scale[train_index], x_scale[test_index]
     y_train, y_test = encoded_y[train_index], encoded_y[test_index]

     pca = PCA()
     pca.fit_transform(X_train)

     explained = pca.explained_variance_ratio_
     print(explained)

     cum = np.cumsum(np.round(explained, decimals=3))
     cum_perc = cum * 100
     pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3'], columns=['PC'])
     explained_df = pd.DataFrame(explained, columns=['Explained variance'])
     cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
     total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)
     print(total_explained)


pca = PCA()
pca.fit(x)
explained = pca.explained_variance_ratio_
print(explained)

cum = np.cumsum(np.round(explained, decimals=3))
cum_perc = cum * 100
pc_df = pd.DataFrame(['PC1', 'PC2', 'PC3'], columns=['PC'])
explained_df = pd.DataFrame(explained, columns=['Explained variance'])
cum_df = pd.DataFrame(cum_perc, columns=['Cumulative variance (in %)'])
total_explained = pd.concat([pc_df, explained_df, cum_df], axis=1)
print(total_explained)
percent_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
# calc of variation % of each PC


labels = ['PC'+str(p) for p in range(1,len(percent_var)+1)]
f, ax = plt.subplots(figsize=(23, 5))
plt.bar(x=range(1,len(percent_var)+1), height=percent_var, tick_label=labels)
plt.xlabel('Principal Components')
plt.ylabel('Variation %')
plt.title('Scree Plot: All n PCs')
plt.show()


# NOTE THAT INSTEAD of doing standard scalar over all features, do it over each sample since there are multiple chromosomes

# refer to the mathios paper (OR SLACK) for more information.