from sklearn.linear_model import LogisticRegression
# https://www.statology.org/leave-one-out-cross-validation-in-python/
# logistic regression using PCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from numpy import *

from pipeline.data_path.data_transformation import *

sys.path.insert(0, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/data_path')

path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv'
data = read_data(path)

X = get_X(data)
y = get_y(data)

# log = LogisticRegression(penalty='l1', solver='liblinear')
# log.fit(X, y)
scaled_data = standard_scaling(X)
encode_y = encode_target(y)

pca = PCA(n_components=26)

X_pca = pca.fit_transform(scaled_data)

log = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear', random_state=42).fit(X_pca, y)
cv = LeaveOneOut()
scoresMEAN = cross_val_score(log, X_pca, encode_y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
acc = cross_val_score(log,X_pca,y,cv=5,scoring='accuracy',n_jobs=-1)

print(mean(absolute(scoresMEAN)))
print(acc)
#https://fixexception.com/scikit-learn/multi-class-must-be-in-ovo-ovr/
scoresSQUARED = cross_val_score(log, X_pca, encode_y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

print(sqrt(mean(absolute(scoresSQUARED))))