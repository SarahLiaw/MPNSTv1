from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.model_selection import cross_val_score
from numpy import *

import sys

from sklearn.tree import DecisionTreeClassifier

from pipeline.data_path.data_transformation import *

sys.path.insert(0, '/home/sarahl/PycharmProjects/MPNST_v1/pipeline/data_path')

path = '/home/sarahl/PycharmProjects/MPNST_v1/data_v1/MPNST_v1_rm2otlr.csv'
data = read_data(path)

X = get_X(data)
y = get_y(data)

y = encode_target(y)
count = 0.05
# for i in range(1, 20):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=count, random_state=101)
#     print(count, "training mean is", y_train.mean())
#     print(count, "testing mean is", y_test.mean())
#     count += 0.05


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print("training mean is", y_train.mean())
print("testing mean is", y_test.mean())

# base estimator: a weak learner with max_depth=2
shallow_tree = DecisionTreeClassifier(max_depth=2, random_state = 100)
# fit the shallow decision tree
shallow_tree.fit(X_train, y_train)

# test error
y_pred = shallow_tree.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print(score)

estimators = list(range(1, 50, 3))

abc_scores = []
for n_est in estimators:
    ABC = AdaBoostClassifier(
        base_estimator=shallow_tree,
        n_estimators=n_est)

    ABC.fit(X_train, y_train)
    y_pred = ABC.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    abc_scores.append(score)
print(abc_scores)
print("Max", max(abc_scores))

plt.plot(estimators, abc_scores)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.ylim([0.55, 1])
plt.show(block=True)
plt.interactive(False)

# Checking acuracy:
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Kappa Stats:",metrics.cohen_kappa_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
# print("Recall:",metrics.recall_score(y_test, y_pred))
# print("Mean Absolute Error:",metrics.mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error:",metrics.mean_squared_error(y_test, y_pred))
# print("F-Measure:",metrics.recall_score(y_test, y_pred))