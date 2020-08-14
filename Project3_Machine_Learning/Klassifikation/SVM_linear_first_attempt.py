"""
SVM classification with "linear kernel" (that is, no kernel) using Sklearn.


I use 5-fold stratisfied cross validation (which is very simple using Sklearn!) to find best parameters.
Note that this is not the standard use of cross validation (normally cross validation is used on the whole data set
to estimate the out-of-sample error of the model (insted of using a separate test-set))

Read THIS before I proceed: http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
Also read Bogdanovich answer here:  https://stats.stackexchange.com/questions/52274/how-to-choose-a-predictive-model-after-k-fold-cross-validation
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('input3.csv')
print data.head()

X = np.array(data)[:,:2]
y = np.array(data)[:,[2]]

#plt.scatter(X[:,[0]][y==1], X[:,[1]][y==1])
#plt.scatter(X[:,[0]][y==0], X[:,[1]][y==0])
plt.scatter(X[:, 0], X[:, 1], c = y.flatten(), s=25, cmap=plt.cm.Paired)

plt.xlabel("A")
plt.ylabel("B")
#plt.show()


# ==========Divide data into 60% training data and 40% test data ==============
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# ====== Use 5-fold cross-validation on the training data (the 60%) to find best parameters=============
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X_train, y)

from sklearn import svm
from sklearn.metrics import accuracy_score

average_scores = []
for C in [0.1, 0.5, 1, 5, 10, 50]:
    scores = []
    for train_index, test_index in skf.split(X, y):
        #print "\nTRAIN:", train_index
        #print "\nTEST:", test_index
        X_cv_train, X_cv_test = X[train_index], X[test_index]
        y_cv_train, y_cv_test = y[train_index], y[test_index]

        clf = svm.SVC(C=C, kernel='rbf')
        clf.fit(X_cv_train, y_cv_train.flatten())
        predictions = clf.predict(X_cv_test)
        scores.append(accuracy_score(y_cv_test, predictions, normalize=True))

    mean_score = np.mean(scores)
    average_scores.append(mean_score)
    print "Test accuracy with C={}, computed with 5-fold cross validation: {}".format(C, mean_score)

print average_scores


# ================Compute test set accuracy=====================
predictions = clf.predict(X_test)
print "Test set accuracy:", accuracy_score(y_test, predictions, normalize=True)
