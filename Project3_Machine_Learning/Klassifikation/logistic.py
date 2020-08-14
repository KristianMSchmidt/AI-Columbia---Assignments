    """
Logistic classification (linear decision boundary) using Sklearn
I tune parameters using Sklearn
I use Sklearn GridSearch with 5-fold stratisfied cross validation to find best values for X.

Read THIS before I proceed: http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
Also read Bogdanovich answer here:  https://stats.stackexchange.com/questions/52274/how-to-choose-a-predictive-model-after-k-fold-cross-validation
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('input3.csv')

X = np.array(data)[:,:2]
y = np.array(data)[:,[2]]

plt.scatter(X[:, 0], X[:, 1], c = y.flatten(), s=25, cmap=plt.cm.Paired)
#plt.scatter(X[:,[0]][y==1], X[:,[1]][y==1], label = "1")
#plt.scatter(X[:,[0]][y==0], X[:,[1]][y==0], label = "0")
#plt.legend()
plt.xlabel("A")
plt.ylabel("B")


# ==========Divide data into 60% training data and 40% test data ==============
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]




# ====== Use 5-fold cross-validation on the training data (the 60%) to find best parameters=============

from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

#clf = LogisticRegression()
#clf.fit(X_train, y_train.flatten())
#print clf.predict(np.array([[0,0]]))
#print clf.score(X_test, y_test)


# Set the parameters by cross-validation
parameters = [{'C':[0.1, 0.5, 1, 5, 10, 50, 100]}]

clf = GridSearchCV(LogisticRegression(), parameters, cv=5) #CV is number of (stratisfied ) folds in k-fold cross val
clf.fit(X_train, y_train.flatten())

print "Best parameters", clf.best_params_
best_C = clf.best_params_["C"]

print "Best training score", clf.best_score_
#print clf.best_estimator_
#print pd.DataFrame.from_dict(clf.cv_results_, orient="columns")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
           % (mean, std * 2, params))

#  ================Compute test set accuracy of selected best model =====================
from sklearn.metrics import accuracy_score

clf = clf.best_estimator_
clf.fit(X_train, y_train.flatten())
predictions = clf.predict(X_test)

print "Final test set accuracy:", accuracy_score(y_test, predictions, normalize=True)


##  =============== Plot decision boundary  =====================

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0]-1, xlim[1]+1, 30)
yy = np.linspace(ylim[0]-1, ylim[1]+1, 30)

YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
CS = ax.contour(XX, YY, Z, colors='k', levels=[0.5])
plt.clabel(CS, inline=1, fontsize=10)


plt.title("LogisticRegression Decision Boundary \n C={}".format(best_C))
plt.show()

print clf.predict(X_test)
