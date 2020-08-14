"""
Random forest classification using Sklearn

I use Sklearn GridSearch with 5-fold stratisfied cross validation to find best parameter values.

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
from sklearn.ensemble import RandomForestClassifier

# Set the parameters by cross-validation
parameters = [{'max_depth': range(1,51), 'min_samples_split': range(2,11)}]
forest = RandomForestClassifier(max_depth=2, random_state=0)


clf = GridSearchCV(forest, parameters, cv = 5) #CV is number of (stratisfied ) folds in k-fold cross val
clf.fit(X_train, y_train.flatten())

print "Best parameters", clf.best_params_

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


##  =============== Plot classification  =====================
