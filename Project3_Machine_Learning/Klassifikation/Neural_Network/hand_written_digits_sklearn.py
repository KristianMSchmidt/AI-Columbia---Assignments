"""
Handwritten image recognition.
Solution using Sklearn's library for neural nets. Insanely easy to use.
"""
# Imports
from __future__ import division
import numpy as np
import scipy.io

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

## Load Training Data
#print 'Loading and Visualizing Data ...\n'

mat = scipy.io.loadmat('ex3data1.mat')
y = mat["y"].flatten()
X = mat["X"]
m, n = X.shape  # m = 5000, n = 400


# ==========Divide data into 60% training data and 40% test data ==============
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# ================= A first naive attempt ===========================
from sklearn.neural_network import MLPClassifier   #multiple layer perceptron
clf = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(10  ))
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print "Test set accuracy in first naive attempt:", accuracy_score(y_test, clf.predict(X_test), normalize=True)
print "Train set accuracy in first naive attempt:", accuracy_score(y_train, clf.predict(X_train), normalize=True)


# ====== Use 5-fold cross-validation on the training data (the 60%) to find best parameters=============
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier   #multiple layer perceptron

# Set the parameters by cross-validation
parameters = [{'hidden_layer_sizes':[(25), (25,5), 20], 'alpha':[2,3,4]}]
nn = MLPClassifier()


clf = GridSearchCV(nn, parameters, cv = 5) #CV is number of (stratisfied ) folds in k-fold cross val
clf.fit(X_train, y_train.flatten())

print "Best parameters", clf.best_params_
print "Best test score during cross validation on the 60%", clf.best_score_
#print clf.best_estimator_
#print pd.DataFrame.from_dict(clf.cv_results_, orient="columns")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
           % (mean, std * 2, params))





#  ============== Fit final model using best found parameters and calculate test cted best model =====================
nn = clf.best_estimator_
nn.fit(X_train, y_train)
print "Final test set accuracy:", accuracy_score(y_test, nn.predict(X_test), normalize=True)
print "Final train set accuracy:", accuracy_score(y_train, nn.predict(X_train), normalize=True)
