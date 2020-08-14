"""
I use Sklearn GridSearch stratisfied cross validation to find best parameter values.
Read THIS before I proceed: http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
Also read Bogdanovich answer here:  https://stats.stackexchange.com/questions/52274/how-to-choose-a-predictive-model-after-k-fold-cross-validation
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier


# load train data from csv
train_data = pd.read_csv("imdb_tr.csv", sep=',', encoding = "ISO-8859-1")
X_train = train_data["text"]
y_train = train_data["polarity"]

# load cross-valiaton (cv) and test data from csv
test_data = pd.read_csv("imdb_labeled_test_data.csv", sep=',', encoding = "ISO-8859-1")
X_test = test_data["text"]
y_test = test_data["polarity"]


# print("Training and testing unigram model")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X_train = vectorizer.fit_transform(train_data["text"])
X_test =  vectorizer.transform(test_data["text"])


from sklearn.model_selection import GridSearchCV
# Set the parameters by cross-validation
parameters = [{'alpha':[0.00005, 0.00002, 0.00001]}]

clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="l1"), parameters, cv = 20)
clf.fit(X_train, y_train)

print ("Best parameters", clf.best_params_)
print ("Best score", clf.best_score_)
print ("Best estimator", clf.best_estimator_)


final_clf = clf.best_estimator_
final_clf.fit(X_train, y_train)

# train_set_accuracy
print("Train accurary:", final_clf.score(X_train, y_train))

# Final test set accuracy
print("Test score", final_clf.score(X_test, y_test))
