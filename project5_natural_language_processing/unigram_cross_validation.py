"""
I use Sklearn GridSearch stratisfied cross validation to find best parameter values.
Read THIS before I proceed: http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
Also read Bogdanovich answer here:  https://stats.stackexchange.com/questions/52274/how-to-choose-a-predictive-model-after-k-fold-cross-validation
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

with open("stopwords.en.txt", "r") as f:
    stop_words = []
    for line in f:
        stop_words.append(line.strip())

# load train data from csv
train_data = pd.read_csv("imdb_tr.csv", sep=',', encoding = "ISO-8859-1")
X_train = train_data["text"]
y_train = train_data["polarity"]

# load cross-valiaton (cv) and test data from csv
test_data = pd.read_csv("imdb_labeled_test_data.csv", sep=',', encoding = "ISO-8859-1")
X_test = test_data["text"]
y_test = test_data["polarity"]


# print("Training and testing unigram model")
vectorizer = CountVectorizer(stop_words = None)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


from sklearn.model_selection import GridSearchCV

# Set the parameters by cross-validation
#parameters = [{'alpha':[0.00001, 0.000005, 0.0000025, 0.000001, 0.0000005, 0.00000025, 0.0000001, 0.00000005, 0.000000025]}]
parameters = [{'alpha':[0.00003, 0.00002, 0.0001, 0.00001]}]

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
