"""
Below I've tried to implement stochastic gradient descent from scratch. Though without
the regularization termsn. It seemsto work - but slowly. Scikit learn is really impressively fast.
"""
import sys
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

PATH = "C:/Users/kimar/Google Drev/Python programmer/AI Columbia - Assignments/project5_natural_language_processing/"

def load_stop_words():
    with open(PATH + "stopwords.en.txt", "r") as f:
        stop_words = []
        for line in f:
            stop_words.append(line.strip())
    return stop_words

def predict(x, coefs, bias):
    if decision_function(x, coefs, bias) >= 0:
        return 1
    else:
        return 0

def decision_function(x, coefs, bias):
    return(x.dot(coefs) + bias)

def compute_total_loss(y_train, X_train, coefs, bias):
    coefs = np.asarray(coefs).reshape(75391,1)
    a = np.maximum(np.zeros((25000,1)), 1 - decision_function(X_train, coefs, bias)*y_train)
    return (np.sum(a)/25000)

def gradient(zero_coefs, current_coefs, bias, y, x):
    decision = float(y*decision_function(x, current_coefs, bias))

    if decision < 1:
        coefs_gradient = (int(-y)*x).T
    else:
        coefs_gradient = zero_coefs
    bias_gradient = -y

    return bias_gradient, coefs_gradient

def SGD(X_train, y_train):

    learning_rate = 0.00040
    zero_coefs = np.zeros((75391,1))
    current_coefs = np.zeros((75391,1))
    bias = 0

    print("original loss", compute_total_loss(y_train, X_train, current_coefs, bias))

    #print(X_train[i].shape)
    #for i in range(10):
    #    print(decision_function(X_train[i], coefs, bias))
    #    print(predict(X_train[i], coefs, bias))
    for epoch in range(100):
        for i in range(25000):
           bias_gradient, coefs_gradient = gradient(zero_coefs, current_coefs, bias, y_train[i], X_train[i])
           current_coefs -= learning_rate * coefs_gradient
           bias -= learning_rate * bias_gradient
        print("Loss after {} epochs:".format(epoch+1), compute_total_loss(y_train, X_train, current_coefs, bias))

if __name__ == "__main__":

    # Make list with stop words
    stop_words = load_stop_words()

    # load train data from csv
    train_data = pd.read_csv(PATH + "imdb_tr.csv", sep=',', encoding = "ISO-8859-1")
    y_train = train_data["polarity"]
    y_train = np.array(y_train).reshape((25000,1))
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1

    # load test data from csv
    test_data = pd.read_csv(PATH + "imdb_labeled_test_data.csv", sep=',', encoding = "ISO-8859-1")
    y_test = test_data["polarity"]

    # Unigram representation model
    #print("Training and testing unigram model")
    #vectorizer = CountVectorizer(stop_words = stop_words)
    #X_train = vectorizer.fit_transform(train_data["text"])
    #X_test =  vectorizer.transform(test_data["text"])

    #SGD(X_train, y_train)

    # Unigram Tf-idf model
    print("Training and testing Tf-idf unigram model")
    vectorizer = TfidfVectorizer(stop_words = stop_words, ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(train_data["text"])
    X_test =  vectorizer.transform(test_data["text"])
    SGD(X_train, y_train)
