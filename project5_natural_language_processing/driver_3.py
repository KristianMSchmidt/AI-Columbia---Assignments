"""
NLP: Natural language processing.
Prediction of movie reviews. Supervised learning.
25000 labeled examples: 0 means negative review. 1 means positive review.
Test set consisting of 25000 labeled examples.
I use unigram and bigram models. Both models are based on occurances of words in review.
Unigram looks at single words (isolated), while bigram looks at 2 words following each other.
Neutral stopwords are removed.

"""
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


def load_stop_words():
    with open("stopwords.en.txt", "r") as f:
        stop_words = []
        for line in f:
            stop_words.append(line.strip())
    return stop_words


def some_examples(vectorizer, clf):
    bad_movie1 = vectorizer.transform(["Bad movie!"])
    bad_movie2 = vectorizer.transform(["Lousy low budget sentimental"])
    good_movie1 = vectorizer.transform(['Brilliant acters'])
    good_movie2 = vectorizer.transform(['Fabulous beautifull'])
    good_movie3 = vectorizer.transform(['Subtle characters'])
    medium_movie = vectorizer.transform([""])

    print(clf.predict(bad_movie1))  #expect 0
    print(clf.predict(bad_movie2))  #expect 0
    print(clf.predict(good_movie1)) #expect 1
    print(clf.predict(good_movie2)) #expect 1
    print(clf.predict(good_movie3)) #expect 1
    print(clf.predict(medium_movie))


    #Prdict confidence scores
    coef = clf.coef_            # coefficients in decision function
    intercept = clf.intercept_  #constant in decision function

    print(coef.dot(bad_movie1.toarray().T) + intercept)
    print(coef.dot(bad_movie2.toarray().T) + intercept)
    print(coef.dot(good_movie1.toarray().T) + intercept)
    print(coef.dot(good_movie2.toarray().T) + intercept)
    print(coef.dot(good_movie3.toarray().T) + intercept)
    print(coef.dot(medium_movie.toarray().T) + intercept)

    #Predict confidence scores  (if bigger than 0 -> prediction = 1, if less than 1 -> prediction = 0)
    print(clf.decision_function(bad_movie1))
    print(clf.decision_function(bad_movie2))
    print(clf.decision_function(good_movie1))
    print(clf.decision_function(good_movie2))
    print(clf.decision_function(good_movie3))
    print(clf.decision_function(medium_movie))



def train_set_accuracy(X_train, y_train, clf):
    predictions = clf.predict(X_train)
    error_rate = (sum(abs(predictions-y_train))/25000)*100
    print("Train set accuracy: {}%".format(100-error_rate))


def test_set_accuracy(X_test, y_test, clf):
    predictions = clf.predict(X_test)
    error_rate = (sum(abs(predictions-y_test))/25000)*100
    print("Test set accuracy: {}%".format(100-error_rate))

def compute_total_loss(y_train, X_train, clf):
    a = np.maximum(np.zeros((25000,)), 1 - clf.decision_function(X_train)*y_train)
    return (np.sum(a)/25000)

def unigram_model():
    """
    alpha = 0.0001 seems to be best. gives test score aroung 87%
    """
    print("Training and testing unigram model")
    #vectorizer = CountVectorizer(stop_words = stop_words)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data["text"])
    X_test =  vectorizer.transform(test_data["text"])
    clf = SGDClassifier(loss="hinge", penalty="l1", alpha = 0.0001)
    clf.fit(X_train, y_train)
    train_set_accuracy(X_train, y_train, clf) #print(clf.score(X_train, y_train))
    test_set_accuracy(X_test, y_test, clf)  #print(clf.score(X_test, y_test))
    #some_examples(vectorizer, clf)


    # SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
    #        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
    #        learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
    #        n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
    #        shuffle=True, tol=None, verbose=0, warm_start=False)
    #


def bigram_model():
    """
    Best alpha seems to be around e1-8
    """
    print("Training and testing bigram model")
    #bigram_vectorizer = CountVectorizer(stop_words = stop_words, ngram_range=(2, 2))
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words = stop_words)
    X_train = bigram_vectorizer.fit_transform(train_data["text"])
    X_test =  bigram_vectorizer.transform(test_data["text"])
    clf = SGDClassifier(loss="hinge", penalty="l1", alpha = 1e-8)
    clf.fit(X_train, y_train)
    train_set_accuracy(X_train, y_train, clf)
    test_set_accuracy(X_test, y_test, clf)
    #some_examples(bigram_vectorizer, clf)


def unigramtfidf_model():
    """
    Best alpha seems to be arond 0.00002
    """
    print("Training and testing Tf-idf unigram model")
    from sklearn.feature_extraction.text import TfidfVectorizer
    #vectorizer = TfidfVectorizer(stop_words = stop_words, ngram_range=(1, 1))
    vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    X_train = vectorizer.fit_transform(train_data["text"])
    X_test =  vectorizer.transform(test_data["text"])
    clf = SGDClassifier(loss="hinge", penalty="l1", alpha = 0.00002)
    clf.fit(X_train, y_train)
    train_set_accuracy(X_train, y_train, clf)
    test_set_accuracy(X_test, y_test, clf)
    some_examples(vectorizer, clf)
    #print("Total loss of final coefficients:", compute_total_loss(y_train, X_train, clf))


def bigramtfidf_model():
    """
    Best alpha = 0.000001
    """
    print("Training and testing Tf-idf bigram model")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words = stop_words)
    X_train = vectorizer.fit_transform(train_data["text"])
    X_test =  vectorizer.transform(test_data["text"])
    clf = SGDClassifier(loss="hinge", penalty="l1", alpha = 0.000001)
    clf.fit(X_train, y_train)
    train_set_accuracy(X_train, y_train, clf)
    test_set_accuracy(X_test, y_test, clf)
    #some_examples(vectorizer, clf)



if __name__ == "__main__":

    # Make list with stop words
    stop_words = load_stop_words()

    # load train data from csv
    train_data = pd.read_csv("imdb_tr.csv", sep=',', encoding = "ISO-8859-1")
    y_train = train_data["polarity"]

    # load test data from csv
    test_data = pd.read_csv("imdb_labeled_test_data.csv", sep=',', encoding = "ISO-8859-1")
    y_test = test_data["polarity"]

    #Unigram representation model
    unigram_model()

    #Bigram representation model
    #bigram_model()

    #Tf-idf unigram model:
    #unigramtfidf_model()

    # Tf-idf bigram model
    #bigramtfidf_model()
