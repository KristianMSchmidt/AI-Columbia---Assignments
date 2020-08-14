import pandas as pd
import numpy as np
import os
import re

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation
train_path = "train/"
polarity_dict = {"neg":"0", "pos":"1"}


with open("stopwords.en.txt", "r") as f:
    stopwords = []
    for line in f:
        stopwords.append(line.strip())

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

count = 0
def make_data():
    global count
    print ("Loading review data from files")
    with open("imdb_tr.csv", "w") as csv:

        for pol in ["neg", "pos"]:
            path = train_path + pol + "/"
            directory = os.fsencode(path)
            for file in os.listdir(directory):
                count += 1
                if count %1000 == 0:
                    print (count)
                filename = os.fsdecode(file)
                with open (path + filename, "r", errors = "ignore") as myfile:

                    review = myfile.readlines()[0].lower()
                    #print(review)
                    review = cleanhtml(review)
                    #print("")
                    for char in "'`_":
                        review = review.replace(char, '')
                    for char in "?{!}*()[]#-<>=~+^\/.,:;\"":
                        review = review.replace(char, ' ')

                    review = re.sub(r'[0-9]+', 'number', review)

                    #print(review)
                    #print("")
                    current_review = []
                    for word in review.split():
                        if not word in stopwords:
                            current_review.append(word)
                    #print(current_review)

                    current_review = " ".join(current_review)
                    #print(current_review)
                    csv.write(current_review + "," + polarity_dict[pol] + "\n")
#make_data()


def make_dictionary(csv):
    dictionary = set()
    num_reviews = 0

    with open("imdb_tr.csv") as csv:
        #Number of lines in csv
        for line in csv:
            num_reviews += 1
            words = line.split(",")[0].split()
            dictionary.update(words)
    import random
    dictionary = sorted(dictionary)
    print(dictionary)
    input("p")
    return dictionary, num_reviews

make_dictionary(3)

def unigram(review, dictionary):

    vector = np.zeros((1, len(dictionary)))
    words = review.split()
    for i, dict_word in enumerate(dictionary):
        if dict_word in words:
            vector[0][i] += 1
    return vector


def make_review_vectors(review_number):
    count = 0
    dictionary, num_reviews = make_dictionary(3)
    print("num reviews", num_reviews)
    with open("imdb_tr.csv") as csv:
        for line in csv:
            count += 1
            print(count)
            review = line.split(",")[0]
            #print(review)
            review_vector = unigram(review, dictionary)
            #print(np.sum((review_vector)))
            #for i in range(len(dictionary)):
                #if review_vector[0][i] != 0:
                #    print (dictionary[i])

make_review_vectors(3)

from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(X, y)

SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)


def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    """
    Implement this module to extract and combine text files under train_path directory into
    imdb_tr.csv. Each text file in train_path should be stored
    as a row in imdb_tr.csv. And imdb_tr.csv should have two columns, "text" and label
    """
    pass

if __name__ == "__main__":
    """
    train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt
    """
    pass
