import pandas as pd
import numpy as np
import os

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation

def make_csv(local_computer = True, train = True):
    print ("Loading reviews into csv")
    polarity_dict = {"neg":"0", "pos":"1"}

    if local_computer:
        if train:
            path = "train/"
            outputfilename = "imdb_tr.csv"

        else:
            path = "test/"
            outputfilename = "imdb_te.csv"

    else:
        pass

    count = 0
    with open(outputfilename, "w") as csv:
        csv.write("text, polarity\n")
        for pol in ["neg", "pos"]:
            path = path + pol + "/"
            directory = os.fsencode(path)
            for file in os.listdir(directory):
                count += 1
                if count %1000 == 0:
                    print (count)
                filename = os.fsdecode(file)
                with open (path + filename, "r", errors = "ignore") as myfile:
                    review = myfile.readlines()[0]
                    review = review.replace(",", ' ')
                    csv.write(review + "," + polarity_dict[pol] + "\n")

make_csv(local_computer=True, train=True)
#make_csv(local_computer=True, train=False)

with open("stopwords.en.txt", "r") as f:
    stopwords = []
    for line in f:
        stopwords.append(line.strip())
#print(stopwords)

df = pd.read_csv("imdb_tr_version2.csv", sep=',', encoding = "ISO-8859-1")
reviews = df["text"]
y_train = df["polarity"]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = stopwords)

X_train = vectorizer.fit_transform(reviews)
print(X.shape)
#print(type(X))

#import scipy.sparse
#scipy.sparse.save_npz('sparse_matrix.npz', X)
#X = scipy.sparse.load_npz('sparse_matrix.npz')
#print(vectorizer.get_feature_names())

#analyze = vectorizer.build_analyzer()

#print(analyze("This, #is a text$ document to analyze."))
#print(vectorizer.vocabulary_.get('asshole'))

# print(vectorizer.transform(['Something completely new.']).toarray())
# print(vectorizer.transform(['This is a test document.']).toarray())


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(X_train, y_train)

bad_movie1 = vectorizer.transform(["Bad movie!"])
good_movie1 = vectorizer.transform(['Brilliant acters'])
good_movie2 = vectorizer.transform(['Fabulous beautifull'])
good_movie3 = vectorizer.transform(['Subtle characters'])
bad_movie2 = vectorizer.transform(["Lousy low budget sentimental"])

print(clf.predict(bad_movie1))  #expect 0
print(clf.predict(bad_movie2))  #expect 0

print(clf.predict(good_movie1)) #expect 1
print(clf.predict(good_movie2)) #expect 1
print(clf.predict(good_movie3)) #expect 1


# correct = 0
# print("counting...")
# for i in range(25000):
#     if int(clf.predict(X_train[i])) == y_train[i]:   #91% train set accuracy
#         correct +=1
# print(correct/25000)
#


# SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
#        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
#        learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
#        n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
#        shuffle=True, tol=None, verbose=0, warm_start=False)
#



# def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
#     """
#     Implement this module to extract and combine text files under train_path directory into
#     imdb_tr.csv. Each text file in train_path should be stored
#     as a row in imdb_tr.csv. And imdb_tr.csv should have two columns, "text" and label
#     """
#     pass
#
# if __name__ == "__main__":
#     """
#     train a SGD classifier using unigram representation,
#     predict sentiments on imdb_te.csv, and write output to
#     unigram.output.txt
#     """
#     AND NORE::::::: pass
