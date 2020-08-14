"""
Code to read all train and test reviews from text files on my local computer to csv-files.
"""
import os

def make_train_csv():
    print ("Loading 25000 training reviews into csv file...")

    with open("imdb_tr.csv", "w") as csv:
        csv.write("row_number,text,polarity\n")

        #Negative reviews
        directory = os.fsencode("train/neg")
        row_number = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            with open ("train/neg/" + filename, "r", errors = "ignore") as myfile:
                review = myfile.readlines()[0]
                review = review.replace(",", ' ')
                csv.write(str(row_number) + "," + review + "," + "0" + "\n")
            row_number += 1

        #Positive reviews
        directory = os.fsencode("train/pos")
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            with open ("train/pos/" + filename, "r", errors = "ignore") as myfile:
                review = myfile.readlines()[0]
                review = review.replace(",", ' ')
                csv.write(str(row_number) + "," + review + "," + "1" + "\n")
            row_number += 1


def make_test_csv():
    print ("Loading 25000 test reviews into csv file...")

    with open("imdb_labeled_test_data.csv", "w") as csv:
        csv.write("row_number,text,polarity\n")

        #Negative reviews
        directory = os.fsencode("test/neg")
        row_number = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            with open ("test/neg/" + filename, "r", errors = "ignore") as myfile:
                review = myfile.readlines()[0]
                review = review.replace(",", ' ')
                #print(review)
                #input("pause")
                csv.write(str(row_number) + "," + review + "," + "0" + "\n")

            row_number += 1

        #Positive reviews
        directory = os.fsencode("test/pos")
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            with open ("test/pos/" + filename, "r", errors = "ignore") as myfile:
                review = myfile.readlines()[0]
                review = review.replace(",", ' ')
                csv.write(str(row_number) + "," + review + "," + "1" + "\n")
            row_number += 1

make_train_csv()
make_test_csv()
