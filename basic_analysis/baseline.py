import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

DATAPATH = "../emotion_bert/data/daily_dialog_single/csv/"

train = pd.read_csv(DATAPATH+"train.csv")
train = train.drop(columns=["act"])

valid = pd.read_csv(DATAPATH+"validation.csv")
valid = valid.drop(columns=["act"])

test = pd.read_csv(DATAPATH+"test.csv")
test = test.drop(columns=["act"])


tiv = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',stop_words= 'english')

train_features = tiv.fit_transform(train.current_dialog).toarray()
train_labels = train.emotion

valid_features = tiv.transform(valid.current_dialog).toarray()
valid_labels = valid.emotion




def compute_accuracy(true, pred):
    print("Accuracy: ", accuracy_score(true, pred))

def compute_f1(true, pred):
    print("F1: ", f1_score(true, pred, average="macro"))



print("NAIVE BAYES")

clf = MultinomialNB().fit(train_features, train_labels)
valid_predict = clf.predict(valid_features)

compute_accuracy(valid_labels, valid_predict)
compute_f1(valid_labels, valid_predict)

print("\n\n")


print("LOGISTIC REGRESSION")

clf = LogisticRegression(solver="newton-cg", multi_class="ovr").fit(train_features, train_labels)
valid_predict = clf.predict(valid_features)

compute_accuracy(valid_labels, valid_predict)
compute_f1(valid_labels, valid_predict)

print("\n\n")


print("SVC")

clf = LinearSVC(multi_class="ovr").fit(train_features, train_labels)
valid_predict = clf.predict(valid_features)

compute_accuracy(valid_labels, valid_predict)
compute_f1(valid_labels, valid_predict)

print("\n\n")


test_features = tiv.transform(test.current_dialog).toarray()
test_labels = test.emotion

test_predict = clf.predict(test_features)

compute_accuracy(test_labels, test_predict)
compute_f1(test_labels, test_predict)

print(classification_report(test_labels, test_predict))
