import pickle

import nltk
import random
from nltk.corpus import movie_reviews
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify.scikitlearn import  SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from VoteClassifier import VoteClassifier

def read_from_directory(dir_name):
    os.chdir(dir_name)
    review_list = []
    for file in os.listdir():
        f = open(file, 'r',errors='ignore')
        if f.mode == 'r':
            review_list.append(f.read())
    return review_list


def get_labeled_data(list, is_truthful):
    labelled_data = []
    for message in list:
        labelled_data = labelled_data + [(message, is_truthful)]

    return labelled_data


deceitfulListPositive = read_from_directory(
    '/Users/Suiu/Desktop/op_spam_v1.4/positive_polarity/deceptive_from_MTurk/allfolds')

deceitfulListNegative = read_from_directory(
    '/Users/Suiu/Desktop/op_spam_v1.4/negative_polarity/deceptive_from_MTurk/allfolds'
)

truthfulFromWebNegative = read_from_directory(
    '/Users/Suiu/Desktop/op_spam_v1.4/negative_polarity/truthful_from_Web/allfolds'
)

truthfulListPositive = read_from_directory(
    '/Users/Suiu/Desktop/op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/allfolds'
)

labelled_data = get_labeled_data(deceitfulListPositive, False)
labelled_data = labelled_data\
                + get_labeled_data(deceitfulListNegative, False)\
                +get_labeled_data(truthfulFromWebNegative, True)\
                +get_labeled_data(truthfulListPositive, True)

allSentences = deceitfulListNegative + truthfulFromWebNegative + deceitfulListPositive + truthfulListPositive

allWords = nltk.word_tokenize(allSentences[0])

documents = labelled_data
random.shuffle(documents)

word_features = allWords[0:1000]

def get_most_common_words(text_set, number_of_words):
    return nltk.FreqDist(text_set, number_of_words)


def find_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

def get_n_grams_for_sentence(sentence,n):
    all_grams = []
    tokenised_sentence = nltk.word_tokenize(sentence)
    for i in range(len(tokenised_sentence)-n+1):
        all_grams.append(tokenised_sentence[i:i+n])
    return all_grams



featuresets = [(find_features(review), truthful) for (review, truthful) in documents]
training_set = featuresets[:1000]
testing_set = featuresets[1000:1400]
classifier_f = open("naivebayes.pickle","rb")
NB_classifier = pickle.load(classifier_f)
classifier_f.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

classifiers = [("NB_classifier",NB_classifier)
    ,("MNB_classifier",MNB_classifier)
    ,("BernoulliNB_classifier",BernoulliNB_classifier)
    ,("LogisticRegression_classifier",LogisticRegression_classifier)
    ,("SGDClassifier_classifier_",SGDClassifier_classifier)
    ,("SVC_classifier",SVC_classifier)
    ,("LinearSVC_classifier",LinearSVC_classifier)
    ,("NuSVC_classifier",NuSVC_classifier)]

for cl in classifiers:
    print(cl[0]+": " + str(nltk.classify.accuracy(cl[1], testing_set) * 100))
# classifier.show_most_informative_features(5)
voted_classifier = VoteClassifier(NB_classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

for sentence in allSentences:
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
