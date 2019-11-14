import nltk
import random
from nltk.corpus import movie_reviews
import os
from sklearn.feature_extraction.text import CountVectorizer
import time


def read_from_directory(dir_name):
    os.chdir(dir_name)
    review_list = []
    for file in os.listdir():
        f = open(file, 'r')
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
    words = get_n_grams_for_sentence(document[0],2)
    features = {}
    print(word_features)
    print(get_n_grams_for_sentence(word_features,2))
    for word in word_features:
        features[word] = (word in words)

    return features

def get_n_grams_for_sentence(sentence,n):
    all_grams = []
    tokenised_sentence = nltk.word_tokenize(sentence)
    for i in range(len(tokenised_sentence)-n+1):
        all_grams.append(tokenised_sentence[i:i+n])
    return all_grams


print(find_features(documents[0]))
# featuresets = [(find_features(review), truthful) for (review, truthful) in documents]
# training_set = featuresets[:800]
# testing_set = featuresets[800:1000]
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print(nltk.classify.accuracy(classifier, testing_set) * 100)
# classifier.show_most_informative_features(5)
