import pickle
import re
from collections import defaultdict

import textstat
from spellchecker import SpellChecker
from nltk.corpus import stopwords
import nltk
import random
import os
import re
import pandas as pd
from isPassiveVoice import Tagger
from nltk.classify.scikitlearn import  SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from textblob import TextBlob, Word
from string import punctuation
from VoteClassifier import VoteClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


yelpDeceitful = []
yelpGenuine = []
yelpFileMetaNYC = open('/Users/Suiu/PycharmProjects/nlptest/YelpNYC/metadata','r',10,'utf 8')
dict1={}
for line in yelpFileMetaNYC:
    separated_line= line.split("\t")
    if(separated_line[3] == '1'):
        dict1[separated_line[0]] = True
    else :
        dict1[separated_line[0]] = False
yelpFileReviewContentNYC = open('/Users/Suiu/PycharmProjects/nlptest/YelpNYC/reviewContent','r',10,'utf 8')
dict2={}
for line in yelpFileReviewContentNYC:
    separated_line= line.split("\t")
    dict2[separated_line[0]] = separated_line[3].replace("\xa0",'').replace("\n",'')

for key in dict2.keys():
    if (dict1 == True):
        yelpGenuine.append(dict2[str(key)])
    else:
        yelpDeceitful.append(dict2[str(key)])

yelpFileMetaNYC = open('/Users/Suiu/PycharmProjects/nlptest/YelpZIP/metadata','r',10,'utf 8')
dict1={}
for line in yelpFileMetaNYC:
    separated_line= line.split("\t")
    if(separated_line[3] == '1'):
        dict1[separated_line[0]] = True
    else :
        dict1[separated_line[0]] = False
yelpFileReviewContentNYC = open('/Users/Suiu/PycharmProjects/nlptest/YelpZIP/reviewContent','r',10,'utf 8')
dict2={}
for line in yelpFileReviewContentNYC:
    separated_line= line.split("\t")
    dict2[separated_line[0]] = separated_line[3].replace("\xa0",'').replace("\n",'')

for key in dict2.keys():
    if(dict1 == True):
     yelpGenuine.append(dict2[str(key)])
    else:
        yelpDeceitful.append(dict2[str(key)])

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

allWords =[]
for sentence in allSentences:
    allWords+=nltk.word_tokenize(sentence)

documents = labelled_data
f_out = open("C:/Users/Suiu/PycharmProjects/nlptest/test.txt","w+")
for doc in documents:
    f_out.write(doc[0])
    f_out.write("")
f_out.close()


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

featuresets = [(find_features(review), truthful) for (review, truthful) in documents]
training_set = featuresets[:1000]
testing_set = featuresets[1000:1400]
classifier_f = open("naivebayes.pickle","rb")
NB_classifier = pickle.load(classifier_f)
classifier_f.close()
classifier = nltk.NaiveBayesClassifier.train(training_set)
# classifier.show_most_informative_features(30)

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
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
docs = map(lambda x:x[0],documents)


def special_character_removal(s):
    stripped = re.sub('[^a-zA-z\s]', '', s)
    stripped = re.sub('_', '', stripped)

    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
        return stripped.lower()

#we are preparing to extract the features of deceitful text
deceitfulList= list(map(lambda y: y[0],filter(lambda x: x[1]==True,documents)))


stop_words = set(stopwords.words('english'))

filtered_deceitful_sentences = []
for text in deceitfulList:
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence= " ".join([w for w in word_tokens if not w in stop_words])
    filtered_deceitful_sentences.append(filtered_sentence)

def n_grams_ranking(data,n):
    vectorizer_trigrams = CountVectorizer(ngram_range=(n, n))
    x1 = vectorizer_trigrams.fit_transform(data)
    features = (vectorizer_trigrams.get_feature_names())

    # tfidf
    vectorizer_trigrams = TfidfVectorizer(ngram_range=(n, n))
    x2 = vectorizer_trigrams.fit_transform(deceitfulList)
    scores = (x2.toarray())
    sums = x2.sum(axis=0)

    data = []
    for col, term in enumerate(features):
        data.append((term, sums[0, col]))

    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    words = (ranking.sort_values('rank', ascending=False))
    print("\n\nWords head : \n", words.head(15))

# n_grams_ranking(filtered_deceitful_sentences,2)
# n_grams_ranking(filtered_deceitful_sentences,3)

pos_deceptive_list = []
def to_pos(txt):
    return map(lambda tuple: tuple[1],list(nltk.pos_tag(nltk.word_tokenize(txt))))
for text in deceitfulList:
    pos_deceptive_list.append(" ".join(list(to_pos(text))))
    tally = defaultdict(int)
    for i in text.split():
        tally[i] += 1
#
# n_grams_ranking(pos_deceptive_list,1)
# n_grams_ranking(pos_deceptive_list,2)
# n_grams_ranking(pos_deceptive_list,3)
tagger = Tagger()
spell = SpellChecker()

number_of_nouns_list = []
number_of_verbs_list = []
typographical_error_ratio = []
redundancy_list = []
passive_voice_sentences_ratio_list = []
pausality_list = []
nr_of_noun_phrases_list = []
polarity_list = []
emotiveness_list = []
subjectivity_list = []
number_of_determiners_list = []
number_of_possesive_list = []
average_sentence_length_list = []
average_word_length_list = []
lexical_diversity_list = []
ease_of_read_rating_list = []
label_list = []
works = 0
print('starting')
for text in deceitfulList:
    analysis = TextBlob(text)
    number_of_words = len(analysis.words)
    number_of_nouns = 0
    number_of_verbs = 0
    number_of_personal_terms = 0
    number_of_determiners = 0
    punct_number = 0
    function_words = 0
    nr_of_sentences = len(analysis.sentences)
    nr_of_passive_voice_sentences = 0
    number_of_adjectives_and_adverbs = 0
    number_of_possesive = 0
    number_of_mispelled_words = 0

    for sentence in analysis.sentences:
        if(tagger.is_passive(str(sentence))):
            nr_of_passive_voice_sentences += 1
    for letter in text:
        if (letter in punctuation):
            punct_number += 1
    for word in analysis.words:
        if(str(word) in set(stopwords.words('english'))):
            function_words += 1
        check_spell = Word(word)
        #print(check_spell.spellcheck()[0])
        if(check_spell.spellcheck()[0][1]<1):
            number_of_mispelled_words += 1

    #print(number_of_mispelled_words/len(analysis.words))
    typographical_error_ratio.append(number_of_mispelled_words/number_of_words)
   # print(function_words)
    redundancy = function_words/nr_of_sentences
    redundancy_list.append(redundancy)
    #print(nr_of_passive_voice_sentences/nr_of_sentences)
    passive_voice_sentences_ratio = nr_of_passive_voice_sentences/nr_of_sentences
    passive_voice_sentences_ratio_list.append(passive_voice_sentences_ratio)
    #print(spell.unknown((nltk.word_tokenize(text))))
    #print("pausality: ",punct_number/nr_of_sentences)
    pausality = punct_number / nr_of_sentences
    pausality_list.append(pausality)

    #print("noun phrases: ",len(analysis.noun_phrases))
    nr_of_noun_phrases = len(analysis.noun_phrases)
    nr_of_noun_phrases_list.append(nr_of_noun_phrases)
   # print("polarity : ",analysis.sentiment.polarity)
    polarity = analysis.polarity
    polarity_list.append(polarity)
    #print(analysis.sentiment.subjectivity)
    subjectivity = analysis.subjectivity
    subjectivity_list.append(subjectivity)
    for tag in analysis.pos_tags:
        if(tag[1] == 'JJ' or tag[1] == 'JJR' or tag[1] == 'JJS' or tag[1] == 'RB' or tag[1] == 'RBR' or tag[1] == 'RBS' or tag[1] == 'WRB') :
            number_of_adjectives_and_adverbs += 1
        if(tag[1] == 'POS' or tag[1] == 'PRP$' or tag[1] =='WP$'):
            number_of_possesive += 1
        if(tag[1] == 'DT' or tag[1] == 'WDT' ):
            number_of_determiners += 1
        if(tag[1] == 'VB' or tag[1] =='VBD' or tag[1] == 'VBG'
        or tag[1] == 'VBN' or tag[1] =='VBP' or tag[1] == 'VBZ'):
            number_of_verbs += 1
        if(tag[1]=='NN' or tag[1]=='NNS'):
            number_of_nouns += 1

    number_of_verbs_list.append(number_of_verbs)
    number_of_nouns_list.append(number_of_nouns)

    emotiveness = number_of_adjectives_and_adverbs/(number_of_verbs+number_of_nouns)
    emotiveness_list.append(emotiveness)
    number_of_determiners_list.append(number_of_determiners/nr_of_sentences)
    number_of_possesive_list.append(number_of_possesive/number_of_words)

    #print("nr of words / nr of sentences : ",len(analysis.words)/ nr_of_sentences)
    average_sentence_length = number_of_words/ nr_of_sentences
    average_sentence_length_list.append(average_sentence_length)
    #print("nr of characters/nr of words",len(text)/len(analysis.words))
    average_word_length = len(text)/number_of_words
    average_word_length_list.append(average_word_length)
    #print("number of words:",textstat.lexicon_count(text, removepunct=True))
    number_of_different_words = textstat.lexicon_count(text, removepunct=True)
    lexical_diversity = number_of_different_words / number_of_words
    lexical_diversity_list.append(lexical_diversity)
    #print("number of sentences:",nr_of_sentences)
    #contains Dale-Chall ,The Coleman-Liau,Automated Readability
    #print("ease of read rating:",textstat.text_standard(text, float_output=False))
    ease_of_read_rating = textstat.text_standard(text, float_output=False)
    ease_of_read_rating_list.append(ease_of_read_rating)
    label_list.append(False)
    works += 1
    print(works)
print('exit')
dct={'number_of_nouns': number_of_nouns_list ,
'number_of_verbs': number_of_verbs_list ,
'typographical_error_ratio':typographical_error_ratio,
'redundancy':redundancy_list,
'passive_voice':passive_voice_sentences_ratio_list,
'pausality':pausality_list ,
'nr_of_noun_phrases' :nr_of_noun_phrases_list,
'polarity':polarity_list,
'emotiveness':emotiveness_list ,
'subjectivity':subjectivity_list,
'number_of_determiners':number_of_determiners_list,
'number_of_possesive':number_of_possesive_list,
'average_sentence_length':average_sentence_length_list,
'average_word_length':average_word_length_list,
'lexical_diversity':lexical_diversity_list,
'ease_of_reading_rating':ease_of_read_rating_list,
'label':label_list
}
print('test')
df = pd.DataFrame(dct)
print(df)