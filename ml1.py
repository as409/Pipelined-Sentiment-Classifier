
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement

# Requires NLTK. I used the nltk 3.0a
from nltk import bigrams
from nltk import trigrams

# Import classifier libraries
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from operator import itemgetter
import re
import csv


import tokenizer
import codecs


import nltk
from nltk import ConfusionMatrix



# polarity from lexicon used in the feature set
from LexiconClassifier import LexiconClassifier






f=codecs.open('first_edit.tsv','r+',encoding='utf8')
lines=f.readlines()
message=[]
sentiment=[]
tweets=[]
for line in lines:
    x=line.split('\t')
    a=tokenizer.tokenize(x[3])
    b=nltk.pos_tag(a)
    message.append(b)
    sentiment.append(x[1])

for i in range(len(message)):
    d=(message[i],sentiment[i])
    tweets.append(d)

f=codecs.open('twitter-dev-gold-A.tsv','r+',encoding='utf8')
lines=f.readlines()
message1=[]
sentiment1=[]
tweets1=[]
for line in lines:
    x=line.split('\t')
    a=tokenizer.tokenize(x[5])
    b=nltk.pos_tag(a)
    message1.append(b)
    sentiment1.append(x[4])

for i in range(len(message1)):
    d1=(message1[i],sentiment1[i])
    tweets1.append(d1)



tweets=tweets+tweets1

trainset=tweets

#### Provides a Machine Learning Sentiment Analysis classifier
class MachineLearningClassifier(object):
    

    # Constructor
    def __init__(self, tweet):
        print ('Loading training modules')
        self.bag_of_words = []
        self.vectorizer = DictVectorizer(dtype=int, sparse=True)
        self.encoder = LabelEncoder()
        self.lexicon_classifier = LexiconClassifier()
        self.classifier = LinearSVC(C=0.005)
        self.train(trainset)

    # Extract features for ML process
    # Some insights from http://aclweb.org/anthology/S/S13/S13-2053.pdf
    def extract_features(self, tweet_tokens):

        if len(self.bag_of_words) == 0:
            print('Bag-of-Words empty!')

        unigrams = [w.lower() for w,t in tweet_tokens]
        tokens = unigrams
        tokens += ['_'.join(b) for b in bigrams(unigrams)]
        tokens += ['_'.join(t) for t in trigrams(unigrams)]
        tokens += [t1 + '_*_' + t3 for t1,t2,t3 in trigrams(unigrams)]

        tweet_tags =  [tag for token, tag in tweet_tokens]

        feature_set = {}

        # 1st set of features: bag-of-words
        for token in set(tokens).intersection(self.bag_of_words):
            feature_set['has_'+token] = True

        # 2nd set of features: the count for each tag type present in the message
        # Tweet_nlp taget. Info:
        # http://www.ark.cs.cmu.edu/TweetNLP/annot_guidelines.pdf
        for tag in ['CC','CD','DT','EX','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WRB']:
            feature_set['num_'+tag] = sum([1 for t in tweet_tags if t == tag])

        # 3rd feature: negation is present?
        negators = set(LexiconClassifier().read_negation_words())
        if len(negators.intersection(set(tokens))) > 0:
            feature_set['has_negator'] = True

        # 4th feature: character ngrams
        regexp = re.compile(r"([a-z])\1{2,}")
        feature_set['has_char_ngrams'] = False
        for token,tag in tweet_tokens:
            if regexp.search(token):
                feature_set['has_char_ngrams'] = True
                break

        # 5th feature: punctuaion ngrams
        regexp = re.compile(r"([!\?])\1{2,}")
        feature_set['has_punct_ngrams'] = False
        for token,tag in tweet_tokens:
            if regexp.search(token):
                feature_set['has_punct_ngrams'] = True
                break

        # 6th feature: the number of all upper cased words
        feature_set['num_all_caps'] = sum([1 for token,tag in tweet_tokens if token.isupper() and len(token)>=3])

        # 7th and 8th feature: the positive and negative score from lexicon
        # classifier (i.e., number of positive and negative words from lexicon)
        positive_score, negative_score = self.lexicon_classifier.classify(tweet_tokens) 
        feature_set['pos_lexicon'] = positive_score
        feature_set['neg_lexicon'] = -1 * negative_score

        return feature_set


    # train the classifier
    # Tweets argument must be a list of dicitionaries. Each dictionary must
    # have the keys ['MESSAGE'] and ['SENTIMENT'] with the message string and
    # the classificationclass, respectively.
    def train(self,tweets):
        # 1st step: build the bag-of-words model
        tweet_tokens_list = [tweet_tokens for tweet_tokens,label in tweets]
        tokens = []
        print('Computing the trainset vocabulary of n-grams')
        for tweet_tokens in tweet_tokens_list:
            unigrams = [w.lower() for w,t in tweet_tokens]
            tokens += unigrams
            tokens += ['_'.join(b) for b in bigrams(unigrams)]
            tokens += ['_'.join(t) for t in trigrams(unigrams)]
            tokens += [t1 + '_*_' + t3 for t1,t2,t3 in trigrams(unigrams)]

        # build the bag-of-words list using all the tokens
        self.bag_of_words = set(tokens)

        data = list()
        total_tweets = len(tweets)
        features_list = list()
        for index,(tweet_tokens,label) in enumerate(tweets):
            print('Training for tweet n. {}/{}'.format(index+1,total_tweets))
            features_list.append(self.extract_features(tweet_tokens))

        # Train a SVM classifier
        #data = self.vectorizer.fit_transform([features for features,label in self.train_set_features])
        print('Vectorizing the features')
        data = self.vectorizer.fit_transform(features_list)
        target = self.encoder.fit_transform([label for tweet_tokens,label in tweets])
        print('Builing the model')
        self.classifier.fit(data, target)



    # classify a new message. Return the scores (probabilities) for each
    # classification class
    def classify(self, tweet_tokens):
        data = self.vectorizer.transform(self.extract_features(tweet_tokens))
        probs = self.classifier.decision_function(data)
        classes = self.encoder.classes_        
        return {classes.item(i): probs.item(i) for i in range(len(classes))}


    # return the probability of classification into one of the three classes
    def decision_function(self, tweet_tokens):
        data = self.vectorizer.transform(self.extract_features(tweet_tokens))
        probs = self.classifier.decision_function(data)
        classes = self.encoder.classes_  
        return {classes.item(i): probs.item(i) for i in range(len(classes))}








