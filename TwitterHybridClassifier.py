# Python 3 compatibility
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
from __future__ import unicode_literals

# Import the libraries created for this task
from RulesClassifier import RulesClassifier
from LexiconClassifier import LexiconClassifier
from ml1 import MachineLearningClassifier
from PreProcess import pre_process

import tokenizer
import nltk
import csv


# Import other libraries used
import pickle
import codecs
import os
import sys



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



class TwitterHybridClassifier(object):
    predictions=[]

    def __init__(self, tweets=[]):
        # initialize internal variables
        self.rules_classifier = RulesClassifier()
        self.lexicon_classifier = LexiconClassifier()
        self.ml_classifier = None

        # if the ML model has been generated, load the model from model.pkl
        if sys.version_info >= (3,0):
            if os.path.exists('model_python3.pkl'):
                print ('Reading the model from model_python3.pkl')
                self.ml_classifier = pickle.load(open('model_python3.pkl','rb'))
        else:
            if os.path.exists('model_python2.pkl'):
                print ('Reading the model from model_python2.pkl')
                self.ml_classifier = pickle.load(open('model_python2.pkl','rb'))

        if self.ml_classifier == None:
            # Preprocess the data and train a new model
            print ('Preprocessing the training data')
            tweet_messages = [tweet_message for tweet_message,label in tweets]
            tweet_labels = [label for tweet_message,label in tweets]

            # preproces all the tweet_messages (Tokenization, POS and normalization)
            tweet_tokens =pre_process(tweet_messages)
            
            # compile a trainset with tweek_tokens and labels (positive,
            # negative or neutral)

            trainset = [(tweet_tokens[i],tweet_labels[i]) for i in range(len(tweets))]

            # initialize the classifier and train it
            classifier = MachineLearningClassifier(trainset)

            # dump the model into de pickle
            python_version = sys.version_info[0]
            model_name = 'model_python' + str(python_version) + '.pkl'
            print ('Saving the trained model at ' + model_name)
            pickle.dump(classifier, open(model_name, 'wb'))
            self.ml_classifier = classifier

    # Apply the classifier over a tweet message in String format
    def classify(self,tweet_text):

        # 0. Pre-process the teets (tokenization, tagger, normalizations)
        tweet_tokens_list = []
        predictions= []

        print ('Preprocessing the string')
        # pre-process the tweets 

        tweet_tokens = pre_process([tweet_text])
        
        '''print(tweet_tokens_list)

            # 1. Rule-based classifier. Look for emoticons basically
        positive_score,negative_score = self.rules_classifier.classify(tweet_tokens)

            # 1. Apply the rules, If any found, classify the tweet here. If none found, continue for the lexicon classifier.
        if positive_score >= 1 and negative_score == 0:
            sentiment = ('positive','EB')
            predictions.append(sentiment)
            #continue
        elif positive_score == 0 and negative_score <= -1:
            sentiment = ('negative','EB')
            predictions.append(sentiment)
            #continue

            # 2. Lexicon-based classifier
        positive_score, negative_score = self.lexicon_classifier.classify(tweet_tokens)
        lexicon_score = positive_score + negative_score

            
        if positive_score >= 1 and negative_score == 0:
            sentiment = ('positive','LB')
            predictions.append(sentiment)
            #continue

        elif negative_score <= -2:
            sentiment = ('negative','LB')
            predictions.append(sentiment)
        #continue

            # 3. Machine learning based classifier - used the Train+Dev set sto define the best features to classify new instances
        result = self.ml_classifier.classify(tweet_tokens)
        positive_conf = result['positive']
        negative_conf = result['negative']
        neutral_conf = result['nuetral']

        if negative_conf >= -0.4:
            sentiment = ('negative','ML')
        elif positive_conf > neutral_conf:
            sentiment = ('positive','ML')
        else:
            sentiment = ('neutral','ML')

        predictions.append(sentiment)

        '''print("\n")
        print('individual scores are as follows : \n')


        print("emoticon classifier:")
        positive_score,negative_score = self.rules_classifier.classify(tweet_tokens)
        print('Positive: '+str(positive_score) + '\t' + 'Negative: '+str(negative_score) + '\t')



        print("\n")
        print('lexicon classifier:')
        positive_score, negative_score = self.lexicon_classifier.classify(tweet_tokens)
        lexicon_score = positive_score + negative_score
        print('Positive: '+str(positive_score) + '\t' +'Negative: '+ str(negative_score) + '\t')

        print("\n")
        print('machine learning classifier:')
        result = self.ml_classifier.decision_function(tweet_tokens)
        print('Positive: '+str(result['positive']) + '\t' +'Negative: '+ str(result['negative']) + '\t'+ ' Neutral: '+str(result['nuetral']) + '\t')

        print('\n Final output :')'''

            

        return predictions

    











'''text='john is an idiot and a little bit stupid'



myobjectq=TwitterHybridClassifier(trainset)

print(myobjectq.classify('john is an idiot and a little bit stupid :('))'''


