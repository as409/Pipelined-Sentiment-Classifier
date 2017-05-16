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
from TwitterHybridClassifier import TwitterHybridClassifier


from nltk import ConfusionMatrix
from pprint import pprint
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

tweet=tweets+tweets1
trainset=tweet



def confusion_matrix(gold,guess):
    correct = 0
    total = len(gold)
    for i in range(len(gold)):
        if guess[i] == gold[i]:
            correct += 1
    accuracy = float(correct) / float(total)
    print('Accuracy: {:.2%}'.format(accuracy))

    # Confusion Matrix
    cm = ConfusionMatrix(gold, guess)
    print(cm)





f=codecs.open('input.txt','r+',encoding='utf8')
lines=f.readlines()


f1=codecs.open('output.txt','r+',encoding='utf8')
lines1=f1.readlines()

Myobject=TwitterHybridClassifier(trainset)

#count = {'RB':0, 'LB':0, 'ML':0 }

observed = list()
answer = list()

for line in lines:
    
    x=line.split('\t')
    
    prediction=Myobject.classify(x[5])



    if (len(prediction)==1):
        result=prediction[0][0]
    elif (len(prediction)==2):
        result=prediction[1][0]
    else:
        result=prediction[2][0]

    
    #count[method] += 1
    observed.append(result)
    
    #print(prediction)
    
    

for line in lines1:
    line=line.strip()
    line = line.replace('\\\\', '').replace('\\"', '"').replace("\\'", "'").replace('\\u2019', '\'').replace('\\u002c', ',')
    x=line.split('\t')
    SENTIMENT=x[4]
    answer.append(SENTIMENT)

confusion_matrix(answer,observed)

'''print ('Statistics -  Number of instances processed by each method')
print ('Rule Based:       ',count['RB'])
print ('Lexicon Based:    ',count['LB'])
print ('Machine Learning: ',count['ML'])'''





