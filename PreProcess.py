from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement

from subprocess import Popen, PIPE

try:
    # python 2.x
    from string import punctuation, letters
except:
    # python 3.x
    from string import punctuation, ascii_letters


import re
import tokenizer

import nltk

# Some elements from http://en.wikipedia.org/wiki/List_of_emoticons
emoticons = { ':-)'   : 'happy_abc',
              ':)'    : 'happy_abc',
              ':o)'   : 'happy_abc',
              ':]'    : 'happy_abc',
              ':3'    : 'happy_abc',
              ':c)'   : 'happy_abc',
              ':>'    : 'happy_abc',
              '=]'    : 'happy_abc',
              '8)'    : 'happy_abc',
              '=)'    : 'happy_abc',
              ':}'    : 'happy_abc',
              ':^)'   : 'happy_abc',
              ':-))'  : 'happy_abc',
              '|;-)'  : 'happy_abc',
              ":'-)"  : 'happy_abc',
              ":')"   : 'happy_abc',
              '\o/'   : 'happy_abc',
              '*\\0/*': 'happy_abc',
              ':-l'   : 'laugh_abc',
              ':l'    : 'laugh_abc',
              '8-l'   : 'laugh_abc',
              '8D'    : 'laugh_abc',
              ':D'    : 'laugh_abc',
              'x-l'   : 'laugh_abc',
              'xD'    : 'laugh_abc',
              'X-l'   : 'laugh_abc',
              'XD'    : 'laugh_abc',
              '=-l'   : 'laugh_abc',
              '=l'    : 'laugh_abc',
              '=-3'   : 'laugh_abc',
              '=3'    : 'laugh_abc',
              'B^l'   : 'laugh_abc',
              '>:['   : 'sad_abc',
              ':-('   : 'sad_abc',
              ':('    : 'sad_abc',
              ':-c'   : 'sad_abc',
              ':c'    : 'sad_abc',
              ':-<'   : 'sad_abc',
              ':<'    : 'sad_abc',
              ':-['   : 'sad_abc',
              ':['    : 'sad_abc',
              ':{'    : 'sad_abc',
              ':-||'  : 'sad_abc',
              ':@'    : 'sad_abc',
              ":'-("  : 'sad_abc',
              ":'("   : 'sad_abc',
              'l:<'   : 'sad_abc',
              'l:'    : 'sad_abc',
              'D8'    : 'sad_abc',
              'l;'    : 'sad_abc',
              'l='    : 'sad_abc',
              'DX'    : 'sad_abc',
              'v.v'   : 'sad_abc',
              "l-':"  : 'sad_abc',
              '(>_<)' : 'sad_abc',
              ':|'    : 'sad_abc',
              '>:O'   : 'surprise_abc',
              ':-O'   : 'surprise_abc',
              ':-o'   : 'surprise_abc',
              ':O'    : 'surprise_abc',
              ':O'    : 'surprise_abc',
              'o_O'   : 'surprise_abc',
              'o_0'   : 'surprise_abc',
              'o.O'   : 'surprise_abc',
              '8-0'   : 'surprise_abc',
              '|-O'   : 'surprise_abc',
              ';-)'   : 'wink_abc',
              ';)'    : 'wink_abc',
              '*-)'   : 'wink_abc',
              '*)'    : 'wink_abc',
              ';-]'   : 'wink_abc',
              ';]'    : 'wink_abc',
              ';l'    : 'wink_abc',
              ';^)'   : 'wink_abc',
              ':-,'   : 'wink_abc',
              '>:P'   : 'tong_abc',
              ':-P'   : 'tong_abc',
              ':P'    : 'tong_abc',
              'X-P'   : 'tong_abc',
              'x-p'   : 'tong_abc',
              'xp'    : 'tong_abc',
              'XP'    : 'tong_abc',
              ':-p'   : 'tong_abc',
              ':p'    : 'tong_abc',
              '=p'    : 'tong_abc',    
              ':-b'   : 'tong_abc',
              ':b'    : 'tong_abc',
              ':-&'   : 'tong_abc',
              ':&'    : 'tong_abc',
              '>:\\'  : 'annoyed_abc',
              '>:/'   : 'annoyed_abc',
              ':-/'   : 'annoyed_abc',
              ':-.'   : 'annoyed_abc',
              ':/'    : 'annoyed_abc',
              ':\\'   : 'annoyed_abc',
              '=/'    : 'annoyed_abc',
              '=\\'   : 'annoyed_abc',
              ':L'    : 'annoyed_abc',
              '=L'    : 'annoyed_abc',
              ':S'    : 'annoyed_abc',
              '>.<'   : 'annoyed_abc',
              ':-|'   : 'annoyed_abc',
              '<:-|'  : 'annoyed_abc',
              ':-X'   : 'seallips_abc',
              ':X'    : 'seallips_abc',
              ':-#'   : 'seallips_abc',
              ':#'    : 'seallips_abc',
              'O:-)'  : 'angel_abc',
              '0:-3'  : 'angel_abc',
              '0:3'   : 'angel_abc',
              '0:-)'  : 'angel_abc',
              '0:)'   : 'angel_abc',
              '0;^)'  : 'angel_abc',
              '>:)'   : 'devil_abc',
              '>;)'   : 'devil_abc',
              '>:-)'  : 'devil_abc',
              '}:-)'  : 'devil_abc',
              '}:)'   : 'devil_abc',
              '3:-)'  : 'devil_abc',
              '3:)'   : 'devil_abc',
              'o/\o'  : 'highfive_abc',
              '^5'    : 'highfive_abc',
              '>_>^'  : 'highfive_abc',
              '^<_<'  : 'highfive_abc',
              '<3'    : 'heart_abc'
          }




def pre_process(tweet_messages):
    count = 0
    for temp in tweet_messages:
      for i in temp:
        if (i==' '):
          count+=1
    count+=1
    ark_tweet_cmd = ['/home/akhil/Desktop/TwitterHybridClassifier/Tools/ark-tweet-nlp/runTagger.sh', '--input-format', 'text', '--output-format', 'conll', '--no-confidence', '--quiet']

    tweets = '\n'.join(tweet_messages)
    #print (len(tweets))
    tweets = tweets.encode("ascii","ignore")
    # Run the tagger and get the output
    p = Popen(ark_tweet_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,shell=True)
    (stdout, stderr) = p.communicate(input=tweets)
    stdout = stdout.decode('utf8','ignore')
    ark_tweet_output = stdout.strip() + '\n'
    #print (len(ark_tweet_output))

    # Check the return code.
    if p.returncode != 0:
        print ('Tools/ark-tweet-nlp/runTagger.sh command failed! Details: %s\n%s' % (stderr,ark_tweet_output))
        return None

    tweet_tokens_list = list()
    tweet_tokens = list()
    lines = ark_tweet_output.split('\n')
    
    for line in lines:
        values = re.split(r'[ \t]',line)
        values = [t for t in values if len(t) != 0]
        if len(values) == 0:
            tweet_tokens_list.append(tweet_tokens)
            tweet_tokens = list()
            continue
        try:
            for i in range(len(values)):
              tweet_tokens.append(values[i])
            
        except:
            print ('Error reading art tweet tagger output line: ' + line)
    
    #print (tweet_tokens_list)
    for tweet_tokens in tweet_tokens_list:
        l = len(tweet_tokens)-count
        l =  (int(l/3))
        #print (l)
        for i in range(l):
            token = tweet_tokens[i]
            
            
            tag = tweet_tokens[i+l]
            
            # substitute mentions
            if tag == '@':
                tweet_tokens[i] = ('&mention')

            # substitute urls
            if tag == 'U':
                tweet_tokens[i] = ('&url')

            # substitute emoticions
            if tag == 'E':
                tweet_tokens[i] = (emoticons.get(token,'_'))

# return the tweet in the format [(word,tag),...]
    yup=[]
    
    for twe in tweet_tokens_list:
      for i in range(l):

        k = twe[i]
       # print (k)
        k.encode('ascii', 'ignore')
        yup.append(k)
        yup.append(' ')

    str1=''.join(str(i) for i in yup)

    c=tokenizer.tokenize(str1)
    tweet_tokens=nltk.pos_tag(c)


    return tweet_tokens

   
