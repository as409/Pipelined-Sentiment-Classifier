from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement
from __future__ import unicode_literals

import nltk
import tokenizer


class RulesClassifier(object):

    def classify(self, tweet_tokens):

        positive_patterns = []
        negative_patterns = []

        # emoticons are substituted by codes in the pre-process step
        pos_patterns = ['happy_abc',
                        'laugh_abc',
                        'wink_abc',
                        'heart_abc',
                        'highfive_abc',
                        'angel_abc',
                        'tong_abc',
                       ]

        neg_patterns = ['sad_abc',
                        'annoyed_abc',
                        'seallips_abc',
                        'devil_abc',
                       ]

        # how many positive and negative emoticons are in the message?
        matches_pos = [token for token,tag in tweet_tokens if token in pos_patterns]
        matches_neg = [token for token,tag in tweet_tokens if token in neg_patterns]

        # return (positive_score , negative_score). Number of emoticons for
        # each sentiment
        return ( len(matches_pos),-1*len(matches_neg) )



