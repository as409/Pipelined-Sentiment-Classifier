# TwitterHybridClassifier

 - Authors: Akhil Sharma
 - Date: 16/05/17

These python scripts provide the TwitterClassifier I used for my B.tech final year project

## Dependencies :
- python 2.7 or a python 3.4 versions
- Natural Language Toolkit ( NLTK - http://www.nltk.org/ ) 
- scikit-learn ( Machine Learning in Python - http://scikit-learn.org/stable/ )
- ark_twitter_nlp Pos tagger ( http://www.cs.cmu.edu/~ark/TweetNLP/ )

I also used **ark_twitter_nlp Pos tagger** as well as the **NLTK pos tagger**. 

The lexicon this library uses are:

  - NRC Hashtag Sentiment Lexicon http://www.umiacs.umd.edu/~saif/WebPages/Abstracts/NRC-SentimentAnalysis.htm

  - Liu's Opinion Lexicon http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

## Running the code :
- To obtain the end output make an instance of the TwitterHybridClassifier() class providing your pos tagged and tokenzied dataset as arguments

## Output :
The final output of would be the sentiment(i.e. 'positive', 'negative' or 'neutral') predicted by the classifer along with the accuracy and confusion matrix.

Instead of using the NLTK tokenizer I have used the tokenizer built by Brendan O'Connor(http://github.com/brendano/tweetmotif)

The datasets I have used in my codes are of SemEval 2014 which I cannot provide because of legal reasons. You can contact the organisers of the SemEval for the datasets.



